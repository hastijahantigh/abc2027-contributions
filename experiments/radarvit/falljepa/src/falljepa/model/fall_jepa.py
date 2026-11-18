import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregate import frames_mean, frames_to_steps
from .encoder import FrameEncoder
from .layers import Block
from .stem import TCN, RadarStem, contrast_from_context
from .radar_vit import RadarViT
from .recording_head import RecordingDecoder


class PreProjector(nn.Module):
    """MLP with BatchNorm mapping encoder outputs into the space SIGReg shapes (from gir's htr_jepa.py)."""
    def __init__(self, d_model: int, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FallJEPA(nn.Module):
    """
    gir's frame model, read along time instead of along a line. A window of the stream (20 s,
    read with `(w - s) / 2` timesteps of margin on each side) is a video of overlapping frames,
    `frame_width` timesteps wide and moving `frame_stride` timesteps at a time:

        z_i       = Ev(frame_i)          one frame at a time, isolated (gir's FrameEncoder)  -> SIGReg
        h_i       = Cv(z_<=i)            causal context over the frames
        r_i       = Bv(h_1:F)            bidirectional layers (gir w2/w3: they help recognition)
        y_i       = head(r_i)            the state (none / begin / middle / end) of each of the
                                         w timesteps frame i covers                          -> CE
        ẑ_{i+k}   = Hv(h_<=i)            the latent of a later frame (JEPA, no action: there is
                                         no text here)                                       -> MSE

    The frame predictions of every timestep are averaged (Hann over the position in the frame)
    into one prediction per timestep (`aggregate.frames_to_steps`), also trained with CE.

    `regression` adds a boundary head (ActionFormer): each frame predicts, for each of its w
    timesteps, the distances in seconds to the start and the end of the fall that timestep is in.
    They are averaged over frames like the logits (step_bounds [B, L, 2]) and trained with a 1D
    DIoU loss on the timesteps inside falls.

    From gir's ablations (reports/ablation_w1.html, w2.html): MSE on an attached visual target
    squeezed z (effective rank 17 of 128) and cost CER; detaching the target brought the rank back.
    So `detach_target` defaults to True, and the default horizon is w / s frames (the next frame
    that shares no timestep with this one), since a horizon of 1 here can be met by copying.

    `physics_channels` > 0 (Amir's physics features, data/physics.py): each frame's features
    phys [P, w] are averaged over the columns of its patches and projected to d_model, and that
    vector is added to every token of the frame (FrameEncoder's frame_embed), as Amir adds them to
    his tokens.

    `recording_head` adds the second stage (model/recording_head.py, docs/falljepa_recording_head.md):
    on every unit (source recording) the batch holds, P(fall in the unit) and a start and an end
    distribution over its timesteps, from the bidirectional tokens r and the units' labeller channels.

    `trunk` "stem" (JF4, docs/falljepa_JF4_design.md): a convolutional stem over the whole window, per
    radar (model/stem.py: complex L0 + modulus, range pooling, `radar_fusion` "early" or "attention")
    replaces the patch embedding and the frame transformer; its token i covers frame i's steps, so z,
    the heads and the JEPA horizon keep their meaning. It needs the complex layout without amplitude
    channels (data.amplitude false) and `contrast` maps per radar. `local` "tcn" puts a dilated TCN
    between z and the context blocks (no JEPA then: it sees the future); `rope` uses rotary positions in
    the context and bidirectional blocks instead of the learned frame position embedding (kept for the
    fall queries' keys).

    Inputs: x [B, C, n_bins, L + w - s] (C = radars, their differences and the valid channel),
    valid [B, L + w - s], and with physics phys [B, P, L + w - s]. Rows are range bins, columns
    are timesteps.
    """
    def __init__(
        self,
        in_channels: int = 7,
        n_bins: int = 368,
        window_frames: int = 125,
        d_model: int = 128,
        nhead: int = 4,
        # Framing (in timesteps of 20 ms)
        frame_width: int = 32,
        frame_stride: int = 8,
        frame_patch_size: tuple[int, int] = (23, 8),
        frame_pooling: str = "cls",
        # Depths
        frame_encoder_layers: int = 4,
        context_layers: int = 2,
        bidir_layers: int = 1,
        predictor_layers: int = 2,
        # Prediction
        pred_horizons: tuple[int, ...] = (4,),
        detach_target: bool = True,
        pre_projector: bool = True,
        num_states: int = 4,
        regression: bool = False,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
        pre_proj_ratio: int = 4,
        physics_channels: int = 0,
        recording_head: bool = False,
        unit_channels: int = 11,
        unit_scalars: int = 10,
        unit_layers: int = 2,
        unit_max_frames: int = 340,
        trunk: str = "frames",
        radar_fusion: str = "attention",
        stem_channels: tuple[int, ...] = (16, 32, 48, 64),
        range_heads: int = 4,
        contrast_channels: int = 0,
        local: str = "none",
        tcn_dilations: tuple[int, ...] = (1, 2, 4, 8),
        rope: bool = False,
        contrast_scales: list | None = None,
        range_attention: bool = False,
    ):
        super().__init__()
        assert frame_width >= frame_stride and (frame_width - frame_stride) % 2 == 0
        assert n_bins % frame_patch_size[0] == 0, "n_bins must be a multiple of the patch height"
        self.frame_width = frame_width
        self.frame_stride = frame_stride
        self.window_frames = window_frames
        self.num_states = num_states
        # No predictor layers or no horizon: no JEPA branch at all (no Hv, no z_v_pred/target/copy)
        self.pred_horizons = tuple(pred_horizons) if predictor_layers > 0 else ()
        self.detach_target = detach_target

        assert trunk in ("frames", "stem", "radarvit") and local in ("none", "tcn"), (trunk, local)
        if range_attention and trunk != "stem":
            raise ValueError("range_attention requires the radar stem")
        assert local == "none" or not pred_horizons or predictor_layers == 0, "a centred TCN leaks into JEPA"
        self.trunk = trunk
        if trunk == "frames":
            self.frame_encoder = FrameEncoder(
                dim=d_model, depth=frame_encoder_layers, num_heads=nhead, in_channels=in_channels,
                patch_size=tuple(frame_patch_size), frame_width=frame_width,
                max_rows=n_bins // frame_patch_size[0], dropout=dropout, pooling=frame_pooling, mlp_ratio=mlp_ratio,
            )
        elif trunk == "radarvit":
            if in_channels != 13 or contrast_channels:
                raise ValueError("RadarViT expects the 13-channel complex radar layout")
            self.stem = RadarViT(d_model=d_model, nhead=nhead, n_bins=n_bins, frame_width=frame_width,
                                 frame_stride=frame_stride, mlp_ratio=mlp_ratio, dropout=dropout)
        else:
            assert in_channels == 13 + 3 * contrast_channels, \
                f"the stem reads [I, Q, I_c, Q_c] x 3 + {contrast_channels} contrast maps x 3 + valid, got {in_channels}"
            self.stem = RadarStem(d_model, nhead, tuple(stem_channels), contrast_channels, radar_fusion, range_heads,
                                  n_bins, frame_width, frame_stride, mlp_ratio, dropout, range_attention)
        self.tcn = TCN(d_model, tuple(tcn_dilations), dropout) if local == "tcn" else None
        # (gap, width) in timesteps, for contrast computed here (data.contrast_in_model: `ctx` in the batch)
        self.contrast_steps = [(int(round(g * 50)), int(round(w * 50))) for g, w in (contrast_scales or [])]
        self.rope = rope
        self.patch_cols = frame_width // frame_patch_size[1]
        self.physics_embed = nn.Linear(physics_channels * self.patch_cols, d_model) if physics_channels else None
        self.pre_proj = PreProjector(d_model, pre_proj_ratio * d_model) if pre_projector else nn.Identity()
        self.frame_pos_embed = nn.Parameter(torch.randn(window_frames, d_model) * 0.02)
        self.context = nn.ModuleList([Block(d_model, nhead, mlp_ratio, dropout=dropout, rope=rope) for _ in range(context_layers)])
        self.context_norm = nn.LayerNorm(d_model)
        self.bidir = nn.ModuleList([Block(d_model, nhead, mlp_ratio, dropout=dropout, rope=rope) for _ in range(bidir_layers)])
        self.bidir_norm = nn.LayerNorm(d_model) if bidir_layers > 0 else nn.Identity()

        if self.pred_horizons:
            self.predictor = nn.ModuleList([Block(d_model, nhead, mlp_ratio, dropout=dropout) for _ in range(predictor_layers)])
            self.predictor_norm = nn.LayerNorm(d_model)
            self.horizon_embed = nn.Parameter(torch.zeros(len(self.pred_horizons), d_model))

        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, frame_width * num_states),
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, frame_width * 2),
        ) if regression else None
        self.recording = RecordingDecoder(
            d_model, nhead, frame_width, frame_stride, unit_channels, unit_scalars, unit_layers, mlp_ratio,
            dropout, unit_max_frames,
        ) if recording_head else None

    # ------------------------------------------------------------------ parts

    def extract_frames(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, L + w - s] -> [B, F, C, H, w], frame i over columns [i s, i s + w)."""
        frames = x.unfold(-1, self.frame_width, self.frame_stride)       # [B, C, H, F, w]
        return frames.permute(0, 3, 1, 2, 4)

    def encode(self, x: torch.Tensor, valid: torch.Tensor, phys: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        B = x.size(0)
        Fn = (x.size(-1) - self.frame_width) // self.frame_stride + 1
        assert Fn <= self.window_frames, f"{Fn} frames, the model was built for {self.window_frames}"
        frame_embed = None
        if self.physics_embed is not None:
            ph = phys.unfold(-1, self.frame_width, self.frame_stride)                 # [B, P, F, w]
            ph = ph.reshape(*ph.shape[:3], self.patch_cols, -1).mean(-1)              # per patch column
            frame_embed = self.physics_embed(ph.permute(0, 2, 1, 3).reshape(B * Fn, -1))
        if self.trunk == "frames":
            frames = self.extract_frames(x)
            z = self.frame_encoder(frames.reshape(B * Fn, *frames.shape[2:]), frame_embed)
        else:
            z = self.stem(x, Fn).reshape(B * Fn, -1)
            if frame_embed is not None:
                z = z + frame_embed
        z = self.pre_proj(z).view(B, Fn, -1)

        # A frame with no real timestep (past the stream end) is never attended to
        frame_valid = valid.unfold(-1, self.frame_width, self.frame_stride).amax(-1) > 0   # [B, F]
        h = z if self.rope else z + self.frame_pos_embed[:Fn]
        if self.tcn is not None:
            h = self.tcn(h, frame_valid)
        for blk in self.context:
            h = blk(h, causal=True, key_valid=frame_valid)
        h = self.context_norm(h)
        r = h
        for blk in self.bidir:
            r = blk(r, key_valid=frame_valid)
        r = self.bidir_norm(r)
        return {"z": z, "h": h, "r": r, "frame_valid": frame_valid}

    def predict_latents(self, h: torch.Tensor, frame_valid: torch.Tensor, hi: int) -> torch.Tensor:
        p = h + self.horizon_embed[hi]
        for blk in self.predictor:
            p = blk(p, causal=True, key_valid=frame_valid)
        return self.predictor_norm(p)

    def with_contrast(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """With `ctx` in the batch (data.contrast_in_model): the contrast channels computed from it and put
        in x after the 12 signal channels, where the data loader would have put them."""
        if "ctx" not in batch:
            return batch
        assert self.contrast_steps, "ctx in the batch but no contrast_scales"
        d = contrast_from_context(batch["ctx"], batch["ctx_valid"], self.contrast_steps)
        x = batch["x"]
        out = {k: v for k, v in batch.items() if k not in ("ctx", "ctx_valid")}
        out["x"] = torch.cat([x[:, :12], d.to(x.dtype), x[:, 12:]], 1)
        return out

    # ---------------------------------------------------------------- forward

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self.with_contrast(batch)
        x, valid = batch["x"], batch["valid"]
        enc = self.encode(x, valid, batch.get("phys"))
        z, h, r, frame_valid = enc["z"], enc["h"], enc["r"], enc["frame_valid"]
        B, Fn, _ = z.shape
        w, s, K = self.frame_width, self.frame_stride, self.num_states
        margin = (w - s) // 2

        frame_logits = self.state_head(r).view(B, Fn, w, K)
        step_logits, spread = frames_to_steps(frame_logits, s)             # [B, L + w - s, K]
        step_logits = step_logits[:, margin:step_logits.size(1) - margin]  # the window's own timesteps
        spread = spread[:, margin:spread.size(1) - margin]

        # Frames the self-supervised losses may use: real, and (cross-validation) wholly outside the
        # held-out fold
        ssl = frame_valid
        if "fit" in batch:
            ssl = ssl & (batch["fit"].unfold(-1, w, s).amin(-1) > 0)

        # JEPA: frame i predicts the latent of frame i + k
        z_pred, z_target, z_copy = [], [], []
        for hi, k in enumerate(self.pred_horizons):
            if k >= Fn:
                continue
            pred = self.predict_latents(h, frame_valid, hi)
            ok = ssl[:, :-k] & ssl[:, k:]
            z_pred.append(pred[:, :-k][ok])
            z_target.append(z[:, k:][ok])
            z_copy.append(z[:, :-k][ok].detach())

        output = {
            "z_v_proj": z[ssl],
            "frame_logits": frame_logits.reshape(-1, K),
            "step_logits": step_logits,
            "step_spread": spread,
            "tokens": r,
            "frame_valid": frame_valid,
        }
        if z_pred:
            z_target = torch.cat(z_target)
            output["z_v_pred"] = torch.cat(z_pred)
            output["z_v_target"] = z_target.detach() if self.detach_target else z_target
            output["z_v_copy"] = torch.cat(z_copy)
        if self.boundary_head is not None:
            # Distances (seconds) to the fall's start and end, >= 0, averaged over frames like the logits
            frame_bounds = F.softplus(self.boundary_head(r).float()).view(B, Fn, w, 2)
            step_bounds = frames_mean(frame_bounds, s)
            output["step_bounds"] = step_bounds[:, margin:step_bounds.size(1) - margin]
            if "bounds" in batch:
                output["bound_pred"] = output["step_bounds"].reshape(-1, 2)
                output["bound_target"] = batch["bounds"][:, margin:batch["bounds"].size(1) - margin].reshape(-1, 2)
                output["bound_weight"] = batch["bound_weight"][:, margin:batch["bound_weight"].size(1) - margin].reshape(-1)
        if self.recording is not None and "units" in batch and bool(batch["unit_mask"].any()):
            rec = self.recording(r, frame_valid, batch["units"], batch["unit_mask"], batch["unit_scalars"], batch["uphys"])
            output.update(rec)
            rows, slots, origin = rec["unit_rows"], rec["unit_slots"], rec["span_origin"]
            if "unit_fall" in batch:
                output["unit_fall"] = batch["unit_fall"][rows, slots]
                output["unit_fall_w"] = batch["unit_fall_w"][rows, slots]
                output["span_target"] = batch["unit_span"][rows, slots] - origin[:, None]
                output["span_w"] = batch["unit_span_w"][rows, slots]
                output["rule_target"] = batch["unit_rule"][rows, slots] - origin[:, None]
                output["rule_w"] = batch["unit_rule_w"][rows, slots]
        if "state" in batch:
            state, weight = batch["state"], batch["weight"]
            idx = (torch.arange(Fn, device=x.device)[:, None] * s + torch.arange(w, device=x.device)[None, :])
            output["frame_state"] = state[:, idx].reshape(-1)
            output["frame_weight"] = weight[:, idx].reshape(-1)
            output["step_logits_flat"] = step_logits.reshape(-1, K)
            output["step_state"] = state[:, margin:state.size(1) - margin].reshape(-1)
            output["step_weight"] = weight[:, margin:weight.size(1) - margin].reshape(-1)
        return output

    @torch.no_grad()
    def window_log_probs(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-timestep state log-probabilities [B, L, K] and frame disagreement [B, L] of each window."""
        out = self.forward({k: batch[k] for k in ("x", "valid", "phys", "ctx", "ctx_valid") if k in batch})
        return F.log_softmax(out["step_logits"].float(), dim=-1), out["step_spread"]
