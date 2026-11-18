import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import TrainerFn

from .loss import BaseLoss
from .model.aggregate import NONE, StreamAccumulator, UnitAccumulator, decode_events, decode_units, mbr_span
from .model import sets
from .official import evaluate as official
from .optimizer import BaseOptimizer
from . import postprocess as amir


# load_from_checkpoint rebuilds `model` from the saved config and passes the module in again
warnings.filterwarnings("ignore", message="Attribute 'model' is an instance of `nn.Module`")


def in_mask(mask, t_start: float, t_end: float) -> bool:
    """Whether the midpoint of [t_start, t_end) lies in a timestep the mask keeps."""
    i = min(int((t_start + t_end) / 2 * 50.0), len(mask) - 1)
    return bool(mask[i])


def score_predictions(predictions: list[dict], ground: list[dict], lengths: dict[str, int],
                      masks: dict | None = None) -> dict:
    """
    The official report for these streams. With masks (cross-validation), only predictions and
    ground-truth events whose midpoint lies in a kept timestep count, and false alarms per hour
    are over the kept duration.
    """
    ground = [e for e in ground if e["stream_id"] in lengths]
    records = [{k: p[k] for k in ("stream_id", "t_start", "t_end", "score", "t_alarm")} for p in predictions]
    if masks is None:
        durations = {sid: T / 50.0 for sid, T in lengths.items()}
    else:
        durations = {sid: float(masks[sid].sum()) / 50.0 for sid in lengths}
        ground = [e for e in ground if in_mask(masks[e["stream_id"]], e["t_start"], e["t_end"])]
        records = [r for r in records if in_mask(masks[r["stream_id"]], r["t_start"], r["t_end"])]
    return official.evaluate(ground, records, durations)


def effective_rank(z: torch.Tensor) -> torch.Tensor:
    """exp(entropy) of the normalized singular values of the centred latents (d_model when isotropic)."""
    s = torch.linalg.svdvals((z - z.mean(0)).float())
    p = s / s.sum().clamp_min(1e-12)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())


class FallModule(LightningModule):
    """
    Trains the frame model on windows and evaluates it on whole streams, the way it is scored.
    Adapted from gir's HTRModule: the losses are the same named, weighted terms read from the
    model's output dict; CER decoding is replaced by stream decoding and the official evaluator.

    losses / loss_weights:   named loss terms and their weights (missing names weigh 1.0; 0 still
                             computes and logs the term)
    decoder:                 arguments of `aggregate.decode_events` (kind, switch_penalty, ...), or
                             {kind: amir, theta, gate, per_recording}: Amir's post-processing
                             (postprocess.py) on P(fall) and the raw CIR of each stream; or
                             {kind: units, threshold, mbr, also_amir, theta, gate}: the recording
                             head, one event per unit with P(fall) > threshold (aggregate.decode_units);
                             also_amir additionally decodes the dense head with Amir's chain, written
                             as <name>.amir.csv and logged as <stage>/amir/*, for comparison; or
                             {kind: sets, threshold, proposal_threshold, group_iou, nms_iou, batch}: the
                             join-free model (model/fall_set.py), its stream stage on the groups of every
                             stream's proposals, one event per slot with P(fall) > threshold, MBR span,
                             duplicates removed at tIoU nms_iou; it also writes the ladder <name>.queries.csv
                             (stage 1, non-learned merge) and <name>.dense.csv (the dense head, Viterbi),
                             logged as <stage>/queries/* and <stage>/dense/*
    window_floor:            the floor of the Hann weight that merges overlapping windows
    window_weight / window_merge: "hann" or "triangular" weights; mean of "logp" or of "prob"
                             (Amir: triangular, prob)
    decode_every_n_epochs:   whole-stream decoding and scoring during fit (validate/test: always)
    ignore_flagged:          score like the official test (flagged falls are ignore zones)
    sam:                     {rho, clip}: sharpness-aware minimization (Foret et al. 2021) with
                             manual optimization: the gradient is taken again at the weights moved
                             by rho along the normalized first gradient, then clipped to `clip`
                             (norm) and stepped from the original weights. Set the trainer's
                             gradient_clip_val to null with it (Lightning cannot clip manual steps)

    Validation/test write `<run dir>/predictions/<stage>-....csv`, a valid submission, and log
    `<stage>/event/F1_mean` (the ranking score), F1 at each theta, the median start error and
    false alarms per hour. Streams without ground truth (the test set) are only written.
    """
    def __init__(
        self,
        model: nn.Module,
        losses: dict[str, BaseLoss],
        optimizer: BaseOptimizer,
        loss_weights: dict[str, float] | None = None,
        decoder: dict | None = None,
        window_floor: float = 1e-3,
        decode_every_n_epochs: int = 1,
        ignore_flagged: bool = True,
        window_weight: str = "hann",
        window_merge: str = "logp",
        sam: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        loss_weights = loss_weights or {}
        unknown = set(loss_weights) - set(losses)
        if unknown:
            raise ValueError(f"loss_weights names {sorted(unknown)} are not losses: {sorted(losses)}")
        self.model = model
        self.losses = nn.ModuleDict(losses)
        self.loss_weights = {name: float(loss_weights.get(name, 1.0)) for name in losses}
        self.optimizer_setup = optimizer
        self.decoder = decoder or {}
        self.window_floor = window_floor
        self.decode_every_n_epochs = decode_every_n_epochs
        self.ignore_flagged = ignore_flagged
        self.window_weight, self.window_merge = window_weight, window_merge
        self.sam = dict(sam or {})
        if float(self.sam.get("rho", 0.0)) > 0:
            self.automatic_optimization = False
        self._acc: StreamAccumulator | None = None
        self._uacc: UnitAccumulator | None = None
        self._uacc_done: UnitAccumulator | None = None
        self._joins: amir.StreamJoins | None = None
        self._tacc: sets.TokenAccumulator | None = None
        self._pacc: sets.ProposalAccumulator | None = None
        self._sets_done: tuple | None = None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(batch)

    def configure_optimizers(self):
        return self.optimizer_setup.setup(self, self.trainer.estimated_stepping_batches)

    def on_train_epoch_start(self) -> None:
        # New random window starts every epoch (the workers get a fresh copy of the dataset)
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None and "train" in getattr(datamodule, "datasets", {}):
            datamodule.datasets["train"].set_epoch(self.current_epoch)

    # ------------------------------------------------------------------ steps

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> tuple[torch.Tensor, dict]:
        output = self.model(batch)
        terms = {name: loss(output) for name, loss in self.losses.items()}
        total = sum(self.loss_weights[name] * value for name, value in terms.items())

        batch_size = batch["x"].size(0)
        on_step = stage == "train"
        logs = {f"{stage}/loss/{name}": value.detach() for name, value in terms.items()}
        logs[f"{stage}/loss/Total"] = total.detach()
        with torch.no_grad():
            if "z_v_pred" in output:  # absent without a latent predictor (predictor_layers 0)
                # < 1: the latent predictor beats copying the frame it starts from (gir's VisualCopyRatio)
                pred_err = (output["z_v_pred"].float() - output["z_v_target"].float()).square().mean()
                copy_err = (output["z_v_copy"].float() - output["z_v_target"].float()).square().mean()
                logs[f"{stage}/metric/CopyRatio"] = pred_err / copy_err.clamp_min(1e-12)
            if stage != "train":
                logs[f"{stage}/metric/EffRank_z"] = effective_rank(output["z_v_proj"])
        self.log_dict(logs, on_step=on_step, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log(f"{stage}_loss", total.detach(), on_step=on_step, on_epoch=not on_step, prog_bar=True,
                 logger=False, batch_size=batch_size)
        return total, output

    def training_step(self, batch, batch_idx):
        if not self.automatic_optimization:
            return self._sam_step(batch)
        total, _ = self._shared_step(batch, "train")
        return total

    def _sam_step(self, batch) -> torch.Tensor:
        """One SAM step: ascend to w + rho g / |g|, take the gradient there, step from w."""
        opt = self.optimizers()
        rho, clip = float(self.sam["rho"]), self.sam.get("clip", 1.0)
        total, _ = self._shared_step(batch, "train")
        opt.zero_grad()
        self.manual_backward(total)
        params = [p for g in opt.optimizer.param_groups for p in g["params"] if p.grad is not None]
        norm = torch.norm(torch.stack([p.grad.detach().norm(2) for p in params]), 2)
        scale = rho / (norm + 1e-12)
        moves = []
        with torch.no_grad():
            for p in params:
                e = p.grad * scale
                p.add_(e)
                moves.append(e)
        opt.zero_grad()
        output = self.model(batch)
        sharp = sum(self.loss_weights[name] * loss(output) for name, loss in self.losses.items())
        self.manual_backward(sharp)
        with torch.no_grad():
            for p, e in zip(params, moves):
                p.sub_(e)
        if clip:
            self.clip_gradients(opt, gradient_clip_val=float(clip), gradient_clip_algorithm="norm")
        opt.step()
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
        self.log_dict({"train/metric/SAM_grad_norm": norm.detach(), "train/loss/SAM_sharp": sharp.detach()},
                      on_step=True, on_epoch=True, batch_size=batch["x"].size(0))
        return total.detach()

    def _eval_step(self, batch, stage: str) -> None:
        if "state" in batch:
            _, output = self._shared_step(batch, stage)
        else:
            output = self.model(batch)
        if not self._decoding_epoch():
            return
        logp = torch.log_softmax(output["step_logits"].float(), dim=-1).cpu().double().numpy()
        spread = output["step_spread"].float().cpu().double().numpy()
        bounds = None
        if "step_bounds" in output:  # distances -> absolute [start, end] in stream seconds
            L = logp.shape[1]
            t = (batch["start"].cpu().double()[:, None] + torch.arange(L, dtype=torch.float64)[None, :]) / 50.0
            d = output["step_bounds"].float().cpu().double()
            bounds = torch.stack([t - d[..., 0], t + d[..., 1]], dim=-1).numpy()
        store = self._dataset(stage).store
        if self._acc is None:
            self._acc = StreamAccumulator(store.lengths, logp.shape[1], logp.shape[2], self.window_floor,
                                          self.window_weight, self.window_merge)
        for i in range(logp.shape[0]):
            sid = store.ids[int(batch["stream"][i])]
            self._acc.add(sid, int(batch["start"][i]), logp[i], spread[i], None if bounds is None else bounds[i])
        if "unit_logit" in output:
            self._add_units(batch, output, store, self._dataset(stage).window)
        if "q_logits" in output:
            self._add_sets(batch, output, store)

    def _add_sets(self, batch, output, store) -> None:
        """Every window's tokens (onto the stream's token grid) and its proposals."""
        tokens = output["tokens"].float().cpu().numpy()
        frame_valid = output["frame_valid"].cpu().numpy()
        if self._tacc is None:
            self._tacc = sets.TokenAccumulator(store.lengths, self.model.frame_stride, tokens.shape[-1])
            self._pacc = sets.ProposalAccumulator()
        valid = output["q_step_valid"]
        logp = torch.log_softmax(output["q_spans"].float().masked_fill(~valid[:, None, :, None], -torch.inf), dim=2)
        logp, valid = logp.cpu().numpy(), valid.cpu().numpy()
        logits = output["q_logits"].float().cpu().numpy()
        hidden = output["q_hidden"].float().cpu().numpy()
        threshold = float(self.decoder.get("proposal_threshold", 0.05))
        for i in range(len(tokens)):
            sid, start = store.ids[int(batch["stream"][i])], int(batch["start"][i])
            self._tacc.add(sid, start, tokens[i], frame_valid[i])
            rows = sets.proposals(logits[i], logp[i], hidden[i], start, logp.shape[2], threshold, valid[i])
            for r in rows:                   # past the stream end (a padded last window)
                r["end"] = min(r["end"], store.lengths[sid])
            self._pacc.add(sid, [r for r in rows if r["start"] < r["end"]])

    def _add_units(self, batch, output, store, window: int) -> None:
        """Every unit's fall logit and start / end log-probabilities (over its domain, stream timesteps)."""
        if self._uacc is None:
            self._uacc = UnitAccumulator()
        logits = output["unit_logit"].float().cpu().numpy()
        span = output["span_logits"].float()
        domain = output["span_domain"]
        logp = torch.log_softmax(span.masked_fill(~domain[..., None], -torch.inf), dim=1).cpu().numpy()
        domain = domain.cpu().numpy()
        rows, slots = output["unit_rows"].cpu().numpy(), output["unit_slots"].cpu().numpy()
        origin = output["span_origin"].cpu().numpy()
        units = batch["units"].cpu().numpy()
        for n, (b, k) in enumerate(zip(rows, slots)):
            sid = store.ids[int(batch["stream"][b])]
            start = int(batch["start"][b])
            lo, hi = units[b, k]
            centrality = int(min(lo, window - hi))
            local = np.flatnonzero(domain[n])
            self._uacc.add(sid, int(batch["unit_index"][b, k]), centrality, float(logits[n]),
                           start + origin[n] + local, logp[n, local, 0], logp[n, local, 1])

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        self._finish("val")

    def on_test_epoch_end(self) -> None:
        self._finish("test")

    # --------------------------------------------------------------- decoding

    def _dataset(self, stage: str):
        return self.trainer.datamodule.datasets[stage]

    def _decoding_epoch(self) -> bool:
        fn = self.trainer.state.fn
        if fn in (TrainerFn.VALIDATING, TrainerFn.TESTING):
            return True
        if self.trainer.sanity_checking:
            return False
        n = self.decode_every_n_epochs
        return n > 0 and (self.current_epoch + 1) % n == 0

    def decode_streams(self, acc: StreamAccumulator, stage: str = "val") -> list[dict]:
        if self.decoder.get("kind") == "amir":
            return self._decode_amir(acc, stage)
        if self.decoder.get("kind") == "units":
            return self._decode_units(acc, stage)
        if self.decoder.get("kind") == "sets":
            return self._decode_sets(acc, stage)
        predictions = []
        for sid in acc.lengths:
            logp, spread = acc.result(sid)
            for event in decode_events(logp, spread, bounds=acc.result_bounds(sid), **self.decoder):
                predictions.append({"stream_id": sid, **event})
        return predictions

    def _decode_units(self, acc: StreamAccumulator, stage: str) -> list[dict]:
        """The recording head: one event per unit with P(fall) > threshold, its span by MBR."""
        uacc = self._uacc_done or UnitAccumulator()
        threshold, mbr = float(self.decoder.get("threshold", 0.5)), bool(self.decoder.get("mbr", True))
        predictions = []
        for sid, T in acc.lengths.items():
            for event in decode_units(uacc, sid, T, threshold, mbr=mbr):
                event.pop("unit")
                predictions.append({"stream_id": sid, **event})
        return predictions

    def _decode_sets(self, acc: StreamAccumulator, stage: str) -> list[dict]:
        """The join-free model's stream stage: group every stream's proposals, run the refiner on the
        groups (their region's stream tokens and signal), one event per slot with P > threshold."""
        tacc, pacc = self._sets_done or (None, None)
        if tacc is None:
            return []
        model, store = self.model, self._dataset(stage).store
        threshold = float(self.decoder.get("threshold", 0.5))
        iou, nms_iou = float(self.decoder.get("group_iou", 0.2)), float(self.decoder.get("nms_iou", 0.5))
        centre = self.decoder.get("centre_s")
        size = int(self.decoder.get("batch", 64))
        alpha = float(self.decoder.get("quality_alpha", 1.0))
        refine = bool(self.decoder.get("boundary", True))   # False: keep the MBR spans (ablates JF7b's refiner)
        s, g = model.frame_stride, model.signal_groups
        predictions, self._slots = [], {}
        for sid, T in acc.lengths.items():
            tokens, _ = tacc.result(sid)
            pending, meta, events, slots = [], [], [], []

            def flush():
                if not pending:
                    return
                res = model.refine(pending)
                logit, logp = res["logit"].cpu().numpy(), res["logp"].cpu().numpy()
                valid = res["step_valid"].cpu().numpy()
                # JF7: the event score is P x predicted tIoU ^ alpha, so NMS keeps the better-timed slot
                quality = 1.0 / (1.0 + np.exp(-res["quality"].cpu().numpy())) if "quality" in res else None
                found = []
                for i, (gi, lo) in enumerate(meta):
                    keep = np.flatnonzero(valid[i])
                    for q in range(logit.shape[1]):
                        p = 1.0 / (1.0 + np.exp(-logit[i, q]))
                        if p <= 0.05:
                            continue
                        a, b = mbr_span(lo + keep, logp[i, q, keep, 0], logp[i, q, keep, 1], tie_iou=sets.TIE_IOU)
                        found.append((i, q, gi, lo, p, a, b))
                spans = [(i, q, a - lo, b - lo) for i, q, gi, lo, p, a, b in found]
                moved = model.refine_bounds(res, spans) if refine else [(a, b) for _, _, a, b in spans]
                for (i, q, gi, lo, p, _, _), (a, b) in zip(found, moved):
                    a, b = a + lo, b + lo
                    slots.append((gi, q, logit[i, q], a, b))
                    score = p if quality is None else p * quality[i, q] ** alpha
                    e = sets.event(a, b, score, T) if p > threshold else None
                    if e is not None:
                        events.append(e)
                pending.clear()
                meta.clear()

            for gi, (members, span) in enumerate(sets.stream_groups(pacc.get(sid), iou, model.group_gap)):
                starts = np.array([r["start"] for r in members])
                ends = np.array([r["end"] for r in members])
                j0, j1 = sets.group_region(starts, ends, span, model.group_margin, s, model.group_max_tokens, len(tokens))
                lo, hi = j0 * s, j1 * s
                x, valid = store.read(sid, lo, hi)
                p = 1.0 / (1.0 + np.exp(-np.array([r["logit"] for r in members])))
                pending.append(sets.group_item(
                    torch.from_numpy(tokens[j0:j1]).to(self.device), torch.from_numpy(x[:3 * g]).to(self.device),
                    torch.from_numpy(valid > 0).to(self.device), (span[0] - lo, span[1] - lo),
                    torch.from_numpy(np.stack([r["hidden"] for r in members])).to(self.device),
                    sets.member_features(members, lo, hi - lo, model.window), sets.group_stats(p, span)))
                meta.append((gi, lo))
                if len(pending) >= size:
                    flush()
            flush()
            self._slots[sid] = np.array(slots, dtype=np.float64).reshape(-1, 5)
            predictions.extend({"stream_id": sid, **e} for e in sets.nms(events, nms_iou, centre))
        return predictions

    def _decode_queries(self, acc: StreamAccumulator) -> list[dict]:
        """Stage 1 alone, with the non-learned merge (sets.decode_queries)."""
        _, pacc = self._sets_done or (None, None)
        threshold, iou = float(self.decoder.get("threshold", 0.5)), float(self.decoder.get("group_iou", 0.2))
        return [{"stream_id": sid, **e} for sid, T in acc.lengths.items()
                for e in sets.decode_queries(pacc.get(sid) if pacc else [], T, threshold, iou,
                                             float(self.decoder.get("nms_iou", 0.5)), self.model.group_gap,
                                             self.decoder.get("centre_s"))]

    def _decode_dense(self, acc: StreamAccumulator) -> list[dict]:
        """The dense head with the Viterbi decoder: the join-free baseline."""
        predictions = []
        for sid in acc.lengths:
            logp, spread = acc.result(sid)
            predictions.extend({"stream_id": sid, **e} for e in decode_events(logp, spread))
        return predictions

    def _proposal_recall(self, ground: list[dict], masks: dict | None) -> float:
        """The share of scored falls that some proposal overlaps at tIoU >= .1 (the stream stage's ceiling)."""
        _, pacc = self._sets_done or (None, None)
        hits = total = 0
        for e in ground:
            if pacc is None or e["stream_id"] not in self._sets_lengths or e.get("ignore"):
                continue
            if masks is not None and not in_mask(masks[e["stream_id"]], e["t_start"], e["t_end"]):
                continue
            total += 1
            hits += any(sets.tiou(e["t_start"], e["t_end"], r["start"] / 50.0, r["end"] / 50.0) >= 0.1
                        for r in pacc.get(e["stream_id"]))
        return hits / total if total else 0.0

    def _decode_amir(self, acc: StreamAccumulator, stage: str) -> list[dict]:
        """Amir's chain on every stream: threshold, refine with the official labeller on the detected
        source recording, merge, NMS 5.9 s and the persistence gate."""
        datamodule = self.trainer.datamodule
        files = self._dataset(stage).store.files
        A = amir.load_annotate(getattr(datamodule, "annotate_release", datamodule.release))
        if self._joins is None:
            self._joins = amir.StreamJoins(datamodule.stats_path.parent / "joins")
        args = {k: v for k, v in self.decoder.items() if k != "kind"}
        theta, use_gate, per_recording = args.get("theta", 0.3), args.get("gate", True), args.get("per_recording", False)
        predictions = []
        for sid in acc.lengths:
            logp, spread = acc.result(sid)
            p = 1 - np.exp(logp[:, NONE])
            cir, peaks = self._joins.cir(files[sid]), self._joins.peaks(files[sid])
            for a, b, c in amir.postprocess(A, p, cir, peaks, theta, True, use_gate, per_recording):
                predictions.append({"stream_id": sid, "t_start": a / 50.0, "t_end": b / 50.0, "t_alarm": a / 50.0,
                                    "score": c, "confidence": c, "disagreement": float(spread[a:b].mean()) if b > a else 0.0})
        return predictions

    def _finish(self, stage: str) -> None:
        acc, self._acc = self._acc, None
        uacc, self._uacc = self._uacc, None
        tacc, pacc = self._tacc, self._pacc
        self._tacc = self._pacc = None
        if not self._decoding_epoch() or acc is None:
            return
        if self.trainer.world_size > 1:  # every rank decoded a share of the windows
            gathered = [None] * self.trainer.world_size
            dist.all_gather_object(gathered, (acc, uacc, tacc, pacc))
            acc, uacc, tacc, pacc = gathered[0]
            for other, other_units, other_tokens, other_props in gathered[1:]:
                acc.merge(other)
                if other_units is not None:
                    uacc = uacc or UnitAccumulator()
                    uacc.merge(other_units)
                if other_tokens is not None:
                    tacc.merge(other_tokens)
                    pacc.merge(other_props)
        self._uacc_done = uacc
        self._sets_done = (tacc, pacc)
        self._sets_lengths = acc.lengths
        datamodule = self.trainer.datamodule
        masks = datamodule.score_masks.get(stage)
        predictions = self.decode_streams(acc, stage)
        extra = {}
        if self.decoder.get("kind") == "units" and self.decoder.get("also_amir", False):
            extra["amir"] = self._decode_amir(acc, stage)
        if self.decoder.get("kind") == "sets":
            extra["queries"] = self._decode_queries(acc)
            extra["dense"] = self._decode_dense(acc)
        if self.trainer.is_global_zero:
            self._write_predictions(stage, predictions)
            for name, preds in extra.items():
                self._write_predictions(stage, preds, suffix=f".{name}")
            if self.trainer.state.fn in (TrainerFn.VALIDATING, TrainerFn.TESTING):
                self._save_probabilities(stage, acc, masks, uacc)
        if uacc is not None:
            store = self._dataset(stage).store
            total = sum(len(u["units"]) for u in store.units.values()) if store.units else 0
            self.log(f"{stage}/units/missed", float(total - len(uacc.best)), sync_dist=False)

        if not datamodule.events_path.exists():
            return
        ground = official.load_ground_truth(datamodule.events_path, ignore_flagged=self.ignore_flagged)
        if not any(event["stream_id"] in acc.lengths for event in ground):
            return
        if self.decoder.get("kind") == "sets":
            self.log(f"{stage}/sets/proposal_recall", self._proposal_recall(ground, masks), sync_dist=False)
        for name, preds in extra.items():
            other = score_predictions(preds, ground, acc.lengths, masks)
            self.log_dict({f"{stage}/{name}/F1_mean": float(other["ranking_score"]["f1_mean"]),
                           f"{stage}/{name}/F1_0.5": float(other["by_theta"]["0.5"]["f1"]),
                           f"{stage}/{name}/FalseAlarmsPerHour_0.3": float(other["by_theta"]["0.3"]["false_alarms_per_hour"] or 0.0)},
                          sync_dist=False)
            if self.trainer.is_global_zero:
                (self._out_dir() / f"{self._name(stage)}.{name}.report.json").write_text(json.dumps(other, indent=2))
        report = score_predictions(predictions, ground, acc.lengths, masks)
        by = report["by_theta"]
        values = {
            "F1_mean": report["ranking_score"]["f1_mean"],
            "F1_0.1": by["0.1"]["f1"], "F1_0.3": by["0.3"]["f1"], "F1_0.5": by["0.5"]["f1"], "F1_0.7": by["0.7"]["f1"],
            "Precision_0.3": by["0.3"]["precision"], "Recall_0.3": by["0.3"]["recall"],
            "FalseAlarmsPerHour_0.3": by["0.3"]["false_alarms_per_hour"] or 0.0,
            "StartErrorMedian_0.3": by["0.3"]["start_error_median_abs"] or 0.0,
            "MeanTIoU_0.3": by["0.3"]["mean_tiou"] or 0.0,
            "Predictions": float(by["0.3"]["tp"] + by["0.3"]["fp"] + by["0.3"]["n_ignored_predictions"]),
        }
        # Every rank holds the merged predictions, so every rank logs the same values (the checkpoint
        # callback needs the monitored key on all of them)
        self.log_dict({f"{stage}/event/{k}": float(v) for k, v in values.items()}, sync_dist=False)
        if self.trainer.is_global_zero:
            out = self._out_dir() / f"{self._name(stage)}.report.json"
            out.write_text(json.dumps(report, indent=2))

    def _out_dir(self) -> Path:
        out = Path(self.trainer.default_root_dir) / "predictions"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _name(self, stage: str) -> str:
        fn = self.trainer.state.fn
        suffix = self.decoder.get("suffix", "")   # e.g. -noboundary: an ablation beside the run's own outputs
        return f"{stage}-epoch{self.current_epoch:03d}" if fn == TrainerFn.FITTING else f"{stage}-{fn.value}{suffix}"

    def _save_probabilities(self, stage: str, acc: StreamAccumulator, masks: dict | None,
                            uacc: UnitAccumulator | None = None) -> None:
        """Merged per-timestep state log-probabilities (float16) and disagreement per stream, for
        tools/tune_decoder.py; with the score mask in cross-validation. With the recording head also,
        per stream, `units/<sid>` [U, 4] (unit index, start, stop, fall logit) and `span/<sid>`
        [n, 4] (unit index, stream timestep, log P(start), log P(end)) of every unit it decoded."""
        arrays = {}
        if uacc is not None:
            store = self._dataset(stage).store
            for sid in acc.lengths:
                rows, spans = [], []
                for u, (_, logit, pos, ls, le) in sorted(uacc.units(sid).items()):
                    lo, hi, _ = store.units[sid]["units"][u]
                    rows.append((u, lo, hi, logit))
                    spans.append(np.stack([np.full(len(pos), u), pos, ls, le], 1))
                arrays[f"units/{sid}"] = np.array(rows, dtype=np.float64).reshape(-1, 4)
                arrays[f"span/{sid}"] = (np.concatenate(spans) if spans else np.zeros((0, 4))).astype(np.float32)
        _, pacc = self._sets_done or (None, None)
        if pacc is not None:
            for sid in acc.lengths:
                rows = [(r["logit"], r["start"], r["end"], r["centrality"]) for r in pacc.get(sid)]
                arrays[f"proposals/{sid}"] = np.array(rows, dtype=np.float64).reshape(-1, 4)
                arrays[f"slots/{sid}"] = getattr(self, "_slots", {}).get(sid, np.zeros((0, 5)))
        for sid in acc.lengths:
            logp, spread = acc.result(sid)
            arrays[f"logp/{sid}"] = logp.astype(np.float16)
            arrays[f"spread/{sid}"] = spread.astype(np.float16)
            if acc.bounds is not None:  # absolute seconds: float16 is too coarse over an hour
                arrays[f"bounds/{sid}"] = acc.result_bounds(sid).astype(np.float32)
            if masks is not None:
                arrays[f"mask/{sid}"] = masks[sid]
        np.savez_compressed(self._out_dir() / f"{self._name(stage)}.probs.npz", **arrays)

    def _write_predictions(self, stage: str, predictions: list[dict], suffix: str = "") -> None:
        """A submission CSV (stream_id, t_start, t_end, score, t_alarm) plus confidence and disagreement."""
        path = self._out_dir() / f"{self._name(stage)}{suffix}.csv"
        fields = ["stream_id", "t_start", "t_end", "score", "t_alarm", "confidence", "disagreement"]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for p in predictions:
                writer.writerow({k: (f"{p[k]:.3f}" if k.startswith("t_") else (f"{p[k]:.6f}" if k != "stream_id" else p[k]))
                                 for k in fields})

    # ----------------------------------------------------------- diagnostics

    def on_before_optimizer_step(self, optimizer) -> None:
        """Gradient norm of each top-level part of the model (before clipping), on logging steps."""
        if not self.trainer._logger_connector.should_update_logs:
            return
        norms = {}
        for name, child in self.model.named_children():
            grads = [p.grad.detach().float().norm() for p in child.parameters() if p.grad is not None]
            if grads:
                norms[name] = torch.stack(grads).norm()
        loose = [p.grad.detach().float().norm() for p in self.model.parameters(recurse=False) if p.grad is not None]
        if loose:
            norms["params"] = torch.stack(loose).norm()
        if norms:
            norms["total"] = torch.stack(list(norms.values())).norm()
            self.log_dict({f"train/grad_norm/{k}": v for k, v in norms.items()}, on_step=True, on_epoch=False)
