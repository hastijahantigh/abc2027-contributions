"""JF4 (docs/falljepa_JF4_design.md, section 7): the stem's alignment and phase invariance, the radar
mask, the TCN's reach, RoPE, the contrast maps, the step swap and the four JF4 configs end to end."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from falljepa.cli import main
from falljepa.data.uwb import STEP_GUARD, UWBDataModule, contrast_maps
from falljepa.model import FallSetJEPA
from falljepa.model.layers import rotary
from falljepa.model.stem import TCN, RadarStem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def _stem(fusion="attention", contrast=0):
    torch.manual_seed(0)
    return RadarStem(16, 2, (8, 8, 16, 16), contrast, fusion, 2, n_bins=184, frame_width=32, frame_stride=4).eval()


def _x(L=228, contrast=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 13 + 3 * contrast, 184, L, generator=g)
    if contrast:
        x[:, 12:12 + 3 * contrast] = x[:, 12:12 + 3 * contrast].abs()
    x[:, -1] = 1.0
    return x


@pytest.mark.parametrize("fusion", ["early", "attention"])
def test_stem_token_i_sees_exactly_frame_i(fusion):
    stem = _stem(fusion)
    x = _x().requires_grad_(True)
    Fn = (228 - 32) // 4 + 1
    out = stem(x, Fn)
    assert out.shape == (1, Fn, 16)
    for i in (0, 10, Fn - 1):
        g, = torch.autograd.grad(out[0, i].sum(), x, retain_graph=True)
        steps = torch.nonzero(g.abs().sum((0, 1, 2)) > 0).flatten()
        assert int(steps.min()) >= 4 * i and int(steps.max()) <= 4 * i + 32, (i, steps.min(), steps.max())
        assert int(steps.max()) - int(steps.min()) >= 28


@pytest.mark.parametrize("fusion", ["early", "attention"])
def test_stem_ignores_each_radars_phase(fusion):
    stem = _stem(fusion, contrast=3)
    x = _x(contrast=3)
    ref = stem(x, 50)
    y = x.clone()
    for r, phi in ((0, 0.7), (2, -2.1)):                 # z and c of a radar rotate together
        for i, q in ((r, 3 + r), (6 + r, 9 + r)):
            y[:, i] = x[:, i] * np.cos(phi) - x[:, q] * np.sin(phi)
            y[:, q] = x[:, i] * np.sin(phi) + x[:, q] * np.cos(phi)
    assert torch.allclose(stem(y, 50), ref, atol=1e-4)
    y[:, 0] = x[:, 0] * 0.5                              # a change that is not a rotation does matter
    assert not torch.allclose(stem(y, 50), ref, atol=1e-4)


def test_dropped_radar_gets_no_fusion_weight():
    stem = _stem()
    x = _x()
    x[:, [1, 4, 7, 10]] = 0.0                            # radar 1, as radar dropout leaves it
    out = stem(x, 50)
    assert torch.isfinite(out).all()
    w = stem.fuse.last_weights
    assert torch.all(w[..., 1] == 0) and torch.allclose(w.sum(-1), torch.ones_like(w[..., 0]))


def test_tcn_reaches_61_tokens_and_keeps_padding_zero():
    tcn = TCN(8).eval()
    x = torch.randn(1, 120, 8, requires_grad=True)
    valid = torch.ones(1, 120, dtype=torch.bool)
    g, = torch.autograd.grad(tcn(x, valid)[0, 60].sum(), x)
    reach = torch.nonzero(g.abs().sum(-1)[0] > 0).flatten()
    assert int(reach.min()) == 30 and int(reach.max()) == 90                # 1 + 4 (1 + 2 + 4 + 8) = 61
    valid[:, 100:] = False
    assert torch.all(tcn(x, valid)[0, 100:] == 0)


def test_rope_scores_depend_only_on_the_offset():
    q, k = torch.randn(1, 1, 1, 16).expand(1, 1, 40, 16), torch.randn(1, 1, 1, 16).expand(1, 1, 40, 16)
    s = rotary(q) @ rotary(k).transpose(-1, -2)
    assert torch.allclose(s[0, 0, 5, 2], s[0, 0, 25, 22], atol=1e-4)
    assert torch.allclose(rotary(q).norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_contrast_ignores_the_background_and_peaks_at_a_scene_step():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.05, (3, 400, 368)).astype(np.float32)
    z[1, 200:, 10:20] += 2.0                             # radar 1: the scene changes at t = 200
    scales = [(15, 25), (15, 50), (25, 100)]
    d = contrast_maps(z, scales)
    assert d.shape == (9, 400, 184)
    for k in range(3):
        energy = d[3 * k + 1].sum(-1)
        top = np.flatnonzero(energy >= 0.98 * energy.max())      # flat over [step - g, step + g]
        assert abs(top.mean() - 200) <= 2 and top.min() <= 200 <= top.max()
        assert d[3 * k + 1][:, 10:20].max() > 10 * d[3 * k][:, 10:20].max()   # the other radars do not move
    background = np.zeros_like(z)
    background[:, :, :] = rng.normal(0, 3, (3, 1, 368))
    assert np.allclose(contrast_maps(z + background, scales), d, atol=1e-4)
    assert np.all(d[:, 0] == 0) and np.all(d[:, -1] == 0)                     # one side empty at the ends


def _datamodule(release, **kw):
    sampler = {"fall_slots": 1, "hard_slots": 0, "groups": str(release / "folds.csv"), **kw.pop("sampler", {})}
    return UWBDataModule(release=str(release), stats_path=str(release / "stats_jf4.json"), window_s=16.0,
                         train_hop_s=2.0, infer_hop_s=4.0, batch_size=2, eval_batch_size=4, num_workers=0,
                         frame_stride=4, diff_channel=False, complex_input=True, physics=True, num_states=2,
                         fall_sets=True, chunk_windows=3, train_sampler="balanced", infer_tail="pad",
                         sampler=sampler, clutter_s=2.0, amplitude=False, **kw)


def test_layout_contrast_channels_and_step_swap(release):
    scales = [[0.3, 0.5], [0.3, 1.0], [0.5, 2.0]]
    dm = _datamodule(release, contrast=scales, augment={"step_swap_p": 1.0, "radar_drop_p": 0.0})
    assert dm.in_channels == 22
    dm.setup("fit")
    ds = dm.datasets["train"]
    store = ds.store
    sid = store.ids[0]
    x, _ = store.read(sid, 300, 700)
    assert x.shape == (22, 184, 400)
    z = store.read_z(sid, 0, 900)
    ref = contrast_maps(z, store.contrast)[:, 300:700]
    assert np.allclose(x[12:21], ref.transpose(0, 2, 1), atol=1e-4)
    # The step swap: never near a fall, seen by the clutter and contrast channels, labels unchanged
    hits = 0
    for idx in range(len(ds.index)):
        stream, start = ds.index[idx]
        sid = store.ids[stream]
        lo, hi = start - ds.margin, start + ds.window + ds.margin
        step = ds._step(sid, lo, hi, idx)
        if step is None:
            continue
        t = step[0]
        assert max(lo, 0) + 100 <= t < min(hi, store.lengths[sid])
        for fa, fb, _ in ds.falls[sid]:
            assert not (fa - STEP_GUARD[0] <= t < fb + STEP_GUARD[1]), (t, fa, fb)
        if hits == 0:
            plain, _ = store.read(sid, lo, hi)
            swapped, _ = store.read(sid, lo, hi, step)
            k = t - lo
            assert np.allclose(plain[:6, :, :k], swapped[:6, :, :k], atol=1e-5)   # raw I / Q change only from t on
            assert not np.allclose(plain[6:12], swapped[6:12]) and not np.allclose(plain[12:21], swapped[12:21])
        hits += 1
    assert hits > 0
    batch = next(iter(dm.train_dataloader()))
    assert batch["x"].shape[1] == 22


def test_stem_model_trains_both_stages_on_chunks(release):
    torch.manual_seed(0)
    dm = _datamodule(release, contrast=[[0.3, 0.5]], augment={"radar_drop_p": 0.5})
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    model = FallSetJEPA(in_channels=16, n_bins=184, window_frames=200, d_model=16, nhead=2, frame_width=32,
                        frame_stride=4, context_layers=0, bidir_layers=1, predictor_layers=0, pred_horizons=(),
                        num_states=2, mlp_ratio=2, physics_channels=12, query_layers=1, group_layers=1,
                        raw_drop_p=0.5, raw_channels=6, trunk="stem", stem_channels=(8, 8, 16, 16), range_heads=2,
                        contrast_channels=1, local="tcn", rope=True)
    out = model(batch)
    assert out["tokens"].shape == (6, 200, 16) and "z_v_pred" not in out
    loss = out["q_logit"].sum() + out["q_span_logits"].float().logsumexp(1).sum() + out["g_logit"].sum()
    loss.backward()
    assert model.stem.l0.wr.grad.abs().sum() > 0 and model.tcn.blocks[0].conv1.weight.grad.abs().sum() > 0
    assert model.frame_pos_embed.grad.abs().sum() > 0          # the queries' keys use it


@pytest.mark.parametrize("config", ["JF4_Se", "JF4_S", "JF4_ST", "JF4", "JF5T", "JF6Range", "JF6Radar"])
def test_cli_jf4_fits_and_validates(release, tmp_path, config):
    from compose_config import main as compose_main
    composed = tmp_path / "composed.yaml"
    compose_main([str(composed), str(ROOT / f"configs/uwb/{config}.yaml")])
    tiny = tmp_path / "tiny.yaml"
    tiny.write_text(
        "run: {wandb: disabled}\n"
        "trainer: {max_epochs: 2, accelerator: cpu, devices: 1, log_every_n_steps: 1, enable_progress_bar: false,\n"
        "          check_val_every_n_epoch: 1, callbacks: [{class_path: falljepa.callbacks.WeightAveraging,\n"
        "          init_args: {start_epoch: 0, bn_batches: 2, every_n_steps: 2}}]}\n"
        "data: {window_s: 16.0, infer_hop_s: 4.0, train_hop_s: 2.0, chunk_windows: 3, batch_size: 2, eval_batch_size: 4,\n"
        "       num_workers: 0, clutter_s: 2.0}\n"
        "model: {model: {init_args: {d_model: 32, nhead: 2, query_layers: 1, group_layers: 1, group_max_s: 12.0,\n"
        "        stem_channels: [8, 16, 16, 32]}},\n"
        "        decoder: {kind: sets, threshold: 0.5, proposal_threshold: 0.0, centre_s: 3.0}}\n")
    args = ["-c", str(composed), "-c", str(tiny), "--data.release", str(release),
            "--data.stats_path", str(tmp_path / "cache" / "stats.json"), "--data.sampler.groups", str(release / "folds.csv"),
            "--data.sampler.fall_slots", "1", "--data.sampler.hard_slots", "0", "--data.sampler.steps_scale", "1.0",
            "--run.root", str(tmp_path / "runs"), "--run.name", config, "--run.id", "t"]
    main(["fit"] + args)
    run_dir = next((tmp_path / "runs").glob(f"UWB/*/{config}/t"))
    metrics = next((run_dir / "csv").glob("fit-*/metrics.csv")).read_text()
    for key in ("train/loss/Query_IoU", "train/loss/Group_IoU", "val/event/F1_mean", "val/queries/F1_mean"):
        assert key in metrics, key
    assert ("train/loss/MSE_Latent" in metrics) == (config in ("JF4_Se", "JF4_S"))
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", str(run_dir / "checkpoints" / "swa.ckpt")])
    assert (run_dir / "predictions" / "val-validate.csv").exists()


def test_contrast_in_the_model_matches_the_data_loader(release):
    scales = [[0.3, 0.5], [0.3, 1.0], [0.5, 2.0]]
    augment = {"step_swap_p": 1.0, "radar_drop_p": 0.5, "bg_swap_p": 0.5}
    cpu, gpu = (_datamodule(release, contrast=scales, augment=augment, contrast_in_model=flag) for flag in (False, True))
    cpu.setup("fit")
    gpu.setup("fit")
    model = FallSetJEPA(in_channels=22, n_bins=184, window_frames=200, d_model=16, nhead=2, frame_width=32,
                        frame_stride=4, context_layers=0, bidir_layers=1, predictor_layers=0, pred_horizons=(),
                        num_states=2, mlp_ratio=2, physics_channels=12, query_layers=1, group_layers=1,
                        raw_channels=6, trunk="stem", stem_channels=(8, 8, 16, 16), range_heads=2,
                        contrast_channels=3, contrast_scales=scales)
    for split in ("train", "val"):
        if split == "val":
            cpu.setup("validate")
            gpu.setup("validate")
        for idx in (0, 3, len(cpu.datasets[split]) - 1):
            a, b = cpu.datasets[split][idx], gpu.datasets[split][idx]
            assert a["x"].shape[0] == 22 and b["x"].shape[0] == 13 and "ctx" in b and "ctx" not in a
            batch = {k: v[None] for k, v in b.items()}
            x = model.with_contrast(batch)["x"][0]
            assert torch.allclose(x, a["x"], atol=2e-3), (split, idx, (x - a["x"]).abs().max())
