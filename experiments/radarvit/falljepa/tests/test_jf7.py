"""JF7 (docs/falljepa_JF7_design.md): the pair-tIoU span loss, the member channels, the slot quality head,
the boundary refiner's crops and the JF7a / JF7b configs end to end."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from falljepa.cli import main
from falljepa.loss import SpanPairIoULoss
from falljepa.model import FallSetJEPA
from falljepa.model.set_heads import BoundaryRefiner, member_bumps
from test_jf4 import _datamodule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def _pair_out(logits, target, L):
    return {"span_logits": logits, "span_target": torch.tensor(target), "span_w": torch.ones(len(target)),
            "span_domain": torch.ones(len(target), L, dtype=torch.bool), "step_logits": torch.zeros(1)}


@pytest.mark.parametrize("max_len", [300, 8])
def test_pair_iou_loss_matches_brute_force_at_stride_1(max_len):
    torch.manual_seed(0)
    L = 40
    logits = torch.randn(2, L, 2) * 2
    target = [[10, 25], [5, 12]]
    loss = SpanPairIoULoss(stride=1, max_len=max_len)(_pair_out(logits, target, L))
    p = torch.softmax(logits, 1)
    want = []
    for n, (a, b) in enumerate(target):
        gain = 0.0
        for s in range(L):
            for e in range(L):
                if e <= s or e - s > max_len:
                    continue
                inter = max(0, min(e, b) - max(s, a))
                iou = inter / ((e - s) + (b - a) - inter)
                g = 0.5 * iou + 0.5 * np.mean([1 / (1 + np.exp(-(iou - th) / 0.05)) for th in (0.1, 0.3, 0.5)])
                gain += float(p[n, s, 0] * p[n, e, 1]) * g
        want.append(1 - gain)
    assert abs(float(loss) - np.mean(want)) < 1e-4


def test_pair_iou_loss_prefers_the_right_span_and_pulls_a_late_start_back():
    L = 200
    logits = torch.full((1, L, 2), -10.0)
    logits[0, 100, 0], logits[0, 150, 1] = 10.0, 10.0         # exactly the target
    good = float(SpanPairIoULoss()(_pair_out(logits, [[100, 150]], L)))
    late = torch.full((1, L, 2), -10.0)
    late[0, 130, 0], late[0, 150, 1] = 10.0, 10.0              # start 0.6 s late: too short
    late.requires_grad_(True)
    bad = SpanPairIoULoss()(_pair_out(late, [[100, 150]], L))
    assert good < 0.05 < float(bad)
    bad.backward()
    assert late.grad[0, 100, 0] < 0 < late.grad[0, 130, 0]       # descent moves start mass to the true start


def test_member_bumps_peak_at_member_boundaries():
    g = {"member_features": torch.tensor([[[3.0, 0.25, 0.5, 0, 0], [-3.0, 0.75, 0.9, 0, 0]]]),
         "member_valid": torch.tensor([[True, True]]), "region_len": torch.tensor([200]),
         "step_valid": (torch.arange(240) < 200)[None]}
    b = member_bumps(g, 240)
    assert b.shape == (1, 2, 240)
    assert int(b[0, 0].argmax()) == 50 and int(b[0, 1].argmax()) == 100   # the confident member
    assert abs(float(b[0, 0, 150]) - torch.sigmoid(torch.tensor(-3.0))) < 1e-4
    assert float(b[0, :, 200:].abs().max()) == 0.0                  # invalid steps (padding) stay 0


def test_boundary_crops_align_and_mask_outside_the_region():
    torch.manual_seed(0)
    br = BoundaryRefiner(8, 5, stride=4, half=20, width=8).eval()
    tokens, steps = torch.randn(2, 25, 8), torch.randn(2, 5, 100)
    valid = torch.ones(2, 100, dtype=torch.bool)
    valid[1, 90:] = False
    hidden = torch.randn(2, 2, 8)
    t = lambda v: torch.tensor(v)  # noqa: E731
    logits, mask = br(tokens, steps, valid, hidden, t([0, 1]), t([0, 1]), t([10, 85]), t([0, 1]))
    assert logits.shape == (2, 41)
    assert not mask[0, :10].any() and mask[0, 10:].all()           # steps < 0
    assert mask[1, :25].all() and not mask[1, 25:].any()           # steps >= 90 invalid
    steps2 = steps.clone()
    steps2[0, :, 25] += 5.0                                        # a step outside the crop's receptive range
    l2, _ = br(tokens, steps2, valid, hidden, t([0, 1]), t([0, 1]), t([10, 85]), t([0, 1]))
    assert torch.allclose(l2[1], logits[1]) and not torch.allclose(l2[0], logits[0])


def test_jf7_model_trains_quality_and_boundaries(release):
    torch.manual_seed(0)
    np.random.seed(0)
    dm = _datamodule(release)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    model = FallSetJEPA(in_channels=13, n_bins=184, window_frames=200, d_model=16, nhead=2, frame_width=32,
                        frame_stride=4, context_layers=0, bidir_layers=1, predictor_layers=0, pred_horizons=(),
                        num_states=2, mlp_ratio=2, physics_channels=12, query_layers=1, group_layers=1,
                        raw_channels=6, trunk="stem", stem_channels=(8, 8, 16, 16), range_heads=2, local="tcn",
                        rope=True, proposal_threshold=0.0, member_channels=True, quality=True, boundary=True,
                        boundary_half_s=1.0)
    out = model(batch)
    assert out["g_quality"].shape == out["g_logit"].shape == out["g_qtarget"].shape
    assert ((out["g_qtarget"] >= 0) & (out["g_qtarget"] <= 1)).all()
    assert "b_logits" in out and out["b_logits"].shape[1] == 101
    tgt = out["b_target"][:, 0].long()
    assert out["b_domain"][torch.arange(len(tgt)), tgt].all()       # the truth is inside every crop
    loss = (out["g_quality"].sum() + out["b_logits"].clamp_min(-50).sum()
            + SpanPairIoULoss(target_key="g_span", weight_key="g_span_w", logits_key="g_span_logits",
                              domain_key="g_domain")(out))
    loss.backward()
    assert model.refiner.quality[0].weight.grad.abs().sum() > 0
    assert model.boundary.out.weight.grad.abs().sum() > 0
    assert model.refiner.token_in.weight.shape[1] == 11 + 3 + 2   # event channels + mixed + member bumps


@pytest.mark.parametrize("config", ["JF7a", "JF7b"])
def test_cli_jf7_fits_and_validates(release, tmp_path, config):
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
    keys = ["train/loss/Query_PairIoU", "train/loss/Group_PairIoU", "train/loss/Group_Quality", "val/event/F1_mean"]
    for key in keys + (["train/loss/Boundary_CE"] if config == "JF7b" else []):
        assert key in metrics, key
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", str(run_dir / "checkpoints" / "swa.ckpt")])
    assert (run_dir / "predictions" / "val-validate.csv").exists()
    # decoder overrides reach a validate from a checkpoint (Lightning would restore the saved ones)
    time.sleep(1.1)   # each invocation dumps config.validate-<second>.yaml and refuses to overwrite one
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", str(run_dir / "checkpoints" / "swa.ckpt"),
          "--model.decoder.boundary", "false", "--model.decoder.suffix=-noboundary"])
    assert (run_dir / "predictions" / "val-validate-noboundary.csv").exists()
