"""The recording head (docs/falljepa_recording_head.md): units, their labeller channels and targets,
the complex input layout, the decoder, its losses, MBR decoding and the RS config end to end."""
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from falljepa import postprocess as pp
from falljepa.cli import main
from falljepa.data.labels import Event
from falljepa.data.units import (CHUNK, MAX_UNIT, NUM_CHANNELS, NUM_SCALARS, contained, make_units, stream_units,
                                 unit_physics, unit_targets)
from falljepa.data.uwb import StreamStore, UWBDataModule, fit_normalizer, stream_files
from falljepa.loss import SpanSoftCELoss, UnitBCELoss
from falljepa.model import FallJEPA
from falljepa.model.aggregate import UnitAccumulator, decode_units, mbr_span

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
ANNOTATE = ROOT / "abc2027_uwb_train_v1_sample" / "abc2027_uwb_train_v1" / "src" / "abc2027" / "annotate.py"
needs_annotate = pytest.mark.skipif(not ANNOTATE.exists(), reason="the official annotate.py is not available")


@pytest.fixture
def official(release):
    (release / "src" / "abc2027").mkdir(parents=True, exist_ok=True)
    shutil.copy(ANNOTATE, release / "src" / "abc2027" / "annotate.py")
    return pp.load_annotate(release)


def test_units_between_joins_chunk_long_spans_and_drop_slivers():
    peaks = np.array([500, 505 + 2000, 2600])
    units = make_units(peaks, 3500)
    assert tuple(units[0]) == (0, 495, 0)
    long = units[units[:, 2] == 1]                     # 506 .. 2500: 1994 > 22 s, cut in 2
    assert len(long) == 2 and long[0, 0] == 506 and long[-1, 1] == 2500
    assert (long[:, 1] - long[:, 0] <= CHUNK).all() and long[0, 1] == long[1, 0]
    assert tuple(units[-1]) == (2606, 3500, 0)
    assert all(b - a <= MAX_UNIT or c for a, b, c in units)
    assert len(make_units(np.array([100, 108]), 1000)) == 2   # the 0.04 s sliver between them is dropped


def test_a_fall_belongs_to_the_unit_it_starts_in():
    units = np.array([[0, 500, 0], [506, 1000, 0], [1006, 1500, 0], [1506, 2000, 0]])
    rule = np.array([[100, 180], [-1, -1], [1100, 1200], [1600, 1700]])
    t = unit_targets(units, rule, [Event("s", 4.0, 6.0, False),     # 200-300, inside unit 0
                                   Event("s", 29.0, 30.6, False)])  # 1450-1530: a join peak at its landing
    assert t["fall"].tolist() == [1, 0, 1, 0] and t["fall_w"].tolist() == [1, 1, 1, 0.5]
    assert tuple(t["span"][2]) == (1450, 1530)                       # the span runs past its unit
    assert t["rule_w"].tolist() == [0, 0, 0, 1]                      # the tail unit is a no-fall unit
    t = unit_targets(units, rule, [Event("s", 19.8, 21.0, False),   # 990-1050: 1/6 in the unit it starts in
                                   Event("s", 31.0, 32.0, True)])    # 1550-1600, flagged
    assert t["fall"].tolist() == [0, 0, 0, 1] and t["fall_w"].tolist() == [1, 0, 0.5, 0]
    assert t["span_w"][3] == 0.5 and t["rule_w"].tolist() == [1, 0, 1, 0]


def test_contained_keeps_units_away_from_inner_window_edges():
    units = np.array([[0, 400, 0], [406, 900, 0], [906, 1700, 0]])
    assert contained(units, 0, 1000, 5000, edge=50).tolist() == [0, 1]      # the stream start is no edge
    assert contained(units, 0, 980, 5000).tolist() == [0]                   # 2 s by default: 900 > 880
    assert contained(units, 380, 1600, 5000, edge=50).tolist() == [2]       # 406 is only 26 from 380
    assert contained(units, 300, 1600, 1700, edge=50).tolist() == [1, 2]    # the window reaches the stream end


@needs_annotate
def test_unit_physics_is_the_official_labeller(release, official):
    with h5py.File(release / "streams" / "train" / "stream_00.h5", "r") as h:
        cir = h["cir"][:].astype(np.float64)
    lo, hi = 0, 395                                    # the first synthetic recording, with a fall at 5.0-6.6 s
    channels, scalars, rule = unit_physics(official, cir[:, lo:hi])
    assert channels.shape == (9, hi - lo) and scalars.shape == (NUM_SCALARS,)
    left, right = pp.official_bounds(official, cir[:, lo:hi], raw=True)
    assert rule == (left, right)
    grid = np.arange(hi - lo) * 0.02
    before, after = official._profiles(cir[0, lo:hi], grid)
    state = official._smooth(official._state(cir[0, lo:hi], before, after), 0.02, official.SMOOTH_S)
    assert np.allclose(channels[0], state, atol=1e-6)
    change = [np.sum(np.subtract(*official._profiles(cir[r, lo:hi], grid)[::-1]) ** 2) for r in range(3)]
    assert np.allclose(scalars[:3], np.log10(np.array(change) + 1e-12) + 4.67, atol=1e-4)


@needs_annotate
def test_stream_units_cover_the_synthetic_recordings(release, official):
    with h5py.File(release / "streams" / "train" / "stream_00.h5", "r") as h:
        out = stream_units(official, h["cir"])
    units = out["units"]
    assert len(units) >= 4 and out["channels"].shape == (NUM_CHANNELS, 2000)
    inside = np.zeros(2000, bool)
    for a, b, _ in units:
        inside[a:b] = True
    assert (out["channels"][:, ~inside] == 0).all() and np.abs(out["channels"][:, inside]).sum() > 0


def test_complex_layout_pairs_i_and_q_with_one_std(release):
    files = stream_files(release, "train")
    stats = fit_normalizer(list(files.values()))
    plain = StreamStore(files, stats, diff_channel=False)
    cplx = StreamStore(files, stats, diff_channel=False, complex_input=True)
    x, valid = cplx.read("train_00", -10, 90)
    assert x.shape == (10, 184, 100) and valid[:10].sum() == 0
    raw = np.asarray(cplx._cir("train_00")[:, :90, :], dtype=np.float32)
    std = np.sqrt((plain.std[..., :184] ** 2 + plain.std[..., 184:] ** 2) / 2)
    i = (raw[..., :184] - plain.mean[..., :184]) / std
    q = (raw[..., 184:] - plain.mean[..., 184:]) / std
    assert np.allclose(x[0:3, :, 10:], i.transpose(0, 2, 1), atol=1e-5)
    assert np.allclose(x[3:6, :, 10:], q.transpose(0, 2, 1), atol=1e-5)
    assert np.allclose(x[6:9, :, 10:], np.sqrt(i ** 2 + q ** 2).transpose(0, 2, 1), atol=1e-5)
    assert (x[9, :, :10] == 0).all() and (x[9, :, 10:] == 1).all()


def _unit_batch(B=2, window=800, w=32, s=4):
    m = (w - s) // 2
    L = window + 2 * m
    units = torch.tensor([[[60, 460], [470, 760], [0, 0]], [[0, 300], [310, 790], [0, 0]]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    return {"x": torch.randn(B, 10, 184, L), "valid": torch.ones(B, L), "phys": torch.randn(B, 12, L),
            "units": units, "unit_mask": mask, "unit_scalars": torch.randn(B, 3, NUM_SCALARS),
            "uphys": torch.randn(B, NUM_CHANNELS, L),
            "unit_fall": torch.tensor([[1.0, 0, 0], [0, 1, 0]]), "unit_fall_w": torch.ones(B, 3),
            "unit_span": torch.tensor([[[200, 290], [0, 0], [0, 0]], [[0, 0], [400, 480], [0, 0]]]),
            "unit_span_w": torch.tensor([[1.0, 0, 0], [0, 1, 0]]),
            "unit_rule": torch.tensor([[[0, 0], [500, 560], [0, 0]], [[100, 150], [0, 0], [0, 0]]]),
            "unit_rule_w": torch.tensor([[0.0, 1, 0], [1, 0, 0]])}


def _rs_model():
    return FallJEPA(in_channels=10, n_bins=184, window_frames=200, d_model=32, nhead=2, frame_width=32, frame_stride=4,
                    frame_encoder_layers=1, context_layers=1, bidir_layers=1, predictor_layers=0, num_states=2,
                    physics_channels=12, recording_head=True, unit_layers=1, dropout=0.0)


def test_recording_head_rows_domains_and_targets():
    model = _rs_model().eval()
    batch = _unit_batch()
    out = model(batch)
    assert out["unit_logit"].shape == (4,) and out["unit_rows"].tolist() == [0, 0, 1, 1]
    origin, domain = out["span_origin"], out["span_domain"]
    t = origin[:, None] + torch.arange(domain.size(1))[None, :]
    for n, (b, k) in enumerate(zip(out["unit_rows"].tolist(), out["unit_slots"].tolist())):
        lo, hi = batch["units"][b, k].tolist()
        dom = t[n][domain[n]]
        assert dom.min() == max(lo - 25, int(origin[n])) and dom.max() == hi + 99   # 0.5 s before, 2 s after
    assert (out["span_logits"][~domain] <= -1e3).all()
    target = out["span_target"] + origin[:, None]
    assert target[0].tolist() == [200, 290] and target[3].tolist() == [400, 480]
    assert out["unit_fall"].tolist() == [1, 0, 0, 1]


def test_unit_and_span_losses_train_the_head():
    torch.manual_seed(0)
    model = _rs_model()
    batch = _unit_batch()
    unit, span, rule = UnitBCELoss(), SpanSoftCELoss(), SpanSoftCELoss("rule_target", "rule_w")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        out = model(batch)
        loss = unit(out) + span(out) + 0.2 * rule(out)
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    out = model(batch)
    assert float(unit(out)) < 0.05 and float(span(out)) < 3.4     # 3.03: the entropy of the sigma-5 target
    assert (torch.sigmoid(out["unit_logit"]) > 0.5).tolist() == [True, False, False, True]
    best = (out["span_logits"].argmax(1) + out["span_origin"][:, None])
    assert (best[0] - torch.tensor([200, 290])).abs().max() <= 3


def test_mbr_prefers_the_safe_span_and_decode_gives_one_event_per_unit():
    pos = np.arange(0, 400)
    ls = np.full(400, -50.0)
    le = np.full(400, -50.0)
    ls[100], le[180] = 0.0, 0.0
    assert mbr_span(pos, ls, le) == (100, 180)
    # Two start modes (100 and 160, 55/45), one end (200): the earlier start gives tIoU .6 against the
    # later truth and keeps every theta; the later start gives tIoU .4 against the earlier one
    ls[:] = -50.0
    ls[100], ls[160] = np.log(0.55), np.log(0.45)
    le[:] = -50.0
    le[200] = 0.0
    assert mbr_span(pos, ls, le) == (100, 200)
    acc = UnitAccumulator()
    acc.add("s", 0, 10, 3.0, pos, ls, le)
    acc.add("s", 0, 5, -3.0, pos, ls, le)                  # less central: ignored
    acc.add("s", 1, 10, -3.0, pos, ls, le)                 # P(fall) < .5: no event
    events = decode_units(acc, "s", 1000)
    assert len(events) == 1 and events[0]["t_start"] == 2.0 and events[0]["t_end"] == 4.0
    assert events[0]["score"] == pytest.approx(1 / (1 + np.exp(-3.0)))


@needs_annotate
def test_datamodule_windows_carry_units_and_the_fall_slots_hold_a_fall_unit(release, official):
    dm = UWBDataModule(release=str(release), stats_path=str(release / "stats.json"), window_s=16.0, train_hop_s=2.0,
                       infer_hop_s=4.0, batch_size=4, num_workers=0, frame_stride=4, diff_channel=False,
                       complex_input=True, physics=True, units=True, train_sampler="balanced", num_states=2,
                       sampler={"fall_slots": 2, "hard_slots": 0, "groups": str(release / "folds.csv")}, infer_tail="pad")
    dm.setup("fit")
    ds = dm.datasets["train"]
    seen = 0
    for batch in dm.batch_sampler:
        for i in batch:
            item = ds[i]
            assert item["x"].shape[:2] == (10, 184) and item["uphys"].shape[0] == NUM_CHANNELS
            n = int(item["unit_mask"].sum())
            assert n > 0 and (item["units"][:n, 0] >= 0).all() and (item["units"][:n, 1] <= ds.window).all()
            seen += int((item["unit_fall"] * item["unit_fall_w"]).sum() > 0)
        break
    assert seen >= 2                                        # the 2 fall slots
    val = dm.datasets["val"]
    total = len(val.store.units["val_00"]["units"])
    covered = {int(u) for j in range(len(val)) for u, m in zip(val[j]["unit_index"], val[j]["unit_mask"]) if m}
    assert covered == set(range(total))                    # every unit is decoded by some window


@needs_annotate
def test_cli_rs_fits_and_validates_with_both_decoders(release, tmp_path, official):
    from compose_config import main as compose_main
    composed = tmp_path / "composed.yaml"
    compose_main([str(composed), str(ROOT / "configs/uwb/RS.yaml")])
    tiny = tmp_path / "tiny.yaml"
    tiny.write_text(
        "run: {wandb: disabled}\n"
        "trainer: {max_epochs: 2, accelerator: cpu, devices: 1, log_every_n_steps: 1, enable_progress_bar: false,\n"
        "          check_val_every_n_epoch: 1, callbacks: [{class_path: falljepa.callbacks.WeightAveraging,\n"
        "          init_args: {start_epoch: 0, bn_batches: 2, every_n_steps: 2}}]}\n"
        "data: {window_s: 16.0, infer_hop_s: 4.0, batch_size: 4, eval_batch_size: 4, num_workers: 0}\n"
        "model: {model: {init_args: {d_model: 32, nhead: 2, frame_encoder_layers: 1, context_layers: 1,\n"
        "        bidir_layers: 1, predictor_layers: 1, unit_layers: 1}}}\n")
    args = ["-c", str(composed), "-c", str(tiny), "--data.release", str(release), "--data.annotate_release", str(release),
            "--data.stats_path", str(tmp_path / "cache" / "stats.json"), "--data.sampler.groups", str(release / "folds.csv"),
            "--data.sampler.fall_slots", "2", "--data.sampler.hard_slots", "0",
            "--run.root", str(tmp_path / "runs"), "--run.name", "RS", "--run.id", "t"]
    main(["fit"] + args)
    run_dir = tmp_path / "runs" / "UWB" / "RS" / "RS" / "t"
    swa = run_dir / "checkpoints" / "swa.ckpt"
    assert swa.exists()
    metrics = next((run_dir / "csv").glob("fit-*/metrics.csv")).read_text()
    for key in ("train/loss/Unit", "train/loss/Span", "train/loss/Rule", "val/event/F1_mean", "val/amir/F1_mean"):
        assert key in metrics, key
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", str(swa)])
    pred = run_dir / "predictions"
    assert (pred / "val-validate.csv").exists() and (pred / "val-validate.amir.csv").exists()
    assert json.loads((pred / "val-validate.report.json").read_text())["ranking_score"]["f1_mean"] >= 0
    probs = np.load(pred / "val-validate.probs.npz")
    assert probs["units/val_00"].shape[1] == 4 and len(probs["units/val_00"]) > 0
    assert probs["span/val_00"].shape[1] == 4
