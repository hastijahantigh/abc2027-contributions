import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from falljepa.cli import main
from falljepa.data.folds import fit_and_score_masks, load_assignment, stream_fold_bits
from falljepa.module import score_predictions
from falljepa.official import evaluate as official
from falljepa.tune_decoder import main as tune_main
from falljepa.data.labels import BEGIN, END, FALL, MIDDLE, NONE, Event, stream_bounds, stream_targets
from falljepa.data.uwb import StreamStore, UWBDataModule, WindowDataset, fit_normalizer, grid_starts, stream_files
from falljepa.model import FallJEPA
from falljepa.model.aggregate import StreamAccumulator, decode_events, frames_to_steps, propose_events, viterbi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def tiny_model(**kw) -> FallJEPA:
    args = dict(in_channels=7, window_frames=25, d_model=32, nhead=2, frame_width=32, frame_stride=8,
                frame_encoder_layers=1, context_layers=1, bidir_layers=1, predictor_layers=1, dropout=0.0)
    args.update(kw)
    return FallJEPA(**args).eval()


def tiny_data(release, **kw) -> UWBDataModule:
    args = dict(release=str(release), stats_path=str(release / "stats.json"), window_s=4.0, train_hop_s=2.0,
                infer_hop_s=2.0, batch_size=4, num_workers=0)
    args.update(kw)
    dm = UWBDataModule(**args)
    dm.setup("fit")
    return dm


# ------------------------------------------------------------------ labels


def test_states_are_half_open_with_capped_boundaries():
    state, weight = stream_targets([Event("s", 1.0, 2.0, False), Event("s", 3.0, 3.18, True)], 250,
                                   boundary_s=0.3, flagged_weight=0.5)
    assert state[49] == NONE and state[50] == BEGIN and state[64] == BEGIN and state[65] == MIDDLE
    assert state[84] == MIDDLE and state[85] == END and state[99] == END and state[100] == NONE
    # 9 timesteps: begin/end capped at a third of the fall
    assert list(state[150:159]) == [BEGIN] * 3 + [MIDDLE] * 3 + [END] * 3 and state[159] == NONE
    assert weight[150:159].tolist() == [0.5] * 9 and weight[149] == 1 and weight[50] == 1


# -------------------------------------------------------------------- data


def test_grid_covers_every_timestep():
    for T in (1, 999, 1000, 1001, 4321):
        covered = np.zeros(T, dtype=bool)
        for s in grid_starts(T, 1000, 500):
            covered[s:s + 1000] = True
        assert covered.all()


def test_store_reads_normalized_margins_and_padding(release):
    files = stream_files(release, "train")
    stats = fit_normalizer(list(files.values()))
    store = StreamStore(files, stats, diff_channel=True)
    sid = store.ids[0]
    x, valid = store.read(sid, -12, 200)
    assert x.shape == (7, 368, 212) and valid[:12].sum() == 0 and valid[12:].all()
    assert (x[:, :, :12] == 0).all() and (x[6, :, 12:] == 1).all()
    # The difference channel uses the real predecessor, also at a window start inside the stream
    x2, _ = store.read(sid, 100, 110)
    z = x[:3, :, 12 + 99:12 + 110]
    d = (z[:, :, 1:] - z[:, :, :-1]) / np.asarray(stats["diff_std"], dtype=np.float32)[:, :, None]
    np.testing.assert_allclose(x2[3:6], d, atol=1e-4)
    # Training statistics: the z-scored training streams have mean 0 and std 1 per (radar, bin)
    allz = np.concatenate([store.read(s, 0, store.lengths[s])[0][:3] for s in store.ids], axis=2)
    np.testing.assert_allclose(allz.mean(axis=2), 0, atol=1e-3)
    np.testing.assert_allclose(allz.std(axis=2), 1, atol=1e-3)


def test_windows_carry_labels_of_their_margins(release):
    dm = tiny_data(release)
    ds = dm.datasets["val"]
    item = ds[1]
    assert item["x"].shape == (7, 368, 200 + 24) and item["state"].shape == (224,)
    sid = ds.store.ids[int(item["stream"])]
    full, _ = stream_targets([], 1)  # noqa: F841 (just checks the import surface)
    state = dm._targets(ds.store)[sid][0]
    s = int(item["start"])
    assert torch.equal(item["state"][12:212], torch.from_numpy(state[s:s + 200]))


# ------------------------------------------------------------------- model


def test_frames_to_steps_matches_a_loop():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 8, 4)
    out, spread = frames_to_steps(logits, 2)
    w = 0.5 - 0.5 * torch.cos(2 * torch.pi * (torch.arange(8) + 0.5) / 8)
    acc, ws = torch.zeros(2, 16, 4), torch.zeros(16)
    for i in range(5):
        acc[:, 2 * i:2 * i + 8] += logits[:, i] * w[:, None]
        ws[2 * i:2 * i + 8] += w
    torch.testing.assert_close(out, acc / ws[None, :, None])
    assert spread.shape == (2, 16) and (spread >= 0).all()


def _batch(B=2, L=224, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 7, 368, L, generator=g)
    x[:, -1] = 1.0
    return {"x": x, "valid": torch.ones(B, L)}


def test_frame_isolation():
    """A frame's latent depends only on the timesteps inside it."""
    model = tiny_model()
    batch = _batch()
    z = model.encode(batch["x"], batch["valid"])["z"]
    x2 = batch["x"].clone()
    x2[:, :6, :, 40:48] += 5.0           # timesteps 40..47: inside frames 2..5 only (frame i = [8i, 8i + 32))
    z2 = model.encode(x2, batch["valid"])["z"]
    changed = (z - z2).abs().amax(-1).amax(0) > 1e-5
    assert changed.nonzero().flatten().tolist() == [2, 3, 4, 5]


def test_context_is_causal_and_heads_see_both_sides():
    model = tiny_model()
    batch = _batch()
    enc = model.encode(batch["x"], batch["valid"])
    x2 = batch["x"].clone()
    x2[:, :6, :, 200:] += 5.0             # only the last frames change
    enc2 = model.encode(x2, batch["valid"])
    first_changed = int(((enc["z"] - enc2["z"]).abs().amax(-1).amax(0) > 1e-5).nonzero()[0])
    torch.testing.assert_close(enc["h"][:, :first_changed], enc2["h"][:, :first_changed])
    assert not torch.allclose(enc["r"][:, 0], enc2["r"][:, 0])   # the bidirectional layer looks ahead


def test_forward_shapes_losses_and_padding():
    model = tiny_model()
    batch = _batch()
    batch["valid"][:, 150:] = 0           # past the stream end
    batch["x"][:, :, :, 150:] = 0
    batch["x"][:, -1, :, 150:] = 0
    batch["state"] = torch.zeros(2, 224, dtype=torch.long)
    batch["weight"] = batch["valid"].clone()
    out = model(batch)
    assert out["step_logits"].shape == (2, 200, 4) and out["frame_logits"].shape == (2 * 25 * 32, 4)
    assert out["frame_state"].shape == out["frame_weight"].shape == (2 * 25 * 32,)
    # frames entirely past the end take no part in the latents
    frame_valid = batch["valid"].unfold(-1, 32, 8).amax(-1) > 0
    assert out["z_v_proj"].shape[0] == int(frame_valid.sum())
    assert torch.isfinite(out["step_logits"]).all()



def test_no_predictor_means_no_jepa_branch():
    model = tiny_model(predictor_layers=0, pre_projector=False)
    assert not any(n.startswith(("predictor", "horizon_embed", "pre_proj.net")) for n, _ in model.named_parameters())
    out = model(_batch())
    assert not any(k in out for k in ("z_v_pred", "z_v_target", "z_v_copy"))
    assert out["step_logits"].shape == (2, 200, 4) and torch.isfinite(out["step_logits"]).all()

def test_overfits_one_batch(release):
    torch.manual_seed(0)
    dm = tiny_data(release, augment={})
    batch = next(iter(torch.utils.data.DataLoader(dm.datasets["val"], batch_size=8)))
    from falljepa.loss import MSELoss, SIGRegLoss, StateCELoss
    model = tiny_model().train()
    losses = [StateCELoss("frame_logits", "frame_state", "frame_weight"),
              StateCELoss("step_logits_flat", "step_state", "step_weight"),
              MSELoss("z_v_pred", "z_v_target"), SIGRegLoss("z_v_proj", 8, num_projections=64)]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = None
    for _ in range(60):
        out = model(batch)
        ce = losses[0](out) + losses[1](out)
        loss = ce + 0.5 * losses[2](out) + 0.1 * losses[3](out)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = ce.item() if first is None else first
    assert ce.item() < 0.3 * first


# ---------------------------------------------------------------- decoding


def test_viterbi_follows_the_fall_grammar():
    T = 200
    probs = np.tile([0.97, 0.01, 0.01, 0.01], (T, 1))
    for (a, b), k in (((50, 60), BEGIN), ((60, 85), MIDDLE), ((85, 95), END), ((150, 160), MIDDLE)):
        probs[a:b] = 0.03
        probs[a:b, k] = 0.91             # 150-160: a middle with no begin is still entered through begin
    logp = np.log(probs)
    path = viterbi(logp, switch_penalty=1.0)
    assert path[49] == NONE and path[50] == BEGIN and path[60] == MIDDLE and path[85] == END and path[95] == NONE
    for a, b in zip(path[:-1], path[1:]):
        assert (a, b) in {(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (3, 0)}
    events = decode_events(logp, np.zeros(T), switch_penalty=1.0)
    assert events[0]["t_start"] == 1.0 and events[0]["t_end"] == 1.9 and 0 < events[0]["score"] <= 1


def test_window_merge_is_exact_for_identical_windows():
    acc = StreamAccumulator({"s": 30}, window=10)
    logp = np.log(np.tile([0.7, 0.1, 0.1, 0.1], (10, 1)))
    for start in grid_starts(30, 10, 5):
        acc.add("s", start, logp, np.zeros(10))
    merged, _ = acc.result("s")
    np.testing.assert_allclose(merged, np.tile(logp[0], (30, 1)), atol=1e-9)



# ------------------------------------------------- binary labels, boundary regression (A3, A4)


def test_binary_states_and_boundary_targets():
    events = [Event("s", 1.0, 2.0, False)]
    state, _ = stream_targets(events, 150, num_states=2)
    assert set(np.unique(state)) == {NONE, FALL} and (state == FALL).sum() == 50 and state[50] == FALL
    bounds, inside = stream_bounds(events, 150)
    assert inside.sum() == 50 and not inside[49] and not inside[100]
    t = np.arange(50, 100) / 50
    np.testing.assert_allclose(t - bounds[50:100, 0], 1.0, atol=1e-6)    # every timestep points at [1, 2)
    np.testing.assert_allclose(t + bounds[50:100, 1], 2.0, atol=1e-6)
    _, centre = stream_bounds(events, 150, center=0.5)
    assert abs(int(centre.sum()) - 25) <= 1 and centre[75] and not centre[55]   # the central half


def test_two_state_viterbi_smooths_without_the_grammar():
    p = np.full(300, 0.05)
    p[100:200] = 0.9
    p[150] = 0.2                              # a dip inside the fall
    logp = np.log(np.stack([1 - p, p], axis=1))
    path = viterbi(logp, switch_penalty=2.0)
    assert (path[100:200] == FALL).all() and (path[:100] == NONE).all() and (path[200:] == NONE).all()


def test_boundary_loss_is_zero_when_exact_and_grows_with_error():
    from falljepa.loss import BoundaryDIoULoss
    loss = BoundaryDIoULoss("p", "t", "w")
    target = torch.tensor([[0.2, 0.8], [0.5, 0.5]])
    weight = torch.tensor([1.0, 1.0])
    assert loss({"p": target, "t": target, "w": weight}) < 1e-6
    near = loss({"p": target + torch.tensor([[0.1, 0.0], [0.0, 0.1]]), "t": target, "w": weight})
    far = loss({"p": target + torch.tensor([[0.5, 0.0], [0.0, 0.5]]), "t": target, "w": weight})
    assert 0 < near < far
    assert loss({"p": target * 3, "t": target, "w": torch.zeros(2)}) == 0      # no fall timestep


def _falls(T=1000, falls=((400, 500),)):
    """P(fall) high on [a - 20, b + 30) of each fall (a late end, as the model does) and regressed
    bounds pointing at the true [a, b) in seconds."""
    p = np.full(T, 0.02)
    bounds = np.zeros((T, 2))
    for a, b in falls:
        p[a - 20:b + 30] = 0.9
        bounds[a - 20:b + 30] = a / 50, b / 50
    return np.log(np.stack([1 - p, p], axis=1)), np.zeros(T), bounds


def test_refine_moves_the_run_to_the_regressed_boundaries():
    logp, spread, bounds = _falls()
    plain = decode_events(logp, spread, kind="viterbi", switch_penalty=2.0)
    refined = decode_events(logp, spread, kind="viterbi", switch_penalty=2.0, bounds=bounds, refine=True)
    assert len(plain) == len(refined) == 1
    assert plain[0]["t_start"] == pytest.approx(380 / 50) and plain[0]["t_end"] == pytest.approx(530 / 50)
    assert refined[0]["t_start"] == pytest.approx(8.0) and refined[0]["t_end"] == pytest.approx(10.0)


def test_proposals_keep_one_event_per_fall():
    logp, spread, bounds = _falls()
    events = propose_events(logp, spread, bounds, proposal_threshold=0.5, nms_iou=0.3)
    assert len(events) == 1 and events[0]["t_start"] == pytest.approx(8.0) and events[0]["t_end"] == pytest.approx(10.0)
    logp, spread, bounds = _falls(T=2000, falls=((400, 500), (1400, 1450)))
    assert [(e["t_start"], e["t_end"]) for e in propose_events(logp, spread, bounds)] == [(8.0, 10.0), (28.0, 29.0)]
    quiet = np.log(np.tile([0.9, 0.1], (500, 1)))              # nothing above the threshold
    assert propose_events(quiet, np.zeros(500), bounds[:500]) == []


def test_regression_head_outputs_and_window_merge():
    model = tiny_model(regression=True)
    batch = _batch()
    batch["bounds"] = torch.rand(2, 224, 2)
    batch["bound_weight"] = torch.ones(2, 224)
    out = model(batch)
    assert out["step_bounds"].shape == (2, 200, 2) and (out["step_bounds"] >= 0).all()
    assert out["bound_pred"].shape == out["bound_target"].shape == (400, 2) and out["bound_weight"].shape == (400,)
    acc = StreamAccumulator({"s": 300}, 200)
    for start in (0, 100):
        acc.add("s", start, np.log(np.full((200, 4), 0.25)), np.zeros(200), np.full((200, 2), [3.0, 4.0]))
    np.testing.assert_allclose(acc.result_bounds("s"), np.tile([3.0, 4.0], (300, 1)))

# --------------------------------------------------------------------- CLI


def test_cli_fit_writes_a_scored_submission(release, tmp_path):
    main(["fit", "-c", str(ROOT / "configs/uwb/U0.yaml"), "-c", str(ROOT / "tests/configs/tiny.yaml"),
          "--data.release", str(release), "--data.stats_path", str(tmp_path / "stats.json"),
          "--run.root", str(tmp_path / "runs"), "--run.name", "tiny"])
    run_dir = next((tmp_path / "runs").glob("**/tiny/*"))
    assert (run_dir / "config.yaml").exists() and list((run_dir / "checkpoints").glob("best-*.ckpt"))
    sub = run_dir / "predictions" / "val-epoch001.csv"
    assert sub.exists() and (run_dir / "predictions" / "val-epoch001.report.json").exists()
    with sub.open() as f:
        rows = list(csv.DictReader(f))
    assert all(0 <= float(r["t_start"]) < float(r["t_end"]) <= 30.0 for r in rows)
    metrics = next((run_dir / "csv").glob("fit-*/metrics.csv")).read_text()
    assert "val/event/F1_mean" in metrics and "train/loss/MSE_Latent" in metrics

    # validate from the checkpoint alone
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", "best"])
    assert (run_dir / "predictions" / "val-validate.csv").exists()


def test_last_checkpoint_is_the_latest_epoch_and_resumes_from_it(release, tmp_path):
    """last.ckpt is the final epoch, not the best one (Lightning's save_last follows only a new top-k
    save), and a resume continues after it. Monitoring the epoch with mode min pins the best at epoch 0."""
    args = ["-c", str(ROOT / "configs/uwb/U0.yaml"), "-c", str(ROOT / "tests/configs/tiny.yaml"),
            "--data.release", str(release), "--data.stats_path", str(tmp_path / "stats.json"),
            "--run.root", str(tmp_path / "runs"), "--run.name", "tiny", "--run.id", "t",
            "--checkpoint.monitor", "epoch", "--checkpoint.mode", "min"]
    main(["fit", *args, "--trainer.max_epochs", "3"])
    checkpoints = tmp_path / "runs" / "UWB" / "Frame" / "tiny" / "t" / "checkpoints"

    def epoch(path):
        return torch.load(path, map_location="cpu", weights_only=False)["epoch"]

    assert sorted(p.name for p in checkpoints.glob("*.ckpt")) == ["best-epoch=000.ckpt", "last.ckpt"]
    assert epoch(checkpoints / "best-epoch=000.ckpt") == 0
    assert epoch(checkpoints / "last.ckpt") == 2

    main(["fit", *args, "--trainer.max_epochs", "4", "--ckpt_path", "last"])
    assert sorted(p.name for p in checkpoints.glob("*.ckpt")) == ["best-epoch=000.ckpt", "last.ckpt"]
    assert epoch(checkpoints / "last.ckpt") == 3
    resumed = sorted(checkpoints.parent.glob("csv/fit-*/metrics.csv"))[-1]
    with resumed.open() as f:
        epochs = {int(r["epoch"]) for r in csv.DictReader(f) if r.get("val/event/F1_mean")}
    assert epochs == {3}



def test_overlays_deep_merge_and_delete(tmp_path):
    """A later overlay that changes one loss keeps the losses an earlier one removed (LightningCLI's
    -c chaining would replace the whole dict)."""
    from compose_config import compose

    later = tmp_path / "later.yaml"
    later.write_text("model:\n  losses:\n    State_Frame:\n      init_args: {class_weights: [1.0, 3.0]}\n"
                     "  model:\n    init_args: {context_layers: 0}\n")
    config = compose([ROOT / "configs/uwb/U0.yaml", ROOT / "configs/uwb/ablations/A1b.yaml", later])
    model = config["model"]
    assert set(model["losses"]) == set(model["loss_weights"]) == {"State_Frame", "State_Step"}
    assert model["losses"]["State_Frame"]["init_args"] == {
        "logits_key": "frame_logits", "target_key": "frame_state", "weight_key": "frame_weight",
        "class_weights": [1.0, 3.0]}
    args = model["model"]["init_args"]
    assert (args["predictor_layers"], args["pre_projector"], args["context_layers"], args["bidir_layers"]) == (0, False, 0, 2)

def test_cli_ablation_overlays_and_comparison(release, tmp_path, capsys):
    """A1b+A2 (no JEPA, no Hv, no SIGReg, all context bidirectional) fits, validates its last
    checkpoint and is summarized by tools/compare_ablations.py."""
    from compare_ablations import main as compare_main
    from compose_config import main as compose_main

    composed = tmp_path / "composed.yaml"
    compose_main([str(composed), *[str(ROOT / f) for f in ("configs/uwb/U0.yaml", "configs/uwb/ablation.yaml",
                                                           "configs/uwb/ablations/A1b.yaml", "configs/uwb/ablations/A2.yaml")]])
    main(["fit", "-c", str(composed), "-c", str(ROOT / "tests/configs/tiny.yaml"),
          "--model.model.init_args.predictor_layers", "0",
          "--data.release", str(release), "--data.stats_path", str(tmp_path / "stats.json"),
          "--run.root", str(tmp_path / "runs"), "--run.name", "A1b+A2-s0", "--run.id", "t"])
    run_dir = tmp_path / "runs" / "UWB" / "Ablation" / "A1b+A2-s0" / "t"
    metrics = next((run_dir / "csv").glob("fit-*/metrics.csv")).read_text()
    assert "val/event/F1_mean" in metrics
    assert "MSE_Latent" not in metrics and "SIGReg" not in metrics and "CopyRatio" not in metrics
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", "last"])
    assert (run_dir / "predictions" / "val-validate.probs.npz").exists()

    capsys.readouterr()
    compare_main(["--root", str(tmp_path / "runs" / "UWB" / "Ablation"), "--last", "2",
                  "--events", str(release / "streams_events.csv"), "--ref", "A1b+A2"])
    table = capsys.readouterr().out
    assert "| A1b+A2 | 1 | 2 |" in table and "nan" not in table.split("\n")[2].split("|")[-4]

    # A run that stopped before trainer.max_epochs is left out, or kept and marked with --incomplete
    config = run_dir / "config.yaml"
    config.write_text(config.read_text().replace("max_epochs: 2", "max_epochs: 3"))
    with pytest.raises(SystemExit):
        compare_main(["--root", str(tmp_path / "runs" / "UWB" / "Ablation"), "--last", "2", "--best"])
    capsys.readouterr()
    compare_main(["--root", str(tmp_path / "runs" / "UWB" / "Ablation"), "--last", "2", "--incomplete"])
    assert "| A1b+A2 | 1 | 2* |" in capsys.readouterr().out


@pytest.mark.parametrize("spec", ["A1+A3", "A1b+A4a", "A1+A4b", "A0+A4-cs"])
def test_cli_stage2_ablations_fit_and_validate(release, tmp_path, spec):
    """Every stage-2 overlay stacked on a stage-1 one composes, fits, validates its last checkpoint
    and writes probabilities (with the regressed boundaries when the model has them)."""
    from compose_config import main as compose_main

    composed = tmp_path / "composed.yaml"
    compose_main([str(composed), str(ROOT / "configs/uwb/U0.yaml"), str(ROOT / "configs/uwb/ablation.yaml"),
                  *[str(ROOT / f"configs/uwb/ablations/{a}.yaml") for a in spec.split("+")]])
    args = ["-c", str(composed), "-c", str(ROOT / "tests/configs/tiny.yaml")]
    if not spec.startswith("A0"):
        args += ["--model.model.init_args.predictor_layers", "0"]
    main(["fit", *args, "--data.release", str(release), "--data.stats_path", str(tmp_path / "stats.json"),
          "--run.root", str(tmp_path / "runs"), "--run.name", spec, "--run.id", "t"])
    run_dir = tmp_path / "runs" / "UWB" / "Ablation" / spec / "t"
    metrics = next((run_dir / "csv").glob("fit-*/metrics.csv")).read_text()
    assert "val/event/F1_mean" in metrics
    assert ("train/loss/Boundary" in metrics) == ("A4" in spec)
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", "last"])
    probs = np.load(run_dir / "predictions" / "val-validate.probs.npz")
    logp = [k for k in probs.files if k.startswith("logp/")]
    assert probs[logp[0]].shape[1] == (2 if spec.endswith(("A3", "A4b")) else 4)
    assert any(k.startswith("bounds/") for k in probs.files) == ("A4" in spec)

# -------------------------------------------------------- cross-validation


def test_fold_bits_cover_streams_and_joins_belong_to_both_sides(release):
    files = stream_files(release, "train")
    lengths = StreamStore(files, fit_normalizer(list(files.values())), False).lengths
    bits = stream_fold_bits(release / "streams_composition.csv", load_assignment(release / "folds.csv"), lengths)
    b = bits["train_00"]
    assert (b != 0).all()
    assert b[389] == 1 and b[390] == 3 and b[399] == 3 and b[400] == 2   # segment 1 spans 7.8-16 s
    fit, score = fit_and_score_masks(bits, 1)
    assert (fit["train_00"] ^ score["train_00"]).all() and not fit["train_00"][390]


def test_masked_normalizer_uses_only_kept_timesteps(release):
    files = stream_files(release, "train")
    rng = np.random.default_rng(0)
    masks = {}
    for path in files.values():
        import h5py
        with h5py.File(path) as h:
            masks[str(path)] = rng.random(h["cir"].shape[1]) < 0.5
    stats = fit_normalizer(list(files.values()), masks)
    import h5py
    x = np.concatenate([h5py.File(p)["cir"][:, masks[str(p)]] for p in files.values()], axis=1).astype(np.float64)
    np.testing.assert_allclose(stats["mean"], x.mean(axis=1), atol=1e-6)
    np.testing.assert_allclose(stats["std"], x.std(axis=1), atol=1e-6)


def test_scoring_inside_a_mask_ignores_everything_else():
    ground = [{"stream_id": "s", "t_start": 2.0, "t_end": 3.0, "ignore": False},
              {"stream_id": "s", "t_start": 12.0, "t_end": 13.0, "ignore": False}]
    preds = [{"stream_id": "s", "t_start": 2.0, "t_end": 3.0, "score": 1.0, "t_alarm": 2.0},
             {"stream_id": "s", "t_start": 15.0, "t_end": 16.0, "score": 1.0, "t_alarm": 15.0}]
    mask = np.zeros(1000, dtype=bool)
    mask[:500] = True                                  # the first 10 s only
    report = score_predictions(preds, ground, {"s": 1000}, {"s": mask})
    assert report["by_theta"]["0.3"]["tp"] == 1 and report["by_theta"]["0.3"]["fp"] == 0
    assert report["by_theta"]["0.3"]["fn"] == 0
    full = score_predictions(preds, ground, {"s": 1000})
    assert full["by_theta"]["0.3"]["fp"] == 1 and full["by_theta"]["0.3"]["fn"] == 1


def test_cli_cross_validation_and_decoder_tuning(release, tmp_path):
    args = ["-c", str(ROOT / "configs/uwb/U0.yaml"), "-c", str(ROOT / "configs/uwb/cv.yaml"),
            "-c", str(ROOT / "tests/configs/tiny.yaml"), "--data.release", str(release),
            "--data.stats_path", str(tmp_path / "stats.json"), "--data.cv_assignment", str(release / "folds.csv"),
            "--data.cv_fold", "1", "--run.root", str(tmp_path / "runs"), "--run.name", "cv1"]
    main(["fit", *args])
    run_dir = next((tmp_path / "runs").glob("**/cv1/*"))
    assert (tmp_path / "stats_cv1.json").exists()
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", "best"])
    probs = run_dir / "predictions" / "val-validate.probs.npz"
    data = np.load(probs)
    assert {"logp/train_00", "spread/train_00", "mask/train_00"} <= set(data.files)
    assert 0 < data["mask/train_00"].mean() < 1          # only fold 1 is scored
    out = tmp_path / "decoder.json"
    tune_main(["--events", str(release / "streams_events.csv"), "--probs", str(probs), "--out", str(out)])
    import json
    best = json.loads(out.read_text())
    assert best["decoder"]["kind"] in ("viterbi", "threshold") and "ranking_score" in best["report"]


# ------------------------------------------------- Amir's pipeline (AP.yaml)

ANNOTATE = ROOT / "abc2027_uwb_train_v1_sample" / "abc2027_uwb_train_v1" / "src" / "abc2027" / "annotate.py"


def test_boundary_mask_and_flagged_mask_as_amir():
    state, weight = stream_targets([Event("s", 2.0, 4.0, False), Event("s", 8.0, 9.0, True)], 600, num_states=2,
                                   flagged_weight=0.0, flagged_margin_s=2.0, boundary_mask_s=0.2)
    assert state[100:200].tolist() == [FALL] * 100 and state[99] == NONE
    assert weight[89] == 1 and weight[90:110].tolist() == [0] * 20 and weight[110] == 1   # +-0.2 s at the start
    assert weight[190:210].tolist() == [0] * 20 and weight[210] == 1
    assert weight[300:550].tolist() == [0] * 250 and weight[299] == 1 and weight[550] == 1  # flagged +-2 s


def test_grid_tails():
    assert grid_starts(1000, 400, 100, "none") == [0, 100, 200, 300, 400, 500, 600]
    assert grid_starts(1000, 400, 100, "pad")[-1] == 900
    assert grid_starts(1050, 400, 100, "end")[-1] == 650
    assert grid_starts(300, 400, 100, "none") == []


def test_physics_motion_ignores_sample_and_hold():
    from falljepa.data.physics import profile_steps, raw_motion, stream_physics
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(200, 8))
    held = np.repeat(raw, 2, axis=0)                 # every raw frame shown twice on the grid
    m = raw_motion(held)
    grid_diff = np.r_[0, np.square(np.diff(held, axis=0)).mean(1)]
    assert (grid_diff[1::2] == 0).all()                              # the grid difference: a zero at every repeat
    assert (m > 0.05).all() and abs(m.mean() - 2.0) < 0.3            # raw-frame motion: none, E||d||^2 / bins = 2
    z = np.zeros((400, 8))
    z[250:] = 1.0                                    # a lasting profile step at t = 250
    steps = profile_steps(z)                         # sides of 0.2, 1 and 3 s, 0.2 s apart
    assert steps[:, 250].min() > 0.9 and steps[:, 50].max() == 0
    assert steps[0, 230] == 0 and steps[2, 230] > 0  # 10 timesteps before it only the 3 s side reaches it
    assert stream_physics(np.zeros((3, 50, 8), np.float32)).shape == (12, 50)


def test_interior_normalizer_skips_crossfades_and_counts_a_recording_once(tmp_path):
    from falljepa.data.uwb import interior_masks
    comp = tmp_path / "comp.csv"
    comp.write_text("stream_id,path,t_start,t_end\n"
                    "train_00,a,0.0,4.2\ntrain_00,b,4.0,8.2\ntrain_00,a,8.0,10.0\n")
    masks = interior_masks(comp, {"train_00": Path("s0.h5")}, {"train_00": 500})["s0.h5"]
    assert masks[:200].all() and not masks[200:210].any() and masks[210:400].all() and not masks[400:].any()


def test_wandb_id_is_unique_per_run_and_stable():
    """Runs of one tools/run_ablations.sh batch share run.id (the TAG); their W&B ids must differ, and
    every invocation of one run (fit, resume, validate) must get the same one."""
    from types import SimpleNamespace
    from falljepa.cli import wandb_id
    tag = "20261015-122938"
    runs = [SimpleNamespace(group="UWB/AP", name=n, id=tag) for n in ("rshift-s0", "rshift-s1", "rshift+rdrop-s0")]
    ids = [wandb_id(r) for r in runs]
    assert len(set(ids)) == 3 and ids[0] == "UWB_AP-rshift-s0-20261015-122938"
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids)
    assert wandb_id(SimpleNamespace(group="UWB/AP", name="rshift-s0", id=tag)) == ids[0]
    long = [SimpleNamespace(group="UWB/AP", name="rshift+rdrop+adlneg+jepaplus+" + x, id=tag) for x in ("a-s0", "b-s0")]
    assert len({wandb_id(r) for r in long}) == 2 and all(len(wandb_id(r)) <= 64 for r in long)


def test_balanced_sampler_slots():
    from falljepa.data.sampler import BalancedBatchSampler
    n = 60
    sources = [f"s{i % 12}" for i in range(n)]
    groups = [i % 12 % 3 for i in range(n)]
    hard = [i % 12 >= 10 for i in range(n)]
    fall_ok = np.array([(i % 12) < 4 for i in range(n)])
    none_ok = np.ones(n, bool)
    sampler = BalancedBatchSampler(sources, groups, hard, fall_ok, none_ok, batch_size=8, fall_slots=4, hard_slots=2, seed=1)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 8
    for b in batches:
        assert len(b) == 8
        assert sum(fall_ok[i] and not hard[i] for i in b) >= 4      # the fall slots
        assert sum(hard[i] for i in b) == 2                          # hard negatives only through their slots
        assert len({sources[i] for i in b}) >= 7                     # sources distinct within a batch where possible
    sampler.set_epoch(1)
    assert list(sampler) != batches
    sampler.set_epoch(0)
    assert list(sampler) == batches


def test_balanced_sampler_join_slots():
    from falljepa.data.sampler import BalancedBatchSampler
    n = 60
    sources = [f"s{i % 12}" for i in range(n)]
    groups = [i % 12 % 3 for i in range(n)]
    hard = [i % 12 >= 10 for i in range(n)]
    join = [i % 12 in (6, 7) for i in range(n)]                   # two ADL sources start after a join
    fall_ok = np.array([(i % 12) < 4 for i in range(n)])
    none_ok = np.ones(n, bool)
    sampler = BalancedBatchSampler(sources, groups, hard, fall_ok, none_ok, batch_size=8, fall_slots=4, hard_slots=1,
                                   seed=1, join=join, join_slots=2)
    for b in sampler:
        assert len(b) == 8 and sum(hard[i] for i in b) == 1
        assert sum(join[i] for i in b) >= 2                          # the join slots (plain no-fall slots may add more)
    plain = BalancedBatchSampler(sources, groups, hard, fall_ok, none_ok, batch_size=8, fall_slots=4, hard_slots=1, seed=1)
    assert plain.join_slots == 0


def test_range_shift_and_radar_drop_augment(release):
    """A range shift moves all radars by the same bins; a dropped radar loses its physics features too."""
    ds = tiny_data(release, physics=True, augment={"range_shift": 3}).datasets["train"]
    x0 = tiny_data(release, physics=True, augment={}).datasets["train"][5]["x"].numpy()
    x1 = ds[5]["x"].numpy()
    shifts = [s for s in range(-3, 4) if np.allclose(np.roll(x0[:-1], s, axis=1)[:, 3:-3], x1[:-1, 3:-3])]
    assert shifts                                                    # one shift explains every radar
    assert np.array_equal(x0[-1], x1[-1])                            # the valid channel is untouched
    ds.augment = {"radar_drop_p": 1.0}
    item = ds[5]
    zero = [r for r in range(3) if not item["x"][r].any()]
    assert len(zero) == 1
    r = zero[0]
    assert not item["phys"][4 * r:4 * r + 4].any() and item["phys"].abs().sum() > 0


def test_triangular_probability_merge_keeps_a_constant():
    acc = StreamAccumulator({"s": 50}, window=20, num_states=2, weight="triangular", merge="prob")
    lp = np.log(np.tile([0.7, 0.3], (20, 1)))
    for start in range(0, 50, 5):
        acc.add("s", start, lp, np.zeros(20))
    logp, _ = acc.result("s")
    assert np.allclose(np.exp(logp), [0.7, 0.3])


def test_join_peaks_find_the_synthetic_splices(release):
    """Synthetic streams change their background on all radars every 8 s: every splice is found
    (their falls also move all radars at once, so they score as joins too)."""
    from falljepa.postprocess import join_peaks, join_score
    import h5py
    with h5py.File(release / "streams" / "train" / "stream_00.h5") as f:
        cir = f["cir"][:]
    peaks = join_peaks(join_score(cir))
    assert all(np.abs(peaks - k).min() <= 10 for k in (400, 800, 1200, 1600))


def test_physics_model_is_small_and_runs():
    m = FallJEPA(in_channels=4, window_frames=200, d_model=64, nhead=4, frame_width=32, frame_stride=4,
                 frame_encoder_layers=2, context_layers=2, bidir_layers=2, predictor_layers=0, num_states=2,
                 mlp_ratio=2, pre_proj_ratio=2, physics_channels=12).eval()
    assert 250_000 < sum(p.numel() for p in m.parameters()) < 300_000
    out = m({"x": torch.zeros(2, 4, 368, 828), "valid": torch.ones(2, 828), "phys": torch.randn(2, 12, 828)})
    assert out["step_logits"].shape == (2, 800, 2)


@pytest.mark.parametrize("variant", ["JEPA", "noJEPA", "rshift+rdrop+adlneg+jepaplus", "rdrop+remix",
                                     "rdrop+dswa", "rdrop+sam", "rdrop+band", "rdrop30", "rdrop+swa5", "rdrop+gain"])
def test_cli_amir_pipeline_fits_averages_and_validates(release, tmp_path, variant):
    """AP + an overlay: interior normalizer, physics, balanced sampler, SWA checkpoint and Amir's
    post-processing (needs the official annotate.py, from the release sample)."""
    if not ANNOTATE.exists():
        pytest.skip("the official annotate.py is not available")
    import shutil
    from compose_config import main as compose_main
    (release / "src" / "abc2027").mkdir(parents=True, exist_ok=True)
    shutil.copy(ANNOTATE, release / "src" / "abc2027" / "annotate.py")

    composed = tmp_path / "composed.yaml"
    compose_main([str(composed), str(ROOT / "configs/uwb/AP.yaml")]
                 + [str(ROOT / f"configs/uwb/ablations/{part}.yaml") for part in variant.split("+")])
    swa_from_start = tmp_path / "swa.yaml"           # average both epochs of the tiny run
    every = ", every_n_steps: 2" if "dswa" in variant else ""
    swa_from_start.write_text("trainer:\n  callbacks:\n    - class_path: falljepa.callbacks.WeightAveraging\n"
                              f"      init_args: {{start_epoch: 0, bn_batches: 2{every}}}\n")
    main(["fit", "-c", str(composed), "-c", str(ROOT / "tests/configs/tiny.yaml"),
          "--data.release", str(release), "--data.stats_path", str(tmp_path / "stats.json"),
          "--data.sampler.groups", str(release / "folds.csv"), "--data.sampler.fall_slots", "2",
          "--data.window_s", "8.0", "-c", str(swa_from_start),
          "--run.root", str(tmp_path / "runs"), "--run.name", variant, "--run.id", "t"])
    run_dir = tmp_path / "runs" / "UWB" / "AP" / variant / "t"
    swa = run_dir / "checkpoints" / "swa.ckpt"
    assert swa.exists()
    metrics = next((run_dir / "csv").glob("fit-*/metrics.csv")).read_text()
    assert "val/event/F1_mean" in metrics and ("train/loss/MSE_Latent" in metrics) == (variant != "noJEPA")
    assert ("train/metric/SAM_grad_norm" in metrics) == ("sam" in variant)
    assert json.loads((tmp_path / "stats.json").read_text())["normalizer"] == "interior"
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", str(swa)])
    probs = np.load(run_dir / "predictions" / "val-validate.probs.npz")
    assert probs["logp/val_00"].shape[1] == 2
    assert (run_dir / "predictions" / "val-validate.report.json").exists()


def test_remix_moves_labels_with_their_signal(release):
    """A remixed window keeps every training fall whole, and the signal under its label is the signal
    of that fall in its training stream (the labels move with their timesteps)."""
    from falljepa.data.remix import CONTEXT
    dm = tiny_data(release, physics=True, augment={"remix_p": 1.0})
    ds = dm.datasets["train"]
    plain = tiny_data(release, physics=True, augment={}).datasets["train"]
    store, W = ds.store, ds.window + 2 * ds.margin
    falls = {}
    for sid, (state, *_rest) in ds.targets.items():
        edges = np.flatnonzero(np.diff(np.r_[0, (state > 0).astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            falls.setdefault(b - a, []).append((sid, a, b))
    seen = remixed = 0
    for idx in range(len(ds)):
        item, ref = ds[idx], plain[idx]
        assert item.keys() == ref.keys() and all(item[k].shape == ref[k].shape for k in item)
        remixed += not torch.equal(item["x"], ref["x"])
        assert item["valid"].all() and torch.isfinite(item["phys"]).all()
        st = item["state"].numpy()
        edges = np.flatnonzero(np.diff(np.r_[0, (st > 0).astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if a == 0 or b == W:
                continue                                  # a fall of a piece that the window cuts
            x = item["x"][:3, :, a:b].numpy()
            matches = [np.allclose(x, store.read_z(sid, fa, fb).transpose(0, 2, 1), atol=1e-5)
                       for sid, fa, fb in falls.get(b - a, [])]
            assert any(matches), f"window {idx}: the fall at {a}-{b} is not a training fall"
            seen += 1
    assert seen > 0 and remixed > len(ds) // 2
    again = ds[3]
    assert torch.equal(again["x"], ds[3]["x"])          # the same draw for the same (seed, epoch, index)
    assert CONTEXT >= 155


def test_band_gain_and_unknown_augment_keys(release):
    from falljepa.data.uwb import band_gains
    g = band_gains(np.random.default_rng(0), 368, 3.0, (2, 5))
    assert g.shape == (3, 368) and (g >= 10 ** (-3 / 20) - 1e-6).all() and (g <= 10 ** (3 / 20) + 1e-6).all()
    assert not np.allclose(g[0], g[0, 0])
    ds = tiny_data(release, augment={"band_gain_db": 3.0}).datasets["train"]
    x0 = tiny_data(release, augment={}).datasets["train"][2]["x"].numpy()
    x1 = ds[2]["x"].numpy()
    assert np.array_equal(x0[-1], x1[-1]) and not np.allclose(x0[:-1], x1[:-1])
    with pytest.raises(AssertionError, match="unknown augment keys"):
        tiny_data(release, augment={"radar_dropout": 0.1})


def test_dense_weight_averaging_snapshots_every_n_steps():
    from types import SimpleNamespace
    from falljepa.callbacks import WeightAveraging
    module = SimpleNamespace(model=torch.nn.Linear(2, 1, bias=False))
    cb = WeightAveraging(start_epoch=1, every_n_steps=2)
    values = []
    for step in range(1, 9):
        module.model.weight.data.fill_(float(step))
        trainer = SimpleNamespace(current_epoch=step // 4, global_step=step)
        cb.on_train_batch_end(trainer, module, None, None, step)
        if step >= 4 and step % 2 == 0:
            values.append(step)
    cb.on_train_epoch_end(SimpleNamespace(current_epoch=2), module)      # no extra epoch snapshot
    assert cb.count == len(values)
    assert torch.allclose(cb.sum["weight"] / cb.count, torch.full((1, 2), float(np.mean(values)), dtype=torch.float64))
