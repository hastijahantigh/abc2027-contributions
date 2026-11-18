"""The join-free model (docs/falljepa_join_free.md): set matching, grouping, event physics, the heads,
chunks in the data and the model, the stream stage at inference and the JF config end to end; and the
JF2 options (section 11): proximity groups, centre suppression, stable anchors, the expected-tIoU loss,
the clutter-removed channels, the background swap and the JF2 config end to end; and the JF3 options
(section 12): the labeller's transition rule, the transition candidate, the late reference spans and
the JF3 config end to end."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from falljepa.cli import main
from falljepa.data.uwb import HALF_BINS, MAX_FALLS, UWBDataModule, moving_mean
from falljepa.loss import SpanIoULoss, SpanSoftCELoss, UnitBCELoss
from falljepa.model import FallSetJEPA
from falljepa.model.set_heads import (CUT, FLAGGED, SCORED, FallQueries, GroupRefiner, event_physics, match, pad_groups,
                                     transition_bounds)
from falljepa.model.sets import (decode_queries, group_proposals, group_region, nms, proposals, stream_groups,
                                 TokenAccumulator)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def _logp(Q, L, peaks):
    """Log-probabilities [Q, L, 2] peaked at (start, end) per prediction."""
    out = np.full((Q, L, 2), -20.0)
    for q, (a, b) in enumerate(peaks):
        out[q, a, 0] = out[q, b, 1] = 0.0
    return out


def test_match_assigns_by_span_and_ignores_cut_and_flagged_falls():
    logit = np.array([2.0, 2.0, -3.0])
    logp = _logp(3, 100, [(60, 80), (10, 20), (0, 1)])
    target, weight, span, span_w = match(logit, logp, [(10, 20, SCORED), (60, 80, FLAGGED), (90, 120, CUT)])
    assert target.tolist() == [0, 1, 0] and tuple(span[1]) == (10, 20) and span_w[1] == 1
    assert weight[0] == 0 and span_w[0] == 0.5 and tuple(span[0]) == (60, 80)    # the flagged fall: span only
    assert weight[2] == 0 and span_w[2] == 0                                    # the cut fall absorbs the third
    target, weight, _, _ = match(logit, logp, [])
    assert target.sum() == 0 and weight.tolist() == [1, 1, 1]


def test_groups_join_overlapping_spans_and_keep_neighbours_apart():
    starts = np.array([100, 104, 98, 400, 402])
    ends = np.array([180, 178, 185, 470, 468])
    p = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
    groups = group_proposals(starts, ends, p)
    assert [g.tolist() for g in groups] == [[0, 1, 2], [3, 4]]
    j0, j1 = group_region(starts[:3], ends[:3], (100, 180), 150, 4, 250, 1000)
    assert (j0, j1) == (0, 84)                                                 # 98 - 150 clips at 0; 185 + 150
    j0, j1 = group_region(np.array([0]), np.array([3000]), (1400, 1500), 150, 4, 250, 1000)
    assert j1 - j0 == 250 and j0 * 4 <= 1400 and j1 * 4 >= 1500               # long: centred on the span


def test_event_physics_sees_a_new_scene_on_one_radar():
    G, L, H = 1, 600, 8
    sig = torch.randn(G, 6, H, L) * 0.01
    sig[:, 1, :, 300:] += 2.0                       # radar 1's I changes for good from step 300
    valid = torch.ones(G, L, dtype=torch.bool)
    chans, scal = event_physics(sig, valid, torch.tensor([[250, 350]]), groups=2)
    assert chans.shape == (G, 11, L) and scal.shape == (G, 4)
    assert scal[0, 1] > scal[0, 0] + 2 and scal[0, 1] > scal[0, 2] + 2       # log change: radar 1 only
    state = chans[0, 1]
    assert state[150] < 0.2 and state[500] > 0.8                              # before -> after
    assert chans[0, 7, 270] > 0.5                                             # the rise of radar 1 at the change


def test_heads_shapes_and_their_losses_train_a_span():
    torch.manual_seed(0)
    d, Fn, s = 16, 20, 4
    heads = FallQueries(d, 2, num_queries=3, layers=1, stride=s, step_channels=0)
    tokens = torch.randn(2, Fn, d)
    tokens[:, 10] += 3.0                                                       # something at token 10
    valid = torch.ones(2, Fn, dtype=torch.bool)
    step_valid = torch.ones(2, Fn * s, dtype=torch.bool)
    opt = torch.optim.Adam(heads.parameters(), lr=1e-2)
    bce, ce = UnitBCELoss("q_logit", "q_target", "q_w"), SpanSoftCELoss("q_span", "q_span_w", "q_span_logits", "q_domain", sigma=1.0)
    first = None
    for _ in range(60):
        h, logit, span = heads(tokens, valid, step_valid)
        assert h.shape == (2, 3, d) and logit.shape == (2, 3) and span.shape == (2, 3, Fn * s, 2)
        logp = torch.log_softmax(span.detach(), 2).numpy()
        rows = [match(logit[i].detach().numpy(), logp[i], [(41, 47, SCORED)]) for i in range(2)]
        t, w, sp, sw = (torch.from_numpy(np.concatenate(x)) for x in zip(*rows))
        out = {"q_logit": logit.reshape(-1), "q_target": t, "q_w": w, "q_span": sp, "q_span_w": sw,
               "q_span_logits": span.reshape(6, Fn * s, 2), "q_domain": step_valid.repeat_interleave(3, 0),
               "step_logits": logit}
        loss = bce(out) + ce(out)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.5 * first
    assert abs(int(span[0, int(logit[0].argmax()), :, 0].argmax()) - 41) <= 2


def test_refiner_runs_on_padded_groups_with_and_without_members():
    torch.manual_seed(0)
    d, s = 16, 4
    refiner = GroupRefiner(d, 2, slots=2, layers=1, stride=s)
    items = []
    for R, P in ((30, 3), (12, 0)):
        items.append({"tokens": torch.randn(R, d), "signal": torch.randn(6, 8, R * s),
                      "step_valid": torch.ones(R * s, dtype=torch.bool), "span": torch.tensor([20, 60]),
                      "members": torch.randn(P, d), "member_features": torch.randn(P, 5), "stats": torch.randn(4)})
    g = pad_groups(items, groups=2)
    assert g["tokens"].shape == (2, 30, d) and g["member_valid"].sum() == 3 and not g["token_valid"][1, 12:].any()
    res = refiner(g)
    assert res["logit"].shape == (2, 2) and res["span_logits"].shape == (2, 2, 120, 2)
    assert (res["span_logits"][1, :, 48:] < -1e3).all()                       # past the second region
    res["logit"].sum().backward()


def test_proposals_merge_across_windows_into_one_event():
    W, d = 200, 4
    rows = []
    for offset in (0, 50, 100):                    # the same fall at stream steps 120-170, seen by 3 windows
        logp = _logp(2, W, [(120 - offset, 170 - offset), (0, 5)])
        rows += proposals(np.array([1.0 + offset / 100, -5.0]), logp, np.zeros((2, d)), offset, W, 0.05)
    assert len(rows) == 3 and all((r["start"], r["end"]) == (120, 170) for r in rows)
    assert len(stream_groups(rows, 0.2)) == 1
    events = decode_queries(rows, 400)
    assert len(events) == 1 and abs(events[0]["t_start"] - 2.4) < 0.05 and abs(events[0]["t_end"] - 3.4) < 0.05
    assert events[0]["score"] == 1 / (1 + np.exp(-1.5))                     # the most central window (offset 50)
    assert len(nms([{"t_start": 1.0, "t_end": 2.0, "score": 0.9}, {"t_start": 1.1, "t_end": 2.0, "score": 0.6},
                    {"t_start": 5.0, "t_end": 6.0, "score": 0.5}])) == 2


def test_token_accumulator_weights_window_centres():
    acc = TokenAccumulator({"s": 40}, stride=4, d_model=1)
    acc.add("s", 0, np.ones((6, 1)), np.ones(6, bool))
    acc.add("s", 16, 3 * np.ones((6, 1)), np.ones(6, bool))
    mean, valid = acc.result("s")
    assert valid.all() and mean[0, 0] == 1 and mean[9, 0] == 3
    assert 1 < mean[4, 0] < 3 and mean[5, 0] > mean[4, 0]                     # where the windows overlap


def _datamodule(release, **kw):
    sampler = {"fall_slots": 1, "hard_slots": 0, "groups": str(release / "folds.csv"), **kw.pop("sampler", {})}
    return UWBDataModule(release=str(release), stats_path=str(release / "stats_jf.json"), window_s=16.0, train_hop_s=2.0,
                         infer_hop_s=4.0, batch_size=2, eval_batch_size=4, num_workers=0, frame_stride=4,
                         diff_channel=False, complex_input=True, physics=True, num_states=2, fall_sets=True,
                         chunk_windows=3, train_sampler="balanced", infer_tail="pad", sampler=sampler, **kw)


def test_chunks_carry_their_falls_and_the_model_trains_both_stages(release):
    torch.manual_seed(0)
    dm = _datamodule(release)
    dm.setup("fit")
    ds = dm.datasets["train"]
    assert ds.window == 800 + 2 * 200                                          # 3 windows of 16 s, 4 s apart
    batch = next(iter(dm.train_dataloader()))
    assert batch["x"].shape == (2, 10, 184, 1200 + 28) and batch["falls"].shape == (2, MAX_FALLS, 2)
    assert int(batch["chunk_hop"][0]) == 200 and batch["falls_mask"].any()
    model = FallSetJEPA(in_channels=10, n_bins=184, window_frames=200, d_model=16, nhead=2, frame_width=32,
                        frame_stride=4, frame_encoder_layers=1, context_layers=1, bidir_layers=1, predictor_layers=1,
                        pred_horizons=(8,), num_states=2, mlp_ratio=2, physics_channels=12, query_layers=1,
                        group_layers=1)
    out = model(batch)
    assert out["q_logit"].shape == (2 * 3 * 3,) and out["q_span_logits"].shape == (18, 800, 2)
    assert out["tokens"].shape == (6, 200, 16)
    assert "g_logit" in out and out["g_logit"].numel() == 2 * out["g_target"].numel() // 2
    assert out["g_span_logits"].size(0) == out["g_logit"].numel() == out["g_domain"].size(0)
    assert (out["g_target"] * out["g_w"]).sum() >= 1                          # a labelled fall reached stage 2
    losses = [UnitBCELoss("q_logit", "q_target", "q_w"), SpanSoftCELoss("q_span", "q_span_w", "q_span_logits", "q_domain"),
              UnitBCELoss("g_logit", "g_target", "g_w"), SpanSoftCELoss("g_span", "g_span_w", "g_span_logits", "g_domain")]
    total = sum(loss(out) for loss in losses)
    total.backward()
    assert model.queries.queries.grad is not None and model.refiner.slots.grad is not None
    assert model.frame_encoder.patch_embed.weight.grad.abs().sum() > 0

    dm.setup("validate")
    item = dm.datasets["val"][0]
    assert item["x"].shape[-1] == 800 + 28 and "falls" in item
    model.eval()
    with torch.no_grad():
        out = model({k: v[None] for k, v in item.items()})
    assert "g_logit" not in out and out["q_logits"].shape == (1, 3)


def test_cli_jf_fits_and_validates_with_the_ladder(release, tmp_path):
    from compose_config import main as compose_main
    composed = tmp_path / "composed.yaml"
    compose_main([str(composed), str(ROOT / "configs/uwb/JF.yaml")])
    tiny = tmp_path / "tiny.yaml"
    tiny.write_text(
        "run: {wandb: disabled}\n"
        "trainer: {max_epochs: 2, accelerator: cpu, devices: 1, log_every_n_steps: 1, enable_progress_bar: false,\n"
        "          check_val_every_n_epoch: 1, callbacks: [{class_path: falljepa.callbacks.WeightAveraging,\n"
        "          init_args: {start_epoch: 0, bn_batches: 2, every_n_steps: 2}}]}\n"
        "data: {window_s: 16.0, infer_hop_s: 4.0, train_hop_s: 2.0, chunk_windows: 3, batch_size: 2, eval_batch_size: 4,\n"
        "       num_workers: 0}\n"
        "model: {model: {init_args: {d_model: 32, nhead: 2, frame_encoder_layers: 1, context_layers: 1,\n"
        "        bidir_layers: 1, predictor_layers: 1, query_layers: 1, group_layers: 1}},\n"
        "        decoder: {kind: sets, threshold: 0.5, proposal_threshold: 0.0}}\n")   # a dict is replaced, not merged
    args = ["-c", str(composed), "-c", str(tiny), "--data.release", str(release),
            "--data.stats_path", str(tmp_path / "cache" / "stats.json"), "--data.sampler.groups", str(release / "folds.csv"),
            "--data.sampler.fall_slots", "1", "--data.sampler.hard_slots", "0",
            "--run.root", str(tmp_path / "runs"), "--run.name", "JF", "--run.id", "t"]
    main(["fit"] + args)
    run_dir = tmp_path / "runs" / "UWB" / "JF" / "JF" / "t"
    swa = run_dir / "checkpoints" / "swa.ckpt"
    assert swa.exists()
    metrics = next((run_dir / "csv").glob("fit-*/metrics.csv")).read_text()
    for key in ("train/loss/Query_Obj", "train/loss/Query_Span", "train/loss/Group_Obj", "train/loss/Group_Span",
                "val/event/F1_mean", "val/queries/F1_mean", "val/dense/F1_mean", "val/sets/proposal_recall"):
        assert key in metrics, key
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", str(swa)])
    pred = run_dir / "predictions"
    for name in ("val-validate.csv", "val-validate.queries.csv", "val-validate.dense.csv"):
        assert (pred / name).exists(), name
    assert json.loads((pred / "val-validate.report.json").read_text())["ranking_score"]["f1_mean"] >= 0
    probs = np.load(pred / "val-validate.probs.npz")
    assert probs["proposals/val_00"].shape[1] == 4 and len(probs["proposals/val_00"]) > 0
    assert probs["slots/val_00"].shape[1] == 5


# ---------------------------------------------------------------- JF2


def test_proximity_groups_join_neighbouring_bursts_and_centre_suppression_drops_them():
    starts = np.array([500, 330, 900])            # the fall, a burst 1.2 s before it, a fall 6 s later
    ends = np.array([600, 440, 1000])
    p = np.array([0.9, 0.8, 0.7])
    assert len(group_proposals(starts, ends, p)) == 3
    assert [g.tolist() for g in group_proposals(starts, ends, p, gap=150)] == [[0, 1], [2]]
    events = [{"t_start": 10.0, "t_end": 12.0, "score": 0.9}, {"t_start": 7.4, "t_end": 9.0, "score": 0.8},
              {"t_start": 18.0, "t_end": 20.0, "score": 0.7}]
    assert len(nms(events)) == 3 and len(nms(events, centre_s=3.0)) == 2


def test_stable_anchors_skip_a_slow_descent():
    G, L, H = 1, 800, 8
    torch.manual_seed(0)
    sig = torch.randn(G, 6, H, L) * 0.01
    sig[:, 1, :, 250:400] += torch.linspace(0, 1, 150)       # radar 1 slowly descends from step 250 ...
    sig[:, 1, :, 250:400] += torch.randn(H, 150) * 0.3       # ... with motion
    sig[:, 1, :, 400:] += 2.0                                 # and lies still from 400
    valid = torch.ones(G, L, dtype=torch.bool)
    span = torch.tensor([[360, 420]])                         # a proposal that starts late, in the descent
    fixed, _ = event_physics(sig, valid, span, groups=2)
    stable, _ = event_physics(sig, valid, span, groups=2, stable=True)
    # late in the descent: the fixed before-profile [300, 350) lies inside the descent, so the person,
    # almost down, still looks "before"; against the standing scene before step 250 it looks "after"
    assert stable[0, 1, 385] > fixed[0, 1, 385] + 0.2
    assert stable[0, 1, 200] < 0.2 and stable[0, 1, 700] > 0.8


def test_span_iou_loss_is_zero_on_the_target_and_has_gradients():
    L = 100
    logits = torch.full((2, L, 2), -30.0)
    logits[0, 20, 0] = logits[0, 60, 1] = 30.0                # exactly the target
    logits[1, 40, 0] = logits[1, 60, 1] = 30.0                # starts late
    logits.requires_grad_(True)
    out = {"q_span": torch.tensor([[20, 60], [20, 60]]), "q_span_w": torch.tensor([1.0, 0.0]),
           "q_span_logits": logits, "q_domain": torch.ones(2, L, dtype=torch.bool), "step_logits": logits}
    loss = SpanIoULoss("q_span", "q_span_w", "q_span_logits", "q_domain")
    assert float(loss(out)) < 1e-3
    out["q_span_w"] = torch.tensor([0.0, 1.0])
    value = loss(out)
    assert abs(float(value) - 0.5) < 1e-3
    value.backward()
    assert logits.grad[1].abs().sum() > 0


def test_clutter_channels_remove_the_static_scene_and_bg_swap_replaces_it(release):
    dm = _datamodule(release, clutter_s=2.0, augment={"bg_swap_p": 1.0, "bg_swap_s": 1.0})
    assert dm.in_channels == 19
    dm.setup("fit")
    ds = dm.datasets["train"]
    store = ds.store
    sid = store.ids[0]
    x, valid = store.read(sid, 100, 400)
    assert x.shape == (19, HALF_BINS, 300)
    z = store.read_z(sid, 50, 450)
    ref = (z - moving_mean(z, 50))[:, 50:350]
    assert np.allclose(x[9:12], ref[..., :HALF_BINS].transpose(0, 2, 1), atol=1e-4)
    assert np.allclose(x[12:15], ref[..., HALF_BINS:].transpose(0, 2, 1), atol=1e-4)
    swapped = x.copy()
    ds._swap_background(swapped, np.random.default_rng(1))
    own, new = x[0:3].mean(-1), swapped[0:3].mean(-1)
    assert not np.allclose(own, new) and np.allclose(swapped[9:18], x[9:18])  # only the raw channels move
    assert np.allclose(x[0:3] - own[..., None], swapped[0:3] - new[..., None], atol=1e-4)
    batch = next(iter(dm.train_dataloader()))
    assert batch["x"].shape[1] == 19


def test_cli_jf2_fits_and_validates(release, tmp_path):
    from compose_config import main as compose_main
    composed = tmp_path / "composed.yaml"
    compose_main([str(composed), str(ROOT / "configs/uwb/JF2.yaml")])
    tiny = tmp_path / "tiny.yaml"
    tiny.write_text(
        "run: {wandb: disabled}\n"
        "trainer: {max_epochs: 2, accelerator: cpu, devices: 1, log_every_n_steps: 1, enable_progress_bar: false,\n"
        "          check_val_every_n_epoch: 1, callbacks: [{class_path: falljepa.callbacks.WeightAveraging,\n"
        "          init_args: {start_epoch: 0, bn_batches: 2, every_n_steps: 2}}]}\n"
        "data: {window_s: 16.0, infer_hop_s: 4.0, train_hop_s: 2.0, chunk_windows: 3, batch_size: 2, eval_batch_size: 4,\n"
        "       num_workers: 0, clutter_s: 2.0}\n"
        "model: {model: {init_args: {d_model: 32, nhead: 2, frame_encoder_layers: 1, context_layers: 1,\n"
        "        bidir_layers: 1, predictor_layers: 1, query_layers: 1, group_layers: 1, group_max_s: 12.0}},\n"
        "        decoder: {kind: sets, threshold: 0.5, proposal_threshold: 0.0, centre_s: 3.0}}\n")
    args = ["-c", str(composed), "-c", str(tiny), "--data.release", str(release),
            "--data.stats_path", str(tmp_path / "cache" / "stats.json"), "--data.sampler.groups", str(release / "folds.csv"),
            "--data.sampler.fall_slots", "1", "--data.sampler.hard_slots", "0", "--data.sampler.steps_scale", "1.0",
            "--run.root", str(tmp_path / "runs"), "--run.name", "JF2", "--run.id", "t"]
    main(["fit"] + args)
    run_dir = tmp_path / "runs" / "UWB" / "JF2" / "JF2" / "t"
    swa = run_dir / "checkpoints" / "swa.ckpt"
    metrics = next((run_dir / "csv").glob("fit-*/metrics.csv")).read_text()
    for key in ("train/loss/Query_IoU", "train/loss/Group_IoU", "val/event/F1_mean"):
        assert key in metrics, key
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", str(swa)])
    assert (run_dir / "predictions" / "val-validate.csv").exists()


def test_split_sampler_yields_the_unsplit_batches_in_pieces():
    from falljepa.data.sampler import BalancedBatchSampler
    n = 40
    sources = [f"s{i % 8}" for i in range(n)]
    args = dict(sources=sources, groups=[i % 4 for i in range(n)], hard=[i % 8 == 7 for i in range(n)],
                fall_ok=np.arange(n) % 2 == 0, none_ok=np.ones(n, bool), batch_size=4, fall_slots=2, hard_slots=1, seed=3)
    whole, split = BalancedBatchSampler(**args), BalancedBatchSampler(**args, split=2)
    pieces = list(split)
    assert len(split) == 2 * len(whole) and all(len(p) == 2 for p in pieces)
    assert [pieces[2 * i] + pieces[2 * i + 1] for i in range(len(whole))] == list(whole)


def test_transition_bounds_match_the_labeller():
    package = ROOT / "abc2027_uwb_train_v1" / "src"
    if not (package / "abc2027" / "annotate.py").exists():
        pytest.skip("the official package (abc2027/annotate.py) is not next to the repo")
    sys.path.insert(0, str(package))
    from abc2027 import annotate
    rng = np.random.default_rng(0)
    for _ in range(40):
        L = int(rng.integers(200, 1200))
        t = np.arange(L)
        c = int(rng.integers(80, L - 80))
        state = 1 / (1 + np.exp(-(t - c) / rng.uniform(5, 40))) + rng.normal(0, 0.03, L)
        motion = np.exp(rng.normal(0, 0.5, L)) + 20 * np.exp(-0.5 * ((t - c) / rng.uniform(10, 60)) ** 2)
        if rng.random() < 0.5:                              # a second, later burst
            motion += 30 * np.exp(-0.5 * ((t - min(L - 1, c + int(rng.integers(50, 200)))) / 20) ** 2)
        state = annotate._smooth(state, 0.02, 0.4)
        motion = annotate._smooth(motion, 0.02, 0.4)
        left, right, score, _ = transition_bounds(state, motion)
        assert (left, right) == annotate._transition_bounds(state, motion)[:2]
        assert len(score) == L - 75


def test_transition_candidate_finds_the_step_behind_a_late_reference():
    G, L, H = 1, 900, 8
    torch.manual_seed(0)
    sig = torch.randn(G, 6, H, L) * 0.01
    # The JF2 timing errors: the fall (scene change at 330) runs on into stronger motion on the floor,
    # and stage 1 places the reference on that later motion. The stable before-profile still finds the
    # standing scene, so the rule returns the step. (With a long still stretch between the step and the
    # later burst, the before-profile lands on the floor and the rule follows the reference.)
    sig[:, 1, :, 280:330] += torch.randn(H, 50) * 0.4          # the fall: motion on radar 1 ...
    sig[:, 1, :, 330:] += 2.0                                  # ... and a new scene from step 330
    sig[:, 1, :, 330:460] += torch.randn(H, 130) * 0.6         # stronger motion on the floor after it
    valid = torch.ones(G, L, dtype=torch.bool)
    late = torch.tensor([[380, 460]])                          # a reference on the later motion
    chans, scal, rule = event_physics(sig, valid, late, groups=2, stable=True, transition=True)
    assert chans.shape == (G, 13, L) and rule.shape == (G, 6) and rule[0, 0] == 1
    a, b = float(rule[0, 1]) * L, float(rule[0, 2]) * L
    assert 250 <= a <= 300 and 320 <= b <= 430, (a, b)        # the step's start; the end widens over motion (<= 2 s)
    assert chans[0, 12, 300] == 1 and chans[0, 12, 600] == 0   # the span channel
    plain, _ = event_physics(sig, valid, late, groups=2, stable=True)
    assert torch.allclose(plain, chans[:, :11])                # the other channels do not change


def test_refiner_with_the_transition_candidate_has_one_more_member():
    torch.manual_seed(0)
    d, s = 16, 4
    refiner = GroupRefiner(d, 2, slots=2, layers=1, stride=s, stable_anchors=True, transition=True)
    items = []
    for R, P in ((60, 3), (12, 0)):
        items.append({"tokens": torch.randn(R, d), "signal": torch.randn(6, 8, R * s),
                      "step_valid": torch.ones(R * s, dtype=torch.bool), "span": torch.tensor([20, 60]),
                      "members": torch.randn(P, d), "member_features": torch.randn(P, 5), "stats": torch.randn(4)})
    g = pad_groups(items, groups=2)
    res = refiner(g)
    assert res["logit"].shape == (2, 2) and res["span_logits"].shape == (2, 2, 240, 2)
    (res["logit"].sum() + res["span_logits"][res["span_logits"] > -1e3].mean()).backward()
    assert refiner.rule_in.weight.grad is not None and refiner.rule_in.weight.grad.abs().sum() > 0


def test_late_reference_shift_moves_only_fall_groups():
    model = FallSetJEPA(in_channels=7, n_bins=368, window_frames=40, d_model=16, nhead=2, frame_encoder_layers=1,
                        context_layers=1, bidir_layers=1, predictor_layers=1, num_queries=2, query_layers=1,
                        group_layers=1, ref_shift_p=1.0, ref_shift_s=(1.0, 2.0), ref_shift_late=1.0)
    rng = np.random.default_rng(0)
    falls = np.array([[400, 480, 1]])
    extra, n_fall = model._extra_groups(falls, [], 2000, rng)
    assert n_fall == 1 and len(extra) == 1 + model.negative_groups
    assert model.ref_shift == (50.0, 100.0)


def test_cli_jf3_fits_and_validates(release, tmp_path):
    from compose_config import main as compose_main
    composed = tmp_path / "composed.yaml"
    compose_main([str(composed), str(ROOT / "configs/uwb/JF3.yaml")])
    tiny = tmp_path / "tiny.yaml"
    tiny.write_text(
        "run: {wandb: disabled}\n"
        "trainer: {max_epochs: 2, accelerator: cpu, devices: 1, log_every_n_steps: 1, enable_progress_bar: false,\n"
        "          check_val_every_n_epoch: 1, callbacks: [{class_path: falljepa.callbacks.WeightAveraging,\n"
        "          init_args: {start_epoch: 0, bn_batches: 2, every_n_steps: 2}}]}\n"
        "data: {window_s: 16.0, infer_hop_s: 4.0, train_hop_s: 2.0, chunk_windows: 3, batch_size: 2, eval_batch_size: 4,\n"
        "       num_workers: 0, clutter_s: 2.0}\n"
        "model: {model: {init_args: {d_model: 32, nhead: 2, frame_encoder_layers: 1, context_layers: 1,\n"
        "        bidir_layers: 1, predictor_layers: 1, query_layers: 1, group_layers: 1, group_max_s: 12.0}},\n"
        "        decoder: {kind: sets, threshold: 0.5, proposal_threshold: 0.0, centre_s: 3.0}}\n")
    args = ["-c", str(composed), "-c", str(tiny), "--data.release", str(release),
            "--data.stats_path", str(tmp_path / "cache" / "stats.json"), "--data.sampler.groups", str(release / "folds.csv"),
            "--data.sampler.fall_slots", "1", "--data.sampler.hard_slots", "0", "--data.sampler.steps_scale", "1.0",
            "--run.root", str(tmp_path / "runs"), "--run.name", "JF3", "--run.id", "t"]
    main(["fit"] + args)
    run_dir = tmp_path / "runs" / "UWB" / "JF3" / "JF3" / "t"
    metrics = next((run_dir / "csv").glob("fit-*/metrics.csv")).read_text()
    for key in ("train/loss/Group_IoU", "val/event/F1_mean"):
        assert key in metrics, key
    main(["validate", "-c", str(run_dir / "config.yaml"), "--ckpt_path", str(run_dir / "checkpoints" / "swa.ckpt")])
    assert (run_dir / "predictions" / "val-validate.csv").exists()
