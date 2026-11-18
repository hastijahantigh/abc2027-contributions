"""The recoded post-processing chain (src/falljepa/postprocess.py): steps on hand-made candidates."""
import numpy as np

from falljepa import postprocess as pp


def cand(a, b, s=0.9, parents=None, analyses=()):
    return pp.Candidate(a, b, s, tuple(parents or [(a, b, s)]), tuple(analyses))


def analysis(lo, hi, anchor, spans_rises):
    return pp.Analysis((lo, hi), anchor, tuple(pp.Transition(r, a, b, rise) for r, (a, b, rise) in enumerate(spans_rises)))


def test_decode_joins_gaps_before_the_minimum_length():
    p = np.zeros(100)
    p[10:16] = 0.5                      # 6 steps
    p[20:26] = 0.8                      # 6 steps, 4 apart: joined into 16 steps
    p[60:65] = 0.9                      # 5 steps alone: dropped
    assert pp.threshold_runs(p, 0.3, 10, 10) == [(10, 26, 0.8)]


def test_merge_keeps_lineage_and_max_score_nms_by_centre_and_iou():
    merged = pp.merge_step([cand(0, 100, 0.4), cand(50, 150, 0.9), cand(150, 200, 0.5)])   # touching is not overlap
    assert [(c.start, c.stop, c.score, len(c.parents)) for c in merged] == [(0, 150, 0.9, 2), (150, 200, 0.5, 1)]
    kept = pp.nms_step([cand(0, 100, 0.9), cand(200, 300, 0.8), cand(400, 500, 0.7)], 5.9)
    assert [c.score for c in kept] == [0.9, 0.7]                                            # centres 4 s apart
    kept = pp.nms_step([cand(0, 100, 0.9), cand(10, 100, 0.8), cand(200, 300, 0.7)], None, 0.5)
    assert [c.score for c in kept] == [0.9, 0.7]


def test_extend_only_short_candidates_toward_the_parent_envelope():
    out = pp.extend_step([cand(100, 150, parents=[(60, 220, 0.9)]), cand(100, 172, parents=[(0, 400, 0.9)]),
                          cand(5, 40, parents=[(0, 45, 0.9)])], T=1000)
    assert [c.interval for c in out] == [(80, 175), (100, 172), (2, 43)]   # min(25, (s-a+1)//2), (b-e+1)//2


def test_local_gate_and_its_fallback():
    T = 2000
    cir = np.zeros((3, T, 4), np.float32)
    cir[1, 1000:] = 0.01                          # a scene change on radar 1 at step 1000
    s = pp.Stream(None, cir)
    cfg = pp.Config(joins=False, gate="local")
    kept = pp.gate_step(s, [cand(950, 1000), cand(400, 450)], cfg)
    assert [c.interval for c in kept] == [(950, 1000)]
    assert s.local_change(950, 1000, (0, T)) > pp.LOCAL_CHANGE
    assert s.local_change(2, 50, (0, T)) is None  # 2 steps before: falls back to the broad gate


def test_move_to_an_earlier_stronger_supported_transition():
    an = analysis(0, 1000, 0, [(500, 560, 0.2), (380, 460, 0.5), (300, 600, 0.9)])
    p = np.zeros(1000)
    p[380:460] = 0.8
    p[500:560] = 0.8
    c = pp.Candidate(500, 560, 0.9, ((370, 470, 0.8), (495, 565, 0.9)), (an, an))
    moved = pp.move_step(None, [c], pp.Config(), p)[0]
    assert moved.interval == (380, 460)          # radar 2 overlaps the candidate, so only radar 1 qualifies
    weak = pp.Candidate(500, 560, 0.9, c.parents, (analysis(0, 1000, 0, [(500, 560, 0.6), (380, 460, 0.5), (0, 5, 0.9)]),) * 2)
    assert pp.move_step(None, [weak], pp.Config(), p)[0].interval == (500, 560)   # rise not above the anchor's
    no_p = p.copy()
    no_p[380:460] = 0.0
    assert pp.move_step(None, [c], pp.Config(), no_p)[0].interval == (500, 560)   # no model support


def test_full_config_is_amirs_chain_and_old_is_the_earlier_one():
    assert (pp.Config().gate, pp.Config().extend, pp.Config().move, pp.Config().min_score) == ("local", True, True, 0.5)
    assert (pp.OLD.gate, pp.OLD.extend, pp.OLD.move, pp.OLD.min_score) == ("broad", False, False, None)
