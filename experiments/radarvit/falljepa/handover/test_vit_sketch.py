"""Checks of handover/vit_sketch.py on CPU: uv run pytest handover/test_vit_sketch.py"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit_sketch import LocalViT, RadarViT  # noqa: E402

from falljepa.model import FallSetJEPA  # noqa: E402


def _x(L=228, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 13, 184, L, generator=g)
    x[:, -1] = 1.0
    return x


def test_radar_vit_token_i_sees_exactly_frame_i():
    torch.manual_seed(0)
    vit = RadarViT(d_model=32, nhead=2, depth=1).eval()
    x = _x().requires_grad_(True)
    Fn = (228 - 32) // 4 + 1
    out = vit(x, Fn)
    assert out.shape == (2, Fn, 32)
    for i in (0, 10, Fn - 1):
        g, = torch.autograd.grad(out[0, i].sum(), x, retain_graph=True)
        steps = torch.nonzero(g.abs().sum((0, 1, 2)) > 0).flatten()
        assert int(steps.min()) == 4 * i and int(steps.max()) == 4 * i + 31, (i, steps.min(), steps.max())


def test_radar_vit_ignores_each_radars_phase():
    torch.manual_seed(0)
    vit = RadarViT(d_model=32, nhead=2, depth=1).eval()
    x = _x()
    ref = vit(x, 50)
    y = x.clone()
    for r, phi in ((0, 0.7), (2, -2.1)):
        for i, q in ((r, 3 + r), (6 + r, 9 + r)):
            y[:, i] = x[:, i] * np.cos(phi) - x[:, q] * np.sin(phi)
            y[:, q] = x[:, i] * np.sin(phi) + x[:, q] * np.cos(phi)
    assert torch.allclose(vit(y, 50), ref, atol=1e-4)


def test_local_vit_reach_is_the_band():
    torch.manual_seed(0)
    loc = LocalViT(d_model=32, nhead=2, depth=2, radius=5).eval()
    h = torch.randn(1, 60, 32, requires_grad=True)
    valid = torch.ones(1, 60, dtype=torch.bool)
    g, = torch.autograd.grad(loc(h, valid)[0, 30].sum(), h)
    reach = torch.nonzero(g.abs().sum(-1)[0] > 0).flatten()
    assert int(reach.min()) == 30 - 10 and int(reach.max()) == 30 + 10    # 2 blocks x radius 5


def test_swapped_into_fall_set_jepa():
    """JF6's trunk settings, small: the ViT parts replace model.stem and model.tcn, encode() is unchanged."""
    torch.manual_seed(0)
    model = FallSetJEPA(in_channels=13, n_bins=184, window_frames=100, d_model=32, nhead=2, frame_width=32,
                        frame_stride=4, context_layers=0, bidir_layers=1, predictor_layers=0, pred_horizons=(),
                        num_states=2, mlp_ratio=2, pre_proj_ratio=2, trunk="stem", radar_fusion="early",
                        stem_channels=(8, 16, 16, 32), local="tcn", rope=True, raw_channels=6)
    model.stem = RadarViT(d_model=32, nhead=2, depth=1)
    model.tcn = LocalViT(d_model=32, nhead=2, depth=1, radius=30)
    x = _x(L=100 * 4 + 28)
    valid = torch.ones(2, x.size(-1))
    valid[1, -60:] = 0                                                     # a window past the stream end
    enc = model.encode(x, valid)
    assert enc["r"].shape == (2, 100, 32) and torch.isfinite(enc["r"]).all()
    enc["r"].sum().backward()
    assert model.stem.embed.wr.grad.abs().sum() > 0
