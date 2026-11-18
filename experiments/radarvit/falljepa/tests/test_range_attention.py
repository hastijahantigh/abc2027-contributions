from pathlib import Path

import torch
import yaml

from falljepa.model import FallSetJEPA
from falljepa.model.stem import RangePool


def test_range_block_attends_across_range_within_one_frame():
    torch.manual_seed(1)
    pool = RangePool(16, 23, heads=4, range_attention=True).eval()
    u = torch.randn(2, 16, 23, 4, requires_grad=True)
    # Observe one range token before pooling. It must already depend on another range.
    activations = []
    hook = pool.range_block.register_forward_hook(lambda _m, _args, out: activations.append(out))
    y = pool(u)
    hook.remove()
    assert y.shape == (2, 4, 16)
    grad, = torch.autograd.grad(activations[0][2, 0, 0], u)
    assert grad[0, :, -1, 2].abs().sum() > 0
    assert grad[1].count_nonzero() == 0
    assert grad[0, :, :, [0, 1, 3]].count_nonzero() == 0


def test_range_config_and_both_event_stages():
    configs = Path(__file__).resolve().parents[1] / "configs" / "uwb"
    base = yaml.safe_load((configs / "JF5T.yaml").read_text(encoding="utf-8"))
    variant = yaml.safe_load((configs / "JF6Range.yaml").read_text(encoding="utf-8"))
    args = variant["model"]["model"]["init_args"]
    assert args["range_attention"] is True
    base["model"]["model"]["init_args"]["range_attention"] = True
    for name in ("group", "name", "notes"):
        base["run"][name] = variant["run"][name]
    assert base == variant

    torch.manual_seed(7)
    model = FallSetJEPA(**args, window_frames=50, in_channels=13, n_bins=184,
                        frame_stride=4, physics_channels=12)
    length = 200 + 100 + 28  # two overlapping windows with margins
    batch = {"x": torch.randn(1, 13, 184, length), "valid": torch.ones(1, length),
             "phys": torch.randn(1, 12, length), "chunk_hop": torch.tensor([100]),
             "falls": torch.tensor([[[70, 130]]]), "falls_w": torch.ones(1, 1),
             "falls_mask": torch.ones(1, 1, dtype=torch.bool)}
    out = model(batch)
    assert out["tokens"].shape == (2, 50, 64)
    loss = out["q_logit"].sum() + out["g_logit"].sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.stem.pool.range_block.attn.q_proj.weight.grad.abs().sum() > 0
