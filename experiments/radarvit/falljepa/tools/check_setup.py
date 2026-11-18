"""
Checks a machine and a release before training:

    uv run python tools/check_setup.py --release /path/to/abc2027_uwb_train_v1

GPU and bf16 support, every stream's shape and radar order, the training-only normalizer (fitted
and cached on first use, as training would), the fold assignment covering every training
recording, the window counts, and one real batch through the model (on the GPU when there is one).
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from falljepa.data import UWBDataModule  # noqa: E402
from falljepa.data.folds import load_assignment, stream_fold_bits  # noqa: E402
from falljepa.model import FallJEPA  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--stats-path", default="runs/_cache/normalizer_train.json")
    parser.add_argument("--folds", default="configs/folds/stage04_source_folds.csv")
    args = parser.parse_args()

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  bf16 supported: {torch.cuda.is_bf16_supported()} (if not, pass --trainer.precision 16-mixed)")

    t = time.time()
    dm = UWBDataModule(release=args.release, stats_path=args.stats_path, window_s=25.6, train_hop_s=12.8,
                       infer_hop_s=6.4, frame_width=32, frame_stride=4, batch_size=4, num_workers=0)
    dm.setup("fit")
    train, val = dm.datasets["train"], dm.datasets["val"]
    hours = lambda ds: sum(ds.store.lengths.values()) / 50 / 3600  # noqa: E731
    print(f"train: {len(train.store.ids)} streams, {hours(train):.2f} h, {len(train)} windows an epoch")
    print(f"val:   {len(val.store.ids)} streams, {hours(val):.2f} h, {len(val)} windows")
    print(f"normalizer at {args.stats_path} ({time.time() - t:.0f} s, fitted on first use)")

    bits = stream_fold_bits(dm.composition_path, load_assignment(args.folds), train.store.lengths)
    counts = [sum(int(((b >> f) & 1).sum()) for b in bits.values()) / 50 / 3600 for f in range(6)]
    print("folds cover every training timestep; hours per fold: " + ", ".join(f"{c:.2f}" for c in counts))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = next(iter(torch.utils.data.DataLoader(train, batch_size=4)))
    model = FallJEPA(in_channels=dm.in_channels, window_frames=320, frame_width=32, frame_stride=4,
                     bidir_layers=2, pred_horizons=(8,)).to(device)
    out = model({k: v.to(device) for k, v in batch.items()})
    out["step_logits"].float().sum().backward()
    falls = (batch["state"] > 0).float().mean().item()
    print(f"batch x {tuple(batch['x'].shape)} -> step logits {tuple(out['step_logits'].shape)} on {device}; "
          f"fall timesteps in batch: {falls:.1%}")
    print("OK")


if __name__ == "__main__":
    main()
