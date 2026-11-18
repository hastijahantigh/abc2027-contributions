"""Entry point for training and evaluation; see src/falljepa/cli.py for the commands and the run folder."""
import os
# Expandable segments let the CUDA caching allocator grow/reuse segments in place.
# Must be set before CUDA is initialized.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from falljepa.cli import main  # noqa: E402


torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    main()
