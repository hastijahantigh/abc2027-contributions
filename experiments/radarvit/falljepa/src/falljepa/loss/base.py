import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """
    One loss term. It reads the keys it needs from the model's output dict and returns a
    scalar; it keeps no running state (Lightning's `self.log` does the epoch averaging).
    Its name and its weight in the total are given by the module that owns it.
    """
    def forward(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
