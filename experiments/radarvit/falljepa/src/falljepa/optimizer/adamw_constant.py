import torch.optim as optim
from lightning.pytorch import LightningModule

from .base import BaseOptimizer


class AdamWConstant(BaseOptimizer):
    """AdamW at a constant learning rate (Amir's recipe: 1e-3, weight decay 1e-4, PyTorch's betas);
    weight decay on weight matrices only, as everywhere in this repo."""
    def __init__(self, learning_rate: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.999)):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = tuple(betas)

    def setup(self, model: LightningModule, total_steps: int) -> dict:
        groups = self.weight_decay_groups(model, self.weight_decay)
        return {"optimizer": optim.AdamW(groups, lr=self.learning_rate, betas=self.betas)}
