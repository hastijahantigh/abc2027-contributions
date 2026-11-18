import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from lightning.pytorch import LightningModule

from .base import BaseOptimizer


class AdamWWithCosineScheduler(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        min_lr: float,
        warmup_ratio: float,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio

    def setup(self, model: LightningModule, total_steps: int) -> dict:
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer, total_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _setup_optimizer(self, model: LightningModule) -> optim.Optimizer:
        optim_groups = self.weight_decay_groups(model, self.weight_decay)

        # JEPA FIX: Restore standard Transformer momentum.
        # beta2 = 0.95 is often preferred for ViT/JEPA over 0.999
        optimizer = optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer

    def _setup_scheduler(
        self, optimizer: optim.Optimizer, total_steps: int
    ) -> optim.lr_scheduler.LRScheduler:
        if total_steps is None or total_steps == 0:
            raise ValueError("total_steps must be > 0 for this scheduler.")

        warmup_steps = int(total_steps * self.warmup_ratio)
        decay_steps = total_steps - warmup_steps

        schedulers = []
        milestones = []

        if warmup_steps > 0:
            scheduler1 = LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
            )
            schedulers.append(scheduler1)
            milestones.append(warmup_steps)

        scheduler2 = CosineAnnealingLR(
            optimizer, T_max=decay_steps, eta_min=self.min_lr
        )
        schedulers.append(scheduler2)

        if len(schedulers) == 1:
            return scheduler2

        return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
