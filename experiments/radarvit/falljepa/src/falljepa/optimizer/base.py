import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import rank_zero_info


# Layers whose `weight` is a projection matrix: the only parameters that get weight decay
DECAY_MODULES = (
    nn.Linear, nn.Bilinear,
    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)


class BaseOptimizer:
    """Builds the optimizer and LR schedule once the trainer knows the total number of steps."""
    def setup(self, model: LightningModule, total_steps: int) -> dict:
        """Returns what `LightningModule.configure_optimizers` returns."""
        raise NotImplementedError

    @staticmethod
    def weight_decay_groups(model: nn.Module, weight_decay: float, decay_modules: tuple = DECAY_MODULES) -> list[dict]:
        """
        The usual transformer convention (GPT-2/minGPT, timm, MAE, I-JEPA): weight decay on the
        weight matrices of linear and convolution layers only, none on everything else, i.e. biases,
        normalization parameters, embedding tables and free parameters such as positional
        embeddings, class tokens or learned null vectors. Decided by layer type, so every
        parameter lands in exactly one group.
        """
        decay, no_decay, seen = [], [], set()
        decay_count = no_decay_count = 0
        for module in model.modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad or id(param) in seen:
                    continue
                seen.add(id(param))
                if name == "weight" and isinstance(module, decay_modules):
                    decay.append(param)
                    decay_count += param.numel()
                else:
                    no_decay.append(param)
                    no_decay_count += param.numel()

        rank_zero_info(
            f"Weight decay {weight_decay} on {len(decay)} weight matrices ({decay_count:,} parameters); "
            f"none on {len(no_decay)} other tensors ({no_decay_count:,} parameters)."
        )
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
