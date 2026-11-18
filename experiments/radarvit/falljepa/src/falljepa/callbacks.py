import json
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import lightning
import torch
import yaml
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _stage(stage) -> str:
    """"fit", "validate", ... (Lightning passes a str-valued enum)."""
    return str(getattr(stage, "value", stage))


def _free_path(path: Path, stage) -> Path:
    """`path` itself on the first write, `<stem>.<stage>-<time><suffix>` once it exists (resume, test, ...)."""
    return path if not path.exists() else path.with_name(f"{path.stem}.{_stage(stage)}-{_timestamp()}{path.suffix}")


class BestAndLastCheckpoint(ModelCheckpoint):
    """
    ModelCheckpoint whose last.ckpt is the latest epoch. Lightning's own save_last (2.6) writes last
    only right after a new top-k checkpoint, so with save_top_k: 1 last.ckpt was the best epoch again
    (validate --ckpt_path last selected on val, and a resume restarted from the best epoch).
    """

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if self._should_save_on_train_epoch_end(trainer):
            self._save_latest(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_end(trainer, pl_module)
        if not self._should_save_on_train_epoch_end(trainer):
            self._save_latest(trainer)

    def _save_latest(self, trainer: Trainer) -> None:
        # Skipped outside fit, in the sanity check, and when this step was already saved (a new
        # top-k checkpoint, which Lightning follows with last itself)
        if self.save_last and not self._should_skip_saving_checkpoint(trainer):
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))


class WeightAveraging(Callback):
    """
    Amir's weight averaging (SWA): the uniform mean of the model's weights at the end of every
    epoch from `start_epoch` (0-based) on. At the end of fit the mean is loaded into the model,
    BatchNorm statistics (the pre-projector's) are recomputed on `bn_batches` training batches, and
    the result is saved as `<run dir>/checkpoints/swa.ckpt`; validate it with
    `--ckpt_path <run dir>/checkpoints/swa.ckpt`. The running sum is part of the checkpoint, so a
    resume continues it.

    every_n_steps > 0 averages densely instead (SWAD, Cha et al. 2021): a snapshot after every
    `every_n_steps`-th optimizer step from `start_epoch` on, none at epoch ends.
    """
    def __init__(self, start_epoch: int = 10, bn_batches: int = 50, every_n_steps: int = 0):
        self.start_epoch, self.bn_batches, self.every_n_steps = start_epoch, bn_batches, every_n_steps
        self.sum: dict[str, torch.Tensor] | None = None
        self.count = 0

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int) -> None:
        if self.every_n_steps > 0 and trainer.current_epoch >= self.start_epoch and trainer.global_step % self.every_n_steps == 0:
            self._add(pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.every_n_steps == 0 and trainer.current_epoch >= self.start_epoch:
            self._add(pl_module)

    def _add(self, pl_module: LightningModule) -> None:
        state = {k: v.detach().double().cpu() for k, v in pl_module.model.state_dict().items()}
        if self.sum is None:
            self.sum = state
        else:
            for k, v in state.items():
                self.sum[k] += v
        self.count += 1

    def state_dict(self) -> dict:
        return {"sum": self.sum, "count": self.count}

    def load_state_dict(self, state_dict: dict) -> None:
        self.sum, self.count = state_dict["sum"], state_dict["count"]

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.count:
            return
        model = pl_module.model
        current = model.state_dict()
        model.load_state_dict({k: (v / self.count).to(current[k].dtype) for k, v in self.sum.items()})
        norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        if norms and self.bn_batches > 0:
            for m in norms:
                m.reset_running_stats()
                m.momentum = None                     # a cumulative mean over the batches below
            was_training = model.training
            model.train()
            with torch.no_grad():
                for i, batch in enumerate(trainer.datamodule.train_dataloader()):
                    if i >= self.bn_batches:
                        break
                    model({k: v.to(pl_module.device) if torch.is_tensor(v) else v for k, v in batch.items()})
            model.train(was_training)
        trainer.save_checkpoint(Path(trainer.default_root_dir) / "checkpoints" / "swa.ckpt")


class SaveRunConfig(SaveConfigCallback):
    """
    Writes the full resolved config to `<run dir>/config.yaml` (later invocations on the same run,
    such as a resume or a test, add `config.<stage>-<time>.yaml` next to it) and sends it to every
    logger, so W&B shows each config value as a column you can filter and group by.
    """
    def __init__(self, parser, config, **kwargs) -> None:
        super().__init__(parser, config, save_to_log_dir=False)

    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        run_dir = Path(trainer.default_root_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.parser.save(self.config, _free_path(run_dir / "config.yaml", stage), skip_none=False, multifile=False)

        config = yaml.safe_load(self.parser.dump(self.config, skip_none=False))
        config.pop("config", None)
        for logger in trainer.loggers:
            logger.log_hyperparams(config)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


class RunMetadata(Callback):
    """
    Records what produced a run: git commit, branch and uncommitted diff, the command line,
    host and library versions. Each invocation (fit, resume, test, ...) writes
    `<run dir>/meta/<time>-<stage>.json` (plus `.diff` when the tree is dirty), and the commit
    also goes to the loggers.
    """
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if not trainer.is_global_zero:
            return
        commit = (_git("rev-parse", "HEAD") or "").strip() or None
        diff = _git("diff", "HEAD")
        stage = _stage(stage)
        meta = {
            "stage": stage,
            "time": datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "git": {
                "commit": commit,
                "branch": (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
                "dirty": bool(diff),
            },
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "lightning": lightning.__version__,
            "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }

        meta_dir = Path(trainer.default_root_dir) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        name = f"{_timestamp()}-{stage}"
        (meta_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
        if diff:
            (meta_dir / f"{name}.diff").write_text(diff)

        for logger in trainer.loggers:
            logger.log_hyperparams({"git": meta["git"]})

        # In the W&B runs table, show each run's best value of these, not its last
        checkpoint = trainer.checkpoint_callback
        summaries = {"val/event/F1_0.3": "max", "val/event/StartErrorMedian_0.3": "min"}
        if checkpoint is not None and checkpoint.monitor:
            summaries[checkpoint.monitor] = checkpoint.mode
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                for key, summary in summaries.items():
                    logger.experiment.define_metric(key, summary=summary, step_metric="trainer/global_step")
