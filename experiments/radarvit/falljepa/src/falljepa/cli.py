"""
Command line for training and evaluation, built on LightningCLI. A config holds everything
about a run (model, losses, metrics, optimizer, data, trainer, tracking); any value can be
overridden on the command line and `--print_config` shows the resolved result.

    python run.py fit      -c configs/uwb/U0.yaml                            # new run
    python run.py fit      -c configs/uwb/U0.yaml --model.loss_weights.MSE_Latent 0 --run.name U0-noJEPA
    python run.py fit      -c runs/<group>/<name>/<id>/config.yaml --ckpt_path last  # resume
    python run.py validate -c runs/<group>/<name>/<id>/config.yaml --ckpt_path best  # real decoding
    python run.py test     -c runs/<group>/<name>/<id>/config.yaml --ckpt_path best

Each run gets its own folder, `<run.root>/<run.group>/<run.name>/<run.id>`:

    config.yaml          the full resolved config; its run.id points back to this folder
    checkpoints/         best-epoch=NNN.ckpt (by checkpoint.monitor) and last.ckpt (the latest epoch)
    predictions/         one submission CSV (and the official report) per decoded epoch/stage
    meta/                git commit, branch and diff, command, host, versions per invocation
    csv/<stage>-<time>/  metrics.csv: a local copy of everything logged
    wandb/               the W&B run files (offline runs are uploaded with `wandb sync`)
"""

import hashlib
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from .callbacks import BestAndLastCheckpoint, RunMetadata, SaveRunConfig
from .data.physics import NUM_FEATURES
from .data import UWBDataModule
from .data.uwb import in_channels
from .module import FallModule


@dataclass
class Run:
    """
    Where a run lives and how it is tracked.

    name, group, project, notes, tags:  the W&B run name, group and project, its description
                                        and tags. The folder is <root>/<group>/<name>/<id>.
    id:     left empty, a new one is made (time + random suffix) and written into the saved
            config, so that config points back at this run (resume, validate, test). Runs of one
            batch may share it (tools/run_ablations.sh passes its TAG), so the W&B run id is
            <group>-<name>-<id> (see `wandb_id`), the same for every invocation of one run.
    wandb:  "online" logs live, "offline" writes locally for a later `wandb sync`, "disabled"
            skips W&B. The WANDB_MODE environment variable, when set, takes precedence.
    tensorboard: also write TensorBoard event files to <run dir>/tensorboard.
    """
    name: str = "debug"
    group: str = "default"
    project: str = "abc2027-fall-jepa"
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    id: str | None = None
    root: str = "runs"
    wandb: Literal["online", "offline", "disabled"] = "offline"
    tensorboard: bool = False


def _run_dir(run) -> Path:
    return Path(run.root) / run.group / run.name / run.id


def wandb_id(run, max_len: int = 64) -> str:
    """The W&B run id of a run: <group>-<name>-<id> with every character W&B does not take (such as
    "/" and "+") made "_"; past max_len, a prefix and a hash of the whole id, so it stays unique."""
    full = re.sub(r"[^A-Za-z0-9_-]", "_", f"{run.group}-{run.name}-{run.id}")
    if len(full) <= max_len:
        return full
    return f"{full[:max_len - 9]}-{hashlib.sha1(full.encode()).hexdigest()[:8]}"


def _in_channels(diff_channel: bool, complex_input: bool = False, clutter_s: float = 0.0, amplitude: bool = True,
                 contrast: list | None = None) -> int:
    return in_channels(diff_channel, complex_input, clutter_s, amplitude, contrast)


def _contrast_channels(contrast: list | None) -> int:
    return len(contrast or [])


def _n_bins(complex_input: bool) -> int:
    return 184 if complex_input else 368


def _physics_channels(physics: bool) -> int:
    return 3 * NUM_FEATURES if physics else 0


def _window_frames(window_s: float, frame_stride: int) -> int:
    return int(round(window_s * 50)) // frame_stride


class FallCLI(LightningCLI):
    def __init__(self, **kwargs) -> None:
        self._started = datetime.now().strftime("%Y%m%d-%H%M%S")
        super().__init__(
            FallModule,
            UWBDataModule,
            save_config_callback=SaveRunConfig,
            trainer_defaults={"callbacks": [LearningRateMonitor(logging_interval="step"), RunMetadata()]},
            auto_configure_optimizers=False,
            **kwargs,
        )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_class_arguments(Run, "run")

        # last.ckpt is the latest epoch, best-epoch=NNN.ckpt the best by checkpoint.monitor
        parser.add_lightning_class_args(BestAndLastCheckpoint, "checkpoint")
        parser.set_defaults({
            "checkpoint.monitor": "val/event/F1_mean",
            "checkpoint.mode": "max",
            "checkpoint.filename": "best-{epoch:03d}",
            "checkpoint.save_top_k": 1,
            "checkpoint.save_last": True,
        })

        # Values the data decides, so they are never set twice by hand: the frames (they set the
        # margin every window is read with), the input channels and the frames per window
        parser.link_arguments("data.frame_width", "model.model.init_args.frame_width")
        parser.link_arguments("data.frame_stride", "model.model.init_args.frame_stride")
        parser.link_arguments(("data.diff_channel", "data.complex_input", "data.clutter_s", "data.amplitude", "data.contrast"),
                              "model.model.init_args.in_channels", compute_fn=_in_channels)
        parser.link_arguments("data.contrast", "model.model.init_args.contrast_channels", compute_fn=_contrast_channels)
        parser.link_arguments("data.contrast", "model.model.init_args.contrast_scales")
        parser.link_arguments("data.complex_input", "model.model.init_args.n_bins", compute_fn=_n_bins)
        parser.link_arguments("data.units", "model.model.init_args.recording_head")
        parser.link_arguments("data.physics", "model.model.init_args.physics_channels", compute_fn=_physics_channels)
        parser.link_arguments(("data.window_s", "data.frame_stride"), "model.model.init_args.window_frames",
                              compute_fn=_window_frames)

    def _parse_ckpt_path(self) -> None:
        # Lightning would overwrite the parsed model arguments with the checkpoint's saved
        # hyperparameters, silently dropping command-line overrides (e.g. --model.decoder.*).
        # The run's config.yaml is already the full resolved config, so keep what was parsed.
        return

    def _subconfig(self):
        return self.config[self.subcommand] if self.subcommand else self.config

    def before_instantiate_classes(self) -> None:
        config = self._subconfig()
        run = config.run
        # The id reaches the DDP worker processes (which re-run this command, with LOCAL_RANK > 0)
        # through the environment; the main process always makes its own, so a second run in the
        # same Python process never inherits the first one's.
        if run.id is None:
            inherited = os.environ.get("FALL_RUN_ID") if int(os.environ.get("LOCAL_RANK", "0")) > 0 else None
            run.id = inherited or f"{self._started}-{secrets.token_hex(2)}"
        os.environ["FALL_RUN_ID"] = run.id
        checkpoints = _run_dir(run) / "checkpoints"

        if self.subcommand == "fit" and config.ckpt_path is None and any(checkpoints.glob("*.ckpt")):
            raise ValueError(
                f"Run {_run_dir(run)} already has checkpoints. Pass --ckpt_path last to resume it, "
                "or --run.id null to start a new run from this config."
            )
        if self.subcommand in ("validate", "test", "predict") and config.ckpt_path in ("best", "last"):
            # Lightning only knows "best" inside the fit that made it: look it up in the run folder.
            found = sorted(checkpoints.glob(f"{config.ckpt_path}*.ckpt"))
            if len(found) != 1:
                raise ValueError(f"Expected one {config.ckpt_path}*.ckpt in {checkpoints}, found {[p.name for p in found]}")
            config.ckpt_path = str(found[0])

    def instantiate_trainer(self, **kwargs):
        run = self._get(self.config_init, "run")
        run_dir = _run_dir(run)
        run_dir.mkdir(parents=True, exist_ok=True)

        loggers = []
        mode = os.environ.get("WANDB_MODE", run.wandb)
        if mode != "disabled":
            # Online, a resume or a test continues the same W&B run. Offline, each invocation writes
            # its own segment under the run's id, and `wandb sync` uploads the segments in order.
            loggers.append(WandbLogger(
                project=run.project, name=run.name, group=run.group, id=wandb_id(run), save_dir=str(run_dir),
                offline=mode == "offline", resume="allow" if mode == "online" else None,
                notes=run.notes or None, tags=run.tags or None, job_type=self.subcommand,
            ))
        # A fresh CSV folder per invocation: CSVLogger would otherwise overwrite an earlier metrics.csv
        loggers.append(CSVLogger(save_dir=run_dir, name="csv", version=f"{self.subcommand}-{self._started}"))
        if run.tensorboard:
            loggers.append(TensorBoardLogger(save_dir=run_dir, name="tensorboard", version="", default_hp_metric=False))

        trainer = super().instantiate_trainer(logger=loggers, default_root_dir=str(run_dir), **kwargs)
        for callback in trainer.checkpoint_callbacks:
            if callback.dirpath is None:
                callback.dirpath = os.path.realpath(run_dir / "checkpoints")
        return trainer


def main(args=None) -> None:
    FallCLI(args=args)
