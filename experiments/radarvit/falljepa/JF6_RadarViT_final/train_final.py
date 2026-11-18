"""Train the JF6 RadarViT detector on a prepared final training release."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="Train JF6 RadarViT on the prepared final set.")
    parser.add_argument("final_config", type=Path, help="Complete final-set fit YAML, including data paths")
    parser.add_argument("seeds", type=int, nargs=4, help="The team's four distinct seed IDs")
    parser.add_argument("--check-only", action="store_true", help="Resolve all four configs without training")
    args = parser.parse_args()
    if len(set(args.seeds)) != 4:
        parser.error("Supply four different seeds.")
    repo = Path.cwd()
    if not (repo / "run.py").is_file():
        parser.error("Run from the FallJEPA repo containing run.py.")
    final = yaml.safe_load(args.final_config.read_text(encoding="utf-8"))
    src = final["data"]
    cfg = yaml.safe_load(Path(__file__).with_name("recipe.yaml").read_text(encoding="utf-8"))
    data = cfg["data"]
    if src.get("cv_fold") is not None or Path(src["release"]).name in ("A", "B", "C", "D"):
        parser.error("Supply the final training release, not a CV fold.")
    for key in ("release", "stats_path", "train_splits", "val_splits", "test_splits"):
        if key not in src:
            parser.error(f"The complete final data config must specify data.{key}")
        data[key] = src[key]
    if not src.get("sampler", {}).get("groups"):
        parser.error("Set data.sampler.groups to the final release's groups CSV.")
    data["sampler"]["groups"] = src["sampler"]["groups"]
    if not Path(data["release"]).is_dir() or not Path(data["sampler"]["groups"]).is_file():
        parser.error("Check the final release and groups paths from the repository root.")
    if not data["train_splits"]:
        parser.error("The final training split list is empty.")
    for split in data["train_splits"]:
        if not (Path(data["release"]) / "streams" / split).is_dir():
            parser.error(f"Missing training stream directory for {split}")
    if "num_workers" in src:
        data["num_workers"] = src["num_workers"]
    if "limit_val_batches" in final.get("trainer", {}):
        cfg["trainer"]["limit_val_batches"] = final["trainer"]["limit_val_batches"]
    if not data["val_splits"]:
        cfg["trainer"]["limit_val_batches"] = 0
    run = cfg["run"]
    root = Path(final.get("run", {}).get("root", "runs")).resolve()
    run["root"] = str(root)
    if "wandb" in final.get("run", {}):
        run["wandb"] = final["run"]["wandb"]
    directory = root / run["group"] / "launch_configs"
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(repo / "src"), env.get("PYTHONPATH")]))
    print("Training release:", data["release"], "| splits:", data.get("train_splits"), flush=True)
    for seed in args.seeds:
        cfg["seed_everything"] = data["seed"] = seed
        run.update(name=f"RadarViT-s{seed}", id="final-v1")
        path = directory / f"seed-{seed}.yaml"
        content = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Existing config differs: {path}. Use a separate run.root for a new experiment.")
        path.write_text(content, encoding="utf-8")
        cmd = [sys.executable, "run.py", "fit", "-c", str(path)]
        if args.check_only:
            subprocess.run(cmd + ["--print_config"], env=env, check=True, stdout=subprocess.DEVNULL)
            print("Config accepted:", path, flush=True)
            continue
        checkpoints = root / run["group"] / run["name"] / run["id"] / "checkpoints"
        if (checkpoints / "swa.ckpt").is_file():
            print("Already has final averaged checkpoint:", checkpoints, flush=True)
            continue
        if (checkpoints / "last.ckpt").is_file():
            cmd += ["--ckpt_path", str(checkpoints / "last.ckpt")]
        subprocess.run(cmd, env=env, check=True)
        if not (checkpoints / "swa.ckpt").is_file():
            raise SystemExit(f"Training exited without the expected SWA checkpoint in {checkpoints}.")


if __name__ == "__main__":
    main()
