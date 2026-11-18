"""
Deep-merges YAML configs into one, for ablation overlays that must stack:

    uv run python tools/compose_config.py OUT.yaml configs/uwb/U0.yaml configs/uwb/ablation.yaml \\
        configs/uwb/ablations/A1.yaml configs/uwb/ablations/A3.yaml

Later files win. Mappings merge key by key at every depth; any other value (lists included)
replaces the earlier one; the value `__delete__` removes the key. LightningCLI's own `-c` chaining
replaces dict-typed arguments such as `model.losses` and `model.loss_weights` whole, so an overlay
that only changed one loss would silently bring back every loss an earlier overlay removed. Merged
here first, A1 can drop the JEPA loss and A3 change the class weights, in either order.

An overlay that changes a loss's class_path should delete the old entry first (`__delete__`),
or the old init_args are merged into the new class.
"""
import sys
from pathlib import Path

import yaml

DELETE = "__delete__"


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if value == DELETE:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def compose(paths: list[Path]) -> dict:
    config: dict = {}
    for path in paths:
        config = merge(config, yaml.safe_load(Path(path).read_text()) or {})
    return config


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        sys.exit(f"usage: {Path(__file__).name} OUT.yaml CONFIG.yaml...")
    out, paths = Path(argv[0]), [Path(p) for p in argv[1:]]
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "".join(f"# composed from {p}\n" for p in paths)
    out.write_text(header + yaml.safe_dump(compose(paths), sort_keys=False))


if __name__ == "__main__":
    main()
