# JF6 RadarViT final training

Train four independent copies of the tested JF6 RadarViT detector on one prepared final training set. Each seed starts from scratch and produces its own averaged checkpoint. The launcher calls only the FallJEPA fit entry point.

## Requirements

Use a compatible FallJEPA checkout containing run.py, the JF6 (JF5T) model and its training dependencies. Use that checkout's Python environment, including PyYAML, Lightning and GPU-enabled PyTorch. The commands below use python from the active environment. If the repository uses uv, substitute uv run --frozen python. A single supported GPU is required for training.

Prepare a complete final-set YAML accepted by that checkout, with working paths when launched from the repository root. It must specify data.release, data.stats_path, data.sampler.groups, and the train_splits, val_splits and test_splits lists. The release must follow the JF6 data layout (including streams/<training split>, training annotations and the metadata used by its interior normalizer). Use the final training release's normalizer path and groups CSV. Move any dataset-path overrides from a previous shell command into this YAML.

## Install the encoder

Extract the ZIP into the repository root. It creates the directory JF6_RadarViT_final. Run these commands from the directory containing run.py:

```sh
git apply --check JF6_RadarViT_final/radarvit.patch
git apply JF6_RadarViT_final/radarvit.patch
```

Apply the patch only after the check succeeds. It adds the supplied RadarViT class and registers trunk=radarvit. If the checkout already includes this implementation, skip the patch. If a modified checkout rejects the patch, inspect the differences in the model constructor before proceeding.

## Training recipe

The model is JF6, called JF5T in the handover. RadarViT replaces RadarStem. For each frame, two spatial transformer blocks attend over 24 tokens (three radars times eight range patches). The encoder returns 400 frame vectors of width 64 for each 32-second window. The local temporal encoder is the original TCN. The three global attention blocks, event heads, losses, optimizer and decoder use the tested JF6 RadarViT configuration.

The recipe uses 15 epochs, 32-bit precision, AdamW at 0.001, weight decay 0.0001 and the original weight averaging callback. Each update uses eight training chunks, split into four consecutive microbatches of two chunks with four-way gradient accumulation. Evaluation batch size is eight.

The launcher reads only the final release, normalization and group paths, split lists, and optional worker count, validation batch limit, run root and W&B mode from the supplied final-set YAML. The packaged recipe controls the model and training settings. The generated YAMLs are saved under <run.root>/Final/JF6RadarViT/launch_configs for inspection.

## Check and launch

Replace /path/to/final.yaml and the four example seed IDs with the final-set configuration and the four seeds used by your team. Run from the repository root.

```sh
python JF6_RadarViT_final/train_final.py /path/to/final.yaml 0 1 2 3 --check-only
```

This checks the training paths and asks the repository CLI to parse each generated configuration. Check the printed release and split names. To start all four fits in sequence:

```sh
python -u JF6_RadarViT_final/train_final.py /path/to/final.yaml 0 1 2 3
```

## Outputs and restart

For each seed, the run directory is <run.root>/Final/JF6RadarViT/RadarViT-s<seed>/final-v1/. Keep its config.yaml, checkpoints/swa.ckpt and the corresponding training normalizer for later inference. After an interruption, rerun the same command and seed IDs. The launcher skips completed seeds with an averaged checkpoint and resumes interrupted fits with last.ckpt when present. Use one launcher per output root at a time.
