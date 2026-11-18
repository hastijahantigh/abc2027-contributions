# Idea 2 experiment
Extract the ZIP in ~/hasti and cd into hasti_radarvit_v1. Run the following in your remote Bash terminal.
```bash
export PATH="/home/abc2027/.local/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
uv sync --frozen
CUDA_VISIBLE_DEVICES="" uv run --frozen pytest -q 'tests/test_jf4.py::test_cli_jf4_fits_and_validates[JF6Radar]'
```
This test fits a synthetic CPU release and validates the saved checkpoint. For the GPU smoke test run `bash tools/hasti_radar.sh smoke` once the GPU is available. The launcher defaults to four microbatches of two chunks each per optimizer update. The model uses 24 per-frame spatial tokens, width 64, two attention blocks, and the original TCN and heads. Each split has separate result folders. Real CUDA peak memory remains unmeasured. For a matched comparison run the actions baseline, radar and ensemble sequentially with the same MICRO_SPLIT, SEED and FOLDS environment variables. Default fold D and seed 0. To run all four folds set FOLDS="A B C D". Read HASTI_RUN.md only for the older range experiment.
