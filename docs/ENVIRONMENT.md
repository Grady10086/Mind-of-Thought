# Release Environment Summary

This repository was verified in the following environment:

## Required versions

| Component | Version | Notes |
| --- | --- | --- |
| Python | 3.10 | Verified on `Spatial-2` |
| PyTorch | 2.8.0+ | ROCm build on `Spatial-2` |
| Transformers | 5.3.0 | Requires the Qwen3-VL fix below |
| qwen-vl-utils | 0.0.14 | Used for video processing |

## Required Qwen3-VL fix

The verified environment applied a local patch to:

`/opt/conda/envs/py_3.10/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`

The patch adds exception handling around the grid iteration logic so the Qwen3-VL video path does not crash on missing or exhausted grid iterators.

## External assets

- Qwen3-VL-8B-Instruct weights
- VSIBench videos for `arkitscenes`, `scannet`, and `scannetpp`
- A local checkout of Depth-Anything-3

## Validation commands

```bash
python3 -m py_compile runtime_config.py \
  scripts/mind_of_thought.py \
  scripts/mind_of_thought_pipeline.py \
  scripts/mind_of_thought_baseline.py \
  scripts/results_parser.py \
  core/semantic_labeler.py

python3 scripts/mind_of_thought.py --help
python3 scripts/mind_of_thought_baseline.py --help
bash -n scripts/run_parallel.sh
```
