# Reproduce Mind_of_Thought on Spatial-2

This document records the verified paths and commands used on `Spatial-2`.

## Verified paths on Spatial-2

- Repository root: `/home/tione/notebook/tianjungu/Mind_of_Thought`
- Local machine config: `/home/tione/notebook/tianjungu/Mind_of_Thought/config/local.env`
- Qwen3-VL weights: `/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct`
- GroundingDINO cache root: `/home/tione/notebook/tianjungu/hf_cache`
- VSIBench videos:
  - `/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes`
  - `/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet`
  - `/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp`
- Depth-Anything-3 checkout: `/home/tione/notebook/tianjungu/projects/Depth-Anything-3`
- Verified evaluation manifest: `/home/tione/notebook/tianjungu/Mind_of_Thought/data/eval_samples_v7_reference.json`

## Environment prerequisites

- Python 3.10
- PyTorch 2.8.0+ (ROCm build on Spatial-2)
- `transformers==5.3.0`
- `qwen-vl-utils==0.0.14`
- The Qwen3-VL fix already applied in:
  `/opt/conda/envs/py_3.10/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`

## 1. Prepare the local config

```bash
cd /home/tione/notebook/tianjungu/Mind_of_Thought
cp config/local.env.example config/local.env
```

For `Spatial-2`, the verified values are:

```bash
HF_HOME=/home/tione/notebook/tianjungu/hf_cache
HF_ENDPOINT=https://hf-mirror.com
MOT_VL_MODEL=/home/tione/notebook/tianjungu/hf_cache/Qwen/Qwen3-VL-8B-Instruct
MOT_VIDEO_DIRS=/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/arkitscenes:/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannet:/home/tione/notebook/tianjungu/projects/ms-swift/tianjungu/projects/ms-swift/run_sft_deepspeed.sh/vsibench/scannetpp
MOT_DA3_ROOT=/home/tione/notebook/tianjungu/projects/Depth-Anything-3
```

`MOT_GDINO_MODEL` is optional on this machine because the runtime now auto-resolves the cached local snapshot under `HF_HOME`.

## 2. Quick validation

Syntax and entrypoints:

```bash
cd /home/tione/notebook/tianjungu/Mind_of_Thought
python3 -m py_compile runtime_config.py   scripts/grid64_agentic_pipeline.py   scripts/grid64_agentic_pipeline_v21.py   scripts/grid64_real_test.py   scripts/overall_results_parser.py   core/semantic_labeler.py

python3 scripts/grid64_agentic_pipeline.py --help
python3 scripts/grid64_real_test.py --help
bash -n scripts/run_8gpu_agentic_parallel.sh
```

## 3. Run a smoke test

```bash
cd /home/tione/notebook/tianjungu/Mind_of_Thought
bash scripts/run_8gpu_agentic_parallel.sh smoke
```

This writes a new timestamped directory under:

`/home/tione/notebook/tianjungu/Mind_of_Thought/outputs/agentic_pipeline_v21_<timestamp>`

## 4. Run a full 8-GPU job

```bash
cd /home/tione/notebook/tianjungu/Mind_of_Thought
bash scripts/run_8gpu_agentic_parallel.sh full
```

## 5. Parse results

If you have launcher logs:

```bash
cd /home/tione/notebook/tianjungu/Mind_of_Thought
python3 scripts/overall_results_parser.py --base outputs/<run_dir>
```

If you only have worker output directories with `gpu*/detailed_results.json`:

```bash
cd /home/tione/notebook/tianjungu/Mind_of_Thought
python3 scripts/overall_results_parser.py --base /path/to/run_dir_with_gpu_subdirs
```

## 6. Reproduce the shard-to-shard comparison against the original V21

Original verified shard reference:

- `/home/tione/notebook/tianjungu/projects/Spatial-Intelligence-MindCube/outputs/agentic_pipeline_v21_ref/gpu2/detailed_results.json`

Equivalent Mind_of_Thought shard command:

```bash
cd /home/tione/notebook/tianjungu/Mind_of_Thought
python3 scripts/grid64_agentic_pipeline.py   --full   --gpu_id 2   --num_gpus 8   --device cuda:2   --output-dir outputs/compare_gpu2_manual
```

The worker result lands at:

- `/home/tione/notebook/tianjungu/Mind_of_Thought/outputs/compare_gpu2_manual/gpu2/detailed_results.json`

## Verified result

On `2026-03-26`, the live comparison against the original V21 `gpu2` shard matched the first scene (`42899696`) exactly on the first 17 emitted question results, including `question_type`, `prediction`, `ground_truth`, and `score`.
