# Mind_of_Thought

Mind_of_Thought is the open-source-ready packaging of the V21 confidence-aware spatial reasoning pipeline.
It keeps the V21 algorithm, vendors the lightweight runtime `core/` modules it needs, and removes the old hardcoded dependency on the original private project layout.

See `docs/REPRODUCE.md` for a verified reproduction recipe on `Spatial-2`.

## Repository layout

```text
Mind_of_Thought/
├── config/
│   ├── local.env.example      # machine-specific configuration template
│   └── local.env              # local-only paths for the current server (gitignored)
├── core/                      # vendored runtime modules required by Grid64 building
├── data/
│   └── eval_samples_v7_reference.json
├── docs/
│   ├── README.md
│   └── REPRODUCE.md
├── outputs/                   # run outputs and historical references
├── runtime_config.py          # shared path/config resolution helpers
└── scripts/
    ├── grid64_agentic_pipeline.py        # stable public entrypoint (wraps V21)
    ├── grid64_agentic_pipeline_v21.py    # V21 implementation
    ├── grid64_real_test.py               # deterministic Grid64 baseline
    ├── overall_results_parser.py         # parses logs or detailed_results.json
    └── run_8gpu_agentic_parallel.sh      # timestamped 8-GPU launcher
```

## What is self-contained now

- The default evaluation manifest is bundled as `data/eval_samples_v7_reference.json`.
- The runtime `core/` Python modules are vendored into this repository.
- The public entrypoint is consistently V21.
- Multi-GPU runs write into a fresh timestamped output directory by default.
- `overall_results_parser.py` can parse either launcher logs or `gpu*/detailed_results.json`.
- GroundingDINO now prefers a local snapshot under `HF_HOME` when one is already cached.

## External assets you still need

This repository does not vendor large model or dataset assets. You still need:

1. Qwen3-VL-8B-Instruct weights.
2. VSIBench videos (`arkitscenes`, `scannet`, `scannetpp`).
3. A Depth-Anything-3 checkout.
4. The verified environment from `V21_ENVIRONMENT.md`, especially `transformers==5.3.0` with the Qwen3-VL bugfix applied.

## Configuration

Create a machine-local config file:

```bash
cd Mind_of_Thought
cp config/local.env.example config/local.env
```

Fill in these variables:

```bash
HF_HOME=/path/to/hf_cache
HF_ENDPOINT=
MOT_VL_MODEL=/path/to/Qwen3-VL-8B-Instruct
MOT_GDINO_MODEL=/path/to/grounding-dino-base
MOT_VIDEO_DIRS=/path/to/vsibench/arkitscenes:/path/to/vsibench/scannet:/path/to/vsibench/scannetpp
MOT_DA3_ROOT=/path/to/Depth-Anything-3
```

Notes:

- `MOT_GDINO_MODEL` is optional. If it is unset, the runtime first tries to reuse a cached local GroundingDINO snapshot under `HF_HOME`, then falls back to the public repo id `IDEA-Research/grounding-dino-base`.
- `HF_ENDPOINT` is optional. Leave it empty unless you intentionally want to use a mirror.
- `config/local.env` is intentionally gitignored. The tracked repository only keeps the template.

## Running V21

Single process:

```bash
cd Mind_of_Thought
python3 scripts/grid64_agentic_pipeline.py   --n_per_type 1   --vl-nframes 8
```

8-GPU smoke test:

```bash
cd Mind_of_Thought
bash scripts/run_8gpu_agentic_parallel.sh smoke
```

8-GPU full run:

```bash
cd Mind_of_Thought
bash scripts/run_8gpu_agentic_parallel.sh full
```

The launcher creates `outputs/agentic_pipeline_v21_<timestamp>/gpu*.log` unless you override `OUTPUT_DIR`.
It refuses to reuse an existing directory unless `OVERWRITE=1` is set.
When `OVERWRITE=1` is set, it cleans the previous `gpu*.log`, `gpu*/`, and `parsed_summary.json` artifacts inside that output directory before the new run starts.

## Parsing results

From launcher logs:

```bash
cd Mind_of_Thought
python3 scripts/overall_results_parser.py --base outputs/<run_dir>
```

From worker result files:

```bash
cd Mind_of_Thought
python3 scripts/overall_results_parser.py --base outputs/<run_dir_with_gpu_dirs>
```

The parser writes `parsed_summary.json` into the same run directory.

## Notes

- `USE_UNIFIED=true` is optional. If the unified pipeline is not present, V21 falls back to its built-in logic instead of crashing.
- `detailed_results.json` is now treated as a regular run artifact, not as the only trusted source of truth.
- The bundled manifest still contains V7 reference fields (`vl_score`, `rule_score`) so V21 can report deltas against the reference system.
