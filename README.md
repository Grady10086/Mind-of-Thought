# Mind of Thought

Mind of Thought is a release-ready packaging of the confidence-aware spatial reasoning pipeline.
It vendors the lightweight runtime modules it needs and removes the old hardcoded dependency on the original private project layout.

## What is included

- `core/`: vendored runtime modules required by Grid64 building
- `data/eval_samples.json`: bundled evaluation manifest
- `runtime_config.py`: shared path and environment resolution
- `scripts/mind_of_thought.py`: stable public entrypoint
- `scripts/results_parser.py`: parses launcher logs or `gpu*/detailed_results.json`
- `docs/README.md`: usage and repository layout
- `docs/REPRODUCE.md`: verified reproduction notes for `Spatial-2`

## External assets still required

This repository does not vendor large model or dataset assets. You still need:

1. Qwen3-VL-8B-Instruct weights
2. VSIBench videos (`arkitscenes`, `scannet`, `scannetpp`)
3. A Depth-Anything-3 checkout
4. The verified release environment, especially `transformers==5.3.0` with the Qwen3-VL bugfix applied

## Quick start

```bash
git clone https://github.com/Grady10086/Mind-of-Thought.git
cd Mind-of-Thought
cp config/local.env.example config/local.env
```

Fill in `config/local.env`, then run either:

```bash
python3 scripts/mind_of_thought.py --n_per_type 1 --vl-nframes 8
```

or:

```bash
bash scripts/run_parallel.sh smoke
```

## Documentation

- Usage guide: `docs/README.md`
- Environment summary: `docs/ENVIRONMENT.md`
- Reproduction notes: `docs/REPRODUCE.md`
