# EasyWAM · RoboDojo Core

[中文](README_zh.md) | [Workflow guide](docs/instructions/benchmark/robodojo_pipeline_zh.md) | [Documentation](docs/README.md)

A RoboDojo-focused development core derived from [OpenMOSS/EasyWAM](https://github.com/OpenMOSS/EasyWAM), retaining the upstream attribution and [MIT license](LICENSE). This checkout is not a released novel model or an official leaderboard result.

## Baseline

Wan2.2-TI2V-5B + MoT, joint14 and three cameras. Fine-tune the VideoDiT, ActionDiT and state projection; keep the VAE/text encoder frozen. No LoRA in the baseline. The single-GPU recipe uses BF16, micro-batch 1, accumulation 16, gradient checkpointing and CPU optimizer offload.

RoboDojo is the only benchmark integration. Shared training/data/model code and the Wan/Cosmos/FLUX backends remain available for research; their presence does not establish RoboDojo validation for every architecture/backend.

## Workflow

Use a compatible Linux Python 3.10/CUDA environment. Dependencies are declared in `pyproject.toml`.

```bash
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
python scripts/robodojo_pipeline.py doctor --gpu
python scripts/robodojo_pipeline.py download --size-only
python scripts/robodojo_pipeline.py download
python scripts/robodojo_pipeline.py prepare
python scripts/robodojo_pipeline.py probe --run-id first-probe --dry-run
```

Remove `--dry-run` to run the probe. Verify training, checkpoint save/resume and inference before starting a long `train` run. The same entry point provides `serve`, `setup-sim`, `plan`, `smoke-eval`, `evaluate` and `summarize`; see the [complete workflow](docs/instructions/benchmark/robodojo_pipeline_zh.md).

## Development

```bash
python -m unittest discover -s experiments/robodojo/tests -v
git diff --check
```

Core code lives in `configs/`, `src/`, `experiments/robodojo/` and `scripts/`. Datasets, weights, training outputs, simulator assets and local research snapshots are excluded from the core. See [development scope](docs/develop_core_zh.md).

CPU workflow tests do not establish successful full-model training or simulator scores. A reproducible release needs checkpoint weights, training config, normalization statistics, data/source revisions and full evaluation evidence.
