# RoboDojo training configuration

[Configuration](README.md) | [Workflow](../benchmark/robodojo_pipeline_zh.md)

`configs/data/robodojo.yaml` uses LeRobot v3, joint14 state/action and three cameras. A 33-observation window supplies 32 actions and nine sampled video frames. The head image and two wrist images form a 384×320 image. Keep transforms, `shape_meta`, `video_size` and camera order consistent.

The historical `concat_multi_camera: robotwin` value identifies this shared three-camera layout; it does not import or run another benchmark. It remains unchanged for checkpoint/config compatibility. Normalization is z-score using the dataset's `dataset_stats.json`; use the same file during inference.

```bash
python scripts/robodojo_pipeline.py prepare
python scripts/robodojo_pipeline.py probe --run-id first-probe --dry-run
python scripts/robodojo_pipeline.py train --run-id baseline
```

The A800 recipe uses micro-batch 1, gradient accumulation 16, BF16, checkpointing and VAE micro-batch 1. Training updates the visual/action transformers and state projection; VAE/text encoder are frozen. The base task's optimizer settings remain inherited. Check effective batch and processed windows when comparing GPU counts or training durations.

A run contains `config.yaml`, normalization statistics, weight exports and full training states. Resume from a `checkpoints/state/` directory to restore optimizer progress; `checkpoints/weights/*.pt` restores weights only. `checkpoint_save_limit=2` does not eliminate temporary save peaks. Verify CPU RAM, GPU memory, disk space, finite loss, save/resume and inference before a long run.

Training validation loss or predicted-video quality is not an official simulator score. Use the separate [evaluation workflow](../benchmark/robodojo.md).
