# RoboDojo resources and efficiency

[Configuration](README.md) | [Workflow and measured limits](../benchmark/robodojo_pipeline_zh.md)

Start with `python scripts/robodojo_pipeline.py doctor --gpu` and the 20-step `probe`. Measure optimizer-step time after warmup, peak GPU/CPU memory, data wait time, checkpoint size/save time and restore behavior. CPU offload consumes container RAM and PCIe bandwidth; single-GPU ZeRO does not distribute parameters across devices.

Tune `num_workers`, `dataloader_prefetch_factor`, `dataloader_pin_memory` and `dataloader_worker_threads` only after measuring data loading. Keep `vae_micro_batch_size=1` and gradient checkpointing for the initial single-GPU baseline. Text features are precomputed once per matching encoder/config.

Evaluation has separate policy and simulator GPU pools. `MULTIRUN.inference_batch_size`, `inference_batch_wait_ms`, worker counts and `EVALUATION.sim_num_envs` affect memory/latency. Start with one environment and inference batch 1. A fast policy server does not establish simulator compatibility on the same GPU.

Change one setting at a time and record the resolved config. A different policy/inference setting requires a distinct evaluation output identity. Increasing training throughput, validation quality and official score are separate measurements.
