# RoboDojo 资源与效率

[配置索引](README_zh.md) | [全流程与资源检查](../benchmark/robodojo_pipeline_zh.md)

从 `python scripts/robodojo_pipeline.py doctor --gpu` 和 20 步 `probe` 开始。记录预热后的每个优化步耗时、GPU/CPU 峰值、数据等待时间、checkpoint 大小/保存耗时与恢复行为。CPU offload 占用容器内存和 PCIe 带宽；单卡 ZeRO 不会把参数分摊到其他设备。

测量数据读取后，再调整 `num_workers`、`dataloader_prefetch_factor`、`dataloader_pin_memory` 和 `dataloader_worker_threads`。初始单卡配方保留 `vae_micro_batch_size=1` 与梯度检查点；文本特征按匹配的编码器/配置预计算。

评测有独立的策略和仿真 GPU 池。`MULTIRUN.inference_batch_size`、`inference_batch_wait_ms`、worker 数与 `EVALUATION.sim_num_envs` 都影响显存和延迟。先用单环境、推理 batch 1。策略服务器能运行，并不代表同一张 GPU 能运行仿真器。

每次只修改一个设置并记录完整解析后的配置。改变模型或推理设置时，应使用新的评测输出身份。训练吞吐量、验证质量和官方分数需要分别测量。
