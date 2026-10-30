# RoboDojo 训练配置

[配置索引](README_zh.md) | [全流程](../benchmark/robodojo_pipeline_zh.md)

`configs/data/robodojo.yaml` 使用 LeRobot v3、joint14 状态/动作和三相机。33 个观测步对应 32 步动作及 9 帧采样视频；头部和两路腕部图像拼成 384×320 图像。图像变换、`shape_meta`、`video_size` 和相机顺序必须一致。

历史配置值 `concat_multi_camera: robotwin` 表示共享三相机拼接布局，不会导入或运行其他 benchmark。为兼容既有配置与 checkpoint 保留该名称。归一化使用数据集的 `dataset_stats.json` 做 z-score，推理必须使用同一份统计量。

```bash
python scripts/robodojo_pipeline.py prepare
python scripts/robodojo_pipeline.py probe --run-id first-probe --dry-run
python scripts/robodojo_pipeline.py train --run-id baseline
```

A800 配方使用 micro-batch 1、梯度累积 16、BF16、梯度检查点和 VAE micro-batch 1。训练更新视频/动作 Transformer 与状态投影，冻结 VAE/文本编码器；基础任务的优化器设置继续继承。比较不同卡数和训练时长时，应同时核对有效 batch 和处理窗口总数。

训练目录保存 `config.yaml`、归一化统计、权重导出和完整训练状态。恢复优化器进度应使用 `checkpoints/state/` 目录；`checkpoints/weights/*.pt` 仅恢复权重。`checkpoint_save_limit=2` 不会消除新 checkpoint 保存期间的临时空间峰值。长训前检查 CPU/GPU 内存、磁盘、loss 是否有限、保存恢复与推理。

训练验证 loss 或视频预测质量都不是官方仿真成绩，跑分请使用独立的[评测流程](../benchmark/robodojo_zh.md)。
