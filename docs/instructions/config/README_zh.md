# RoboDojo 配置

[文档索引](../../README_zh.md) | [模型](models_zh.md) | [训练](training_zh.md) | [效率](efficiency_zh.md)

`configs/train.yaml` 通过 `task=robodojo_easywam_mot_wan22_a800` 组合 `data=robodojo` 和 `model=easywam_mot_wan22`。A800 配方继承 MoT 全参数任务，并调整 batch 和运行参数。

```bash
python scripts/train.py --cfg job task=robodojo_easywam_mot_wan22_a800
python experiments/robodojo/run_robodojo_manager.py --cfg job task=robodojo_easywam_mot_wan22_a800
```

`configs/benchmark/sim_robodojo.yaml` 在推理时加载文本编码器及训练好的策略权重。manager 将完整解析后的配置传给策略进程。动作维度、相机布局、state token 位置和归一化必须与训练时保存的配置一致。

其余 RoboDojo task 是架构对照接口，不代表已经获得跑分。直接覆盖 model 配置组可用于研究；统一流程入口有意固定为 Wan5B/MoT。
