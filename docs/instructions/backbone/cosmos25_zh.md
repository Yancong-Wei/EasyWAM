# Cosmos-Predict2.5-2B Backbone

[English](cosmos25.md) | [Backbone 索引](README_zh.md) | [模型配置](../config/models_zh.md) | [返回项目 README](../../../README_zh.md)

Cosmos2.5 支持 EasyWAM-Unified、MoT、MoT-Joint、MoT-IDM 和 Hidden，并提供全参数与 LoRA 模型配置。默认路径和运行设置位于 `configs/model/backbone/cosmos25.yaml`。

## 准备模型权重

视频模型提供 post-trained 2B DiT 及其 Wan2.1 tokenizer。Cosmos-Reason1-7B 提供离线文本缓存和运行时任务指令所需的文本编码器及 tokenizer：

```bash
mkdir -p checkpoints

huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --include "base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt" "tokenizer.pth" \
  --local-dir checkpoints/Cosmos-Predict2.5-2B

huggingface-cli download nvidia/Cosmos-Reason1-7B \
  --local-dir checkpoints/Cosmos-Reason1-7B
```

## 初始化 ActionDiT

MoT、MoT-Joint、MoT-IDM 和 Hidden 使用独立的 ActionDiT。生成一次插值初始化权重：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/easywam_mot_cosmos25.yaml \
  --backbone cosmos25 \
  --output checkpoints/ActionDiT_CosmosPredict25_2B_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

EasyWAM-Unified 不使用该文件。自定义权重位置时，可覆盖 `model.backbone.model_id`、`model.backbone.reason_model_id` 或 `model.backbone.action_dit_pretrained_path`。

## RoboDojo

保留的研究 backend；尚未完成本项目 RoboDojo 的训练及闭环验证。

```bash
python scripts/precompute_text_embeds.py task=robodojo_easywam_mot_wan22 model=easywam_mot_cosmos25
python scripts/train.py --cfg job task=robodojo_easywam_mot_wan22 model=easywam_mot_cosmos25
```

模型、文本编码器/缓存、初始化以及训练/评测配置必须相互匹配。切换 model 配置组本身并不证明显存足够或策略有效。固定的 [Wan5B/MoT 流程](../benchmark/robodojo_pipeline_zh.md) 不会自动切换 backend。
