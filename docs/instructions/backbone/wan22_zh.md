# Wan2.2-TI2V-5B Backbone

[English](wan22.md) | [Backbone 索引](README_zh.md) | [模型配置](../config/models_zh.md) | [返回项目 README](../../../README_zh.md)

Wan2.2 支持 EasyWAM-Unified、MoT、MoT-Joint、MoT-IDM 和 Hidden，并提供全参数与 LoRA 模型配置。默认路径和运行设置位于 `configs/model/backbone/wan22.yaml`。

## 准备模型权重

安装 EasyWAM 和 Hugging Face CLI 后，在项目根目录执行：

```bash
mkdir -p checkpoints
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir checkpoints/Wan2.2-TI2V-5B
```

该目录提供 Video DiT、VAE、UMT5 编码器和 tokenizer。默认配置从 `checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl` 加载 UMT5 文件。

## 初始化 ActionDiT

MoT、MoT-Joint、MoT-IDM 和 Hidden 使用独立的 ActionDiT。生成一次插值初始化权重：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/easywam_mot_wan22.yaml \
  --backbone wan22 \
  --output checkpoints/ActionDiT_Wan22_5B_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

EasyWAM-Unified 不使用该文件。自定义权重位置时，可覆盖 `model.backbone.model_id`、`model.backbone.tokenizer_model_id` 或 `model.backbone.action_dit_pretrained_path`。

## RoboDojo

当前 Wan5B/MoT 主线；使用统一流程。

```bash
python scripts/precompute_text_embeds.py task=robodojo_easywam_mot_wan22 model=easywam_mot_wan22
python scripts/train.py --cfg job task=robodojo_easywam_mot_wan22 model=easywam_mot_wan22
```

模型、文本编码器/缓存、初始化以及训练/评测配置必须相互匹配。切换 model 配置组本身并不证明显存足够或策略有效。固定的 [Wan5B/MoT 流程](../benchmark/robodojo_pipeline_zh.md) 不会自动切换 backend。
