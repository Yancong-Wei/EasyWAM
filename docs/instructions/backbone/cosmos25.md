# Cosmos-Predict2.5-2B Backbone

[中文](cosmos25_zh.md) | [Backbone index](README.md) | [Model configuration](../config/models.md) | [Back to project README](../../../README.md)

Cosmos2.5 is supported by EasyWAM-Unified, MoT, MoT-Joint, MoT-IDM, and Hidden, with full-parameter and LoRA model configurations. The default paths and runtime settings live in `configs/model/backbone/cosmos25.yaml`.

## Prepare model weights

The video model supplies the post-trained 2B DiT and its Wan2.1 tokenizer. Cosmos-Reason1-7B supplies the text encoder and tokenizer used for offline caching and runtime instructions:

```bash
mkdir -p checkpoints

huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --include "base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt" "tokenizer.pth" \
  --local-dir checkpoints/Cosmos-Predict2.5-2B

huggingface-cli download nvidia/Cosmos-Reason1-7B \
  --local-dir checkpoints/Cosmos-Reason1-7B
```

## Initialize ActionDiT

MoT, MoT-Joint, MoT-IDM, and Hidden use a separate ActionDiT. Generate its interpolated initialization once:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/easywam_mot_cosmos25.yaml \
  --backbone cosmos25 \
  --output checkpoints/ActionDiT_CosmosPredict25_2B_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

EasyWAM-Unified does not use this ActionDiT file. To store weights elsewhere, override `model.backbone.model_id`, `model.backbone.reason_model_id`, or `model.backbone.action_dit_pretrained_path`.

## RoboDojo

Retained research backend; RoboDojo training and closed-loop validation have not been completed for this project.

```bash
python scripts/precompute_text_embeds.py task=robodojo_easywam_mot_wan22 model=easywam_mot_cosmos25
python scripts/train.py --cfg job task=robodojo_easywam_mot_wan22 model=easywam_mot_cosmos25
```

Use matching model, text encoder/cache, initialization and training/evaluation configs. A model-group override alone does not validate memory use or policy quality. The fixed [Wan5B/MoT workflow](../benchmark/robodojo_pipeline_zh.md) does not switch backends automatically.
