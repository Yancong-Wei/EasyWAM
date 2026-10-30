# Wan2.2-TI2V-5B Backbone

[中文](wan22_zh.md) | [Backbone index](README.md) | [Model configuration](../config/models.md) | [Back to project README](../../../README.md)

Wan2.2 is supported by EasyWAM-Unified, MoT, MoT-Joint, MoT-IDM, and Hidden, with full-parameter and LoRA model configurations. The default paths and runtime settings live in `configs/model/backbone/wan22.yaml`.

## Prepare model weights

Run from the project root after installing EasyWAM and the Hugging Face CLI:

```bash
mkdir -p checkpoints
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir checkpoints/Wan2.2-TI2V-5B
```

This directory supplies the Video DiT, VAE, UMT5 encoder, and tokenizer. The default config expects the UMT5 files under `checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl`.

## Initialize ActionDiT

MoT, MoT-Joint, MoT-IDM, and Hidden use a separate ActionDiT. Generate its interpolated initialization once:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/easywam_mot_wan22.yaml \
  --backbone wan22 \
  --output checkpoints/ActionDiT_Wan22_5B_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

EasyWAM-Unified does not use this ActionDiT file. To store weights elsewhere, override `model.backbone.model_id`, `model.backbone.tokenizer_model_id`, or `model.backbone.action_dit_pretrained_path` instead of editing shared configs.

## RoboDojo

The current Wan5B/MoT baseline uses the unified workflow.

```bash
python scripts/precompute_text_embeds.py task=robodojo_easywam_mot_wan22 model=easywam_mot_wan22
python scripts/train.py --cfg job task=robodojo_easywam_mot_wan22 model=easywam_mot_wan22
```

Use matching model, text encoder/cache, initialization and training/evaluation configs. A model-group override alone does not validate memory use or policy quality. The fixed [Wan5B/MoT workflow](../benchmark/robodojo_pipeline_zh.md) does not switch backends automatically.
