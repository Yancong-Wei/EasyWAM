# FLUX.2 Klein Base 4B Backbone

[中文](flux2_zh.md) | [Backbone index](README.md) | [Model configuration](../config/models.md) | [Back to project README](../../../README.md)

FLUX.2 supports Unified, MoT, MoT-Joint, MoT-IDM, and Hidden, with full-parameter and LoRA model configurations. The backbone combines a double-stream/single-stream DiT, an image VAE, and a Qwen3 text encoder. Default paths and runtime settings are in `configs/model/backbone/flux2_klein_4b.yaml`.

## Prepare model weights

Run from the project root. Accept the autoencoder repository's access conditions on Hugging Face and run `huggingface-cli login` before downloading gated files:

```bash
mkdir -p checkpoints/flux2

huggingface-cli download black-forest-labs/FLUX.2-klein-base-4B \
  --include "flux-2-klein-base-4b.safetensors" \
  --local-dir checkpoints/flux2/FLUX.2-klein-base-4B

huggingface-cli download black-forest-labs/FLUX.2-dev \
  --include "ae.safetensors" \
  --local-dir checkpoints/flux2/FLUX.2-dev

huggingface-cli download Qwen/Qwen3-4B \
  --local-dir checkpoints/Qwen3-4B
```

The text encoder defaults to `Qwen/Qwen3-4B`. To use the local download, pass `model.backbone.qwen3_model_spec=./checkpoints/Qwen3-4B` when precomputing text or evaluating. Transformer and VAE locations are configured with `model.backbone.model_path` and `model.backbone.ae_model_path`.

## Initialize ActionDiT

Action experts use the `staged` preset, width `1024`, five sequential blocks, and twenty parallel blocks. Hidden enables cross-attention to visual features and also supports an independently configured action depth. The general component options are documented in [model configuration](../config/models.md#configure-actiondit).

`model.action_dit_pretrained_path=null` selects random initialization. To initialize a staged ActionDiT from visual weights:

```bash
python scripts/preprocess_action_dit_backbone.py \
  --backbone flux2 \
  --model-config configs/model/easywam_mot_flux2_klein_4b.yaml \
  --output checkpoints/ActionDiT_FLUX2_4B_alphascale_1024hdim.pt
```

Pass `model.action_dit_pretrained_path=./checkpoints/ActionDiT_FLUX2_4B_alphascale_1024hdim.pt` to training. For Hidden, use `configs/model/easywam_hidden_flux2_klein_4b.yaml` and a separate output file. MoT-family default action structures share an initialization file; Unified has no separate ActionDiT.

Alpha scaling is enabled by default (`--apply-alpha-scaling true`): weights resized along the input dimension are scaled by `sqrt(source_dim / target_dim)`. The converter transfers compatible visual parameters and resizes Q/K/V and gated-FFN sections separately. Its output includes parameter sources and coverage; new conditioning modules retain their initialization. Conversion supports the staged preset, and the resulting file must match the selected action structure.

## RoboDojo

Retained research backend; RoboDojo training and closed-loop validation have not been completed for this project.

```bash
python scripts/precompute_text_embeds.py task=robodojo_easywam_mot_wan22 model=easywam_mot_flux2_klein_4b
python scripts/train.py --cfg job task=robodojo_easywam_mot_wan22 model=easywam_mot_flux2_klein_4b
```

Use matching model, text encoder/cache, initialization and training/evaluation configs. A model-group override alone does not validate memory use or policy quality. The fixed [Wan5B/MoT workflow](../benchmark/robodojo_pipeline_zh.md) does not switch backends automatically.
