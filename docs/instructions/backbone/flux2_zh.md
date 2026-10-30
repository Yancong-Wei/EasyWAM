# FLUX.2 Klein Base 4B Backbone

[English](flux2.md) | [Backbone 索引](README_zh.md) | [模型配置](../config/models_zh.md) | [返回项目 README](../../../README_zh.md)

FLUX.2 支持 Unified、MoT、MoT-Joint、MoT-IDM 和 Hidden，并提供全参数与 LoRA 模型配置。Backbone 由双流/单流 DiT、图像 VAE 和 Qwen3 文本编码器组成。默认路径和运行设置位于 `configs/model/backbone/flux2_klein_4b.yaml`。

## 准备模型权重

在项目根目录执行。下载受限文件前，需在 Hugging Face 上接受自编码器仓库的访问条件，并执行 `huggingface-cli login`：

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

文本编码器默认使用 `Qwen/Qwen3-4B`。若使用本地下载，在文本预计算和评测命令中加入 `model.backbone.qwen3_model_spec=./checkpoints/Qwen3-4B`。Transformer 和 VAE 路径分别由 `model.backbone.model_path` 和 `model.backbone.ae_model_path` 配置。

## 初始化 ActionDiT

动作专家使用 `staged` 预设，宽度为 `1024`，包含 5 个顺序 block 和 20 个并行 block。Hidden 启用视觉特征交叉注意力，也支持独立配置动作层数。通用组件选项见[模型配置](../config/models_zh.md#配置-actiondit)。

`model.action_dit_pretrained_path=null` 表示随机初始化。使用视觉权重初始化 staged ActionDiT：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --backbone flux2 \
  --model-config configs/model/easywam_mot_flux2_klein_4b.yaml \
  --output checkpoints/ActionDiT_FLUX2_4B_alphascale_1024hdim.pt
```

训练时加入 `model.action_dit_pretrained_path=./checkpoints/ActionDiT_FLUX2_4B_alphascale_1024hdim.pt`。Hidden 使用 `configs/model/easywam_hidden_flux2_klein_4b.yaml` 并输出到独立文件。MoT 系列的默认动作结构共用初始化文件；Unified 没有独立 ActionDiT。

默认启用 alpha scaling（`--apply-alpha-scaling true`）：沿输入维度缩放的权重乘以 `sqrt(source_dim / target_dim)`。转换器迁移兼容的视觉参数，分别缩放 Q/K/V 与门控 FFN 的各部分。输出包含参数来源和覆盖率，新条件模块保留其初始化。转换支持 staged 预设，生成的文件必须与所选动作结构匹配。

## RoboDojo

保留的研究 backend；尚未完成本项目 RoboDojo 的训练及闭环验证。

```bash
python scripts/precompute_text_embeds.py task=robodojo_easywam_mot_wan22 model=easywam_mot_flux2_klein_4b
python scripts/train.py --cfg job task=robodojo_easywam_mot_wan22 model=easywam_mot_flux2_klein_4b
```

模型、文本编码器/缓存、初始化以及训练/评测配置必须相互匹配。切换 model 配置组本身并不证明显存足够或策略有效。固定的 [Wan5B/MoT 流程](../benchmark/robodojo_pipeline_zh.md) 不会自动切换 backend。
