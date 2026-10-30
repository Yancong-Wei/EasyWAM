# EasyWAM · RoboDojo Core

[English](README.md) | [完整操作流程](docs/instructions/benchmark/robodojo_pipeline_zh.md) | [文档索引](docs/README_zh.md)

这个开发核心基于 [OpenMOSS/EasyWAM](https://github.com/OpenMOSS/EasyWAM)，专注 RoboDojo 的数据准备、训练、策略服务、仿真评测与可复现模型发布。保留上游来源及 [MIT 许可证](LICENSE)。当前不代表已发布的新模型或已获得官方榜单成绩。

## 当前基线

- Wan2.2-TI2V-5B + MoT，joint14，三相机；训练 VideoDiT、ActionDiT 和状态投影，冻结 VAE/文本编码器，不启用 LoRA。
- 单卡配方：micro-batch 1、梯度累积 16、BF16、梯度检查点，默认 ZeRO-2 CPU optimizer offload。
- 唯一 benchmark：RoboDojo。共享训练器、LeRobot 数据层、模型组件及 Wan/Cosmos/FLUX backend 保留；其他架构与 backend 是研究扩展，不能据此宣称已在 RoboDojo 验证。
- 下载/配置/评分的回归测试可在 CPU 上运行。完整模型训练、保存恢复及 Isaac Sim 闭环仍需硬件验证。

## 使用

Linux 训练环境建议沿用已验证依赖的 Python 3.10 + CUDA 12.8/PyTorch 2.7.1 环境。项目依赖声明位于 `pyproject.toml`；安装 CUDA PyTorch 时需要匹配的官方 wheel 源。

```bash
python -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
python scripts/robodojo_pipeline.py doctor --gpu
python scripts/robodojo_pipeline.py download --size-only
python scripts/robodojo_pipeline.py download
python scripts/robodojo_pipeline.py prepare
python scripts/robodojo_pipeline.py probe --run-id first-probe --dry-run
```

`probe` 去掉 `--dry-run` 才开始短训练；通过保存/恢复/推理验证后再运行 `train`。独立策略服务使用 `serve`；仿真准备、规划、单任务检查、完整评测和汇总分别使用 `setup-sim`、`plan`、`smoke-eval`、`evaluate`、`summarize`。参数与资源要求见[全流程指南](docs/instructions/benchmark/robodojo_pipeline_zh.md)。

## 开发与复现

```bash
python -m unittest discover -s experiments/robodojo/tests -v
git diff --check
```

主要目录：`configs/` 配方、`src/` 训练与模型、`experiments/robodojo/` 推理和评分、`scripts/robodojo_pipeline.py` 流程入口、`docs/` 使用说明。`data/`、`checkpoints/`、`runs/`、仿真源码/资产与 `research/` 本地资料不纳入开发核心。

清理边界及分支说明见 [开发核心范围](docs/develop_core_zh.md)。对外发布应同时提供权重、训练配置、归一化统计、数据/源码版本和完整评测证据。

研究方案见 [Wan-MoT 改进方向与实验设计](design/robodojo-wan-mot-research.md)。
