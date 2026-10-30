# A800 上的 Wan2.2-5B / MoT 全流程

本方案固定为 Wan2.2-TI2V-5B + MoT，训练 VideoDiT、ActionDiT 和状态投影，冻结 VAE/文本编码器，不启用 LoRA，不增加历史帧分支。统一入口为 `scripts/robodojo_pipeline.py`。这是已有 EasyWAM 的训练与评测工程完善，不是已经训练好的 XiconWAM 模型。

## 资源与验证边界

请使用 `doctor --gpu` 检查当前机器的 GPU、实际容器内存上限和存储配额。2026-10-30 核对的公开资源元数据为：

- `OpenMOSS-Team/robodojo-lerobot-v3.0`，revision `9724a6f5f2dca85b3c0116f376475e8073bdf329`，111.46 GiB，3,500 episodes、1,856,102 frames、joint14、三相机 AV1 视频。
- `Wan-AI/Wan2.2-TI2V-5B`，revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`，整个仓库约 31.85 GiB。

建议先准备至少 500 GB 的训练数据盘（数据、基础权重、动作初始化、文本缓存、权重导出和完整优化器状态），实测保存峰值后再确定保留数量。仿真资产及评测视频另外预算空间。500 GB 是工程预算，不是实测最低值。

可以将 `data/`、`checkpoints/` 和 `runs/` 链接到容量足够的持久存储。200 GiB 能容纳上述原始资源，但不够稳妥保留完整训练状态。不要把底层共享磁盘的容量当作用户存储配额。

本地回归测试不加载完整模型。依赖导入、CUDA 小矩阵、CPUAdam 小张量操作和少量数据解码检查，也不能替代完整模型前向/反向、checkpoint 保存恢复、整集解码及仿真验证。

单卡使用 micro-batch 1、梯度累积 16、BF16、梯度检查点、ZeRO-2 CPU optimizer offload。单卡 ZeRO 不会凭空分摊参数；CPU offload 也会占容器内存与 PCIe 带宽。80GB 显存并不能直接保证全参数训练成功或速度可接受。先测 20 步、一次保存、恢复、推理后，再决定八卡租期。单卡 30,000 步 × 全局 batch 16 约 48 万窗口，与原配方八卡 × 每卡 16 × 30,000 步约 384 万窗口不是等量训练。

实际结构统计：VideoDiT 4,999,787,712 参数，ActionDiT 1,021,004,686 参数，合计 6,020,792,398 参数（不含小型外部状态投影与冻结模块）。BF16 参数约 11.21 GiB；按 FP32 主权重加两份 Adam 状态估算约 67.29 GiB，尚未计入梯度、激活、VAE、数据与通信缓冲。该数字来自 meta 构建与字节数计算，不是测得的训练峰值。

A800 可用于策略训练/推理；这里没有验证它能运行 RoboDojo 的 Isaac Sim。准备满足官方要求的 RTX 仿真机器。下面 manager 默认模型 GPU 0、仿真 GPU 1，要求在**同一主机**；单卡 A800 不能直接照抄该默认值完成仿真。独立 `serve` 可启动策略服务，但本入口没有实现跨主机自动调度与结果回收。

## 1. 安装与下载

在仓库根目录、已安装 EasyWAM 依赖的环境运行：

```bash
python scripts/robodojo_pipeline.py doctor --gpu
python scripts/robodojo_pipeline.py download --size-only
python scripts/robodojo_pipeline.py download
python scripts/robodojo_pipeline.py check-data
```

若服务器无法访问 Hugging Face，可指定公共兼容镜像：

```bash
python scripts/robodojo_pipeline.py download --size-only --endpoint https://hf-mirror.com
python scripts/robodojo_pipeline.py download --endpoint https://hf-mirror.com
```

脚本不向镜像发送 Hugging Face token。下载会解析并记录 commit、文件清单/大小，重试默认继续 manifest 中的 revision；磁盘不足时在下载权重前失败。`--size-only` 只读元数据；磁盘预留默认 30 GiB 只是下载保护，不代表足够训练。模型和数据保存在仓库的 `checkpoints/`、`data/`，可先将这两个目录链接到扩容后的持久盘。

`check-data` 验证 joint14、相机键、统计量及 parquet/视频文件存在；它不替代完整解码、动作数值和训练 loss 检查。请保留下载的同一份 `dataset_stats.json` 用于训练、推理与评分。

## 2. 初始化与训练

```bash
python scripts/robodojo_pipeline.py prepare
python scripts/robodojo_pipeline.py probe --run-id a800-probe --dry-run
python scripts/robodojo_pipeline.py probe --run-id a800-probe
```

`prepare` 从视频权重映射/缩放初始化 ActionDiT，然后预计算文本特征。ActionDiT 初始化不是经过机器人数据预训练的动作策略。已存在初始化文件不会覆盖；若更换基础权重，需使用相应的新初始化文件与配置。

probe 默认 20 个优化步并保存 checkpoint。检查 `runs/robodojo_easywam_mot_wan22_a800/a800-probe/train.log` 中 loss 有限、GPU/CPU 峰值、秒/步和保存耗时。通过后启动：

```bash
python scripts/robodojo_pipeline.py train --run-id a800-baseline
# 未来八卡：保持全局 batch=16 以便比较，不等于复现上游全局 batch=128 的预算
python scripts/robodojo_pipeline.py train --run-id h800x8-baseline --gpus 8 --launcher zero2
```

恢复训练用完整状态目录（位于 run 的 `checkpoints/state/`）；`checkpoints/weights/*.pt` 用于推理，传给 resume 只加载权重而不恢复优化器进度：

```bash
python scripts/robodojo_pipeline.py train --run-id a800-baseline --resume /absolute/path/to/checkpoints/state/step_000020
```

上面状态目录名为占位示例，以实际保存目录为准。恢复时指定与原训练一致的超参数；修改总步数可能影响学习率调度。默认保留 2 组 checkpoint，但保存新组期间仍有临时峰值占用。微调学习率等参数可在命令末尾传 Hydra override；本入口拒绝切换模型或启用 LoRA。

## 3. 推理与仿真评测

在具备仿真条件的机器准备固定版本：

```bash
python scripts/robodojo_pipeline.py setup-sim --install
python scripts/robodojo_pipeline.py plan --output evaluate_results/robodojo/plan
```

`plan` 只生成 162 个任务/种子组合，不加载权重、不启动仿真。完整协议为 42 个任务，3 个种子；其中 12 个泛化任务各拆为标准/随机两组、各 25 局，其他任务各 50 局，总计 6,300 局。

先做单任务单局检查，再跑完整协议：

```bash
python scripts/robodojo_pipeline.py smoke-eval \
  --checkpoint /absolute/path/to/checkpoints/weights/step_000020.pt \
  --stats /absolute/path/to/dataset_stats.json \
  --output evaluate_results/robodojo/smoke --policy-gpu 0 --sim-gpu 1

python scripts/robodojo_pipeline.py evaluate \
  --checkpoint /absolute/path/to/checkpoints/weights/step_030000.pt \
  --stats /absolute/path/to/dataset_stats.json \
  --output evaluate_results/robodojo/full --policy-gpu 0 --sim-gpu 1

python scripts/robodojo_pipeline.py summarize --output evaluate_results/robodojo/full
```

独立启动策略服务：

```bash
python scripts/robodojo_pipeline.py serve \
  --checkpoint /absolute/path/to/model.pt --stats /absolute/path/to/dataset_stats.json \
  --output evaluate_results/robodojo/policy-server --host 127.0.0.1 --port 23000
```

服务仍需准备 RoboDojo 的 XPolicyLab 子模块；远端仿真客户端的连接与网络隧道需按其部署环境配置。不要把启动服务等同于已通过一次仿真闭环。

## 4. 结果与复现

当前先完善本地代码；完整模型训练和 Isaac Sim 闭环验证留待服务器条件就绪。仅需 Hydra/OmegaConf/PyYAML 的本地回归测试：

```bash
python -m unittest discover -s experiments/robodojo/tests -v
python scripts/robodojo_pipeline.py prepare --dry-run
python scripts/robodojo_pipeline.py probe --run-id preview --dry-run
```

测试覆盖：无 LoRA 配方、解析后配置传递、真实结果文件格式的汇总、缺项抑制、跨种子标准差、下载空间/完整性保护、checkpoint 身份隔离、互斥续评。它们不运行真实模型或仿真。训练会保存 `config.yaml`；本入口使用固定配方，若自行修改相机布局、归一化或模型结构，须将推理配置同步到训练时配置，不能只替换 checkpoint。

评测输出记录权重 SHA256、统计量 SHA256、代码摘要、版本、解析后的推理配置、任务列表和逐种子指标。更换权重/统计量/推理设置应使用新输出目录；同一输出目录的并行 evaluator 会被拒绝。可使用相同命令续评，已完成任务按身份标识匹配。

泛化的标准与随机场景合并成任务指标，再按任务、维度、种子计算；跨种子标准差采用 ddof=0。缺失任务不会得到完整总分；smoke 的局部指标不能用于榜单对比。`full_protocol_complete` 仅表示当前汇总覆盖完整协议，`officially_verified` 始终为 false，官方认可需另走官方提交审核流程。RoboDojo 自带 `_summary.md` 可能展示该 checkout 下多个历史实验，当前实验以本次输出中的 `summary.json` 和原始结果为准。

对外发布至少包含：训练/数据版本、配置和训练预算、checkpoint 与统计量、原始评测结果和视频、每维度/每种子指标、复现命令、上游许可证与来源。单纯改名或包装框架不是新方法贡献。

## 为什么 VPP2 分数不同，历史帧是否是关键？

目前不能把领先归因于历史帧。公开仓库有历史条件，但没有找到只开/关历史、其余预算完全一致的 RoboDojo 消融。此前优先强调历史过早，本方案先不加入它。

可核对的差异同时包括：VPP2 当前 RoboDojo 配方使用 Wan2.1 14B + Action2B，EasyWAM 此方案使用 Wan2.2 5B + 较小动作专家；VPP2 论文还包含机器人/人类操作视频、事件切分、细粒度指令及持续视频预训练，并包含后训练与蒸馏阶段。当前代码另有历史采样、关键阶段混采、任务采样、训练预算和推理实现差异。相同 MoT 大类只是 token 交互结构相似，不能保证学习到相同的世界模型或控制能力。

VPP2 论文中视频指令遵循消融支持其数据/预训练方案有效，但不是 RoboDojo 历史帧的因果消融。论文、README 官榜和当前公开 checkpoint 的分数对应不同时间/版本，也不应混成一个对比数字。要判断你自己的瓶颈，先获得 Wan5B 的可复现基线，再依次测试任务采样/关键阶段采样、训练预算、控制重规划参数；只有 memory 维度与回放确实暴露时序歧义时，再做有/无历史的等预算消融。

来源：[VPP2 论文 v1](https://arxiv.org/html/2610.10270v1)、[VPP2 官方仓库](https://github.com/roboterax/video-prediction-policy-2)、[RoboDojo 官方仓库](https://github.com/RoboDojo-Benchmark/RoboDojo)、[数据集](https://huggingface.co/datasets/OpenMOSS-Team/robodojo-lerobot-v3.0)、[Wan5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)。VPP2 代码审查快照为 `a103a1e1784d3e45014549a82238894d3ed85959`。
