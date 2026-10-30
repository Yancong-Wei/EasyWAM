# RoboDojo develop 核心范围

[项目入口](../README_zh.md) | [使用流程](instructions/benchmark/robodojo_pipeline_zh.md)

## 主线

固定 Wan2.2-5B / MoT 全参数方案，以 RoboDojo 下载、数据校验、初始化、训练、推理、仿真与评分为开发主线。保留 `configs/data/robodojo.yaml`、`configs/benchmark/sim_robodojo.yaml`、RoboDojo task 配方及 `experiments/robodojo/`。

共享的 LeRobot 数据层、归一化、视频处理、训练器、分布式 launchers、批量推理、prompt 缓存、任务分发和模型组件均保留。Wan/Cosmos/FLUX 模型配置与 backend 是模型库接口；当前统一流程只支持 Wan5B/MoT，其他组合的配置可解析不代表 GPU 或仿真验证通过。

保留 RoboDojo 的 Hidden、Unified、MoT-Joint、MoT-IDM 配方用于后续架构对照。模型库中的 LoRA 能力未从共享代码中强拆，但默认全流程不启用 LoRA，旧 XiconWAM LoRA 草案已退出活动配置。

## 移除与归档

- 移除其他 benchmark 的专属评测代码、数据配置、任务配方、仿真配置、教程、历史结果与相关博客。
- 移除不再使用的展示图片与性能图，不把上游其他 benchmark 成绩当作本项目成果。
- 删除核心源码未引用的直接依赖：`h5py`、`matplotlib`、`modelscope`、`opencv-python`。没有重新构建完整 GPU 环境，依赖安装兼容性仍需后续验证。
- 本地 `research/` 保存调研快照及 `local-archive/` 下的旧部署/配置资料，已加入忽略规则，不进入开发核心。未删除原始调研与部署文件。
- 保留 MIT LICENSE 和上游 EasyWAM 作者归属。

`concat_multi_camera: robotwin` 是现有三相机拼接模式的兼容名称，仍被 RoboDojo 使用；它不依赖已移除的评测代码，因此保留名称和计算行为。

## 分支与交付边界

当前交付直接落在原有 `develop` 分支。通过合并纳入此前已经测试的 RoboDojo 核心变更，保留 `develop` 的既有提交历史，以及其 epoch 训练进度/训练预算逻辑、单卡 launcher 与模型发布辅助脚本。合并中的训练器冲突同时保留 epoch 进度和每步进度条更新。

临时的 `codex/robodojo-develop-core` 分支在合并完成后删除，不作为后续开发入口。未推送远端。旧分支独有的其他 benchmark 配方和机器专属 `data` / `runs` 符号链接不进入核心；删除链接不会删除其指向的数据。训练数据和输出路径应在每个部署环境自行配置。

ModelScope 发布脚本作为可选工具保留，使用前安装 `publish` extra；日常训练无需安装它。历史删除内容仍可从 Git 提交恢复。

## 验证

```bash
python -m unittest discover -s experiments/robodojo/tests -v
python -m compileall -q src scripts experiments/robodojo
git diff --check
```

范围检查覆盖活动 benchmark、共享模块导入、模型/task 配置解析与本地文档链接；流程测试覆盖配置传递、评分、续评和下载保护。这些都不替代真实 GPU 训练、保存恢复及 Isaac Sim 闭环测试。服务器不在本轮清理范围内。
