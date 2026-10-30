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

本次工作分支为 `codex/robodojo-develop-core`，从包含此前 RoboDojo 流程修改的当前 `main` 工作区建立。检查时 `main` 与已有 `develop` 已分叉，分别有 31 和 4 个独有提交。本次未重置、覆盖或强制移动 `develop`。

这份核心变更在独立的本地工作分支提交供审阅，尚未合并到 `develop`，也未推送远端。后续整合 `develop` 时需逐项处理其数据层、训练器和模型实现差异，不能直接把整个旧分支覆盖回来，否则会恢复已删除的 benchmark。Git 中已跟踪的删除内容可从原基线 `a778b6f103f37065c042b36895234dc4ac04736b` 恢复。

## 验证

```bash
python -m unittest discover -s experiments/robodojo/tests -v
python -m compileall -q src scripts experiments/robodojo
git diff --check
```

范围检查覆盖活动 benchmark、共享模块导入、模型/task 配置解析与本地文档链接；流程测试覆盖配置传递、评分、续评和下载保护。这些都不替代真实 GPU 训练、保存恢复及 Isaac Sim 闭环测试。服务器不在本轮清理范围内。
