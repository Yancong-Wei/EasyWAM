# RoboDojo configuration

[Documentation](../../README.md) | [Models](models.md) | [Training](training.md) | [Efficiency](efficiency.md)

`configs/train.yaml` composes `data=robodojo` and `model=easywam_mot_wan22` through `task=robodojo_easywam_mot_wan22_a800`. The A800 recipe inherits the full-parameter MoT task and changes execution/batch settings.

```bash
python scripts/train.py --cfg job task=robodojo_easywam_mot_wan22_a800
python experiments/robodojo/run_robodojo_manager.py --cfg job task=robodojo_easywam_mot_wan22_a800
```

`configs/benchmark/sim_robodojo.yaml` enables runtime text encoding and loads trained policy weights. The manager forwards its resolved config to policy workers. Keep action dimensions, camera layout, state placement and normalization aligned with the saved training config.

Other retained RoboDojo task recipes are architecture ablations, not scored baselines. Model-group overrides are for research; the unified workflow intentionally fixes the Wan5B/MoT recipe.
