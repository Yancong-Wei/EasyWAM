#!/usr/bin/env python3
"""Wan2.2-5B/MoT RoboDojo workflow. Dry-run commands need only the stdlib."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASK = "robodojo_easywam_mot_wan22_a800"
DATA = ROOT / "playground/datasets/robodojo"
WAN = ROOT / "playground/pretrain_models/Wan2.2-TI2V-5B"
ACTION = ROOT / "playground/pretrain_models/ActionDiT_Wan22_5B_alphascale_1024hdim.pt"
SIM_REVISION = "08b7ee46034c3d0c8a389b2b7bfc138f4d55ee4d"


def run(command, *, dry_run=False, env=None, cwd=ROOT):
    print(shlex.join([str(value) for value in command]), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty: {path}")


def require_completed_download(root):
    manifest = root / "easywam_download.json"
    if manifest.exists() and not json.loads(manifest.read_text()).get("complete", False):
        raise RuntimeError(f"Download is incomplete at {root}; resume download before preparing/training")


def check_data(root=DATA):
    """Fail on the wrong action schema before any GPU allocation."""
    require_completed_download(root)
    require_file(root / "meta/info.json")
    require_file(root / "dataset_stats.json")
    info = json.loads((root / "meta/info.json").read_text())
    features = info.get("features", {})
    for key in ("action", "observation.state"):
        if features.get(key, {}).get("shape") != [14]:
            raise ValueError(f"{key} must be joint14; got {features.get(key)}")
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        key = "observation.images." + camera
        if key not in features:
            raise ValueError(f"Missing camera {key}")
    if not any((root / "data").rglob("*.parquet")) or not any((root / "videos").rglob("*.mp4")):
        raise ValueError("Dataset must contain parquet trajectories and MP4 camera videos")
    stats = json.loads((root / "dataset_stats.json").read_text())
    if not isinstance(stats, dict) or not stats:
        raise ValueError("Empty/malformed normalization statistics")
    print(f"joint14 dataset metadata OK: {info.get('total_episodes')} episodes, {info.get('total_frames')} frames")
    print("Metadata check only; full video decoding and numerical action checks require training-environment validation.")
    return info


def download(args):
    assets = [("dataset", "OpenMOSS-Team/robodojo-lerobot-v3.0", DATA),
              ("model", "Wan-AI/Wan2.2-TI2V-5B", WAN)]
    assets = [asset for asset in assets if args.asset == "all" or asset[0] == args.asset]
    if args.dry_run:
        for kind, repo, target in assets:
            print(f"Resolve {repo}@{args.revision} to commit; size-check and snapshot_download(repo_type={kind}, local_dir={target})")
        return
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(endpoint=args.endpoint, token=False)
    for kind, repo, target in assets:
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = target / "easywam_download.json"
        old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        revision = old["revision"] if old and args.revision == "main" else args.revision
        info = api.repo_info(repo, repo_type=kind, revision=revision, files_metadata=True, timeout=30)
        if old and old["revision"] != info.sha:
            raise ValueError(f"{target} belongs to another revision; choose a clean asset directory")
        files = [item for item in info.siblings if not item.rfilename.startswith(("assets/", "examples/"))]
        if any(item.size is None for item in files):
            raise ValueError("Hub did not return all file sizes; cannot safely budget download")
        needed = sum(item.size for item in files if not (target / item.rfilename).is_file()
                     or (target / item.rfilename).stat().st_size != item.size)
        free = shutil.disk_usage(target).free
        reserve = int(args.reserve_gib * 1024**3)
        print(f"{repo}@{info.sha}: missing {needed / 1024**3:.1f} GiB; free {free / 1024**3:.1f} GiB; reserve {args.reserve_gib} GiB")
        if args.size_only:
            continue
        if needed + reserve > free:
            raise RuntimeError("Insufficient disk for download plus reserve. Expand/mount storage; no weights downloaded.")
        manifest = {"repo_id": repo, "repo_type": kind, "revision": info.sha,
                    "files": [{"path": item.rfilename, "size": item.size} for item in files], "complete": False}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        snapshot_download(repo_id=repo, repo_type=kind, revision=info.sha, local_dir=str(target), endpoint=args.endpoint, token=False,
                          allow_patterns=[item.rfilename for item in files], max_workers=args.workers)
        for item in files:
            local = target / item.rfilename
            if not local.is_file() or local.stat().st_size != item.size:
                raise RuntimeError(f"Incomplete asset: {local}")
        manifest["complete"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if kind == "dataset":
            check_data(target)


def doctor(args):
    print(f"Project: {ROOT}\nPython: {sys.executable}")
    for name in ("torch", "hydra", "deepspeed", "accelerate", "huggingface_hub", "av", "pyarrow", "peft", "ftfy"):
        print(f"{name}: {'installed' if importlib.util.find_spec(name) else 'missing'}")
    for path in (ROOT, DATA, WAN):
        existing = path
        while not existing.exists():
            existing = existing.parent
        print(f"Disk {path}: {shutil.disk_usage(existing).free / 1024**3:.1f} GiB free")
    if args.gpu:
        run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv"])
        import torch
        print(f"torch={torch.__version__}, CUDA={torch.version.cuda}, available={torch.cuda.is_available()}")
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        limit = Path(name)
        if limit.is_file():
            value = limit.read_text().strip()
            print(f"Container RAM limit: {int(value) / 1024**3:.1f} GiB" if value.isdigit() else f"Container RAM limit: {value}")
            break
    print("Full fine-tuning needs CPU optimizer memory as well as GPU memory; host free -h may overstate container RAM.")
    print("Isaac Sim requires a compatible RT GPU; an A800 policy host alone is not a validated simulator host.")


def common_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
    return env


def eval_command(args):
    command = [sys.executable, "experiments/robodojo/run_robodojo_manager.py", f"task={TASK}",
               f"EVALUATION.robodojo_root={json.dumps(str(args.sim_root.resolve()))}",
               f"EVALUATION.output_dir={json.dumps(str(args.output.resolve()))}",
               f"EVALUATION.sim_env={args.sim_env}", "MULTIRUN.num_gpus=1",
               "MULTIRUN.policy_num_gpus=1", "MULTIRUN.env_num_gpus=1",
               f"MULTIRUN.policy_gpu_ids=[{args.policy_gpu}]", f"MULTIRUN.env_gpu_ids=[{args.sim_gpu}]",
               "MULTIRUN.inference_batch_size=1", f"EVALUATION.sim_num_envs={args.sim_num_envs}"]
    if args.command == "plan":
        command += ["MULTIRUN.create_only=true"]
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required")
        command += [f"ckpt={json.dumps(str(args.checkpoint.resolve()))}"]
        if args.stats:
            command += [f"EVALUATION.dataset_stats_path={json.dumps(str(args.stats.resolve()))}"]
    if args.command == "smoke-eval":
        command += [f"EVALUATION.task_name={args.task_name}", "EVALUATION.seeds=[0]", "EVALUATION.eval_num_episodes=1"]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--gpu", action="store_true")
    p = sub.add_parser("download")
    p.add_argument("--asset", choices=["all", "dataset", "model"], default="all")
    p.add_argument("--endpoint", help="Optional public HF-compatible endpoint; no token is sent")
    p.add_argument("--size-only", action="store_true")
    p.add_argument("--revision", default="main"); p.add_argument("--reserve-gib", type=float, default=30)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--dry-run", action="store_true")
    sub.add_parser("check-data")
    p = sub.add_parser("prepare"); p.add_argument("--dry-run", action="store_true")
    for name in ("probe", "train"):
        p = sub.add_parser(name)
        p.add_argument("--gpus", type=int, default=1); p.add_argument("--run-id", required=True)
        p.add_argument("--launcher", choices=["zero1", "zero2", "zero2_offload"], default="zero2_offload")
        p.add_argument("--steps", type=int); p.add_argument("--resume", type=Path)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("overrides", nargs="*")
    p = sub.add_parser("setup-sim"); p.add_argument("--sim-root", type=Path, default=ROOT / "third_party/RoboDojo")
    p.add_argument("--install", action="store_true"); p.add_argument("--dry-run", action="store_true")
    for name in ("plan", "smoke-eval", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", type=Path); p.add_argument("--stats", type=Path)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--sim-root", type=Path, default=ROOT / "third_party/RoboDojo")
        p.add_argument("--sim-env", default="RoboDojo"); p.add_argument("--policy-gpu", type=int, default=0)
        p.add_argument("--sim-gpu", type=int, default=1); p.add_argument("--sim-num-envs", type=int, default=1)
        p.add_argument("--task-name", default="cover_blocks"); p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("serve")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--sim-root", type=Path, default=ROOT / "third_party/RoboDojo")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=23000)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("summarize"); p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    env = common_env()
    if args.command == "doctor":
        doctor(args)
    elif args.command == "download":
        if args.reserve_gib < 0 or args.workers < 1:
            raise ValueError("reserve must be nonnegative; workers must be positive")
        download(args)
    elif args.command == "check-data":
        check_data()
    elif args.command == "prepare":
        if not args.dry_run:
            check_data()
            require_completed_download(WAN)
            for name in ("Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "diffusion_pytorch_model.safetensors.index.json"):
                require_file(WAN / name)
        if not ACTION.exists():
            run([sys.executable, "scripts/preprocess_action_dit_backbone.py", "--model-config", "configs/model/easywam_mot_wan22.yaml",
                 "--backbone", "wan22", "--output", str(ACTION), "--device", "cpu", "--dtype", "bfloat16"], dry_run=args.dry_run, env=env)
        else:
            require_file(ACTION)
            print(f"Keeping existing action initialization: {ACTION}")
        run([sys.executable, "scripts/precompute_text_embeds.py", f"task={TASK}"], dry_run=args.dry_run, env=env)
    elif args.command in ("probe", "train"):
        if args.gpus < 1 or not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
            raise ValueError("Positive GPU count and a single-component run ID are required")
        output = ROOT / "playground/runs" / TASK / args.run_id
        if output.exists() and not args.resume and not args.dry_run:
            raise FileExistsError(f"Use a new run ID or --resume: {output}")
        if not args.dry_run:
            check_data(); require_completed_download(WAN); require_file(ACTION)
        env.update(NPROC_PER_NODE=str(args.gpus), NNODES="1", NODE_RANK="0", RUN_ID=args.run_id)
        # Batch size and gradient accumulation come from the task config.
        overrides = [f"task={TASK}"]
        steps = args.steps if args.steps is not None else (20 if args.command == "probe" else None)
        if steps is not None:
            if steps <= 0:
                raise ValueError("steps must be positive")
            overrides += [f"max_steps={steps}"]
        if args.command == "probe":
            overrides += ["eval_every=0", f"save_every={steps}"]
        if args.resume:
            overrides += [f"resume={json.dumps(str(args.resume.resolve()))}"]
        for value in args.overrides:
            key = value.split("=", 1)[0].lstrip("+")
            if key in ("task", "model", "output_dir", "resume") or key.startswith("model.lora"):
                raise ValueError(f"Use the fixed full-parameter Wan/MoT recipe; forbidden override: {key}")
        print(f"NPROC_PER_NODE={args.gpus} NNODES=1 RUN_ID={args.run_id}")
        run(["bash", f"scripts/train_{args.launcher}.sh", *overrides, *args.overrides], dry_run=args.dry_run, env=env)
    elif args.command == "setup-sim":
        target = args.sim_root.resolve()
        if target.exists():
            if not args.dry_run:
                revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
                if revision != SIM_REVISION:
                    raise ValueError("Existing simulator revision differs; use a new --sim-root")
        else:
            run(["git", "clone", "https://github.com/RoboDojo-Benchmark/RoboDojo.git", str(target)], dry_run=args.dry_run)
            run(["git", "-C", str(target), "checkout", SIM_REVISION], dry_run=args.dry_run)
        run(["git", "-C", str(target), "submodule", "update", "--init", "--recursive"], dry_run=args.dry_run)
        if args.install:
            for script in ("install.sh", "init_assets.sh"):
                run(["bash", f"scripts/{script}"], cwd=target, dry_run=args.dry_run)
            run(["bash", "scripts/robodojo.sh", "doctor"], cwd=target, dry_run=args.dry_run)
    elif args.command in ("plan", "smoke-eval", "evaluate"):
        run(eval_command(args), dry_run=args.dry_run, env=env)
    elif args.command == "serve":
        if args.dry_run:
            print(f"Compose {TASK} inference config; serve {args.checkpoint} on {args.host}:{args.port}; stats={args.stats}")
            return
        sys.path.insert(0, str(ROOT))
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from experiments.robodojo.run_robodojo_manager import _model_overrides
        require_file(args.checkpoint); require_file(args.stats)
        if args.output.exists():
            raise FileExistsError("Use a new --output for the policy server configuration")
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            cfg = compose(config_name="benchmark/sim_robodojo", overrides=[f"task={TASK}",
                f"EVALUATION.output_dir={json.dumps(str(args.output.resolve()))}", "MULTIRUN.inference_batch_size=1"])
        args.output.mkdir(parents=True)
        resolved = args.output.resolve() / "manager_config.yaml"
        OmegaConf.save(cfg, resolved, resolve=True)
        run([sys.executable, "experiments/robodojo/policy_server.py", "--robodojo-root", str(args.sim_root.resolve()),
             "--config", "experiments/robodojo/easywam_policy/deploy_policy.yml", "--host", args.host, "--port", str(args.port),
             "--overrides", *_model_overrides(cfg, args.checkpoint.resolve(), args.stats.resolve(), resolved)], env=env)
    elif args.command == "summarize":
        sys.path.insert(0, str(ROOT))
        from experiments.robodojo.result_utils import aggregate_rows, summarize_jobs
        from experiments.robodojo.run_identity import evaluation_lock
        from experiments.robodojo.run_robodojo_manager import _resolve_path
        from omegaconf import OmegaConf
        with evaluation_lock(args.output):
            cfg = OmegaConf.load(args.output / "manager_config.yaml")
            identity = json.loads((args.output / "evaluation_identity.json").read_text())
            jobs = [json.loads(line) for line in (args.output / "tasks.jsonl").read_text().splitlines() if line.strip()]
            if jobs != identity["settings"]["jobs"]:
                raise ValueError("Task list no longer matches the recorded evaluation identity")
            summary = aggregate_rows(summarize_jobs(jobs, _resolve_path(str(cfg.EVALUATION.robodojo_root)), str(cfg.EVALUATION.env_cfg_type), identity["tag"]))
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
