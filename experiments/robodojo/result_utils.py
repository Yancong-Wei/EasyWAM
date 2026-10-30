"""Result validation for the published RoboDojo simulation protocol."""

from __future__ import annotations

import json
from pathlib import Path


def expected_episodes(
    task_name: str, override: int | None = None, dimension: str | None = None
) -> int:
    native = 25 if dimension == "generalization" or task_name.endswith("_random") else 50
    if override is not None:
        if override <= 0:
            raise ValueError("eval_num_episodes must be positive.")
        return min(override, native)
    return native


def result_parent(root: Path, task_name: str, env_cfg_type: str, seed: int, tag: str) -> Path:
    return (
        root / "eval_result" / "RoboDojo" / task_name / "easywam_policy"
        / env_cfg_type / f"{seed}_ckpt_name={tag},action_type=joint"
    )


def valid_result(path: Path, expected: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        details = payload["details"]
        if not isinstance(details, dict) or len(details) < expected:
            return False
        if int(payload["eval_time"]) < expected:
            return False
        entries = sorted((int(key), item) for key, item in details.items())
        for _, item in entries[:expected]:
            if not isinstance(item, dict) or not isinstance(item.get("success"), bool):
                return False
            score = float(item["score"])
            if not 0.0 <= score <= 1.0:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def latest_valid_result(parent: Path, expected: int) -> Path | None:
    if not parent.is_dir():
        return None
    candidates = sorted(
        (directory / "_result.json" for directory in parent.iterdir() if (directory / "_result.json").is_file()),
        reverse=True,
    )
    return candidates[0] if candidates and valid_result(candidates[0], expected) else None


def summarize_jobs(jobs: list[dict], root: Path, env_cfg_type: str, tag: str) -> list[dict]:
    rows = []
    for job in jobs:
        task, seed, count = job["task_name"], job["seed"], job["episodes"]
        result = latest_valid_result(result_parent(root, task, env_cfg_type, seed, tag), count)
        if result is None:
            rows.append({**job, "complete": False, "result_path": None})
            continue
        payload = json.loads(result.read_text(encoding="utf-8"))
        details = [item for _, item in sorted(
            (int(key), item) for key, item in payload["details"].items()
        )[:count]]
        rows.append({
            **job,
            "complete": True,
            "success_rate": 100.0 * sum(bool(item["success"]) for item in details) / count,
            "score": 100.0 * sum(float(item["score"]) for item in details) / count,
            "result_path": str(result),
        })
    return rows


def aggregate_rows(rows: list[dict]) -> dict:
    """Dimension-balanced metrics, with no headline score for incomplete runs.

    Generalization standard/random variants form one task cell. Report
    population standard deviation (ddof=0) across evaluation seeds.
    """
    import statistics

    grouped = {}
    for row in rows:
        key = (row["task_name"].removesuffix("_random"), row["seed"])
        phase = "random" if row["task_name"].endswith("_random") else "standard"
        phases = grouped.setdefault(key, {})
        if phase in phases:
            raise ValueError(f"Duplicate result cell: {key}/{phase}")
        phases[phase] = row
    cells = []
    for (task, seed), phases in sorted(grouped.items()):
        first = next(iter(phases.values()))
        required = ("standard", "random") if first["dimension"] == "generalization" else ("standard",)
        complete = all(phases.get(phase, {}).get("complete", False) for phase in required)
        cell = dict(task_name=task, seed=seed, dimension=first["dimension"], complete=complete)
        if complete:
            entries = [phases[phase] for phase in required]
            count = sum(item["episodes"] for item in entries)
            for metric in ("success_rate", "score"):
                cell[metric] = sum(item[metric] * item["episodes"] for item in entries) / count
        cells.append(cell)
    complete = bool(cells) and all(cell["complete"] for cell in cells)
    seeds = sorted({row["seed"] for row in rows})
    dimensions = sorted({row["dimension"] for row in rows})
    per_seed, dimension_scores = {}, {}
    for dimension in dimensions:
        selected = [cell for cell in cells if cell["dimension"] == dimension]
        ready = bool(selected) and all(cell["complete"] for cell in selected)
        dimension_scores[dimension] = {"complete": ready, "completed_cells": sum(cell["complete"] for cell in selected)}
        for metric in ("success_rate", "score"):
            values = [statistics.mean(cell[metric] for cell in selected if cell["seed"] == seed)
                      for seed in seeds if any(cell["seed"] == seed for cell in selected)] if ready else []
            dimension_scores[dimension][metric] = statistics.mean(values) if values else None
    if complete:
        for seed in seeds:
            selected = [cell for cell in cells if cell["seed"] == seed]
            seed_dims = {cell["dimension"] for cell in selected}
            if seed_dims != set(dimensions):
                complete = False
                break
            per_seed[str(seed)] = {
                metric: statistics.mean(statistics.mean(cell[metric] for cell in selected if cell["dimension"] == dim)
                                        for dim in dimensions)
                for metric in ("success_rate", "score")
            }
    full_protocol = complete and seeds == [0, 1, 2] and len(dimensions) == 5
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        full_protocol = full_protocol and len(seed_rows) == 54 and sum(row["episodes"] for row in seed_rows) == 2100
        full_protocol = full_protocol and len([cell for cell in cells if cell["seed"] == seed]) == 42
        full_protocol = full_protocol and all(row["episodes"] == expected_episodes(row["task_name"], dimension=row["dimension"]) for row in seed_rows)
    overall = {"complete": complete, "full_protocol_complete": bool(full_protocol),
               "officially_verified": False, "completed_cells": sum(cell["complete"] for cell in cells),
               "expected_cells": len(cells), "std_ddof": 0}
    for metric in ("success_rate", "score"):
        values = [value[metric] for value in per_seed.values()] if complete else []
        overall[metric] = statistics.mean(values) if values else None
        overall[metric + "_std"] = statistics.pstdev(values) if values else None
    return {"jobs": rows, "per_task_seed": cells, "dimensions": dimension_scores,
            "per_seed": per_seed if complete else {}, "overall": overall}
