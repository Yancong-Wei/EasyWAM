"""Content-based identity and exclusive ownership of evaluation outputs."""
from __future__ import annotations

import hashlib
import json
import fcntl
from contextlib import contextmanager
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_identity(checkpoint: Path, stats: Path, settings: dict) -> dict:
    payload = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "stats_sha256": sha256_file(stats),
        "settings": settings,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, "tag": f"easywam_{digest[:24]}"}


def bind_output(output: Path, identity: dict) -> None:
    path = output / "evaluation_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Evaluation identity changed; use a new output directory.")
    else:
        if (output / "summary.json").exists() or (output / "manager_config.yaml").exists():
            raise ValueError("Legacy output has no identity; use a new output directory.")
        path.write_text(json.dumps(identity, indent=2) + "\n")


@contextmanager
def evaluation_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".evaluation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another evaluator owns {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for folder in ("src", "experiments", "configs"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml"):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()
