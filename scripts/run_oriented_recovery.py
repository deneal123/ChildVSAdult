"""Detached recovery supervisor: fresh CPU smoke, then fresh six-cell campaign.

Never resumes orphan checkpoints. Status/logs survive loss of the interactive
tool handle; native training manifests, not process status, prove completion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from scripts.run_oriented_campaign import verified


def paths(tag):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("safe nonempty recovery tag required")
    return {
        "logs": PROJECT_ROOT / ".work" / f"oriented_recovery_{tag}",
        "smoke": PROJECT_ROOT / "metrics" / f"oriented_runtime_smoke_recovery_{tag}",
        "metrics": PROJECT_ROOT / "metrics" / f"oriented_uniform_fixed10_bn_frozen_recovery_{tag}",
        "models": PROJECT_ROOT / "models" / f"oriented_uniform_fixed10_bn_frozen_recovery_{tag}",
    }


def command(stage, preflight, targets):
    if stage not in ("smoke", "train"):
        raise ValueError("smoke/train stage required")
    result = [
        sys.executable,
        "-u",
        "-m",
        "scripts.run_oriented_campaign",
        stage,
        "--preflight",
        str(preflight),
    ]
    if stage == "smoke":
        return result + ["--out", str(targets["smoke"])]
    return result + [
        "--smoke",
        str(targets["smoke"] / "summary.manifest.json"),
        "--models",
        str(targets["models"]),
        "--out",
        str(targets["metrics"]),
    ]


def require_stage_metrics(stage, metrics):
    if stage == "smoke":
        if metrics.get("smoke_complete") is not True:
            raise ValueError("successful process without completed smoke evidence")
    elif stage == "train":
        if metrics.get("training_complete") is not True or metrics.get("completed_cells") != 6:
            raise ValueError("successful process without six completed training cells")
    else:
        raise ValueError("unknown recovery stage")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    targets = paths(args.tag)
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("all recovery directories must be fresh")
    targets["logs"].mkdir(parents=True)
    state = dict(
        supervisor_pid=os.getpid(),
        tag=args.tag,
        stage="startup",
        status="running",
        targets={k: str(v) for k, v in targets.items()},
        child_pid=None,
        training_complete=False,
        evaluation_complete=False,
        publication_ready=False,
    )

    def record():
        state["updated_at_utc"] = datetime.now(UTC).isoformat()
        destination = targets["logs"] / "status.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    record()
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES="-1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    try:
        for stage in ("smoke", "train"):
            state["stage"] = stage
            state["command"] = command(stage, args.preflight.resolve(), targets)
            record()
            with (
                (targets["logs"] / f"{stage}.stdout.log").open("wb") as stdout,
                (targets["logs"] / f"{stage}.stderr.log").open("wb") as stderr,
            ):
                child = subprocess.Popen(
                    state["command"],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                )
                state["child_pid"] = child.pid
                record()
                code = child.wait()
            state["returncode"] = code
            state["child_pid"] = None
            if code != 0:
                raise RuntimeError(f"{stage} child exited with code{code}")
            artifact = (
                targets["smoke"] / "summary.manifest.json"
                if stage == "smoke"
                else targets["metrics"] / "training-bound.manifest.json"
            )
            native = verified(
                artifact,
                "oriented-cpu-runtime-smoke"
                if stage == "smoke"
                else "oriented-uniform-serial-training",
            )
            require_stage_metrics(stage, native["metrics"])
            state[stage + "_manifest"] = str(artifact)
            record()
        state.update(status="completed", training_complete=True)
        record()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        record()
        raise


if __name__ == "__main__":
    main()
