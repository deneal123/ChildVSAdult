"""Explicit, fail-closed sequential fixed8 CUDA matrix; no retries or polling."""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record
from scripts.evaluate_strong_cuda_fgnet_v1 import readiness
from scripts.run_oriented_campaign import paths_from, verified


def key(cell):
    return (cell["negative"], cell["scope"], cell["lr"], cell["seed"])


def validate_plan(plan):
    canonical = set(itertools.product(("random", "lookalike"), ("head", "tail", "full"),
                                      (1e-6, 1e-5), (42, 1, 2)))
    pending = plan["cells"]
    completed = plan["completed"]
    cells = [*pending, *(r["cell"] for r in completed)]
    for cell in cells:
        if (set(cell) != {"negative", "scope", "lr", "seed"}
                or type(cell["seed"]) is not int or type(cell["lr"]) not in (int, float)):
            raise ValueError("explicit canonical cell fields required")
    keys = [key(c) for c in cells]
    if len(keys) != 36 or len(set(keys)) != 36 or set(keys) != canonical:
        raise ValueError("pending plus completed must cover exactly the36-cell matrix")
    return pending, completed


def verify_cell(cell, training, external):
    _, natives = readiness([Path(training)])
    if len(natives) != 1:
        raise ValueError("exactly one completed native cell required")
    native = natives[0]
    p = native["parameters"]
    if (p["negative"], p["trainable_scope"], p["learning_rate"], p["seed"]) != key(cell):
        raise ValueError("native training identity differs from explicit cell")
    result = verified(Path(external), "adaface-fixed8-cuda-full-fgnet-roc-v2")
    if result["metrics"].get("execution_complete") is not True:
        raise ValueError("completed external evaluation required")
    if file_record(Path(training)) not in result["inputs"]:
        raise ValueError("external evaluation does not bind this native training")
    return native, result


def verify_snapshot(snapshot):
    if any(file_record(path) != record for path, record in snapshot.items()):
        raise RuntimeError("shared scientific inputs changed during queue")


def extend_snapshot(snapshot, natives):
    """Lock newly encountered verified ancestry as well as the initial exclusions."""
    for native in natives:
        for path in paths_from(native):
            record = file_record(path)
            if path in snapshot and snapshot[path] != record:
                raise RuntimeError("previously bound scientific input changed")
            snapshot.setdefault(path, record)


def run(plan_path, out, *, execute=False):
    if out.exists():
        raise FileExistsError("fresh queue root required; never resume partial outputs")
    original = file_record(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pending, completed = validate_plan(plan)
    if not execute:
        return dict(pending=len(pending), completed_claimed=len(completed),
                    training_launched=False, completion_verified=False)
    # No bypass for incomplete cells, and no acceptance based on directory presence.
    locked = {plan_path, Path(__file__)}
    locked.update(PROJECT_ROOT / p for p in ("models/adaface_ir101.pt",
                  "data/processed/pairs.jsonl", "data/processed/experiments/pairs_lookalike.jsonl",
                  "data/external/fgnet_crops.npz"))
    for item in completed:
        natives = verify_cell(item["cell"], Path(item["training_manifest"]), Path(item["external_manifest"]))
        for native in natives:
            locked.update(paths_from(native))
    snapshot = {path: file_record(path) for path in sorted(locked)}
    if original != file_record(plan_path):
        raise RuntimeError("plan changed during preflight")
    out.mkdir(parents=True)
    ledger = dict(status="running", completed=[], failed=None, plan_record=original)
    target = out / "queue.json"

    def flush():
        target.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    flush()
    for cell in pending:
        name = f"{cell['negative']}_{cell['scope']}_lr{cell['lr']:.0e}_s{cell['seed']}"
        train, score = out / name / "train", out / name / "fgnet"
        try:
            verify_snapshot(snapshot)
            if original != file_record(plan_path) or train.parent.exists():
                raise RuntimeError("plan changed or cell output already exists")
            subprocess.run([sys.executable, "-m", "scripts.run_strong_backbone_cuda_cell_v1",
                            "--out", str(train), "--negative", cell["negative"],
                            "--scope", cell["scope"], "--lr", str(cell["lr"]),
                            "--seed", str(cell["seed"]), "--execute"], cwd=PROJECT_ROOT, check=True)
            # Readiness before any external scoring: failed training is never scored.
            _, natives = readiness([train / "summary.manifest.json"])
            p = natives[0]["parameters"]
            if (p["negative"], p["trainable_scope"], p["learning_rate"], p["seed"]) != key(cell):
                raise ValueError("actual cell differs before external scoring")
            subprocess.run([sys.executable, "-m", "scripts.evaluate_strong_cuda_fgnet_v1",
                            "--bindings", str(train / "summary.manifest.json"),
                            "--out", str(score), "--execute"], cwd=PROJECT_ROOT, check=True)
            native_results = verify_cell(cell, train / "summary.manifest.json", score / "summary.manifest.json")
            extend_snapshot(snapshot, native_results)
            verify_snapshot(snapshot)
            if original != file_record(plan_path):
                raise RuntimeError("plan changed during cell execution")
        except Exception as error:
            ledger.update(status="failed", failed=dict(cell=name, error=f"{type(error).__name__}: {error}"))
            flush()
            raise
        ledger["completed"].append(dict(cell=cell, training_manifest=str(train / "summary.manifest.json"),
                                        external_manifest=str(score / "summary.manifest.json")))
        flush()
    ledger["status"] = "complete"
    flush()
    # Ledger is orchestration status only, never a publication/scientific aggregate manifest.
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.out, execute=args.execute), indent=2))
