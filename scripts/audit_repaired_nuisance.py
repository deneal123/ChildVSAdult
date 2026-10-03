"""Fresh nuisance refit and paired-objective audit for exact-supported candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_coupled_weight_budget import audit, blocks
from scripts.audit_nuisance_overlap import PROTOCOL, diagnose


def refit(arms, groups, quality, *, n_folds=5):
    """No old weight argument: all weights are generated from the current arms."""
    selected = {
        r[f"face_{s}"]
        for rows in arms.values()
        for r in rows
        if r["split"] == "train"
        for s in ("a", "b")
    }
    if not selected <= set(quality):
        raise ValueError("all selected images require unique quality records")
    for rows in arms.values():
        pairs = blocks([r for r in rows if r["split"] == "train"], groups)
        if any((p["age_a"], p["age_b"]) != (n["age_a"], n["age_b"]) for p, n in pairs):
            raise ValueError("exact endpoint ages required before nuisance refit")
    nuisance, weights = diagnose(arms, groups, quality, n_folds=n_folds)
    coupled, ledger = audit(arms, groups, weights)
    for reports in coupled["views"].values():
        for row in reports.values():
            if any(
                v["empirical_weighted_total_variation"] != 0
                for v in row["coupled_candidate_age_balance"].values()
            ):
                raise ValueError("paired exact-age weighted invariant violated")
    return (
        {
            "nuisance": nuisance,
            "coupled": coupled,
            "execution_complete": True,
            "all_coupled_age_views_exact": True,
            "training_weights_ready": False,
            "publication_ready": False,
            "recognition_outcomes_used": False,
            "scope": "fresh candidate-arm positive nuisance fit and full paired loss ledger; not crop readiness or joint quality balance",
            "remaining": [
                "positive/negative joint nuisance distribution assessment",
                "source/crop-bound preflight",
                "BN and actual presentation policy",
                "independent human identity/benchmark overlap audit",
                "fresh three-seed training",
            ],
        },
        weights,
        ledger,
    )


def verified_prerequisite(path, root):
    native = json.loads(path.read_text(encoding="utf-8"))
    if (
        native["experiment"] != "exact-supported-singleton-profile-repair-v2-row-regime"
        or native["metrics"].get("complete") is not True
    ):
        raise ValueError("completed exact-supported preparation required")
    records = native["inputs"] + native["outputs"]
    paths = [
        Path(r["path"]) if Path(r["path"]).is_absolute() else root / r["path"] for r in records
    ]
    if [file_record(p) for p in paths] != records:
        raise ValueError("prerequisite input/output checksums changed")
    required = [path.parent / f"private/{a}_candidate_arm.jsonl" for a in ("low", "cross")]
    required.extend(
        [
            root / "data/processed/person_clusters.jsonl",
            root / "data/interim/face_quality_audit.jsonl",
        ]
    )
    if any(file_record(p) not in records for p in required):
        raise ValueError("consumed candidate/source not bound by prerequisite")
    summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
    if summary != native["metrics"]:
        raise ValueError("prerequisite JSON/native metrics mismatch")
    return paths


def selected_quality(rows, selected):
    quality = {}
    for row in rows:
        face = row["face_id"]
        if face in selected:
            if face in quality:
                raise ValueError("duplicate selected quality record")
            quality[face] = row
    if set(quality) != selected:
        raise ValueError("missing selected quality record")
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("fresh output directory required")
    root = PROJECT_ROOT
    prerequisite = root / "metrics/exact_supported_repair_v3_20261003/summary.manifest.json"
    inherited = verified_prerequisite(prerequisite, root)
    dependencies = [
        root / f"scripts/{name}.py"
        for name in (
            "audit_nuisance_overlap",
            "audit_matched_arm_balance",
            "audit_coupled_weight_budget",
        )
    ]
    inputs = list(
        dict.fromkeys(
            p.resolve()
            for p in [
                *inherited,
                prerequisite,
                Path(__file__),
                *dependencies,
                root / "uv.lock",
                root / "pyproject.toml",
            ]
        )
    )
    before = [file_record(p) for p in inputs]
    groups = {}
    for row in read_jsonl(root / "data/processed/person_clusters.jsonl"):
        key, person = row["identity_group_id"], row["person_id"]
        if not isinstance(person, str) or not person or (key in groups and groups[key] != person):
            raise ValueError("invalid/conflicting recorded-person mapping")
        groups[key] = person
    arms = {
        a: list(read_jsonl(prerequisite.parent / f"private/{a.lower()}_candidate_arm.jsonl"))
        for a in ("LOW", "CROSS")
    }
    selected = {
        r[f"face_{s}"]
        for rows in arms.values()
        for r in rows
        if r["split"] == "train"
        for s in ("a", "b")
    }
    quality = selected_quality(read_jsonl(root / "data/interim/face_quality_audit.jsonl"), selected)
    result, weights, ledger = refit(arms, groups, quality)
    if before != [file_record(p) for p in inputs]:
        raise ValueError("inputs changed during nuisance refit")
    (out / "private").mkdir(parents=True)
    output = out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    weight_path, ledger_path = (
        out / "private/diagnostic_weights.jsonl",
        out / "private/candidate_pair_weights.jsonl",
    )
    write_jsonl(weight_path, weights)
    write_jsonl(ledger_path, ledger)
    target = out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="exact-supported-fresh-nuisance-coupled-audit",
        parameters={"nuisance_protocol": PROTOCOL, "training_executed": False},
        metrics=result,
        inputs=inputs,
        outputs=[output, weight_path, ledger_path],
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during manifest write; marker withdrawn")
    print(
        json.dumps(
            {
                "all_coupled_age_views_exact": True,
                "views": {
                    v: {
                        "oof_arm_auc": row["oof_arm_auc"],
                        "pair_ess": {a: x["pair_effective_size"] for a, x in row["arms"].items()},
                    }
                    for v, row in result["nuisance"]["views"].items()
                },
                "training_weights_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
