"""Compare old/new candidate quality in one frozen positive-image feature frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_coupled_weight_budget import blocks
from scripts.audit_joint_pair_quality import joint_mmd
from scripts.audit_nuisance_overlap import VIEWS
from scripts.audit_repaired_nuisance import selected_quality
from scripts.match_exact_negative_quality import positive_image_map


def compare(versions, groups, features, ledgers):
    report = {}
    for version, arms in versions.items():
        maps = {v: {} for v in VIEWS}
        for row in ledgers[version]:
            key = row["pair_id"]
            if key in maps[row["view"]]:
                raise ValueError("duplicate ledger weight")
            maps[row["view"]][key] = row["candidate_loss_weight"]
        matrices, weights = {}, {}
        for arm, rows in arms.items():
            paired = blocks([r for r in rows if r["split"] == "train"], groups)
            ordered = [r for pair in paired for r in pair]
            matrices[arm] = np.asarray(
                [
                    np.concatenate(
                        (
                            features[r["face_a"]][:9],
                            features[r["face_b"]][:9],
                            features[r["face_a"]][9:],
                            features[r["face_b"]][9:],
                        )
                    )
                    for r in ordered
                ]
            )
            weights[arm] = np.asarray(
                [[1.0] * len(ordered)] + [[maps[v][r["pair_id"]] for r in ordered] for v in VIEWS]
            )
        comparisons = {}
        for name, a, sa, b, sb in [
            ("LOW_class", "LOW", slice(0, None, 2), "LOW", slice(1, None, 2)),
            ("CROSS_class", "CROSS", slice(0, None, 2), "CROSS", slice(1, None, 2)),
            ("cross_arm_combined", "LOW", slice(None), "CROSS", slice(None)),
        ]:
            mmd, _ = joint_mmd(
                matrices[a][sa],
                matrices[b][sb],
                weights[a][:, sa],
                weights[b][:, sb],
                sigmas=[3.0, 6.0, 12.0],
            )
            comparisons[name] = {
                view: values.tolist()
                for view, values in zip(["uniform", *VIEWS], mmd.T, strict=True)
            }
        report[version] = comparisons
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("fresh destination required")
    root = PROJECT_ROOT
    prerequisites = [
        root / "metrics/repaired_nuisance_20261003/summary.manifest.json",
        root / "metrics/quality_matched_exact_20261003/summary.manifest.json",
    ]
    inherited = []
    for prerequisite in prerequisites:
        native = json.loads(prerequisite.read_text(encoding="utf-8"))
        if native["metrics"].get("execution_complete") is not True:
            raise ValueError("completed prerequisite required")
        records = native["inputs"] + native["outputs"]
        paths = [
            Path(r["path"]) if Path(r["path"]).is_absolute() else root / r["path"] for r in records
        ]
        if [file_record(p) for p in paths] != records:
            raise ValueError("prerequisite records changed")
        inherited.extend([*paths, prerequisite])
    dependencies = [
        Path(__file__),
        *[
            root / f"scripts/{n}.py"
            for n in (
                "match_exact_negative_quality",
                "audit_coupled_weight_budget",
                "audit_joint_pair_quality",
                "audit_nuisance_overlap",
                "audit_repaired_nuisance",
                "audit_matched_arm_balance",
            )
        ],
    ]
    inputs = list(dict.fromkeys(p.resolve() for p in [*inherited, *dependencies]))
    before = [file_record(p) for p in inputs]
    groups = {}
    for row in read_jsonl(root / "data/processed/person_clusters.jsonl"):
        key, person = row["identity_group_id"], row["person_id"]
        if key in groups and groups[key] != person:
            raise ValueError("conflicting mapping")
        groups[key] = person
    directories = {
        "previous": root / "metrics/exact_supported_repair_v3_20261003/private",
        "quality_cost": prerequisites[1].parent / "private",
    }
    versions = {
        v: {a: list(read_jsonl(d / f"{a.lower()}_candidate_arm.jsonl")) for a in ("LOW", "CROSS")}
        for v, d in directories.items()
    }

    def positive(rows):
        return [r for r in rows if r["split"] == "train" and r["label"] == 1]

    for arm in ("LOW", "CROSS"):
        if positive(versions["previous"][arm]) != positive(versions["quality_cost"][arm]):
            raise ValueError("positive pools differ")
    selected = {
        r[f"face_{s}"]
        for rows in versions["previous"].values()
        for r in rows
        if r["split"] == "train"
        for s in ("a", "b")
    }
    quality = selected_quality(read_jsonl(root / "data/interim/face_quality_audit.jsonl"), selected)
    ledgers = {
        v: list(read_jsonl(p.parent / "private/candidate_pair_weights.jsonl"))
        for v, p in zip(("previous", "quality_cost"), prerequisites, strict=True)
    }
    with threadpool_limits(limits=1):
        features, fitted = positive_image_map(versions["previous"], groups, quality)
        result = {
            "comparisons": compare(versions, groups, features, ledgers),
            "standardization": fitted,
            "sigmas": [3.0, 6.0, 12.0],
            "execution_complete": True,
            "training_ready": False,
            "scope": "same positive-image map for both versions and every view; descriptive empirical MMD only",
        }
    if before != [file_record(p) for p in inputs]:
        raise ValueError("inputs changed")
    out.mkdir(parents=True)
    output = out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="same-frame-negative-quality-comparison",
        parameters={"training_executed": False},
        metrics=result,
        inputs=inputs,
        outputs=[output],
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during manifest write")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
