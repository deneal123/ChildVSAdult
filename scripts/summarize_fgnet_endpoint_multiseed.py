"""Summarize existing three-seed endpoint-age-matched FG-NET re-evaluations.

This does not reconstruct the training provenance of legacy checkpoints. It
verifies the evaluation manifests and records only aggregate statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT, data_path
from age_gap.common.manifest import sha256_file, write_experiment_manifest

SEED_FILES = {
    42: "fgnet_endpoint_subject_stats.json",
    1: "fgnet_endpoint_subject_stats_s1.json",
    2: "fgnet_endpoint_subject_stats_s2.json",
}
STRATA = ("overall", "large_gap_25plus")


def _artifact_path(record: dict) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def summarize(seed_paths: dict[int, Path]) -> dict:
    """Require identical evaluation protocols and live, matching input manifests."""
    rows: dict[int, dict] = {}
    common: tuple | None = None
    for seed, path in sorted(seed_paths.items()):
        row = json.loads(path.read_text(encoding="utf-8"))
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment") != "fgnet-endpoint-age-matched-subject-statistics":
            raise ValueError(f"unexpected experiment in {manifest_path}")
        if not any(
            _artifact_path(record) == path and record.get("sha256") == sha256_file(path)
            for record in manifest.get("outputs", [])
        ):
            raise ValueError(f"evaluation output checksum mismatch: {path}")
        checkpoint_name = f"bb_facenet_seed{seed}.pt"
        if not any(
            _artifact_path(record).name == checkpoint_name
            and _artifact_path(record).is_file()
            and record.get("sha256") == sha256_file(_artifact_path(record))
            for record in manifest.get("inputs", [])
        ):
            raise ValueError(f"checkpoint checksum mismatch for seed {seed}")
        signature = (
            row["protocol"],
            row["endpoint_age_tolerance"],
            row["negative_seed"],
            row["n_bootstrap"],
            tuple(
                (
                    stratum,
                    row[stratum]["class_counts"]["positive"],
                    row[stratum]["class_counts"]["negative"],
                    row[stratum]["subject_bootstrap"]["frozen_auc"],
                )
                for stratum in STRATA
            ),
        )
        if common is not None and signature != common:
            raise ValueError("evaluation protocols or frozen baselines differ between seeds")
        common = signature
        rows[seed] = row

    if len(rows) != 3 or set(rows) != {1, 2, 42}:
        raise ValueError("the three pre-specified FaceNet seeds 42, 1 and 2 are required")
    result = {
        "protocol": "endpoint_age_matched",
        "endpoint_age_tolerance": rows[42]["endpoint_age_tolerance"],
        "negative_seed": rows[42]["negative_seed"],
        "seeds": [42, 1, 2],
        "checkpoint_training_provenance": "legacy checkpoints; training manifests absent",
        "strata": {},
    }
    for stratum in STRATA:
        subject_rows = {
            str(seed): rows[seed][stratum]["subject_bootstrap"] for seed in result["seeds"]
        }
        tuned = np.array([subject_rows[str(seed)]["tuned_auc"] for seed in result["seeds"]])
        delta = np.array([subject_rows[str(seed)]["delta_auc"] for seed in result["seeds"]])
        result["strata"][stratum] = {
            "frozen_auc": float(subject_rows["42"]["frozen_auc"]),
            "tuned_auc_mean": float(np.mean(tuned)),
            "tuned_auc_std_across_seeds": float(np.std(tuned, ddof=1)),
            "delta_auc_mean": float(np.mean(delta)),
            "delta_auc_std_across_seeds": float(np.std(delta, ddof=1)),
            "subject_bootstrap_by_seed": subject_rows,
            "all_per_seed_delta_ci95_above_zero": all(
                item["delta_ci95"][0] > 0 for item in subject_rows.values()
            ),
        }
    return result


def main() -> None:
    metrics_dir = Path(str(data_path("metrics_dir")))
    paths = {seed: metrics_dir / filename for seed, filename in SEED_FILES.items()}
    result = summarize(paths)
    output = metrics_dir / "fgnet_endpoint_multiseed.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    inputs = [item for path in paths.values() for item in (path, path.with_suffix(".manifest.json"))]
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="fgnet-endpoint-age-matched-three-seed-reevaluation",
        parameters={
            "seeds": result["seeds"],
            "negative_seed": result["negative_seed"],
            "endpoint_age_tolerance": result["endpoint_age_tolerance"],
            "training_provenance": result["checkpoint_training_provenance"],
        },
        metrics=result,
        inputs=inputs,
        outputs=[output],
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
