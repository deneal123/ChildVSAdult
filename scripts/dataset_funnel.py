"""Generate the unified data funnel and retained/rejected quality distributions."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from age_gap.common.io import data_path, read_jsonl
from age_gap.common.manifest import write_experiment_manifest


def _distribution(rows: list[dict], field: str, *, absolute: bool = False) -> dict[str, float | int | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if absolute:
        values = [abs(value) for value in values]
    if not values:
        return {"n": 0, "p05": None, "p25": None, "median": None, "p75": None, "p95": None}
    array = np.asarray(values)
    q = np.percentile(array, [5, 25, 50, 75, 95])
    return {
        "n": len(values),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
    }


def main() -> None:
    posts_path = Path(str(data_path("data_dir", "raw", "posts.jsonl")))
    faces_path = Path(str(data_path("data_dir", "interim", "faces.jsonl")))
    quality_path = Path(str(data_path("data_dir", "interim", "face_quality_audit.jsonl")))
    groups_path = Path(str(data_path("data_dir", "processed", "identity_groups.jsonl")))
    clusters_path = Path(str(data_path("data_dir", "processed", "person_clusters.jsonl")))
    pairs_path = Path(str(data_path("data_dir", "processed", "pairs.jsonl")))
    gender_age_path = Path(str(data_path("data_dir", "interim", "face_genderage.jsonl")))
    required = [posts_path, faces_path, quality_path, groups_path, clusters_path, pairs_path]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    posts = list(read_jsonl(posts_path))
    faces = list(read_jsonl(faces_path))
    quality = list(read_jsonl(quality_path))
    groups = list(read_jsonl(groups_path))
    clusters = list(read_jsonl(clusters_path))
    pairs = list(read_jsonl(pairs_path))
    gender_age = list(read_jsonl(gender_age_path)) if gender_age_path.exists() else []
    retained = [row for row in quality if row["is_usable"]]
    rejected = [row for row in quality if not row["is_usable"]]
    positive = [row for row in pairs if row["label"] == 1]
    negative = [row for row in pairs if row["label"] == 0]

    fields = {
        "image_width": False,
        "image_height": False,
        "face_width": False,
        "face_height": False,
        "blur_var": False,
        "det_score": False,
        "face_quality_score": False,
        "pose_yaw_proxy_abs": True,
        "pose_roll_rad_abs": True,
    }
    comparison = {}
    for label, absolute in fields.items():
        source = label.removesuffix("_abs")
        comparison[label] = {
            "retained": _distribution(retained, source, absolute=absolute),
            "rejected": _distribution(rejected, source, absolute=absolute),
        }

    age_values = np.asarray([float(row["age_est"]) for row in gender_age if row.get("age_est") is not None])
    gaps = np.asarray([float(row["age_gap"]) for row in positive if row.get("age_gap") is not None])
    result = {
        "funnel": {
            "posts": len(posts),
            "photos": sum(len(post.get("photos", [])) for post in posts),
            "face_records": len(faces),
            "usable_faces": sum(bool(face.get("is_usable")) for face in faces),
            "rejected_face_records": sum(not bool(face.get("is_usable")) for face in faces),
            "curated_faces": len({face_id for group in groups for face_id in group.get("faces", [])}),
            "identity_groups": len(groups),
            "person_clusters": len({row["person_id"] for row in clusters}),
            "positive_pairs": len(positive),
            "negative_pairs": len(negative),
            "final_pairs": len(pairs),
        },
        "rejection_reasons": dict(Counter(face.get("reject_reason") or "none" for face in faces)),
        "retained_vs_rejected": comparison,
        "apparent_age": {
            "n": int(len(age_values)),
            "p05_p25_median_p75_p95": np.percentile(age_values, [5, 25, 50, 75, 95]).tolist(),
        },
        "positive_age_gap": {
            "n": int(len(gaps)),
            "p05_p25_median_p75_p95": np.percentile(gaps, [5, 25, 50, 75, 95]).tolist(),
            "buckets": {
                "0-4": int(((gaps >= 0) & (gaps < 5)).sum()),
                "5-9": int(((gaps >= 5) & (gaps < 10)).sum()),
                "10-14": int(((gaps >= 10) & (gaps < 15)).sum()),
                "15-24": int(((gaps >= 15) & (gaps < 25)).sum()),
                "25+": int((gaps >= 25).sum()),
            },
        },
    }
    output = Path(str(data_path("metrics_dir", "data_funnel.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inputs = required + ([gender_age_path] if gender_age_path.exists() else [])
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="dataset-funnel-and-quality-audit",
        parameters={"quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]},
        metrics=result,
        inputs=inputs,
        outputs=[output],
    )
    print(json.dumps(result["funnel"], ensure_ascii=False, indent=2))
    print(f"OK: {output}")


if __name__ == "__main__":
    main()
