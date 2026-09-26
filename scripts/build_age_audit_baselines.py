"""Freeze regex and reproducible local-model predictions for the blinded age audit.

The local comparator is deliberately modest and fully offline: a character-ngram
ridge regressor trained on the non-audit caption-to-age anchors already present in
the corpus. Its targets are weak automatic labels, so it is a reproducibility
baseline, not independent ground truth. Human adjudication remains the only gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from age_gap.common.io import data_path, read_jsonl
from age_gap.common.manifest import write_experiment_manifest
from age_gap.datasets.age_anchors import RegexAgeExtractor


def _task_id(group_id: str, face_id: str, age: int) -> str:
    evidence = f"{group_id}|{face_id}|{age}"
    return "age-" + hashlib.sha256(f"age|{evidence}".encode()).hexdigest()[:16]


def _feature(caption: str, target_index: int, n_photos: int) -> str:
    return f"__target_{target_index}__ __count_{n_photos}__ {caption.strip().lower()}"


def regex_age(caption: str, target_index: int, n_photos: int) -> int | None:
    labels = RegexAgeExtractor().extract(caption, n_photos)
    explicit = {
        f"position_{target_index}",
        "first" if target_index == 0 else "second" if target_index == 1 else "",
        "left" if target_index == 0 else "right" if target_index == 1 else "",
    }
    for label in labels:
        if label.photo_reference in explicit and label.age is not None:
            return int(label.age)
    if len(labels) == n_photos and target_index < len(labels):
        return int(labels[target_index].age) if labels[target_index].age is not None else None
    if len(labels) == 1 and labels[0].age is not None:
        return int(labels[0].age)
    return None


def _training_rows(audit_ids: set[str]) -> tuple[list[str], list[int]]:
    posts = list(read_jsonl(data_path("data_dir", "raw", "posts.jsonl")))
    post_by_photo = {
        str(photo["photo_id"]): post for post in posts for photo in post.get("photos", [])
    }
    features: list[str] = []
    targets: list[int] = []
    for group in map(
        dict, read_jsonl(data_path("data_dir", "processed", "identity_groups.jsonl"))
    ):
        faces = list(group.get("faces", []))
        for label in group.get("age_labels", []):
            face_id = label.get("face_id")
            age = label.get("age")
            if not face_id or age is None or _task_id(group["identity_group_id"], face_id, age) in audit_ids:
                continue
            photo_id = str(face_id).rsplit("_f", 1)[0]
            post = post_by_photo.get(photo_id)
            if not post:
                continue
            order = [str(photo["photo_id"]) for photo in post.get("photos", [])]
            context = sorted(
                (item for item in faces if item.rsplit("_f", 1)[0] in order),
                key=lambda item: order.index(item.rsplit("_f", 1)[0]),
            )
            if face_id not in context:
                continue
            features.append(_feature(post.get("caption", ""), context.index(face_id), len(context)))
            targets.append(int(age))
    return features, targets


def build(tasks_path: Path, key_path: Path, output: Path, model_path: Path) -> dict:
    tasks = [row for row in read_jsonl(tasks_path) if row["audit_type"] == "age_extraction"]
    key = {row["task_id"]: row for row in read_jsonl(key_path)}
    if not tasks or any(row["task_id"] not in key for row in tasks):
        raise RuntimeError("age audit tasks and key are empty or inconsistent")
    train_x, train_y = _training_rows({row["task_id"] for row in tasks})
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=3,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            ("ridge", Ridge(alpha=10.0, solver="lsqr")),
        ]
    )
    model.fit(train_x, np.asarray(train_y, dtype=float))
    audit_x = [
        _feature(row["caption"], int(row["target_image_index"]), len(row["images"]))
        for row in tasks
    ]
    local = np.clip(np.rint(model.predict(audit_x)), 1, 99).astype(int)
    predictions = []
    for row, local_age in zip(tasks, local, strict=True):
        target = int(row["target_image_index"])
        predictions.append(
            {
                "task_id": row["task_id"],
                "giga_age": key[row["task_id"]]["auto_age"],
                "giga_source": key[row["task_id"]].get("auto_source", "unknown"),
                "regex_age": regex_age(row["caption"], target, len(row["images"])),
                "local_age": int(local_age),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    coverage = sum(row["regex_age"] is not None for row in predictions) / len(predictions)
    metrics = {
        "audit_tasks": len(predictions),
        "local_training_rows": len(train_y),
        "regex_coverage": coverage,
        "gold_scoring_status": "pending adjudicated human responses",
    }
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="age-extraction-audit-baselines",
        parameters={
            "local_model": "TF-IDF char 2-5gram + ridge",
            "ridge_alpha": 10.0,
            "max_features": 50_000,
            "audit_rows_excluded_from_training": True,
            "training_targets": "non-audit automatic age anchors",
        },
        metrics=metrics,
        inputs=[
            tasks_path,
            key_path,
            data_path("data_dir", "raw", "posts.jsonl"),
            data_path("data_dir", "processed", "identity_groups.jsonl"),
        ],
        outputs=[output, model_path],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = data_path("data_dir", "interim", "human_audit")
    parser.add_argument("--tasks", type=Path, default=base / "annotator_a.jsonl")
    parser.add_argument("--key", type=Path, default=base / "audit_key.jsonl")
    parser.add_argument("--output", type=Path, default=base / "age_baseline_predictions.jsonl")
    parser.add_argument(
        "--model", type=Path, default=data_path("models_dir", "audit_age_local.joblib")
    )
    args = parser.parse_args()
    print(json.dumps(build(args.tasks, args.key, args.output, args.model), indent=2))


if __name__ == "__main__":
    main()
