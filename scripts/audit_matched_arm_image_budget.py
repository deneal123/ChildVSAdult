"""Aggregate actual positive/all-training image counts of existing LOW/CROSS arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest


def counts(rows):
    train = [r for r in rows if r.get("split") == "train"]
    positive = [r for r in train if r.get("label") == 1]
    negative = [r for r in train if r.get("label") == 0]
    if not positive or len(positive) != len(negative) or len(train) != len(positive) + len(negative):
        raise ValueError("nonempty binary balanced training required")
    positive_faces = {r[f"face_{s}"] for r in positive for s in ("a", "b")}
    train_faces = {r[f"face_{s}"] for r in train for s in ("a", "b")}
    return {"positive_pairs": len(positive), "negative_pairs": len(negative),
            "positive_images": len(positive_faces), "all_train_images": len(train_faces),
            "negative_only_images": len(train_faces - positive_faces)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", type=Path, default=PROJECT_ROOT / "data/interim/matched_agegap_arms")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "metrics/matched_arm_image_budget_20261002")
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("new output directory required")
    inputs = [args.arms / "low_arm.jsonl", args.arms / "cross_arm.jsonl", Path(__file__)]
    before = [file_record(path) for path in inputs]
    arms = {name: counts(list(read_jsonl(path))) for name, path in zip(("LOW", "CROSS"), inputs[:2], strict=True)}
    result = {"arms": arms,
              "positive_image_budget_equal": arms["LOW"]["positive_images"] == arms["CROSS"]["positive_images"],
              "all_train_image_budget_equal": arms["LOW"]["all_train_images"] == arms["CROSS"]["all_train_images"],
              "scope": "count audit of existing arm files, not retraining or causal evidence",
              "publication_ready": False}
    if before != [file_record(path) for path in inputs]:
        raise ValueError("arm inputs changed")
    args.out.mkdir(parents=True)
    target = args.out / "summary.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = write_experiment_manifest(target.with_suffix(".manifest.json"),
        experiment="matched-arm-actual-image-budget-audit", parameters={"splits_counted": ["train"]},
        metrics=result, inputs=inputs, outputs=[target])
    if json.loads(manifest.read_text(encoding="utf-8"))["inputs"] != before:
        manifest.unlink()
        raise ValueError("inputs changed while writing completed manifest")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
