"""Score two frozen blinded-annotation files and an adjudicated gold file."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from age_gap.common.io import data_path, read_jsonl
from age_gap.common.manifest import write_experiment_manifest
from age_gap.evaluation.human_audit import binary_metrics, cohens_kappa, percentile_ci

BOOTSTRAP_SEED = 20260922
BOOTSTRAP_ITERATIONS = 2000


def _indexed(path: Path) -> dict[str, dict]:
    raw = list(read_jsonl(path))
    rows = {row["task_id"]: row for row in raw}
    if not rows:
        raise RuntimeError(f"no annotations in {path}")
    if len(rows) != len(raw):
        raise RuntimeError(f"duplicate task IDs in {path}")
    return rows


def _validate_responses(
    annotations: dict[str, dict], key: dict[str, dict], source: str
) -> None:
    identity_labels = {"same_identity", "different_identity", "uncertain"}
    duplicate_labels = {"near_duplicate", "distinct_photo", "uncertain"}
    for task_id, key_row in key.items():
        value = annotations[task_id].get("response")
        if value is None:
            raise RuntimeError(f"null response for {task_id} in {source}")
        kind = key_row["audit_type"]
        if kind == "age_extraction":
            if value == "uncertain":
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid age response for {task_id} in {source}") from exc
            if not parsed.is_integer() or parsed < 0 or parsed > 120:
                raise RuntimeError(f"invalid age response for {task_id} in {source}")
        elif kind == "near_duplicate":
            if value not in duplicate_labels:
                raise RuntimeError(f"invalid duplicate response for {task_id} in {source}")
        elif value not in identity_labels:
            raise RuntimeError(f"invalid identity response for {task_id} in {source}")


def _source_components(task_ids: list[str], key: dict[str, dict]) -> list[list[str]]:
    """Connected task components induced by shared source images."""
    parent = {task_id: task_id for task_id in task_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    owner: dict[str, str] = {}
    for task_id in task_ids:
        for source in key[task_id].get("source_images", []):
            if source in owner:
                union(task_id, owner[source])
            else:
                owner[source] = task_id
    components: dict[str, list[str]] = defaultdict(list)
    for task_id in task_ids:
        components[find(task_id)].append(task_id)
    return list(components.values())


def _cluster_ci(components: list[list[str]], statistic) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float64)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sampled = rng.integers(0, len(components), len(components))
        task_ids = [task_id for index in sampled for task_id in components[int(index)]]
        estimates[iteration] = float(statistic(task_ids))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _age_system_metrics(truth: dict[str, float], predictions: dict[str, object]) -> dict:
    common = [task_id for task_id in truth if predictions.get(task_id) is not None]
    errors = np.asarray(
        [abs(truth[task_id] - float(predictions[task_id])) for task_id in common], dtype=float
    )
    if not len(errors):
        return {"n": 0, "coverage": 0.0}
    return {
        "n": len(errors),
        "coverage": len(errors) / len(truth),
        "mae": float(errors.mean()),
        "mae_95ci": list(percentile_ci(errors, np.mean)),
        "within_1y": float(np.mean(errors <= 1)),
        "exact": float(np.mean(errors == 0)),
    }


def score(
    a_path: Path,
    b_path: Path,
    gold_path: Path,
    key_path: Path,
    age_baselines_path: Path | None = None,
) -> dict:
    a, b, gold, key = map(_indexed, [a_path, b_path, gold_path, key_path])
    age_baselines = _indexed(age_baselines_path) if age_baselines_path else {}
    if not (set(a) == set(b) == set(gold) == set(key)):
        raise RuntimeError("annotation/gold/key task IDs are incomplete or inconsistent")
    _validate_responses(a, key, str(a_path))
    _validate_responses(b, key, str(b_path))
    _validate_responses(gold, key, str(gold_path))
    common = set(key)
    by_type: dict[str, list[str]] = defaultdict(list)
    for task_id in sorted(common):
        by_type[key[task_id]["audit_type"]].append(task_id)

    report: dict[str, dict] = {}
    for kind, ids in sorted(by_type.items()):
        aa = [str(a[i]["response"]) for i in ids]
        bb = [str(b[i]["response"]) for i in ids]
        components = _source_components(ids, key)
        entry: dict[str, object] = {
            "n": len(ids),
            "raw_agreement": float(np.mean(np.asarray(aa) == np.asarray(bb))),
            "cohens_kappa": cohens_kappa(aa, bb),
            "cohens_kappa_95ci": _cluster_ci(
                components,
                lambda sample_ids: cohens_kappa(
                    [str(a[i]["response"]) for i in sample_ids],
                    [str(b[i]["response"]) for i in sample_ids],
                ),
            ),
            "bootstrap_unit": "source-image connected component",
            "n_source_components": len(components),
        }
        if kind == "age_extraction":
            numeric_truth: dict[str, float] = {}
            for task_id in ids:
                try:
                    numeric_truth[task_id] = float(gold[task_id]["response"])
                except (TypeError, ValueError):
                    continue
            systems: dict[str, dict] = {
                "gigachat": _age_system_metrics(
                    numeric_truth, {task_id: key[task_id]["auto_age"] for task_id in ids}
                )
            }
            if age_baselines:
                for name in ("regex_age", "local_age"):
                    systems[name.removesuffix("_age")] = _age_system_metrics(
                        numeric_truth,
                        {task_id: age_baselines.get(task_id, {}).get(name) for task_id in ids},
                    )
            entry["n_numeric_gold"] = len(numeric_truth)
            entry["n_uncertain_gold"] = len(ids) - len(numeric_truth)
            entry["systems"] = systems
        else:
            positive_label = "near_duplicate" if kind == "near_duplicate" else "same_identity"
            scored_ids = [i for i in ids if gold[i]["response"] != "uncertain"]
            truth = [gold[i]["response"] == positive_label for i in scored_ids]
            predicted = [key[i]["auto_label"] == positive_label for i in scored_ids]
            if not scored_ids:
                raise RuntimeError(f"no adjudicated, non-uncertain gold rows for {kind}")
            correct = np.asarray(
                [gold[i]["response"] == key[i]["auto_label"] for i in scored_ids], dtype=float
            )
            scored_components = _source_components(scored_ids, key)
            if len(set(predicted)) == 2:
                metrics = binary_metrics(truth, predicted)
                entry.update(metrics)
                for metric_name in ("precision", "recall", "f1"):
                    entry[f"{metric_name}_95ci"] = _cluster_ci(
                        scored_components,
                        lambda sample_ids, name=metric_name, label=positive_label: binary_metrics(
                            [gold[i]["response"] == label for i in sample_ids],
                            [key[i]["auto_label"] == label for i in sample_ids],
                        )[name],
                    )
                entry["sampling_design"] = "both automatic decisions sampled"
            elif all(predicted):
                truth_by_id = {
                    task_id: gold[task_id]["response"] == positive_label
                    for task_id in scored_ids
                }
                entry["positive_predictive_value"] = float(np.mean(truth))
                entry["positive_predictive_value_95ci"] = _cluster_ci(
                    scored_components,
                    lambda sample_ids, values=truth_by_id: np.mean(
                        [values[i] for i in sample_ids]
                    ),
                )
                entry["sampling_design"] = (
                    "automatic-positive audit; recall and F1 are not identifiable"
                )
            else:
                negative_truth_by_id = {
                    task_id: gold[task_id]["response"] != positive_label
                    for task_id in scored_ids
                }
                entry["negative_predictive_value"] = float(np.mean(~np.asarray(truth)))
                entry["negative_predictive_value_95ci"] = _cluster_ci(
                    scored_components,
                    lambda sample_ids, values=negative_truth_by_id: np.mean(
                        [values[i] for i in sample_ids]
                    ),
                )
                entry["sampling_design"] = (
                    "automatic-negative audit; recall and F1 are not identifiable"
                )
            entry["positive_label"] = positive_label
            entry["n_scored"] = len(scored_ids)
            entry["n_uncertain_gold"] = len(ids) - len(scored_ids)
            entry["accuracy"] = float(np.mean(correct))
            correctness = {
                task_id: gold[task_id]["response"] == key[task_id]["auto_label"]
                for task_id in scored_ids
            }
            entry["accuracy_95ci"] = _cluster_ci(
                scored_components,
                lambda sample_ids, values=correctness: np.mean(
                    [values[i] for i in sample_ids]
                ),
            )
        report[kind] = entry
    return {"audit": report, "total_tasks": len(common)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = data_path("data_dir", "interim", "human_audit")
    parser.add_argument("--annotator-a", type=Path, default=base / "annotator_a.jsonl")
    parser.add_argument("--annotator-b", type=Path, default=base / "annotator_b.jsonl")
    parser.add_argument("--gold", type=Path, default=base / "adjudicated_gold.jsonl")
    parser.add_argument("--key", type=Path, default=base / "audit_key.jsonl")
    parser.add_argument("--age-baselines", type=Path, default=base / "age_baseline_predictions.jsonl")
    parser.add_argument(
        "--output", type=Path, default=data_path("metrics_dir", "human_audit.json")
    )
    args = parser.parse_args()
    baselines = args.age_baselines if args.age_baselines.exists() else None
    report = score(args.annotator_a, args.annotator_b, args.gold, args.key, baselines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.output.with_suffix(".manifest.json"),
        experiment="blinded-human-supervision-audit",
        parameters={
            "annotators": 2,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "source-image connected component",
            "class_conditional_metrics": (
                "PPV/NPV for one-sided automatic-decision audits; precision/recall/F1 otherwise"
            ),
        },
        metrics=report,
        inputs=[
            args.annotator_a,
            args.annotator_b,
            args.gold,
            args.key,
            *([baselines] if baselines else []),
        ],
        outputs=[args.output],
    )
    print(args.output)


if __name__ == "__main__":
    main()
