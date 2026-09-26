"""Re-evaluate FG-NET with endpoint-age-matched negatives and subject uncertainty.

The old random-negative primary endpoint remains a separate, explicitly legacy
protocol. This script writes only aggregate statistics, never images or scores.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, resolve_path
from age_gap.common.manifest import write_experiment_manifest
from age_gap.evaluation import benchmark_external as bx
from age_gap.evaluation import fgnet
from age_gap.evaluation.subject_bootstrap import (
    leave_one_subject_out_auc,
    paired_subject_bootstrap_auc,
)
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import load_finetuned


def _row(
    frozen: np.ndarray,
    tuned: np.ndarray,
    labels: np.ndarray,
    subject_a: np.ndarray,
    subject_b: np.ndarray,
    observed_age_gap: np.ndarray,
    age_a: np.ndarray,
    age_b: np.ndarray,
    *,
    n_boot: int,
    bootstrap_seed: int,
) -> dict:
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    if n_positive == 0 or n_positive != n_negative:
        raise ValueError(
            f"endpoint-age-matched protocol must be balanced: "
            f"positive={n_positive}, negative={n_negative}"
        )
    age_gap_auc = float(roc_auc_score(labels, observed_age_gap))
    interval = paired_subject_bootstrap_auc(
        frozen,
        tuned,
        labels,
        subject_a,
        subject_b,
        n_boot=n_boot,
        seed=bootstrap_seed,
    )
    sensitivity = leave_one_subject_out_auc(frozen, tuned, labels, subject_a, subject_b)
    return {
        "subject_bootstrap": asdict(interval),
        "leave_one_subject_out": asdict(sensitivity),
        "class_counts": {
            "positive": n_positive,
            "negative": n_negative,
        },
        "observed_age_gap": {
            "positive_mean": float(np.mean(observed_age_gap[labels == 1])),
            "negative_mean": float(np.mean(observed_age_gap[labels == 0])),
            "gap_only_auc": age_gap_auc,
            "gap_only_predictive_auc": max(age_gap_auc, 1.0 - age_gap_auc),
        },
        "endpoint_age_only_auc": {
            "age_a": float(roc_auc_score(labels, age_a)),
            "age_b": float(roc_auc_score(labels, age_b)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/bb_facenet_seed42.pt")
    parser.add_argument("--negative-seed", type=int, default=42)
    parser.add_argument("--endpoint-age-tolerance", type=int, default=2)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--output", default="metrics/fgnet_endpoint_subject_stats.json")
    args = parser.parse_args()
    if args.endpoint_age_tolerance < 0:
        parser.error("--endpoint-age-tolerance must be nonnegative")

    images_a, images_b, labels, _, meta = fgnet.load_pairs(
        protocol="endpoint_age_matched",
        endpoint_age_tolerance=args.endpoint_age_tolerance,
        seed=args.negative_seed,
        return_metadata=True,
    )
    device = torch_device()
    frozen_model = make_backbone("facenet", pretrained=True).to(device).eval()
    checkpoint = Path(resolve_path(args.checkpoint))
    tuned_model = load_finetuned(checkpoint, device)
    frozen = bx.pair_scores(frozen_model, images_a, images_b, device, rgb=False)
    tuned = bx.pair_scores(tuned_model, images_a, images_b, device, rgb=False)
    y = np.asarray(labels, dtype=int)
    subject_a = np.asarray(meta["subject_a"])
    subject_b = np.asarray(meta["subject_b"])
    observed_gap = np.asarray(meta["observed_age_gap"], dtype=float)
    age_a = np.asarray(meta["age_a"], dtype=float)
    age_b = np.asarray(meta["age_b"], dtype=float)
    stratum_gap = np.asarray(meta["stratum_age_gap"], dtype=float)
    endpoint_errors = np.asarray(meta["negative_endpoint_match_error"], dtype=float)
    large = stratum_gap >= 25
    if np.sum(large & (y == 1)) == 0 or np.sum(large & (y == 0)) == 0:
        raise RuntimeError("matched large-gap stratum has no positive or negative pairs")

    result = {
        "protocol": "endpoint_age_matched",
        "endpoint_age_tolerance": args.endpoint_age_tolerance,
        "negative_seed": args.negative_seed,
        "bootstrap_seed": args.bootstrap_seed,
        "n_bootstrap": args.n_bootstrap,
        "overall": _row(
            frozen,
            tuned,
            y,
            subject_a,
            subject_b,
            observed_gap,
            age_a,
            age_b,
            n_boot=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        ),
        "large_gap_25plus": _row(
            frozen[large],
            tuned[large],
            y[large],
            subject_a[large],
            subject_b[large],
            observed_gap[large],
            age_a[large],
            age_b[large],
            n_boot=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        ),
        "negative_construction": {
            "all_negatives_have_different_subjects": bool(
                np.all(subject_a[y == 0] != subject_b[y == 0])
            ),
            "positive_source_count": int(meta["n_positive_source"]),
            "positive_retained_count": int(meta["n_positive_retained"]),
            "positive_unmatched_count": int(meta["n_positive_unmatched"]),
            "positive_coverage": float(meta["positive_coverage"]),
            "large_gap_positive_source_count": int(meta["n_large_gap_positive_source"]),
            "large_gap_positive_retained_count": int(meta["n_large_gap_positive_retained"]),
            "large_gap_positive_unmatched_count": int(meta["n_large_gap_positive_unmatched"]),
            "large_gap_positive_coverage": float(meta["large_gap_positive_coverage"]),
            "endpoint_age_error_mean_years": float(np.mean(endpoint_errors)),
            "endpoint_age_error_max_years": float(np.max(endpoint_errors)),
            "positive_negative_stratum_counts": {
                "positive": int(np.sum(large & (y == 1))),
                "negative": int(np.sum(large & (y == 0))),
            },
        },
    }
    output = Path(resolve_path(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    inputs = [checkpoint, Path(str(data_path("data_dir", "external", "fgnet_crops.npz")))]
    model_inventory = Path(str(data_path("metrics_dir", "model_inventory.json")))
    if model_inventory.is_file():
        inputs.append(model_inventory)
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="fgnet-endpoint-age-matched-subject-statistics",
        parameters={
            "checkpoint": str(checkpoint),
            "endpoint_age_tolerance": args.endpoint_age_tolerance,
            "negative_seed": args.negative_seed,
            "bootstrap_seed": args.bootstrap_seed,
            "n_bootstrap": args.n_bootstrap,
        },
        metrics=result,
        inputs=inputs,
        outputs=[output],
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
