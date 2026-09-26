"""Build and evaluate the corrected internal endpoint-age-matched protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from age_gap.common.device import torch_device
from age_gap.common.io import PROJECT_ROOT, data_path, read_jsonl, write_jsonl
from age_gap.common.manifest import write_experiment_manifest
from age_gap.common.schemas import IdentityGroup, Pair
from age_gap.evaluation.internal_age_matched import (
    build_endpoint_age_matched_pairs,
    evaluate_endpoint_age_matched,
    filter_matched_blocks_by_faces,
    leave_one_subject_out,
    paired_subject_bootstrap,
    split_matched_blocks,
)
from age_gap.evaluation.metrics import verification_metrics
from age_gap.models.backbones import make_backbone
from age_gap.models.embeddings import load_embeddings
from age_gap.training.finetune import _bb_prep, _crop_path, load_finetuned


def _embed_faces(model: torch.nn.Module, face_ids: list[str], device: str) -> dict[str, np.ndarray]:
    prep = _bb_prep(model)
    result: dict[str, np.ndarray] = {}
    for start in range(0, len(face_ids), 32):
        batch_ids = face_ids[start : start + 32]
        arrays = []
        for face_id in batch_ids:
            image = cv2.imread(str(_crop_path(face_id)))
            if image is None:
                raise RuntimeError(f"Crop became unavailable during evaluation: {face_id}")
            arrays.append(prep(image))
        inputs = torch.from_numpy(np.stack(arrays)).to(device)
        with torch.no_grad():
            vectors = model(inputs).detach().cpu().numpy()
        result.update({face_id: vector for face_id, vector in zip(batch_ids, vectors, strict=True)})
    return result


def _pair_score_map(pairs: list[Pair], embeddings: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        pair.pair_id: float(np.dot(embeddings[pair.face_a], embeddings[pair.face_b]))
        for pair in pairs
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Internal test evaluation with counterpart-endpoint age-matched negatives"
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=data_path("data_dir", "processed", "pairs.jsonl"),
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=data_path("data_dir", "processed", "identity_groups.jsonl"),
    )
    parser.add_argument(
        "--group-splits",
        type=Path,
        default=data_path("splits_dir", "group_splits.jsonl"),
    )
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--tolerance-years", type=int, default=2)
    parser.add_argument("--min-positive-gap", type=int, default=25)
    parser.add_argument(
        "--tuned-checkpoint",
        type=Path,
        default=data_path("models_dir", "bb_facenet_seed42.pt"),
    )
    parser.add_argument(
        "--pairs-out",
        type=Path,
        default=data_path(
            "data_dir", "processed", "experiments", "pairs_internal_endpoint_age_matched.jsonl"
        ),
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        default=data_path("metrics_dir", "internal_endpoint_age_matched.json"),
    )
    args = parser.parse_args()

    source_pairs = [Pair.from_dict(row) for row in read_jsonl(args.pairs)]
    groups = [IdentityGroup.from_dict(row) for row in read_jsonl(args.groups)]
    group_splits = {
        row["identity_group_id"]: row["split"] for row in read_jsonl(args.group_splits)
    }
    evaluation_pairs, diagnostics = build_endpoint_age_matched_pairs(
        source_pairs,
        groups,
        group_splits=group_splits,
        seed=args.seed,
        tolerance_years=args.tolerance_years,
        min_positive_gap=args.min_positive_gap,
    )
    positives, negatives = split_matched_blocks(evaluation_pairs)

    embedding_path = args.embeddings or data_path("embeddings_cache_dir", "baseline_arcface.npz")
    embeddings = load_embeddings(embedding_path)
    available_faces = {
        face_id
        for pair in evaluation_pairs
        for face_id in (pair.face_a, pair.face_b)
        if _crop_path(face_id).is_file() and face_id in embeddings
    }
    positives, negatives, dropped = filter_matched_blocks_by_faces(
        positives, negatives, available_faces
    )
    if not positives:
        raise RuntimeError("No complete, class-balanced matched observations have all crops/embeddings")
    evaluation_pairs = [*positives, *negatives]
    diagnostics["matched_positive_count_before_crop_filter"] = diagnostics["matched_positive_count"]
    diagnostics["matched_positive_count"] = len(positives)
    diagnostics["unmatched_positive_count"] = diagnostics["eligible_positive_count"] - len(positives)
    diagnostics["coverage"] = len(positives) / diagnostics["eligible_positive_count"]
    diagnostics["dropped_for_missing_crops_or_arcface_embedding"] = dropped
    write_jsonl(args.pairs_out, (pair.to_dict() for pair in evaluation_pairs))

    metrics, skipped = evaluate_endpoint_age_matched(evaluation_pairs, embeddings)
    clean_metrics = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in metrics.items()
    }
    device = torch_device()
    frozen = make_backbone("facenet", pretrained=True).to(device).eval()
    tuned = load_finetuned(args.tuned_checkpoint, device)
    needed_faces = sorted({pair.face_a for pair in evaluation_pairs} | {pair.face_b for pair in evaluation_pairs})
    frozen_embeddings = _embed_faces(frozen, needed_faces, device)
    tuned_embeddings = _embed_faces(tuned, needed_faces, device)
    del frozen, tuned
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scores = {
        "facenet_frozen": {
            "positive": _pair_score_map(positives, frozen_embeddings),
            "negative": _pair_score_map(negatives, frozen_embeddings),
        },
        "facenet_tuned": {
            "positive": _pair_score_map(positives, tuned_embeddings),
            "negative": _pair_score_map(negatives, tuned_embeddings),
        },
        "arcface": {
            "positive": _pair_score_map(positives, embeddings),
            "negative": _pair_score_map(negatives, embeddings),
        },
    }
    comparison_metrics: dict[str, dict[str, float | None]] = {}
    for tag, grouped_scores in scores.items():
        ordered_scores = [grouped_scores["positive"][p.pair_id] for p in positives] + [
            grouped_scores["negative"][p.pair_id] for p in negatives
        ]
        raw = verification_metrics(
            np.asarray(ordered_scores),
            np.r_[np.ones(len(positives), dtype=int), np.zeros(len(negatives), dtype=int)],
        )
        comparison_metrics[tag] = {
            key: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in raw.items()
        }

    score_dicts = {
        tag: {
            **grouped_scores["positive"],
            **grouped_scores["negative"],
        }
        for tag, grouped_scores in scores.items()
    }
    bootstrap = paired_subject_bootstrap(
        positives,
        negatives,
        {tag: {pair_id: score for pair_id, score in grouped.items()} for tag, grouped in score_dicts.items()},
        {tag: {pair_id: score for pair_id, score in grouped.items()} for tag, grouped in score_dicts.items()},
        reference="facenet_frozen",
        comparison="facenet_tuned",
        n_boot=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    loso = leave_one_subject_out(
        positives,
        negatives,
        score_dicts,
        score_dicts,
        reference="facenet_frozen",
        comparison="facenet_tuned",
    )

    prior_payload = {}
    if args.results_out.is_file():
        prior_payload = json.loads(args.results_out.read_text(encoding="utf-8"))
    payload = {
        "protocol": "internal endpoint-age-matched evaluation; distinct from legacy age-bucket negatives",
        "parameters": {
            "seed": args.seed,
            "endpoint_age_tolerance_years": args.tolerance_years,
            "min_positive_gap_years": args.min_positive_gap,
            "split": "test",
        },
        "diagnostics": diagnostics,
        "verification_metrics": clean_metrics,
        "model_comparison_same_matched_subset": comparison_metrics,
        "facenet_tuned_checkpoint": args.tuned_checkpoint.resolve().relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix(),
        "facenet_paired_subject_bootstrap": bootstrap,
        "facenet_leave_one_subject_out": loso,
        "previous_arcface_point_estimate": prior_payload.get("verification_metrics", clean_metrics),
        "skipped_missing_embeddings": skipped,
    }
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    inventory = data_path("metrics_dir", "model_inventory.json")
    crop_inputs = sorted({_crop_path(face_id) for face_id in needed_faces})
    inputs = [
        args.pairs,
        args.groups,
        args.group_splits,
        embedding_path,
        args.tuned_checkpoint,
        *crop_inputs,
    ]
    if inventory.is_file():
        inputs.append(inventory)
    write_experiment_manifest(
        args.results_out.with_suffix(".manifest.json"),
        experiment="internal-endpoint-age-matched-evaluation",
        parameters=payload["parameters"] | {
            "age_source": "IdentityGroup.age_labels",
            "tuned_checkpoint": args.tuned_checkpoint.resolve().relative_to(
                PROJECT_ROOT.resolve()
            ).as_posix(),
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        metrics=payload,
        inputs=inputs,
        outputs=[args.pairs_out, args.results_out],
    )
    print(
        f"eligible={diagnostics['eligible_positive_count']} "
        f"matched={diagnostics['matched_positive_count']} "
        f"coverage={diagnostics['coverage']:.3f} "
        f"age_gap_only_auc={diagnostics['age_gap_only_roc_auc']} "
        f"arcface_auc={clean_metrics['roc_auc']} "
        f"facenet_frozen_auc={comparison_metrics['facenet_frozen']['roc_auc']} "
        f"facenet_tuned_auc={comparison_metrics['facenet_tuned']['roc_auc']} "
        f"paired_delta={bootstrap['point_estimates']['delta_comparison_minus_reference']:.4f}"
    )


if __name__ == "__main__":
    main()
