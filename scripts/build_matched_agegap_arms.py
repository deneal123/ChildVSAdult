"""Build private LOW/CROSS train arms with age-bin-matched impostor negatives.

LOW positives have age gap 1-2 years; CROSS positives have age gap >=25 years.
Each arm receives the same number of sampled positives and negatives. Negatives
are generated from a shared pool of age-known endpoints seen in canonical train
positives; the negative endpoint B is matched to the target positive's B age
using fixed-width age bins. Canonical validation/test rows are copied unchanged
to both private outputs. No canonical files are written.

Run with ``uv run python scripts/build_matched_agegap_arms.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from age_gap.common.io import data_path, read_jsonl, write_jsonl
from age_gap.common.manifest import write_experiment_manifest

DEFAULT_OUTPUT = Path("data/interim/matched_agegap_arms")
DEFAULT_TARGET = 600
DEFAULT_IDENTITIES = 500
DEFAULT_FACES = 1096
AGE_BIN_YEARS = 10
MIN_B_BIN_IDENTITY_SUPPORT = 20


def _stable_jsonl_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _person(group_id: str | None, group_to_person: dict[str, str]) -> str | None:
    return group_to_person.get(group_id, group_id) if group_id is not None else None


def _eligible_positive(row: dict[str, Any]) -> bool:
    return (
        row.get("split") == "train"
        and row.get("label") == 1
        and row.get("status", "ok") == "ok"
        and row.get("age_a") is not None
        and row.get("age_b") is not None
        and row.get("identity_group_a") is not None
        and row.get("identity_group_b") is not None
        and row.get("face_a") != row.get("face_b")
    )


def _profile_matched_sample(
    candidates: list[dict[str, Any]],
    *,
    target_identities: int,
    target_pairs: int,
    target_faces: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select exact identity/pair/image counts with seeded tie-breaking."""
    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_person[str(row["_person_id"])].append(row)
    if len(by_person) < target_identities:
        raise ValueError(
            f"need {target_identities} eligible identities, found only {len(by_person)}"
        )
    if target_pairs < target_identities or target_faces < 2 * target_identities:
        raise ValueError("profile targets require at least one positive pair and two faces per identity")

    rng = random.Random(seed)
    person_tie = {person: rng.random() for person in by_person}
    enriched = [
        person
        for person, rows in by_person.items()
        if len(rows) > 1 or len({f for row in rows for f in (row["face_a"], row["face_b"])}) > 2
    ]
    enriched.sort(
        key=lambda person: (
            -len({f for row in by_person[person] for f in (row["face_a"], row["face_b"])}),
            -len(by_person[person]),
            person_tie[person],
        )
    )
    selected_people = enriched[:target_identities]
    if len(selected_people) < target_identities:
        remaining = [p for p in by_person if p not in set(selected_people)]
        remaining.sort(key=lambda person: (person_tie[person], person))
        selected_people.extend(remaining[: target_identities - len(selected_people)])

    selected_rows: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    for person in selected_people:
        row_tie = {str(item["pair_id"]): rng.random() for item in by_person[person]}
        row = min(by_person[person], key=lambda item: (row_tie[str(item["pair_id"])], str(item["pair_id"])))
        selected_rows.append(row)
        chosen_ids.add(str(row["pair_id"]))
    face_set = {f for row in selected_rows for f in (row["face_a"], row["face_b"])}

    remaining_pair_count = target_pairs - len(selected_rows)
    remaining_face_count = target_faces - len(face_set)
    if remaining_face_count < 0:
        raise ValueError("selected identity baseline already exceeds target unique face count")
    available = [
        row
        for person in selected_people
        for row in by_person[person]
        if str(row["pair_id"]) not in chosen_ids
    ]
    row_tie = {str(row["pair_id"]): rng.random() for row in available}
    while remaining_pair_count:
        scored = []
        for row in available:
            delta = len({row["face_a"], row["face_b"]} - face_set)
            if delta <= remaining_face_count:
                scored.append((delta, row_tie[str(row["pair_id"])], str(row["pair_id"]), row))
        if not scored:
            raise ValueError(
                "cannot meet exact positive profile: "
                f"need {remaining_pair_count} more pairs and {remaining_face_count} more faces"
            )
        # Grow the image set one endpoint at a time where possible; once its
        # target is reached, use pairs whose endpoints are already represented.
        preferred_delta = 1 if remaining_face_count > 0 else 0
        eligible = [item for item in scored if item[0] == preferred_delta]
        if not eligible and remaining_face_count > 0:
            eligible = [item for item in scored if item[0] == 2]
        if not eligible:
            raise ValueError(
                "cannot meet exact positive image count with available endpoint edges: "
                f"need {remaining_face_count} more faces across {remaining_pair_count} pairs"
            )
        _delta, _tie, pair_id, row = min(eligible, key=lambda item: (item[1], item[2]))
        selected_rows.append(row)
        chosen_ids.add(pair_id)
        available = [item for item in available if str(item["pair_id"]) != pair_id]
        face_set.update((row["face_a"], row["face_b"]))
        remaining_pair_count -= 1
        remaining_face_count -= _delta

    if remaining_face_count != 0:
        raise ValueError(f"positive profile finished with {remaining_face_count} face-count mismatch")
    return selected_rows


def _endpoint_pool(
    positives: list[dict[str, Any]], group_to_person: dict[str, str]
) -> tuple[list[dict[str, Any]], int]:
    observations: dict[str, Counter[tuple[int, str, str]]] = defaultdict(Counter)
    for row in positives:
        for side in ("a", "b"):
            face_id = str(row[f"face_{side}"])
            age = int(row[f"age_{side}"])
            group_id = str(row[f"identity_group_{side}"])
            person_id = _person(group_id, group_to_person)
            if person_id is not None:
                observations[face_id][(age, str(person_id), group_id)] += 1

    endpoints: list[dict[str, Any]] = []
    conflicts = 0
    for face_id, counts in sorted(observations.items()):
        if len({entry[0] for entry in counts}) > 1:
            conflicts += 1
        # Deterministic majority; age then person/group IDs resolve ties.
        age, person_id, group_id = min(
            counts,
            key=lambda value: (-counts[value], value[0], value[1], value[2]),
        )
        endpoints.append(
            {"face_id": face_id, "age": age, "person_id": person_id, "group_id": group_id}
        )
    return endpoints, conflicts


def _sample_impostors(
    targets: list[dict[str, Any]],
    endpoints: list[dict[str, Any]],
    *,
    arm: str,
    seed: int,
    bin_years: int,
    max_age_error: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    tie = {endpoint["face_id"]: rng.random() for endpoint in endpoints}
    by_bin: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for endpoint in endpoints:
        by_bin[int(endpoint["age"]) // bin_years].append(endpoint)

    face_use: Counter[str] = Counter()
    identity_use: Counter[str] = Counter()
    used_unordered_pairs: set[tuple[str, str]] = set()
    negatives: list[dict[str, Any]] = []
    positive_people = {str(row["_person_id"]) for row in targets}
    anchor_inside = 0
    impostor_inside = 0
    age_error_histogram: Counter[int] = Counter()
    for index, target in enumerate(targets):
        target_bin = int(target["age_b"]) // bin_years
        anchor_face = str(target["face_a"])
        anchor_person = str(target["_person_id"])
        anchor_group = str(target["identity_group_a"])
        b_options = sorted(
            (
                endpoint
                for endpoint in by_bin.get(target_bin, [])
                if endpoint["person_id"] != anchor_person
                and endpoint["face_id"] != anchor_face
                and abs(int(endpoint["age"]) - int(target["age_b"])) <= max_age_error
                and tuple(sorted((anchor_face, endpoint["face_id"]))) not in used_unordered_pairs
            ),
            key=lambda e: (
                abs(int(e["age"]) - int(target["age_b"])),
                face_use[e["face_id"]] + identity_use[e["person_id"]],
                tie[e["face_id"]],
                e["face_id"],
            ),
        )
        if not b_options:
            raise ValueError(
                f"cannot construct distinct-person impostor for {arm} target {index} "
                f"at endpoint-B age bin {target_bin}"
            )
        endpoint_b = b_options[0]
        age_error_histogram[abs(int(endpoint_b["age"]) - int(target["age_b"]))] += 1
        used_unordered_pairs.add(tuple(sorted((anchor_face, endpoint_b["face_id"]))))
        face_use.update((anchor_face, endpoint_b["face_id"]))
        identity_use.update((anchor_person, endpoint_b["person_id"]))
        anchor_inside += int(anchor_person in positive_people)
        impostor_inside += int(endpoint_b["person_id"] in positive_people)
        negatives.append(
            {
                "pair_id": f"matchedneg_{arm.lower()}_{seed}_{index:04d}",
                "face_a": anchor_face,
                "face_b": endpoint_b["face_id"],
                "label": 0,
                "pair_type": "negative_matched_source_impostor",
                "identity_group_a": anchor_group,
                "identity_group_b": endpoint_b["group_id"],
                "age_a": int(target["age_a"]),
                "age_b": endpoint_b["age"],
                "age_gap": abs(int(target["age_a"]) - endpoint_b["age"]),
                "hardness": "matched_age_bin",
                "status": "ok",
                "split": "train",
                "matched_target_pair_id": target["pair_id"],
                "endpoint_b_age_bin": target_bin,
            }
        )

    reuse = Counter(face_use.values())
    diagnostics = {
        "negative_pairs": len(negatives),
        "unique_negative_faces": len(face_use),
        "unique_negative_persons": len(identity_use),
        "max_negative_face_reuse": max(face_use.values(), default=0),
        "negative_face_reuse_histogram": {str(k): v for k, v in sorted(reuse.items())},
        "negative_pairs_reused": len(negatives) - len(used_unordered_pairs),
        "positive_persons_in_arm": len(positive_people),
        "negative_endpoint_a_in_arm_positive_person_fraction": anchor_inside / len(negatives)
        if negatives
        else None,
        "negative_endpoint_a_is_its_positive_face_a_fraction": 1.0,
        "negative_endpoint_b_in_arm_positive_person_fraction": impostor_inside / len(negatives)
        if negatives
        else None,
        "negative_endpoint_b_age_bin_match_fraction": 1.0,
        "negative_endpoint_b_exact_age_error_histogram": {
            str(k): v for k, v in sorted(age_error_histogram.items())
        },
    }
    return negatives, diagnostics


def build_arms(
    pairs: list[dict[str, Any]],
    group_to_person: dict[str, str],
    *,
    target_identities: int = DEFAULT_IDENTITIES,
    target_per_class: int = DEFAULT_TARGET,
    target_faces: int = DEFAULT_FACES,
    seed: int = 42,
    bin_years: int = AGE_BIN_YEARS,
    min_bin_support: int = MIN_B_BIN_IDENTITY_SUPPORT,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if bin_years <= 0 or min_bin_support < 2:
        raise ValueError("bin_years must be positive and min_bin_support must be at least two")
    train_positive = [
        {
            **row,
            "_person_id": _person(str(row["identity_group_a"]), group_to_person),
        }
        for row in pairs
        if _eligible_positive(row)
    ]
    predicates = {
        "LOW": lambda gap: 1 <= gap <= 2,
        "CROSS": lambda gap: gap >= 25,
    }
    eligible_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm, predicate in predicates.items():
        eligible_by_arm[arm] = [
            row
            for row in train_positive
            if predicate(
                int(
                    row["age_gap"]
                    if row["age_gap"] is not None
                    else abs(int(row["age_a"]) - int(row["age_b"]))
                )
            )
        ]
    eligible_people_by_arm = {
        arm: {str(row["_person_id"]) for row in rows}
        for arm, rows in eligible_by_arm.items()
    }
    cross_stratum_people = eligible_people_by_arm["LOW"] & eligible_people_by_arm["CROSS"]
    eligible_people_by_arm = {
        arm: people - cross_stratum_people for arm, people in eligible_people_by_arm.items()
    }
    common_pool, age_conflicts = _endpoint_pool(train_positive, group_to_person)
    endpoint_bins_by_person: dict[str, set[int]] = defaultdict(set)
    for endpoint in common_pool:
        endpoint_bins_by_person[str(endpoint["person_id"])].add(
            int(endpoint["age"]) // bin_years
        )
    supported_b_bins: dict[str, set[int]] = {}
    b_bin_support_counts: dict[str, dict[str, int]] = {}
    for arm, people in eligible_people_by_arm.items():
        support: Counter[int] = Counter(
            age_bin
            for person in people
            for age_bin in endpoint_bins_by_person.get(person, set())
        )
        b_bin_support_counts[arm] = {str(k): v for k, v in sorted(support.items())}
        supported_b_bins[arm] = {
            age_bin
            for age_bin, count in support.items()
            if count >= min_bin_support
        }
    candidates_before_b_filter = {
        arm: [row for row in rows if str(row["_person_id"]) in eligible_people_by_arm[arm]]
        for arm, rows in eligible_by_arm.items()
    }
    candidates_by_arm = {
        arm: [
            row
            for row in candidates_before_b_filter[arm]
            if int(row["age_b"]) // bin_years in supported_b_bins[arm]
        ]
        for arm in eligible_by_arm
    }
    heldout = [row for row in pairs if row.get("split") in {"val", "test"}]
    arms: dict[str, list[dict[str, Any]]] = {}
    arm_stats: dict[str, Any] = {}
    selected_people_by_arm: dict[str, set[str]] = {}
    selected_faces_by_arm: dict[str, set[str]] = {}

    for arm in ("LOW", "CROSS"):
        candidates = candidates_by_arm[arm]
        selected = _profile_matched_sample(
            candidates,
            target_identities=target_identities,
            target_pairs=target_per_class,
            target_faces=target_faces,
            seed=seed + (0 if arm == "LOW" else 1),
        )
        positive_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in selected]
        arm_people = {str(row["_person_id"]) for row in selected}
        arm_pool = [endpoint for endpoint in common_pool if endpoint["person_id"] in arm_people]
        negatives, diagnostics = _sample_impostors(
            selected,
            arm_pool,
            arm=arm,
            seed=seed + (10_000 if arm == "LOW" else 20_000),
            bin_years=bin_years,
            max_age_error=1,
        )
        arms[arm] = [*positive_rows, *negatives, *heldout]
        train_labels = np.asarray([1] * len(positive_rows) + [0] * len(negatives))
        train_rows = [*positive_rows, *negatives]
        cue_auc = {
            field: float(roc_auc_score(train_labels, [int(row[field]) for row in train_rows]))
            for field in ("age_a", "age_b", "age_gap")
        }
        selected_people_by_arm[arm] = {str(row["_person_id"]) for row in selected}
        selected_faces_by_arm[arm] = {
            str(face) for row in selected for face in (row["face_a"], row["face_b"])
        }
        positive_endpoint_b_bins = Counter(int(row["age_b"]) // bin_years for row in positive_rows)
        negative_endpoint_b_bins = Counter(row["endpoint_b_age_bin"] for row in negatives)
        arm_stats[arm] = {
            "target_unique_identities": target_identities,
            "excluded_cross_stratum_eligible_identities": len(cross_stratum_people),
            "positive_rows_removed_for_insufficient_endpoint_b_bin_support": len(
                candidates_before_b_filter[arm]
            )
            - len(candidates_by_arm[arm]),
            "eligible_identities_before_endpoint_b_bin_filter": len(
                {str(row["_person_id"]) for row in candidates_before_b_filter[arm]}
            ),
            "eligible_identities_after_endpoint_b_bin_filter": len(
                {str(row["_person_id"]) for row in candidates_by_arm[arm]}
            ),
            "arm_impostor_pool_unique_faces": len(arm_pool),
            "train_positive_count": len(positive_rows),
            "train_negative_count": len(negatives),
            "positive_unique_faces": len(
                {face for row in positive_rows for face in (row["face_a"], row["face_b"])}
            ),
            "positive_unique_persons": len({str(row["_person_id"]) for row in selected}),
            "train_age_only_class_auc": cue_auc,
            "positive_endpoint_b_age_bins": {
                str(k): v for k, v in sorted(positive_endpoint_b_bins.items())
            },
            "negative_endpoint_b_age_bins": {
                str(k): v for k, v in sorted(negative_endpoint_b_bins.items())
            },
            **diagnostics,
        }

    if _stable_jsonl_digest([r for r in arms["LOW"] if r.get("split") in {"val", "test"}]) != _stable_jsonl_digest(heldout):
        raise AssertionError("LOW arm changed canonical validation/test rows")
    if _stable_jsonl_digest([r for r in arms["CROSS"] if r.get("split") in {"val", "test"}]) != _stable_jsonl_digest(heldout):
        raise AssertionError("CROSS arm changed canonical validation/test rows")
    if selected_people_by_arm["LOW"] & selected_people_by_arm["CROSS"]:
        raise AssertionError("positive identity sets overlap across LOW and CROSS arms")
    if selected_faces_by_arm["LOW"] & selected_faces_by_arm["CROSS"]:
        raise AssertionError("positive image sets overlap across LOW and CROSS arms")

    diagnostics = {
        "target_per_class_per_arm": target_per_class,
        "target_unique_identities_per_arm": target_identities,
        "target_unique_positive_faces_per_arm": target_faces,
        "age_bin_years": bin_years,
        "common_train_positive_impostor_pool_unique_faces": len(common_pool),
        "common_impostor_pool_unique_persons": len({e["person_id"] for e in common_pool}),
        "common_pool_age_conflict_faces_majority_resolved": age_conflicts,
        "cross_stratum_eligible_identity_exclusions": len(cross_stratum_people),
        "minimum_other_identity_support_per_endpoint_b_bin": min_bin_support,
        "negative_endpoint_b_max_exact_age_error_years": 1,
        "endpoint_b_bin_identity_support": b_bin_support_counts,
        "supported_endpoint_b_bins": {
            arm: sorted(bins) for arm, bins in supported_b_bins.items()
        },
        "canonical_heldout_rows": len(heldout),
        "canonical_heldout_sha256": _stable_jsonl_digest(heldout),
        "arms": arm_stats,
        "cross_arm_negative_pair_overlap": len(
            {
                tuple(sorted((row["face_a"], row["face_b"])))
                for row in arms["LOW"]
                if row.get("pair_type") == "negative_matched_source_impostor"
            }
            & {
                tuple(sorted((row["face_a"], row["face_b"])))
                for row in arms["CROSS"]
                if row.get("pair_type") == "negative_matched_source_impostor"
            }
        ),
        "cross_arm_negative_face_overlap": len(
            {
                face
                for row in arms["LOW"]
                if row.get("pair_type") == "negative_matched_source_impostor"
                for face in (row["face_a"], row["face_b"])
            }
            & {
                face
                for row in arms["CROSS"]
                if row.get("pair_type") == "negative_matched_source_impostor"
                for face in (row["face_a"], row["face_b"])
            }
        ),
        "positive_identity_overlap_between_arms": len(
            selected_people_by_arm["LOW"] & selected_people_by_arm["CROSS"]
        ),
        "positive_image_overlap_between_arms": len(
            selected_faces_by_arm["LOW"] & selected_faces_by_arm["CROSS"]
        ),
        "covariates_not_matched": [
            "endpoint A age and pairwise age gap for negatives",
            "gender, image quality, pose/blur, platform, capture date and post context",
            "positive identity frequencies beyond greedy face/person coverage balancing",
        ],
        "generic_face_source": "not available in this construction; no generic source arm created",
    }
    return arms, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(str(data_path("data_dir", "interim"))) / "matched_agegap_arms")
    parser.add_argument("--target-per-class", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--target-identities", type=int, default=DEFAULT_IDENTITIES)
    parser.add_argument("--target-faces", type=int, default=DEFAULT_FACES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--age-bin-years", type=int, default=AGE_BIN_YEARS)
    parser.add_argument("--min-bin-support", type=int, default=MIN_B_BIN_IDENTITY_SUPPORT)
    args = parser.parse_args()

    pairs_path = data_path("data_dir", "processed", "pairs.jsonl")
    clusters_path = data_path("data_dir", "processed", "person_clusters.jsonl")
    pairs = list(read_jsonl(pairs_path))
    group_to_person = {
        str(row["identity_group_id"]): str(row["person_id"])
        for row in read_jsonl(clusters_path)
    }
    arms, summary = build_arms(
        pairs,
        group_to_person,
        target_identities=args.target_identities,
        target_per_class=args.target_per_class,
        target_faces=args.target_faces,
        seed=args.seed,
        bin_years=args.age_bin_years,
        min_bin_support=args.min_bin_support,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    low_path = args.output_dir / "low_arm.jsonl"
    cross_path = args.output_dir / "cross_arm.jsonl"
    write_jsonl(low_path, arms["LOW"])
    write_jsonl(cross_path, arms["CROSS"])
    manifest_path = args.output_dir / "manifest.json"
    write_experiment_manifest(
        manifest_path,
        experiment="matched-low-vs-cross-agegap-train-arms",
        parameters={
            "seed": args.seed,
            "target_unique_identities_per_arm": args.target_identities,
            "target_positive_and_negative_pairs_per_arm": args.target_per_class,
            "target_unique_positive_images_per_arm": args.target_faces,
            "positive_age_gap_years": {"LOW": [1, 2], "CROSS": [25, None]},
            "negative_endpoint_b_age_bin_years": args.age_bin_years,
            "minimum_arm_identity_support_per_endpoint_b_age_bin": args.min_bin_support,
            "negative_pool": "unique age-known face endpoints from all canonical training positives; person IDs differ",
            "positive_selection": "exclude identities eligible in both strata; seeded exact-profile selection by identity count, positive-pair count and unique image count",
            "negative_selection": "each positive face_a anchors its negative; endpoint B comes from that arm's selected positive identity/image pool, within 1 year of target positive endpoint-B age, distinct person, exact age preferred, minimum-reuse randomized tie break",
            "canonical_writes": False,
            "generic_face_source_available": False,
        },
        metrics=summary,
        inputs=[pairs_path, clusters_path],
        outputs=[low_path, cross_path],
    )
    print(
        json.dumps(
            {
                "LOW": summary["arms"]["LOW"],
                "CROSS": summary["arms"]["CROSS"],
                "canonical_heldout_rows": summary["canonical_heldout_rows"],
                "manifest": manifest_path.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
