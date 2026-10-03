"""Read-only aggregate reconstruction of historical group coverage and pruning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import write_experiment_manifest

NOISY = {"multi_person", "collage", "meme"}


def summarize(post, mapping, pre, final, redundant):
    def keyed(rows, key):
        result = {row[key]: row for row in rows}
        if len(result) != len(rows):
            raise ValueError(f"Duplicate {key} records")
        return result

    posts = keyed(post, "identity_group_id")
    mapped = keyed(mapping, "identity_group_id")
    before = keyed(pre, "identity_group_id")
    after = keyed(final, "identity_group_id")
    if set(mapped) - set(posts):
        raise ValueError("Mapping contains unknown post groups")
    expected = {mapped[k]["person_id"] if k in mapped else k for k in posts}
    if expected != set(before):
        raise ValueError("Mapped plus fallback groups do not match pre-prune IDs")
    noisy = {k for k, row in before.items() if row.get("identity_review") in NOISY}
    if set(after) != set(before) - noisy:
        raise ValueError("Final group IDs do not match integrity pruning")
    red = {row["face_id"] for row in redundant}
    set_mismatch = order_only = 0
    for key, row in after.items():
        if set(row["faces"]) - set(before[key]["faces"]):
            raise ValueError("Final group contains faces absent from its pre-prune record")
        expected_faces = [f for f in before[key]["faces"] if f not in red]
        if set(row["faces"]) != set(expected_faces):
            set_mismatch += 1
        elif row["faces"] != expected_faces:
            order_only += 1
    missing = set(posts) - set(mapped)
    def low_consistency(rows):
        return sum(
            len(row["faces"]) > 1 and row.get("gender_consistency") is not None
            and row["gender_consistency"] < 0.6 for row in rows
        )
    return {
        "post_groups": len(posts),
        "mapped_post_groups": len(mapped),
        "mapped_person_clusters": len({row["person_id"] for row in mapping}),
        "unmapped_fallback_groups": len(missing),
        "unmapped_fallback_faces": sum(len(posts[k]["faces"]) for k in missing),
        "pre_prune_groups": len(before),
        "noisy_groups_dropped": len(noisy),
        "retained_groups": len(after),
        "pre_prune_faces": sum(len(row["faces"]) for row in pre),
        "faces_in_dropped_noisy_groups": sum(len(before[k]["faces"]) for k in noisy),
        "dedup_faces_removed_from_retained_groups": sum(
            len(before[k]["faces"]) - len(after[k]["faces"]) for k in after
        ),
        "retained_unique_faces": len({f for row in final for f in row["faces"]}),
        "low_consistency_multiface_groups_pre_prune": low_consistency(pre),
        "low_consistency_multiface_groups_retained": low_consistency(final),
        "current_redundant_file_replays_final_faces": set_mismatch == order_only == 0,
        "dedup_face_set_mismatched_groups": set_mismatch,
        "dedup_face_order_only_mismatches": order_only,
        "scope": "record lineage only; no true-person or historical detector validation",
    }


def main():
    base = PROJECT_ROOT / "data/processed"
    names = ["identity_groups.jsonl.post_bak", "person_clusters.jsonl",
             "identity_groups.jsonl.pre_prune_bak", "identity_groups.jsonl",
             "redundant_faces.jsonl"]
    inputs = [base / name for name in names] + [Path(__file__)]
    def digests():
        return [hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]
    before = digests()
    metrics = summarize(*(list(read_jsonl(p)) for p in inputs[:5]))
    if before != digests():
        raise ValueError("Lineage inputs changed during audit")
    out = PROJECT_ROOT / "metrics/curation_lineage_20261003"
    out.mkdir(parents=True, exist_ok=True)
    result = out / "summary.json"
    result.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        out / "summary.manifest.json", experiment="historical-curation-lineage",
        parameters={"mode": "read-only", "noisy_categories": sorted(NOISY)},
        metrics=metrics, inputs=inputs, outputs=[result],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
