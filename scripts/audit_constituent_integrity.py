"""Read-only constituent-caption policy sensitivity, not human label validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import write_experiment_manifest
from scripts.replay_curation_snapshot import digest

CATEGORIES = {"single", "multi_person", "collage", "meme", "unknown"}
NOISY = {"multi_person", "collage", "meme"}


def audit(post_groups, mapping, pre_groups, final_groups, cache, pairs):
    def keyed(rows, key):
        result = {r[key]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError(f"Duplicate {key} records")
        return result

    post = keyed(post_groups, "identity_group_id")
    mapped = keyed(mapping, "identity_group_id")
    pre = keyed(pre_groups, "identity_group_id")
    final = keyed(final_groups, "identity_group_id")
    cached = keyed(cache, "post_id")
    if any(r.get("category") not in CATEGORIES for r in cache):
        raise ValueError("Invalid cached category; missing is not automatically single")
    if set(mapped) - set(post):
        raise ValueError("Mapping contains unknown candidate groups")
    members = defaultdict(list)
    for key, row in post.items():
        members[mapped[key]["person_id"] if key in mapped else key].append(row)
    if set(members) != set(pre) or set(final) - set(pre):
        raise ValueError("Constituent/pre/final group coverage mismatch")
    mismatch = rep_label_mismatch = 0
    all_noisy, retained_affected, complete_single = set(), set(), set()
    candidates = []
    counts = Counter()
    category_matrix = Counter()
    for group_id, rows in members.items():
        group = pre[group_id]
        combined = list(dict.fromkeys(f for r in rows for f in r["faces"]))
        if combined != group["faces"] or group["source_post_id"] != rows[0]["source_post_id"]:
            mismatch += 1
        categories = [cached.get(r["source_post_id"], {}).get("category", "missing") for r in rows]
        rep = cached.get(group["source_post_id"], {}).get("category", "missing")
        if group.get("identity_review") != (None if rep == "missing" else rep):
            rep_label_mismatch += 1
        noisy_posts = [r["source_post_id"] for r, c in zip(rows, categories, strict=True) if c in NOISY]
        if noisy_posts:
            all_noisy.add(group_id)
        if all(c == "single" for c in categories):
            complete_single.add(group_id)
        counts["merged_groups" if len(rows) > 1 else "singleton_groups"] += 1
        counts["constituent_posts"] += len(rows)
        counts["missing_constituent_posts"] += categories.count("missing")
        counts["unknown_constituent_posts"] += categories.count("unknown")
        counts["groups_with_missing_constituents"] += "missing" in categories
        counts["groups_with_unknown_constituents"] += "unknown" in categories
        category_matrix[(rep, "any_noisy" if noisy_posts else "no_noisy")] += 1
        if group_id in final and noisy_posts:
            retained_affected.add(group_id)
            candidates.append({"identity_group_id": group_id, "member_count": len(rows),
                               "representative_category": rep,
                               "noisy_constituent_post_ids": noisy_posts})
    if mismatch or rep_label_mismatch:
        raise ValueError("Saved constituent order/faces or representative labels do not replay")
    expected_retained = {key for key, g in pre.items() if g.get("identity_review") not in NOISY}
    if set(final) != expected_retained:
        raise ValueError("Representative-based final group IDs do not replay")
    before_pairs, retained_pairs = Counter(), Counter()
    seen = set()
    for pair in pairs:
        if pair["pair_id"] in seen:
            raise ValueError("Duplicate pair IDs")
        seen.add(pair["pair_id"])
        if pair["label"] not in (0, 1) or pair.get("split") not in ("train", "val", "test"):
            raise ValueError("Unexpected pair label/split")
        a, b = pair.get("identity_group_a"), pair.get("identity_group_b")
        if a not in final or b not in final:
            raise ValueError("Pair endpoint group absent from final snapshot")
        key = f"{pair['split']}_{'positive' if pair['label'] == 1 else 'negative'}"
        before_pairs[key] += 1
        if a not in retained_affected and b not in retained_affected:
            retained_pairs[key] += 1
    summary = {
        "pre_groups": len(pre), "retained_groups": len(final),
        "constituent_coverage": dict(counts),
        "representative_by_constituent_noisy": {f"{a}|{b}": n for (a, b), n in sorted(category_matrix.items())},
        "any_constituent_noisy_pre_groups": len(all_noisy),
        "retained_groups_with_noisy_constituent": len(retained_affected),
        "retained_unique_faces_in_affected_groups": len({f for k in retained_affected for f in final[k]["faces"]}),
        "all_constituents_cached_single_pre_groups": len(complete_single),
        "all_constituents_cached_single_retained_groups": len(complete_single & set(final)),
        "original_pair_counts": dict(sorted(before_pairs.items())),
        "pair_counts_after_any_noisy_group_exclusion": {k: retained_pairs[k] for k in sorted(before_pairs)},
        "scope": "cached automatic categories; sensitivity only, no true-identity purity or retrained effect",
    }
    return summary, candidates


def main():
    out = PROJECT_ROOT / "metrics/constituent_integrity_20261003"
    if out.exists():
        raise ValueError("Fresh output required")
    base = PROJECT_ROOT / "data/processed"
    data_inputs = [base / "identity_groups.jsonl.post_bak", base / "person_clusters.jsonl",
                   base / "identity_groups.jsonl.pre_prune_bak", base / "identity_groups.jsonl",
                   PROJECT_ROOT / "data/interim/group_validation_cache.jsonl", base / "pairs.jsonl"]
    reference = PROJECT_ROOT / "metrics/group_integrity_audit.manifest.json"
    inputs = data_inputs + [reference, Path(__file__), PROJECT_ROOT / "src/age_gap/datasets/person_clusters.py",
                            PROJECT_ROOT / "scripts/validate_groups.py"]
    before = [digest(p) for p in inputs]
    records = json.loads(reference.read_text(encoding="utf-8"))["inputs"]
    matched = [r for r in records if r["path"] == "data/interim/group_validation_cache.jsonl"]
    if len(matched) != 1 or matched[0]["sha256"] != digest(data_inputs[4]) or matched[0]["bytes"] != data_inputs[4].stat().st_size:
        raise ValueError("Cached integrity decisions differ from the reference artifact")
    # Only post_id/category retained; captions, raw responses and images are never output.
    cache = [{"post_id": r["post_id"], "category": r.get("category")} for r in read_jsonl(data_inputs[4])]
    summary, candidates = audit(*(list(read_jsonl(p)) for p in data_inputs[:4]), cache,
                                list(read_jsonl(data_inputs[5])))
    if before != [digest(p) for p in inputs]:
        raise ValueError("Audit inputs changed")
    (out / "private").mkdir(parents=True)
    result, private = out / "summary.json", out / "private/candidates.jsonl"
    result.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    private.write_text("".join(json.dumps(r) + "\n" for r in candidates), encoding="utf-8")
    write_experiment_manifest(
        out / "summary.manifest.json", experiment="constituent-integrity-policy-audit",
        parameters={"mode": "read-only cached decisions", "policy": "exclude any cached noisy constituent", "seed": None},
        metrics=summary, inputs=inputs, outputs=[result, private],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
