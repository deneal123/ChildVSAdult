"""Extend known-noisy sensitivity with unresolved coverage and explicit denominators."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import write_experiment_manifest
from scripts.audit_constituent_integrity import audit
from scripts.replay_curation_snapshot import digest


def sensitivity(post, mapping, pre, final, cache, pairs):
    base, _ = audit(post, mapping, pre, final, cache, pairs)
    to_person = {r["identity_group_id"]: r["person_id"] for r in mapping}
    categories = {r["post_id"]: r["category"] for r in cache}
    members = defaultdict(list)
    for row in post:
        members[to_person.get(row["identity_group_id"], row["identity_group_id"])].append(row["source_post_id"])
    complete_single = {k for k, values in members.items() if all(categories.get(p) == "single" for p in values)}
    groups = {r["identity_group_id"]: r for r in final}
    positive_groups = set()
    strict_pairs = Counter()
    for pair in pairs:
        a, b = pair["identity_group_a"], pair["identity_group_b"]
        if pair["label"] == 1:
            if a != b:
                raise ValueError("Clean positive pair has different recorded-person groups")
            positive_groups.add(a)
        elif a == b:
            raise ValueError("Negative pair has the same recorded-person group")
        key = f"{pair['split']}_{'positive' if pair['label'] == 1 else 'negative'}"
        if a in complete_single and b in complete_single:
            strict_pairs[key] += 1
    rep_counts = Counter(groups[k].get("identity_review") or "missing" for k in positive_groups)
    total = len(positive_groups)
    cached_total = total - rep_counts["missing"]
    base.update(
        positive_groups=total,
        positive_group_representative_categories=dict(sorted(rep_counts.items())),
        positive_group_single_fraction_all=rep_counts["single"] / total if total else None,
        positive_group_single_fraction_cached=rep_counts["single"] / cached_total if cached_total else None,
        retained_groups_unresolved_after_known_noisy_exclusion=(len(final) - base["retained_groups_with_noisy_constituent"] - len(complete_single & set(groups))),
        pair_counts_after_all_cached_single_filter={k: strict_pairs[k] for k in base["original_pair_counts"]},
        policy_scope="known-noisy exclusion does not certify missing/unknown as clean; all-cached-single is a separate automatic sensitivity arm",
    )
    return base


def main():
    out = PROJECT_ROOT / "metrics/constituent_policy_sensitivity_20261003"
    if out.exists():
        raise ValueError("Fresh output required")
    base = PROJECT_ROOT / "data/processed"
    paths = [base / "identity_groups.jsonl.post_bak", base / "person_clusters.jsonl",
             base / "identity_groups.jsonl.pre_prune_bak", base / "identity_groups.jsonl",
             PROJECT_ROOT / "data/interim/group_validation_cache.jsonl", base / "pairs.jsonl"]
    reference = PROJECT_ROOT / "metrics/group_integrity_audit.manifest.json"
    sources = [Path(__file__), PROJECT_ROOT / "scripts/audit_constituent_integrity.py",
               PROJECT_ROOT / "scripts/replay_curation_snapshot.py"]
    inputs = paths + [reference] + sources
    before = [digest(p) for p in inputs]
    records = json.loads(reference.read_text(encoding="utf-8"))["inputs"]
    matches = [r for r in records if r["path"] == "data/interim/group_validation_cache.jsonl"]
    if len(matches) != 1 or digest(paths[4]) != matches[0]["sha256"] or paths[4].stat().st_size != matches[0]["bytes"]:
        raise ValueError("Integrity cache reference mismatch")
    cache = [{"post_id": r["post_id"], "category": r.get("category")} for r in read_jsonl(paths[4])]
    result = sensitivity(*(list(read_jsonl(p)) for p in paths[:4]), cache, list(read_jsonl(paths[5])))
    if before != [digest(p) for p in inputs]:
        raise ValueError("Sensitivity inputs changed")
    out.mkdir(parents=True)
    summary = out / "summary.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        out / "summary.manifest.json", experiment="constituent-integrity-sensitivity",
        parameters={"mode": "read-only", "policies": ["known noisy", "all cached single"], "seed": None},
        metrics=result, inputs=inputs, outputs=[summary],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
