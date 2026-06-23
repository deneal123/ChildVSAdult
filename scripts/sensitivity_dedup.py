"""Dedup-threshold sensitivity of the curation finding (valid, cheap).

The stored identity_groups.jsonl is ALREADY curated (post-dedup), so re-running
dedup on it finds ~0 near-duplicates. To measure sensitivity we run the dedup on
the PRE-curation groups (identity_groups.jsonl.pre_prune_bak) at several cosine
thresholds and report the near-duplicate faces removed and the resulting
positive-pair count (positives are quadratic in faces per group). No retraining
and no mutation of the canonical processed state.

    uv run python scripts/sensitivity_dedup.py
"""

from __future__ import annotations

import json
from math import comb

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.schemas import IdentityGroup
from age_gap.datasets.dedup import find_redundant_faces
from age_gap.models.embeddings import load_embeddings

THRESHOLDS = [0.93, 0.95, 0.97, 0.98, 0.99]


def main() -> None:
    pre = str(data_path("data_dir", "processed", "identity_groups.jsonl.pre_prune_bak"))
    groups = [IdentityGroup.from_dict(r) for r in read_jsonl(pre)]
    emb = load_embeddings(None)

    def positives(red: dict[str, list[str]]) -> int:
        total = 0
        for g in groups:
            n = len([f for f in g.faces if f in emb])
            nr = len(red.get(g.identity_group_id, []))
            total += comb(max(n - nr, 0), 2)
        return total

    raw = positives({})
    out: dict = {"no_dedup_positives": raw, "thresholds": {}}
    print(f"no-dedup positives={raw}")
    for t in THRESHOLDS:
        red = find_redundant_faces(groups, emb, t)
        nred = sum(len(v) for v in red.values())
        pos = positives(red)
        red_pct = round(100.0 * (1.0 - pos / raw), 1)
        out["thresholds"][f"{t}"] = {"redundant_faces": nred, "positives": pos, "reduction_pct": red_pct}
        print(f"thr={t}: redundant_faces={nred} positives={pos} reduction={red_pct}%")

    dst = resolve_path("docs", "sensitivity_dedup.json")
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
