"""Replay the actual global-face-ID cleanup policy in a fresh shadow directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import write_experiment_manifest
from age_gap.common.schemas import IdentityGroup
from age_gap.datasets.dedup import find_redundant_faces
from scripts.replay_curation_snapshot import (
    EMBEDDINGS,
    PRE,
    compare_replay,
    digest,
    validate_embeddings,
    verify_reference_inputs,
)


def compare_global_replay(pre, final, redundant):
    # clean.prune_groups removes every globally listed face ID from every retained group.
    all_redundant = {f for values in redundant.values() for f in values}
    known_ids = {row["identity_group_id"] for row in pre}
    if set(redundant) - known_ids:
        raise ValueError("Unknown original dedup group")
    for row in pre:
        if set(redundant.get(row["identity_group_id"], [])) - set(row["faces"]):
            raise ValueError("Unknown original redundant face")
    expanded = {row["identity_group_id"]: [f for f in row["faces"] if f in all_redundant]
                for row in pre}
    result = compare_replay(pre, final, expanded)
    result["globally_removed_pre_face_memberships"] = result.pop("all_pre_groups_redundant_faces")
    result["detected_redundant_face_memberships"] = sum(len(v) for v in redundant.values())
    result["detected_redundant_unique_ids"] = len(all_redundant)
    for name, rows in (("pre", pre), ("final", final)):
        counts = Counter(f for row in rows for f in row["faces"])
        result[f"{name}_cross_group_shared_face_ids"] = sum(v > 1 for v in counts.values())
        result[f"{name}_empty_groups"] = sum(not row["faces"] for row in rows)
    result["cleanup_policy"] = "global redundant face-ID set, including shared memberships"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "metrics/curation_replay_global_20261003")
    out = parser.parse_args().out.resolve()
    if not any(out.is_relative_to(PROJECT_ROOT / name) for name in ("metrics", ".work")):
        raise ValueError("Output outside permitted shadow directories")
    if out in (PROJECT_ROOT / "metrics", PROJECT_ROOT / ".work") or out.exists():
        raise ValueError("Output must be a fresh subdirectory")
    reference = PROJECT_ROOT / "metrics/sensitivity_dedup.manifest.json"
    final_path = PROJECT_ROOT / "data/processed/identity_groups.jsonl"
    sources = ["scripts/replay_curation_snapshot.py", "src/age_gap/datasets/dedup.py",
               "src/age_gap/datasets/clean.py", "src/age_gap/datasets/person_clusters.py",
               "src/age_gap/common/schemas.py"]
    inputs = [PROJECT_ROOT / PRE, PROJECT_ROOT / EMBEDDINGS, final_path, reference,
              Path(__file__)] + [PROJECT_ROOT / name for name in sources]
    before = [digest(path) for path in inputs]
    verify_reference_inputs(PROJECT_ROOT, json.loads(reference.read_text(encoding="utf-8")))
    print("Verified reference inputs; global-ID shadow replay", flush=True)
    with np.load(PROJECT_ROOT / EMBEDDINGS, allow_pickle=True) as archive:
        embeddings = validate_embeddings(archive["face_ids"], archive["embeddings"])
        cache_rows = len(archive["face_ids"])
    pre, final = list(read_jsonl(PROJECT_ROOT / PRE)), list(read_jsonl(final_path))
    redundant = find_redundant_faces([IdentityGroup.from_dict(r) for r in pre], embeddings, 0.97)
    result = compare_global_replay(pre, final, redundant)
    result.update(embedding_cache_rows=cache_rows, embedding_unique_ids=len(embeddings),
                  identical_duplicate_extra_rows=cache_rows - len(embeddings),
                  pre_faces_without_embeddings=len({f for r in pre for f in r["faces"]} - set(embeddings)))
    if before != [digest(path) for path in inputs]:
        raise ValueError("Global replay inputs changed")
    out.mkdir(parents=True)
    summary = out / "summary.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        out / "summary.manifest.json", experiment="shadow-global-id-curation-replay",
        parameters={"threshold": 0.97, "policy": "global redundant face IDs", "seed": None},
        metrics=result, inputs=inputs, outputs=[summary],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
