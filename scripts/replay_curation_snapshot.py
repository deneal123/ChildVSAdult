"""Shadow dedup replay: never overwrite historical inputs or assert original provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import write_experiment_manifest
from age_gap.common.schemas import IdentityGroup
from age_gap.datasets.dedup import find_redundant_faces

NOISY = {"multi_person", "collage", "meme"}
PRE = "data/processed/identity_groups.jsonl.pre_prune_bak"
EMBEDDINGS = "cache/embeddings/baseline_arcface.npz"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_reference_inputs(root, manifest):
    records = manifest.get("inputs", [])
    for relative in (PRE, EMBEDDINGS):
        matching = [r for r in records if r.get("path") == relative]
        path = root / relative
        if len(matching) != 1 or path.stat().st_size != matching[0].get("bytes"):
            raise ValueError("Reference input count/size mismatch")
        if digest(path) != matching[0].get("sha256"):
            raise ValueError("Reference input checksum mismatch")


def validate_embeddings(ids, matrix):
    if ids.ndim != 1 or matrix.ndim != 2 or len(ids) != len(matrix) or not matrix.shape[1]:
        raise ValueError("Embedding dimensions do not match face IDs")
    if any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("Invalid embedding face IDs")
    if not np.issubdtype(matrix.dtype, np.floating) or not np.isfinite(matrix).all():
        raise ValueError("Embeddings must be finite floating vectors")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=0.002, rtol=0):
        raise ValueError("Cosine replay requires unit-normalized vectors")
    embeddings = {}
    for face_id, vector in zip(ids, matrix, strict=True):
        if face_id in embeddings and not np.array_equal(embeddings[face_id], vector):
            raise ValueError("Conflicting duplicate embedding face IDs")
        embeddings[face_id] = vector
    return embeddings


def compare_replay(pre, final, redundant):
    def keyed(rows):
        result = {r["identity_group_id"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate group IDs")
        if any(len(set(r["faces"])) != len(r["faces"]) for r in rows):
            raise ValueError("Duplicate faces within a group")
        return result

    before, after = keyed(pre), keyed(final)
    if set(redundant) - set(before):
        raise ValueError("Unknown dedup group")
    for key, faces in redundant.items():
        if set(faces) - set(before[key]["faces"]):
            raise ValueError("Unknown redundant face")
    noisy = {k for k, r in before.items() if r.get("identity_review") in NOISY}
    expected_ids = set(before) - noisy
    missing, extra = expected_ids - set(after), set(after) - expected_ids
    set_mismatch = order_only = metadata = removed = 0
    expected_unique = set()
    for key in expected_ids:
        original = before[key]
        red = set(redundant.get(key, []))
        expected = [f for f in original["faces"] if f not in red]
        removed += len(original["faces"]) - len(expected)
        expected_unique.update(expected)
        if key not in after:
            continue
        actual = after[key]
        if set(expected) != set(actual["faces"]):
            set_mismatch += 1
        elif expected != actual["faces"]:
            order_only += 1
        if {k: v for k, v in original.items() if k != "faces"} != {
            k: v for k, v in actual.items() if k != "faces"
        }:
            metadata += 1
    return {
        "pre_groups": len(before), "expected_retained_groups": len(expected_ids),
        "actual_retained_groups": len(after), "noisy_groups_dropped": len(noisy),
        "missing_group_count": len(missing), "extra_group_count": len(extra),
        "face_set_mismatched_groups": set_mismatch, "face_order_only_mismatches": order_only,
        "metadata_mismatched_groups": metadata, "dedup_removed_from_retained_groups": removed,
        "all_pre_groups_redundant_faces": sum(len(v) for v in redundant.values()),
        "expected_retained_unique_faces": len(expected_unique),
        "actual_retained_unique_faces": len({f for r in final for f in r["faces"]}),
        "ordered_faces_replayed": not (missing or extra or set_mismatch or order_only),
        "full_group_records_replayed": not (missing or extra or set_mismatch or order_only or metadata),
        "scope": "new deterministic reconstruction; original dedup file/provenance not recovered",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "metrics/curation_replay_20261003")
    args = parser.parse_args()
    out = args.out.resolve()
    if not any(out.is_relative_to(PROJECT_ROOT / name) for name in ("metrics", ".work")):
        raise ValueError("Output must be a fresh metrics/ or .work/ subdirectory")
    if out in (PROJECT_ROOT / "metrics", PROJECT_ROOT / ".work") or out.exists():
        raise ValueError("Refusing existing or root output directory")
    reference = PROJECT_ROOT / "metrics/sensitivity_dedup.manifest.json"
    final_path = PROJECT_ROOT / "data/processed/identity_groups.jsonl"
    inputs = [PROJECT_ROOT / PRE, PROJECT_ROOT / EMBEDDINGS, final_path, reference,
              Path(__file__), PROJECT_ROOT / "src/age_gap/datasets/dedup.py",
              PROJECT_ROOT / "src/age_gap/datasets/person_clusters.py",
              PROJECT_ROOT / "src/age_gap/common/schemas.py"]
    before = [digest(p) for p in inputs]
    verify_reference_inputs(PROJECT_ROOT, json.loads(reference.read_text(encoding="utf-8")))
    print("Reference SHA/size verified; loading local cache", flush=True)
    # This pre-existing object-ID cache is loaded only after its reference checksum is verified.
    with np.load(PROJECT_ROOT / EMBEDDINGS, allow_pickle=True) as archive:
        embeddings = validate_embeddings(archive["face_ids"], archive["embeddings"])
        cache_rows = len(archive["face_ids"])
    pre, final = list(read_jsonl(PROJECT_ROOT / PRE)), list(read_jsonl(final_path))
    groups = [IdentityGroup.from_dict(row) for row in pre]
    redundant = find_redundant_faces(groups, embeddings, threshold=0.97)
    result = compare_replay(pre, final, redundant)
    all_faces = {f for row in pre for f in row["faces"]}
    result.update(pre_faces=len(all_faces), embedding_unique_ids=len(embeddings),
                  embedding_cache_rows=cache_rows, identical_duplicate_extra_rows=cache_rows - len(embeddings),
                  pre_faces_without_embeddings=len(all_faces - set(embeddings)))
    if before != [digest(p) for p in inputs]:
        raise ValueError("Replay inputs changed during computation")
    out.mkdir(parents=True)
    path = out / "summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        out / "summary.manifest.json", experiment="shadow-curation-snapshot-replay",
        parameters={"threshold": 0.97, "mode": "read-only shadow", "seed": None},
        metrics=result, inputs=inputs, outputs=[path],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
