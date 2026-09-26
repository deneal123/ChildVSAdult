"""Create the held-out, model-blinded annotation pack requested by reviewers.

The pack stays local. Source crops are copied under opaque filenames so the
annotator files expose neither corpus IDs nor original paths. Two annotator
files contain the same task IDs in independently shuffled order; automatic
decisions and source mappings are kept only in a separate key unavailable to
annotators. No item from the pack should be used to tune thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from age_gap.common.io import data_path, read_jsonl, resolve_path, write_jsonl
from age_gap.common.manifest import sha256_file, write_experiment_manifest
from age_gap.models.embeddings import load_embeddings


def _task_id(kind: str, evidence_id: str) -> str:
    digest = hashlib.sha256(f"{kind}|{evidence_id}".encode()).hexdigest()[:16]
    return f"{kind}-{digest}"


def _crop(face_id: str) -> str:
    return str(resolve_path("data", "interim", "faces", f"{face_id}.jpg"))


def _sample_disjoint(
    rows: list[dict],
    n: int,
    rng: np.random.Generator,
    used_images: set[str],
    *,
    shuffle: bool = True,
) -> list[dict]:
    """Sample tasks without reusing any source crop across audit strata."""
    selected: list[dict] = []
    indices = rng.permutation(len(rows)) if shuffle else range(len(rows))
    for index in indices:
        row = rows[int(index)]
        images = set(row.get("images", []))
        if not images or images.intersection(used_images):
            continue
        selected.append(row)
        used_images.update(images)
        if len(selected) == n:
            return selected
    raise RuntimeError(
        f"audit stratum has only {len(selected)} image-disjoint candidates; requested {n}"
    )


def _sample_minimum_overlap(rows: list[dict], n: int, used_images: set[str]) -> list[dict]:
    """Prefer a matching, then fill an unavoidable shortfall with minimum-overlap pairs."""
    selected: list[dict] = []
    selected_ids: set[str] = set()
    usage: Counter[str] = Counter()
    # The caller orders boundary candidates by distance from the production threshold.
    # Preserve that order: shuffling here silently turned the audit into an arbitrary
    # one-sided sample after high-similarity faces had already been pruned.
    order = rows
    for row in order:
        images = set(row.get("images", []))
        if images and not images.intersection(used_images):
            selected.append(row)
            selected_ids.add(row["evidence_id"])
            used_images.update(images)
            usage.update(images)
            if len(selected) == n:
                return selected

    remaining = [row for row in rows if row["evidence_id"] not in selected_ids]
    remaining.sort(
        key=lambda row: (
            sum(usage[image] for image in set(row.get("images", []))),
            abs(float(row.get("similarity", 0.97)) - 0.97),
        )
    )
    for row in remaining:
        images = set(row.get("images", []))
        if not images:
            continue
        selected.append(row)
        used_images.update(images)
        usage.update(images)
        if len(selected) == n:
            return selected
    raise RuntimeError(f"audit stratum has only {len(selected)} candidates; requested {n}")


def _sample_balanced_disjoint(
    rows: list[dict], n: int, rng: np.random.Generator, used_images: set[str]
) -> list[dict]:
    """Sample both automatic binary decisions while preserving image disjointness."""
    labels = sorted({str(row["auto_label"]) for row in rows})
    if len(labels) != 2:
        raise RuntimeError(f"balanced audit requires two automatic labels, got {labels}")
    targets = {labels[0]: n // 2, labels[1]: n - n // 2}
    selected: list[dict] = []
    for label in labels:
        candidates = [row for row in rows if row["auto_label"] == label]
        selected.extend(_sample_disjoint(candidates, targets[label], rng, used_images))
    return selected


def _assert_no_frozen_responses(paths: list[Path]) -> None:
    """Refuse to replace a pack after either annotator has started working."""
    for path in paths:
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("response") is not None:
                raise RuntimeError(
                    f"refusing to overwrite frozen response in {path} at line {line_number}"
                )


def _materialize_opaque_images(
    tasks: list[dict], key: list[dict], output: Path, seed: int
) -> Path:
    """Copy task images under deterministic opaque names and hide source paths in the key."""
    image_dir = output / "images"
    resolved_output = output.resolve()
    resolved_images = image_dir.resolve()
    if resolved_images.parent != resolved_output:
        raise RuntimeError(f"unsafe audit image directory: {resolved_images}")
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True)

    key_by_task = {row["task_id"]: row for row in key}
    image_records: list[dict] = []
    for task in tasks:
        sources = [Path(item) for item in task.get("images", [])]
        opaque_paths: list[str] = []
        source_records: list[str] = []
        for index, source in enumerate(sources):
            if not source.is_file():
                raise FileNotFoundError(source)
            suffix = source.suffix.lower() if source.suffix else ".jpg"
            digest = hashlib.sha256(f"{seed}|{task['task_id']}|{index}".encode()).hexdigest()[:24]
            destination = image_dir / f"{digest}{suffix}"
            shutil.copyfile(source, destination)
            relative = destination.relative_to(output).as_posix()
            opaque_paths.append(relative)
            source_records.append(str(source))
            image_records.append(
                {
                    "file": relative,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        task["images"] = opaque_paths
        task["notes"] = None
        key_by_task[task["task_id"]]["source_images"] = source_records

    image_manifest = output / "images.manifest.json"
    image_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "n_files": len(image_records),
                "files": image_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return image_manifest


def _sources() -> tuple[dict[str, dict], dict[str, dict], list[dict], list[dict]]:
    posts = {r["post_id"]: r for r in read_jsonl(data_path("data_dir", "raw", "posts.jsonl"))}
    groups = {
        r["identity_group_id"]: r
        for r in read_jsonl(data_path("data_dir", "processed", "identity_groups.jsonl"))
    }
    pairs = list(read_jsonl(data_path("data_dir", "processed", "pairs.jsonl")))
    clusters = list(read_jsonl(data_path("data_dir", "processed", "person_clusters.jsonl")))
    return posts, groups, pairs, clusters


def _dedup_candidates(groups: dict[str, dict]) -> list[dict]:
    embeddings = load_embeddings()
    candidates: list[dict] = []
    for group in groups.values():
        faces = [fid for fid in group.get("faces", []) if fid in embeddings]
        if len(faces) < 2:
            continue
        matrix = np.stack([embeddings[fid] for fid in faces])
        similarities = matrix @ matrix.T
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                similarity = float(similarities[i, j])
                if 0.90 <= similarity <= 0.995:
                    candidates.append(
                        {
                            "evidence_id": f"{faces[i]}|{faces[j]}",
                            "images": [_crop(faces[i]), _crop(faces[j])],
                            "auto_label": "near_duplicate"
                            if similarity >= 0.97
                            else "distinct_photo",
                            "similarity": similarity,
                        }
                    )
    candidates.sort(key=lambda row: abs(row["similarity"] - 0.97))
    return candidates


def build_pack(n: int, seed: int, output: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(seed)
    posts, groups, pairs, clusters = _sources()
    post_by_photo = {
        str(photo["photo_id"]): post
        for post in posts.values()
        for photo in post.get("photos", [])
    }
    tasks: list[dict] = []
    key: list[dict] = []
    used_images: set[str] = set()
    pre_prune_path = Path(
        str(data_path("data_dir", "processed", "identity_groups.jsonl.pre_prune_bak"))
    )
    pre_prune_groups = {
        row["identity_group_id"]: row for row in read_jsonl(pre_prune_path)
    }
    duplicate_rows = _sample_minimum_overlap(
        _dedup_candidates(pre_prune_groups), n, used_images
    )
    duplicate_labels = Counter(row["auto_label"] for row in duplicate_rows)
    minimum_per_side = max(1, n // 5)
    if (
        duplicate_labels["near_duplicate"] < minimum_per_side
        or duplicate_labels["distinct_photo"] < minimum_per_side
    ):
        raise RuntimeError(
            "near-duplicate audit must cover both sides of the 0.97 threshold; "
            f"got {dict(duplicate_labels)}"
        )

    age_candidates = []
    for group in groups.values():
        group_faces = list(group.get("faces", []))
        for label in group.get("age_labels", []):
            face_id = label.get("face_id")
            photo_id = str(face_id).rsplit("_f", 1)[0]
            source_post = post_by_photo.get(photo_id)
            if not source_post:
                continue
            ordered_photo_ids = [str(photo["photo_id"]) for photo in source_post.get("photos", [])]
            context_faces = sorted(
                (item for item in group_faces if item.rsplit("_f", 1)[0] in ordered_photo_ids),
                key=lambda item: ordered_photo_ids.index(item.rsplit("_f", 1)[0]),
            )
            if face_id in context_faces and label.get("age") is not None:
                age_candidates.append(
                    {
                        "evidence_id": f"{group['identity_group_id']}|{face_id}|{label['age']}",
                        "caption": source_post.get("caption", ""),
                        "images": [_crop(item) for item in context_faces],
                        "target_image_index": context_faces.index(face_id),
                        "auto_age": int(label["age"]),
                        "auto_source": str(label.get("source", "unknown")),
                    }
                )
    for row in _sample_disjoint(age_candidates, n, rng, used_images):
        tid = _task_id("age", row.pop("evidence_id"))
        auto_age = row.pop("auto_age")
        auto_source = row.pop("auto_source")
        tasks.append({"task_id": tid, "audit_type": "age_extraction", **row, "response": None})
        key.append(
            {
                "task_id": tid,
                "audit_type": "age_extraction",
                "auto_age": auto_age,
                "auto_source": auto_source,
            }
        )

    integrity_candidates = [
        {
            "evidence_id": gid,
            "caption": posts.get(group.get("source_post_id"), {}).get("caption", ""),
            "images": [_crop(fid) for fid in group["faces"][:8]],
            "auto_label": (
                "same_identity" if group.get("identity_review") == "single" else "different_identity"
            ),
        }
        for gid, group in pre_prune_groups.items()
        if len(group.get("faces", [])) >= 2
        and group.get("identity_review") in {"single", "multi_person", "collage", "meme"}
    ]
    for row in _sample_balanced_disjoint(integrity_candidates, n, rng, used_images):
        tid = _task_id("group", row.pop("evidence_id"))
        auto_label = row.pop("auto_label")
        tasks.append(
            {
                "task_id": tid,
                "audit_type": "group_integrity",
                **row,
                "allowed": ["same_identity", "different_identity", "uncertain"],
                "response": None,
            }
        )
        key.append({"task_id": tid, "audit_type": "group_integrity", "auto_label": auto_label})

    positive_candidates = [
        {
            "evidence_id": p["pair_id"],
            "images": [_crop(p["face_a"]), _crop(p["face_b"])],
            "auto_label": "same_identity",
        }
        for p in pairs
        if int(p["label"]) == 1
    ]
    for row in _sample_disjoint(positive_candidates, n, rng, used_images):
        tid = _task_id("positive", row.pop("evidence_id"))
        auto_label = row.pop("auto_label")
        tasks.append(
            {
                "task_id": tid,
                "audit_type": "positive_identity",
                **row,
                "allowed": ["same_identity", "different_identity", "uncertain"],
                "response": None,
            }
        )
        key.append({"task_id": tid, "audit_type": "positive_identity", "auto_label": auto_label})

    # Cross-post merges: sample pairs of original post-groups assigned to one person.
    original_groups = {
        r["identity_group_id"]: r
        for r in read_jsonl(
            data_path("data_dir", "processed", "identity_groups.jsonl.post_bak")
        )
    }
    by_person: dict[str, list[str]] = defaultdict(list)
    for row in clusters:
        by_person[row["person_id"]].append(row["identity_group_id"])
    merge_candidates = []
    for person, group_ids in by_person.items():
        valid = [g for g in group_ids if g in original_groups]
        for left, right in zip(valid, valid[1:], strict=False):
            merge_candidates.append(
                {
                    "evidence_id": f"{person}|{left}|{right}",
                    "images": [
                        *[_crop(fid) for fid in original_groups[left]["faces"][:4]],
                        *[_crop(fid) for fid in original_groups[right]["faces"][:4]],
                    ],
                    "auto_label": "same_identity",
                }
            )
    for row in _sample_disjoint(merge_candidates, n, rng, used_images):
        tid = _task_id("merge", row.pop("evidence_id"))
        auto_label = row.pop("auto_label")
        tasks.append(
            {
                "task_id": tid,
                "audit_type": "cluster_merge",
                **row,
                "allowed": ["same_identity", "different_identity", "uncertain"],
                "response": None,
            }
        )
        key.append({"task_id": tid, "audit_type": "cluster_merge", "auto_label": auto_label})

    # Borderline same-person face pairs around the production dedup threshold were reserved first,
    # so later strata cannot reuse their crops and create dependent judgments.
    for row in duplicate_rows:
        tid = _task_id("dedup", row.pop("evidence_id"))
        auto_label = row.pop("auto_label")
        similarity = row.pop("similarity")
        tasks.append(
            {
                "task_id": tid,
                "audit_type": "near_duplicate",
                **row,
                "allowed": ["near_duplicate", "distinct_photo", "uncertain"],
                "response": None,
            }
        )
        key.append(
            {
                "task_id": tid,
                "audit_type": "near_duplicate",
                "auto_label": auto_label,
                "cosine_similarity": similarity,
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    annotator_paths = [output / "annotator_a.jsonl", output / "annotator_b.jsonl"]
    _assert_no_frozen_responses(annotator_paths)
    image_manifest = _materialize_opaque_images(tasks, key, output, seed)
    for offset, path in enumerate(annotator_paths, 1):
        order = np.random.default_rng(seed + offset).permutation(len(tasks))
        write_jsonl(path, (tasks[i] for i in order))
    key_path = output / "audit_key.jsonl"
    write_jsonl(key_path, key)
    adjudication_template = output / "adjudicated_gold.template.jsonl"
    write_jsonl(
        adjudication_template,
        (
            {
                "task_id": task["task_id"],
                "audit_type": task["audit_type"],
                "response": None,
                "adjudication_notes": None,
            }
            for task in sorted(tasks, key=lambda row: row["task_id"])
        ),
    )
    protocol = output / "README.md"
    protocol.write_text(
        "# Blinded annotation protocol\n\n"
        "## Independence and blinding\n\n"
        "Work independently, do not discuss cases, and do not open `audit_key.jsonl`. Resolve image "
        "paths relative to this directory. Filenames are deliberately opaque; do not inspect file "
        "metadata or use reverse-image search. The sample was held out after all production "
        "thresholds were fixed and must never be used to tune them. Fill only `response` and, when "
        "useful, `notes`; do not reorder, add, or remove rows.\n\n"
        "## Labels\n\n"
        "- `age_extraction`: images are in source order. Enter the integer age that the caption "
        "explicitly assigns to `target_image_index` (zero-based), or `uncertain`. Never estimate age "
        "from appearance. If the caption gives a range, relative age, or ambiguous image mapping, "
        "choose `uncertain`.\n"
        "- `group_integrity`: decide whether every displayed face belongs to one identity. Use "
        "`different_identity` if any face is another person.\n"
        "- `positive_identity`: decide whether the two faces show the same person across time.\n"
        "- `cluster_merge`: decide whether all displayed faces from the two source groups can be "
        "merged into one identity. A single contradictory face means `different_identity`.\n"
        "- `near_duplicate`: use `near_duplicate` only for the same capture/frame or an effectively "
        "identical burst/repost. Two distinct photographs of the same person are `distinct_photo`.\n\n"
        "For non-age tasks use exactly one value from `allowed`. Choose `uncertain` when occlusion, "
        "resolution, or contradictory evidence prevents a defensible decision; do not guess.\n\n"
        "## Freeze and adjudication\n\n"
        "Return the completed JSONL without changing its filename. Both files are checksum-frozen "
        "before agreement is computed. Only then does an adjudicator inspect disagreements and write "
        "a separate `adjudicated_gold.jsonl`; annotator responses are never edited during "
        "adjudication.\n",
        encoding="utf-8",
    )
    write_experiment_manifest(
        output / "audit_pack.manifest.json",
        experiment="blinded-human-supervision-audit-pack",
        parameters={"n_per_stratum": n, "seed": seed, "annotators": 2},
        metrics={"tasks": len(tasks), "strata": 5},
        inputs=[
            data_path("data_dir", "raw", "posts.jsonl"),
            data_path("data_dir", "processed", "identity_groups.jsonl"),
            pre_prune_path,
            data_path("data_dir", "processed", "pairs.jsonl"),
            data_path("embeddings_cache_dir", "baseline_arcface.npz"),
        ],
        outputs=[
            *annotator_paths,
            key_path,
            adjudication_template,
            protocol,
            image_manifest,
        ],
    )
    return annotator_paths[0], annotator_paths[1], key_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400, help="held-out tasks per audit stratum")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--output", type=Path, default=data_path("data_dir", "interim", "human_audit")
    )
    args = parser.parse_args()
    for path in build_pack(args.n, args.seed, args.output):
        print(path)


if __name__ == "__main__":
    main()
