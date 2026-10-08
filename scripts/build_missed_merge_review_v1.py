"""Blinded group review within an existing cross-split candidate frame, not recall proof."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sample_candidates(rows, *, per_stratum=25, seed=42):
    if type(per_stratum) is not int or per_stratum < 1 or type(seed) is not int:
        raise ValueError("positive sample size and integer seed required")
    strata, seen = defaultdict(list), set()
    for row in rows:
        pair = tuple(sorted((row["group_id_a"], row["group_id_b"])))
        score = row["group_centroid_cosine"]
        splits = tuple(sorted((row["split_a"], row["split_b"])))
        if (pair in seen or pair[0] == pair[1] or len(set(splits)) != 2
                or not set(splits) <= {"train", "val", "test"}
                or row["person_id_a"] == row["person_id_b"]
                or isinstance(score, bool) or not isinstance(score, (float, int))
                or not math.isfinite(score) or not 0.65 <= score <= 1):
            raise ValueError("invalid/duplicate cross-split candidate")
        seen.add(pair)
        band = next((name for cutoff, name in (
            (0.85, "ge085"), (0.80, "080_085"), (0.75, "075_080"),
            (0.70, "070_075"), (0.65, "065_070"),
        ) if score >= cutoff))
        strata["-".join(splits) + ":" + band].append(row)
    selected, counts = [], {}
    for name, population in sorted(strata.items()):
        ordered = sorted(population, key=lambda row: digest(
            f"{seed}|{name}|{row['group_id_a']}|{row['group_id_b']}"))
        keep = ordered if name.endswith(":ge085") else ordered[:per_stratum]
        counts[name] = dict(population=len(population), selected=len(keep),
                            inclusion_probability=len(keep) / len(population))
        selected.extend(dict(candidate=row, stratum=name,
                             inclusion_probability=len(keep) / len(population)) for row in keep)
    return selected, counts


def build(audit: Path, out: Path):
    if out.exists():
        raise FileExistsError("fresh output required")
    audit = audit.resolve()
    native = json.loads((audit / "manifest.json").read_text(encoding="utf-8"))
    if native["experiment"] != "read-only-missed-identity-merge-split-audit":
        raise ValueError("native candidate audit required")
    audit_dependencies = []
    for record in native["inputs"] + native["outputs"]:
        path = Path(record["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if file_record(path) != record:
            raise ValueError("candidate audit binding mismatch")
        audit_dependencies.append(path)
    candidate_path = audit / "private_candidates.jsonl"
    groups_path = PROJECT_ROOT / "data/processed/identity_groups.jsonl.post_bak"
    if file_record(candidate_path) not in native["outputs"] or file_record(groups_path) not in native["inputs"]:
        raise ValueError("exact candidate/group bindings required")
    rows = list(read_jsonl(candidate_path))
    selected, strata = sample_candidates(rows)
    if not selected:
        raise ValueError("empty candidate sample")
    groups = {}
    for row in read_jsonl(groups_path):
        identity = row["identity_group_id"]
        if identity in groups:
            raise ValueError("duplicate group ID")
        groups[identity] = row["faces"]
    face_ids = set()
    for item in selected:
        for key in ("group_id_a", "group_id_b"):
            faces = groups.get(item["candidate"][key])
            if not faces or len(faces) != len(set(faces)):
                raise ValueError("missing/empty/duplicate group faces")
            for face in faces:
                if not isinstance(face, str) or any(c in face for c in "/\\:") or face in (".", ".."):
                    raise ValueError("unsafe face ID")
                face_ids.add(face)
    crop_root = (PROJECT_ROOT / "data/interim/faces").resolve()
    crops = {face: crop_root / f"{face}.jpg" for face in sorted(face_ids)}
    if any(path.resolve().parent != crop_root for path in crops.values()):
        raise ValueError("crop escapes source directory")
    inputs = list(dict.fromkeys([audit / "manifest.json", *audit_dependencies, candidate_path, groups_path, Path(__file__).resolve(),
              PROJECT_ROOT / "src/age_gap/common/io.py", PROJECT_ROOT / "src/age_gap/common/manifest.py",
              *crops.values()]))
    before = {str(path): file_record(path) for path in inputs}
    out.mkdir(parents=True)
    outputs, private_rows = [], []
    for item in selected:
        row = item["candidate"]
        item["task_id"] = "merge-" + digest(row["group_id_a"] + "|" + row["group_id_b"])[:24]
        private_rows.append(item)
    if len({item["task_id"] for item in selected}) != len(selected):
        raise ValueError("task ID collision")
    instructions = (
        "# Blinded cross-split group review\n\n"
        "Work independently; never inspect private/ or the other reviewer. Each task shows "
        "all available source crops from two recorded groups. Do not use reverse-image search. "
        "Edit only response/notes, without reordering rows. Use same_identity only if all "
        "faces support one person; use shared_identity_mixed_groups if some identity occurs "
        "in both groups but either group also includes another person. Use different_identity "
        "only if the groups share no apparent identity, otherwise uncertain. A mixed group "
        "must never be cleared as no-overlap solely because it cannot be merged as a whole. "
        "Never infer identity from machine scores. "
        "Return a separate completed tasks.jsonl copy; do not overwrite frozen inputs.\n\n"
        "This is a sampled machine-candidate frame, not exhaustive leakage detection or "
        "a population recall estimate. Private biometric material; not a public release.\n"
    )
    for reviewer in ("reviewer_a", "reviewer_b"):
        directory = out / reviewer
        (directory / "images").mkdir(parents=True)
        image_names = {}
        for face, path in crops.items():
            relative = "images/" + digest(f"{reviewer}|{face}")[:24] + ".png"
            if relative in image_names.values():
                raise ValueError("opaque image collision")
            image = cv2.imread(str(path))
            if image is None or not cv2.imwrite(str(directory / relative), image):
                raise ValueError("crop decode/encode failed")
            if not (cv2.imread(str(directory / relative)) == image).all():
                raise ValueError("lossless crop roundtrip failed")
            image_names[face] = relative
            outputs.append(directory / relative)
        tasks = []
        for item in selected:
            row = item["candidate"]
            images = [[image_names[face] for face in groups[row[key]]]
                      for key in ("group_id_a", "group_id_b")]
            tasks.append(dict(task_id=item["task_id"], audit_type="cross_split_group_candidate",
                              image_groups=images, allowed=["same_identity", "shared_identity_mixed_groups",
                                                           "different_identity", "uncertain"],
                              response=None, notes=""))
        tasks.sort(key=lambda row: digest(reviewer + "|" + row["task_id"]))
        task_path = directory / "tasks.jsonl"
        write_jsonl(task_path, tasks)
        readme = directory / "README.md"
        readme.write_text(instructions, encoding="utf-8")
        outputs.extend([task_path, readme])
    private = out / "private"
    private.mkdir()
    key_path = private / "key.jsonl"
    write_jsonl(key_path, private_rows)
    outputs.append(key_path)
    for path in inputs:
        if file_record(path) != before[str(path)]:
            raise RuntimeError("source changed during pack generation")
    metrics = dict(pack_complete=True, human_review_complete=False, population_recall_estimated=False,
                   candidate_frame=len(rows), selected_tasks=len(selected), unique_crops=len(crops), strata=strata)
    write_experiment_manifest(
        out / "pack.manifest.json", experiment="blinded-cross-split-missed-merge-candidate-pack",
        parameters=dict(seed=42, per_lower_stratum=25, high_risk="all cosine>=0.85",
                        face_policy="all faces in both snapshot groups; no silent missing-crop omission",
                        estimand="candidate-frame subgroup-positive rate only; not global merge recall",
                        limitation="cross-split top5 centroid cosine>=0.65 frame; not below-threshold/nonretrieved pairs"),
        metrics=metrics, inputs=inputs, outputs=outputs,
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "strata"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=PROJECT_ROOT / "data/interim/missed_identity_merge_audit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.audit, args.out)
