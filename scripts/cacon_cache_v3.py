"""Native generated-cache lineage checks, NOT proof of CACon generator fidelity.

No retroactive manifest creation for undocumented historical caches. A producer
must record its actual execution, weights, code, seed and target bins. The loader
accepts only explicitly adapted/unverified implementations, never paper parity.
CACon section2.2 specifies five-year bins; exact endpoints must be declared, not
invented or borrowed from MTLFace's seven decade groups.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT, read_jsonl
from age_gap.common.manifest import file_record
from scripts.cacon_dataset_v2 import ThreeViewDataset, ThreeViewRecord
from scripts.cacon_views_v2 import DEFAULT_VIEW_POLICY
from scripts.mtlface_epoch_v2 import load_bound_dataset


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _record(record):
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError("exact native file record required")
    if file_record(resolve(record["path"])) != record:
        raise ValueError("native cache/prerequisite hash mismatch")
    return resolve(record["path"])


def validate_rows(rows, face_rows, native, faces_record):
    """Validate execution declarations/bindings, not that a named GAN is correct."""
    if native.get("experiment") != "cacon-three-view-generated-cache":
        raise ValueError("native CACon generated-cache manifest required")
    metrics, params = native["metrics"], native["parameters"]
    if metrics.get("cache_complete") is not True or metrics.get("training_complete") is not False:
        raise ValueError("completed cache, not completed recognition training, required")
    if metrics.get("full_method_parity") is not False:
        raise ValueError("unverified cache cannot claim full-method parity")
    command = native.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) or not arg for arg in command):
        raise ValueError("recorded execution command required")
    if faces_record not in native["inputs"]:
        raise ValueError("exact retained face-list manifest must be input-bound")
    producer = params["producer"]
    if producer.get("status") not in ("adapted", "reference-reproduction-unverified"):
        raise ValueError("explicit adapted/unverified generator status required")
    conditioning = producer.get("conditioning")
    if conditioning not in ("target-image-and-label", "target-label", "label-only-adaptation"):
        raise ValueError("explicit generator conditioning required")
    if conditioning == "label-only-adaptation" and producer["status"] != "adapted":
        raise ValueError("label-only conditioning must remain an adaptation")
    for key in ("family", "version", "reference_url"):
        if not isinstance(producer.get(key), str) or not producer[key].strip():
            raise ValueError("generator family/version/primary reference required")
    if producer["version"].lower() in ("latest", "main", "master", "unknown"):
        raise ValueError("immutable generator version required")
    if not producer["reference_url"].startswith("https://"):
        raise ValueError("primary reference HTTPS locator required")
    for key in ("weights", "implementation"):
        if not isinstance(producer.get(key), list) or not producer[key]:
            raise ValueError("bound generator weights and implementation required")
        for record in producer[key]:
            if record not in native["inputs"]:
                raise ValueError("generator dependency not native-input-bound")
    seed = params["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("nonnegative bounded execution seed required")
    if params.get("sampling") != "continuing-PCG64-uniform-group-with-replacement":
        raise ValueError("explicit reproducible target sampling required")
    bins = params["target_age_bins"]
    if not isinstance(bins, list) or len(bins) < 2:
        raise ValueError("explicit five-year target bins required")
    for i, pair in enumerate(bins):
        if (
            not isinstance(pair, list) or len(pair) != 2
            or any(isinstance(a, bool) or not isinstance(a, int) or a < 0 for a in pair)
            or pair[1] - pair[0] != 4
            or (i and pair[0] != bins[i - 1][1] + 1)
        ):
            raise ValueError("ordered contiguous inclusive five-year bins required")
    if not rows or len(rows) != len(face_rows) or metrics.get("generated_images") != len(rows):
        raise ValueError("full retained source-image coverage required")
    faces = {row["face_id"]: row for row in face_rows}
    if len(faces) != len(face_rows):
        raise ValueError("duplicate retained face IDs refused")
    records, generated_paths, seen = [], set(), set()
    rng = np.random.Generator(np.random.PCG64(seed))
    groups = rng.integers(0, len(bins), size=len(rows)).tolist()
    for index, row in enumerate(rows):
        if set(row) != {"face_id", "source", "generated", "target_group", "target_age_bin", "draw_index",
                        "target_reference", "target_reference_face_id"}:
            raise ValueError("exact generated-row schema required")
        face_id = row["face_id"]
        if face_id not in faces or face_id in seen:
            raise ValueError("unknown/duplicate generated-cache face ID")
        # Cache order follows native source order, never score-based selection.
        if face_id != face_rows[index]["face_id"]:
            raise ValueError("native source-order cache required")
        face = faces[face_id]
        if resolve(row["source"]["path"]) != resolve(face["crop_path"]):
            raise ValueError("source crop differs from retained face list")
        if row["source"] not in native["inputs"]:
            raise ValueError("source crop not native-input-bound")
        if row["generated"] not in native["outputs"]:
            raise ValueError("generated crop not native-output-bound")
        group = row["target_group"]
        if (
            isinstance(group, bool) or not isinstance(group, int) or group != groups[index]
            or isinstance(row["draw_index"], bool) or not isinstance(row["draw_index"], int) or row["draw_index"] != index
            or row["target_age_bin"] != bins[group]
        ):
            raise ValueError("target group/bin/seeded draw mismatch")
        target_id, target = row["target_reference_face_id"], row["target_reference"]
        # TIP2021 Eq.(8) is G(x|target-age-label). Training reference y is
        # not an inference input. Neither mode establishes generator fidelity.
        if conditioning in ("target-label", "label-only-adaptation"):
            if target_id is not None or target is not None:
                raise ValueError("label-only declaration must not conceal target-image conditioning")
        else:
            if target_id not in faces or not isinstance(target, dict) or target not in native["inputs"]:
                raise ValueError("target-reference face and image must be native-input-bound")
            reference = faces[target_id]
            age = reference.get("age")
            if (
                resolve(target["path"]) != resolve(reference["crop_path"])
                or isinstance(age, bool) or not isinstance(age, int)
                or not bins[group][0] <= age <= bins[group][1]
            ):
                raise ValueError("target-reference source age does not match requested group")
        path = resolve(row["generated"]["path"])
        if path in generated_paths:
            raise ValueError("duplicate generated crop path")
        seen.add(face_id)
        generated_paths.add(path)
        records.append(ThreeViewRecord(face_id, row["source"], row["generated"]))
    source_paths = {resolve(row["crop_path"]) for row in face_rows}
    if source_paths & generated_paths:
        raise ValueError("generated images must not reuse any source crop path")
    return records


def load_bound_three_views(cache_path, faces_path, preprocess, rng, *, policy=DEFAULT_VIEW_POLICY):
    """Return dataset + lineage audit; fidelity flag intentionally remains false."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("caller-owned continuing augmentation RNG required")
    cache_path, faces_path = resolve(cache_path), resolve(faces_path)
    cache_record = file_record(cache_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    for record in cache["inputs"] + cache["outputs"]:
        _record(record)
    _, faces = load_bound_dataset(faces_path, preprocess)
    # Must use exactly the source-list output already verified by that loader.
    face_path = faces_path.parent / "private/faces.jsonl"
    face_rows = list(read_jsonl(face_path))
    if file_record(face_path) not in faces["outputs"]:
        raise ValueError("retained face rows changed during load")
    index_record = cache["parameters"]["index"]
    if index_record not in cache["outputs"]:
        raise ValueError("generated row index must be native-output-bound")
    rows = list(read_jsonl(resolve(index_record["path"])))
    _record(index_record)
    records = validate_rows(rows, face_rows, cache, file_record(faces_path))
    source_bindings = {resolve(r["path"]): r for r in faces["inputs"]}
    for row in records:
        if source_bindings.get(resolve(row.source_record["path"])) != row.source_record:
            raise ValueError("source bytes differ from retained-list input binding")
    for row in rows:
        if row["target_reference"] is not None:
            target = row["target_reference"]
            if source_bindings.get(resolve(target["path"])) != target:
                raise ValueError("target-reference bytes differ from retained-list input binding")
    dataset = ThreeViewDataset(records, preprocess, rng, policy=policy)
    for record in cache["inputs"] + cache["outputs"]:
        _record(record)
    if file_record(cache_path) != cache_record:
        raise ValueError("cache manifest changed during load")
    return dataset, dict(
        cache_lineage_verified=True,
        generator_provenance_verified=False,
        full_method_parity=False,
        training_complete=False,
        source_images=len(dataset),
        generator_status=cache["parameters"]["producer"]["status"],
        scope="recorded cache lineage; not generator execution/fidelity or human identity truth",
    )
