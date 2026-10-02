"""Build a deterministic, stratified, blinded review pack for overlap candidates.

Inputs (all read-only, produced by existing audit scripts):

* ``private_candidates.jsonl``           - image-level exact/pHash candidates
* ``private_embedding_candidates.jsonl`` - independent FaceNet cosine candidates
* ``private_training_inventory.jsonl``   - train crop inventory (index -> face_id/path/hashes)

Output layout (all private):

* ``reviewer_a/`` and ``reviewer_b/`` - each contains ONLY ``tasks.jsonl``, ``README.md``
  and an ``images/`` directory of opaque crops. Reviewers never see similarity, rank,
  model name, benchmark label or source identifiers.
* ``private/`` - ``key.jsonl`` (mapping + machine signals) and ``sampling_manifest.json``.
  Kept outside the reviewer directories on purpose.

Stratification and sampling
---------------------------
* High-risk image-level candidates (exact decoded pixels, pHash distance <= 4) are all kept.
* High-risk embedding candidates (independent FaceNet cosine >= 0.80) are all kept.
* Lower cosine bands (0.70-0.80, 0.60-0.70, < 0.60) get a deterministic hash-priority
  sample, and every selected row records its equal inclusion probability (n/N).
* Endpoint reuse across strata is never excluded: a crop may legitimately appear in more
  than one candidate, and dropping it would bias the audit sample.

Nothing here decides an identity. Similarity is a machine signal only; every task stays
``unreviewed_candidate`` until a human files a decision.

Run with::

    uv run python build_overlap_review_pack.py --output <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# Defaults and constants
# --------------------------------------------------------------------------------------

DEFAULT_AUDIT_DIR = Path("data/interim/benchmark_overlap_audit")
DEFAULT_OUTPUT = Path("data/interim/benchmark_overlap_review")
DEFAULT_SEED = 20260922

HIGH_RISK_COSINE = 0.8
COSINE_STRATA = (
    ("cosine_ge_0.80", 0.80, 1.01),
    ("cosine_0.70_0.80", 0.70, 0.80),
    ("cosine_0.60_0.70", 0.60, 0.70),
    ("cosine_lt_0.60", -1.01, 0.60),
)
EXACT_STRATUM = "exact_decoded_pixels"
PHASH_STRATUM = "phash_le_4"

# Full-keep strata carry ``None`` as target; sampled strata carry a deterministic count.
STRATUM_TARGETS: dict[str, int | None] = {
    EXACT_STRATUM: None,
    PHASH_STRATUM: None,
    "cosine_ge_0.80": None,
    "cosine_0.70_0.80": 70,
    "cosine_0.60_0.70": 40,
    "cosine_lt_0.60": 20,
}
STRATUM_ORDER = (
    EXACT_STRATUM,
    PHASH_STRATUM,
    "cosine_ge_0.80",
    "cosine_0.70_0.80",
    "cosine_0.60_0.70",
    "cosine_lt_0.60",
)
FULL_KEEP_STRATA = frozenset(name for name, target in STRATUM_TARGETS.items() if target is None)

PRIVATE_DIRNAME = "private"
REVIEWERS = ("reviewer_a", "reviewer_b")
ALLOWED = ["same_identity", "different_identity", "uncertain"]

BENCHMARK_ARTEFACTS = {
    "FG-NET": "fgnet_crops.npz",
    "AgeDB-30": "agedb_30.bin",
    "CALFW": "calfw.bin",
    "LFW": "lfw_aligned.npz",
    "CACD-VS": "cacd_vs_aligned.npz",
}


# --------------------------------------------------------------------------------------
# Small deterministic helpers
# --------------------------------------------------------------------------------------


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task_id(evidence_id: str) -> str:
    return f"overlap-{_sha256_text('overlap|' + evidence_id)[:16]}"


def _opaque_name(seed: int, reviewer: str, task_id: str, slot: int) -> str:
    digest = _sha256_text(f"{seed}|{reviewer}|{task_id}|{slot}")[:24]
    return f"images/{digest}.jpg"


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def pixel_sha256(image: np.ndarray) -> str:
    """Match scripts/benchmark_image_overlap_audit.pixel_sha256 byte for byte."""
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Candidate loading and stratification
# --------------------------------------------------------------------------------------


def _image_candidates(audit_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(_read_jsonl(audit_dir / "private_candidates.jsonl")):
        match_type = str(raw.get("match_type", "phash_candidate"))
        rows.append(
            {
                "source": "image",
                "evidence_id": f"img:{index}",
                "benchmark": str(raw["benchmark"]),
                "train_index": int(raw["train_index"]),
                "benchmark_index": int(raw["benchmark_index"]),
                "match_type": match_type,
                "phash_distance": int(raw.get("phash_distance", 0)),
                "auto_label": (
                    "exact_duplicate" if match_type == "exact_decoded_pixels" else "phash_near_duplicate"
                ),
                "similarity": None,
                "rank": None,
            }
        )
    return rows


def _embedding_candidates(audit_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(_read_jsonl(audit_dir / "private_embedding_candidates.jsonl")):
        rows.append(
            {
                "source": "embedding",
                "evidence_id": f"emb:{index}",
                "benchmark": str(raw["benchmark"]),
                "train_index": int(raw["train_index"]),
                "benchmark_index": int(raw["benchmark_index"]),
                "match_type": str(raw.get("match_type", "cosine_threshold")),
                "phash_distance": None,
                "auto_label": "cosine_candidate",
                "similarity": float(raw["cosine_similarity"]),
                "rank": int(raw.get("rank", -1)),
            }
        )
    return rows


def _cosine_stratum(cosine: float) -> str:
    for name, low, high in COSINE_STRATA:
        if low <= cosine < high:
            return name
    return COSINE_STRATA[-1][0]


def stratum_of(row: dict[str, Any]) -> str:
    if row["source"] == "image":
        return EXACT_STRATUM if row["match_type"] == "exact_decoded_pixels" else PHASH_STRATUM
    return _cosine_stratum(float(row["similarity"]))


def _stratify(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    strata: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATUM_ORDER}
    for row in candidates:
        strata[stratum_of(row)].append(row)
    return strata


def _sampling_priority(seed: int, stratum: str, evidence_id: str) -> str:
    """Deterministic per-row sort key; the n smallest keys form a uniform sample."""
    return _sha256_text(f"{seed}|{stratum}|{evidence_id}")


def _select_stratum(
    name: str, rows: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], float]:
    """Return (selected rows, inclusion probability).

    High-risk strata keep every candidate (probability 1.0). Sampled strata take the
    deterministic n smallest hash priorities, which is a uniform sample without
    replacement, so every row has the same inclusion probability n/N.
    """
    ordered = sorted(rows, key=lambda row: _sampling_priority(seed, name, row["evidence_id"]))
    target = STRATUM_TARGETS[name]
    if target is None or len(ordered) <= target:
        return ordered, 1.0
    return ordered[:target], round(target / len(ordered), 8)


# --------------------------------------------------------------------------------------
# Image sources (train crops + cached benchmark images)
# --------------------------------------------------------------------------------------


def _train_paths(inventory: list[dict[str, Any]]) -> dict[int, str]:
    return {int(row["train_index"]): str(row["path"]) for row in inventory}


def _load_benchmark_images(benchmark: str, external_dir: Path) -> list[np.ndarray]:
    artefact = external_dir / BENCHMARK_ARTEFACTS[benchmark]
    if not artefact.is_file():
        raise FileNotFoundError(f"missing benchmark artefact for {benchmark}: {artefact}")
    if artefact.suffix == ".bin":
        import pickle

        with artefact.open("rb") as handle:
            bins, _labels = pickle.load(handle, encoding="bytes")
        images: list[np.ndarray] = []
        for blob in bins:
            image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
        return images
    with np.load(artefact, allow_pickle=False) as archive:
        arrays = [archive["crops"]] if benchmark == "FG-NET" else [archive["a"], archive["b"]]
        return [image for array in arrays for image in array]


class _BenchmarkStore:
    """Load each benchmark artefact at most once for the whole pack build."""

    def __init__(self, external_dir: Path) -> None:
        self._external = external_dir
        self._cache: dict[str, list[np.ndarray]] = {}

    def get(self, benchmark: str, index: int) -> np.ndarray | None:
        if benchmark not in self._cache:
            self._cache[benchmark] = _load_benchmark_images(benchmark, self._external)
        images = self._cache[benchmark]
        if 0 <= index < len(images):
            return np.asarray(images[index])
        return None


# --------------------------------------------------------------------------------------
# Pack construction
# --------------------------------------------------------------------------------------


def _assert_no_frozen_responses(output: Path) -> None:
    for reviewer in REVIEWERS:
        for line_number, row in enumerate(_read_jsonl(output / reviewer / "tasks.jsonl"), 1):
            if row.get("response") is not None:
                raise RuntimeError(
                    f"refusing to overwrite frozen response in {output / reviewer / 'tasks.jsonl'} "
                    f"at row {line_number}"
                )


def build_pack(
    output: Path,
    *,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    external_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    output = Path(output)
    _assert_no_frozen_responses(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("output directory is nonempty; choose a fresh private pack path")
    for filename in ("private_training_inventory.jsonl", "private_candidates.jsonl", "private_embedding_candidates.jsonl"):
        if not (audit_dir / filename).is_file():
            raise FileNotFoundError(audit_dir / filename)
    if external_dir is None:
        external_dir = Path("data/external")

    inventory = _read_jsonl(audit_dir / "private_training_inventory.jsonl")
    if not inventory:
        raise RuntimeError(f"missing training inventory under {audit_dir}")
    train_paths = _train_paths(inventory)
    face_by_index = {int(row["train_index"]): str(row["face_id"]) for row in inventory}
    pixel_by_index = {int(row["train_index"]): row.get("pixel_sha256") for row in inventory}

    candidates = _image_candidates(audit_dir) + _embedding_candidates(audit_dir)
    strata = _stratify(candidates)
    available = {name: len(rows) for name, rows in strata.items()}

    blockers: dict[str, Any] = {}
    selected: list[dict[str, Any]] = []
    inclusion: dict[str, float] = {}
    stratum_sampling: dict[str, dict[str, Any]] = {}
    for name in STRATUM_ORDER:
        picked, probability = _select_stratum(name, strata[name], seed)
        inclusion[name] = probability
        selected.extend(picked)
        stratum_sampling[name] = {
            "available": available[name],
            "selected": len(picked),
            "full_keep": name in FULL_KEEP_STRATA,
            "inclusion_probability": probability,
        }
        if name not in FULL_KEEP_STRATA and len(picked) < int(STRATUM_TARGETS[name] or 0):
            blockers.setdefault("strata_below_target", {})[name] = {
                "requested": STRATUM_TARGETS[name],
                "available": available[name],
                "selected": len(picked),
            }

    missing_train = sorted(
        {row["train_index"] for row in selected if row["train_index"] not in train_paths}
    )
    if missing_train:
        blockers["train_index_missing_from_inventory"] = len(missing_train)

    store = _BenchmarkStore(external_dir)
    unresolved: list[dict[str, Any]] = []

    # Canonical task list: same task IDs for both reviewers, blank decisions.
    tasks: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    image_payloads: list[tuple[str, int, np.ndarray]] = []  # (task_id, slot, image)

    for row in selected:
        task_id = _task_id(row["evidence_id"])
        train_index = row["train_index"]
        train_path = train_paths.get(train_index)
        train_image = (
            cv2.imread(train_path, cv2.IMREAD_COLOR)
            if train_path is not None and Path(train_path).is_file()
            else None
        )
        if train_image is None:
            unresolved.append({"evidence_id": row["evidence_id"], "reason": "train_crop_unreadable"})
            continue
        if pixel_by_index.get(train_index) is not None and pixel_sha256(train_image) != pixel_by_index[
            train_index
        ]:
            unresolved.append(
                {"evidence_id": row["evidence_id"], "reason": "train_crop_pixel_hash_mismatch"}
            )
            continue

        bench_image = store.get(row["benchmark"], row["benchmark_index"])
        if bench_image is None:
            blockers.setdefault("benchmark_index_unresolvable", {})[row["benchmark"]] = (
                blockers.get("benchmark_index_unresolvable", {}).get(row["benchmark"], 0) + 1
            )
            unresolved.append(
                {
                    "evidence_id": row["evidence_id"],
                    "reason": "benchmark_image_index_out_of_range",
                    "benchmark": row["benchmark"],
                }
            )
            continue

        stratum = stratum_of(row)
        tasks.append(
            {
                "task_id": task_id,
                "audit_type": "overlap_candidate",
                "images": [None, None],  # filled per reviewer below
                "allowed": list(ALLOWED),
                "response": None,
                "notes": None,
            }
        )
        image_payloads.append((task_id, 0, train_image))
        image_payloads.append((task_id, 1, bench_image))
        key_rows.append(
            {
                "task_id": task_id,
                "audit_type": "overlap_candidate",
                "candidate_source": row["source"],
                "match_type": row["match_type"],
                "benchmark": row["benchmark"],
                "train_index": train_index,
                "benchmark_index": row["benchmark_index"],
                "cosine_similarity": row["similarity"],
                "phash_distance": row["phash_distance"],
                "rank": row["rank"],
                "auto_label": row["auto_label"],
                "stratum": stratum,
                "inclusion_probability": inclusion[stratum],
                "train_face_id": face_by_index.get(train_index),
                "source_images": [
                    str(Path(train_path).resolve()) if train_path else "",
                    f"benchmark://{row['benchmark']}#{row['benchmark_index']}",
                ],
                "review_status": "unreviewed_candidate",
            }
        )

    if unresolved:
        blockers["unresolved_selected_candidates"] = len(unresolved)

    stratum_tasks: dict[str, int] = {}
    for row in key_rows:
        stratum_tasks[row["stratum"]] = stratum_tasks.get(row["stratum"], 0) + 1

    # ---- Materialize two independent reviewer directories.
    payload_by_task: dict[str, dict[int, np.ndarray]] = {}
    for task_id, slot, image in image_payloads:
        payload_by_task.setdefault(task_id, {})[slot] = image

    reviewer_manifests: dict[str, dict[str, Any]] = {}
    for reviewer in REVIEWERS:
        reviewer_dir = output / reviewer
        images_dir = reviewer_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []
        reviewer_tasks: list[dict[str, Any]] = []
        for task in sorted(tasks, key=lambda item: item["task_id"]):
            task_id = task["task_id"]
            names: list[str] = []
            for slot in (0, 1):
                image = payload_by_task[task_id][slot]
                rel = _opaque_name(seed, reviewer, task_id, slot)
                destination = output / reviewer / rel
                if not cv2.imwrite(str(destination), image):
                    raise RuntimeError(f"failed to write opaque image {destination}")
                blob = destination.read_bytes()
                files.append(
                    {
                        "file": rel,
                        "bytes": len(blob),
                        "sha256": hashlib.sha256(blob).hexdigest(),
                    }
                )
                names.append(rel)
            reviewer_tasks.append({**task, "images": names})

        offset = 1 if reviewer == "reviewer_a" else 2
        order = np.random.default_rng(seed + offset).permutation(len(reviewer_tasks))
        shuffled = [reviewer_tasks[int(i)] for i in order]
        _write_jsonl(reviewer_dir / "tasks.jsonl", shuffled)
        _write_text(reviewer_dir / "README.md", _reviewer_readme(len(reviewer_tasks)))
        reviewer_manifests[reviewer] = {
            "schema_version": 1,
            "n_files": len(files),
            "files": files,
        }

    # ---- Private mapping and sampling live outside the reviewer directories.
    private_dir = output / PRIVATE_DIRNAME
    _write_jsonl(private_dir / "key.jsonl", key_rows)
    _write_json(private_dir / "images_manifest.json", {"schema_version": 1, "reviewers": reviewer_manifests})
    sampling_manifest = {
        "schema_version": 1,
        "seed": seed,
        "audit_dir": str(audit_dir),
        "external_dir": str(external_dir),
        "high_risk_cosine_threshold": HIGH_RISK_COSINE,
        "stratum_targets": {name: STRATUM_TARGETS[name] for name in STRATUM_ORDER},
        "strata": stratum_sampling,
        "n_tasks": len(tasks),
        "n_unresolved": len(unresolved),
        "endpoint_reuse_policy": "never excluded; a crop or benchmark image may appear in several strata",
        "privacy": "PRIVATE. Never ship to reviewers. Contains mapping to protected crops.",
    }
    _write_json(private_dir / "sampling_manifest.json", sampling_manifest)

    _write_json(output / "decisions.schema.json", _decision_schema())
    _write_json(output / "adjudicated_gold.schema.json", _adjudication_schema())

    summary = {
        "status": "blinded_review_pack_ready_pending_human_decisions",
        "task_id": _pack_task_id(audit_dir, seed, len(tasks)),
        "check_date_utc": datetime.now(UTC).isoformat(),
        "candidate_sources": {
            "image_exact_and_phash": "scripts/benchmark_image_overlap_audit.py",
            "independent_facenet_cosine": "scripts/benchmark_embedding_overlap_audit.py",
        },
        "n_candidates_available": available,
        "n_tasks": len(tasks),
        "n_reviewers": len(REVIEWERS),
        "strata_tasks": stratum_tasks,
        "stratum_inclusion_probability": inclusion,
        "high_risk_cosine_ge_0.80_tasks": stratum_tasks.get("cosine_ge_0.80", 0),
        "sampled_lower_strata_tasks": sum(
            stratum_tasks.get(name, 0)
            for name in ("cosine_0.70_0.80", "cosine_0.60_0.70", "cosine_lt_0.60")
        ),
        "high_risk_fraction": _ratio(
            stratum_tasks.get(EXACT_STRATUM, 0)
            + stratum_tasks.get(PHASH_STRATUM, 0)
            + stratum_tasks.get("cosine_ge_0.80", 0),
            len(tasks),
        ),
        "reviewer_decisions_status": "pending",
        "adjudication_status": "pending",
        "human_identity_judgements": 0,
        "confirmed_identity_overlaps": None,
        "verified_zero_overlap": False,
        "private_artefacts": [
            f"{PRIVATE_DIRNAME}/key.jsonl",
            f"{PRIVATE_DIRNAME}/sampling_manifest.json",
            f"{PRIVATE_DIRNAME}/images_manifest.json",
            "adjudicated_gold.schema.json",
        ],
        "blockers": blockers,
        "interpretation": (
            "Machine candidate retrieval only. Similarity never confirms an identity; "
            "no row may be scored until both reviewers decide and an adjudicator writes gold."
        ),
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "blockers.json",
        {"status": "blocked" if blockers else "none", "blockers": blockers},
    )

    from age_gap.common.manifest import write_experiment_manifest

    inputs = [
        audit_dir / "private_training_inventory.jsonl",
        audit_dir / "private_candidates.jsonl",
        audit_dir / "private_embedding_candidates.jsonl",
        Path(__file__),
        *[external_dir / BENCHMARK_ARTEFACTS[name] for name in store._cache],
    ]
    write_experiment_manifest(
        output / "pack.manifest.json", experiment="blinded-train-benchmark-overlap-pack",
        parameters={"seed": seed, "full_keep_strata": list(FULL_KEEP_STRATA),
                    "sample_targets": STRATUM_TARGETS},
        metrics=summary, inputs=inputs,
        outputs=[output / "summary.json", private_dir / "key.jsonl",
                 private_dir / "sampling_manifest.json", private_dir / "images_manifest.json",
                 *[output / reviewer / "tasks.jsonl" for reviewer in REVIEWERS]],
    )
    return {
        "output": str(output),
        "n_tasks": len(tasks),
        "strata": stratum_tasks,
        "inclusion_probability": inclusion,
        "blockers": blockers,
        "summary": summary,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _pack_task_id(audit_dir: Path, seed: int, n_tasks: int) -> str:
    return f"overlap-pack-{_sha256_text(f'{audit_dir}|{seed}|{n_tasks}')[:12]}"


def _decision_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Blinded overlap decision (reviewer)",
        "type": "object",
        "required": ["task_id", "audit_type", "response"],
        "additionalProperties": False,
        "properties": {
            "task_id": {"type": "string"},
            "audit_type": {"const": "overlap_candidate"},
            "response": {
                "type": ["string", "null"],
                "enum": ["same_identity", "different_identity", "uncertain", None],
                "description": "Blank (null) until a human files a decision.",
            },
            "notes": {"type": ["string", "null"]},
        },
        "description": "One row per reviewer task; identical to the reviewer tasks.jsonl rows.",
    }


def _adjudication_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Adjudicated gold (separate from reviewer decisions)",
        "type": "object",
        "required": ["task_id", "audit_type", "response"],
        "additionalProperties": False,
        "properties": {
            "task_id": {"type": "string"},
            "audit_type": {"const": "overlap_candidate"},
            "response": {
                "type": ["string", "null"],
                "enum": ["same_identity", "different_identity", "uncertain", None],
            },
            "adjudication_notes": {"type": ["string", "null"]},
        },
        "description": "Written only by an adjudicator after both reviewer files are frozen.",
    }


def _reviewer_readme(n_tasks: int) -> str:
    return f"""# Blinded overlap review pack

## Independence and blinding

Work independently, do not discuss cases, and never open another reviewer directory or
the private key. Image paths are relative to this directory and are deliberately opaque;
do not inspect file metadata or run reverse-image search.

This pack holds {n_tasks} candidate pairs mined from train crops versus external
benchmarks. A pair is a *machine candidate*: pHash distance or an independent FaceNet
cosine may both be wrong. Similarity is not evidence of identity.

## Task format

Every row is `overlap_candidate` with two images: the first is the candidate train crop,
the second the candidate benchmark image. Fill only `response` and, when useful, `notes`.
Use exactly one value from `allowed`:

- `same_identity` - the two faces are the same person.
- `different_identity` - the two faces are different people.
- `uncertain` - occlusion, resolution or blur prevents a defensible decision. Do not guess.

Do not reorder, add or remove rows, and do not edit anything except `response`/`notes`.

## Freeze and adjudication

Return the completed `tasks.jsonl` without renaming it. Both reviewer files are frozen
before agreement is computed. Only then may an adjudicator inspect disagreements and
write a separate `adjudicated_gold.jsonl` (see the shared schema outside this pack).
Reviewer responses are never edited during adjudication. This pack makes no human
identity judgement on its own.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--external-dir", type=Path, default=Path("data/external"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    if args.output.is_dir():
        for reviewer in REVIEWERS:
            for line in _read_jsonl(args.output / reviewer / "tasks.jsonl"):
                if line.get("response") is not None:
                    print(
                        f"refusing to overwrite frozen response in {args.output / reviewer}",
                        file=sys.stderr,
                    )
                    return 2

    try:
        result = build_pack(
            args.output, audit_dir=args.audit_dir, external_dir=args.external_dir, seed=args.seed
        )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": result["output"],
                "n_tasks": result["n_tasks"],
                "strata": result["strata"],
                "inclusion_probability": result["inclusion_probability"],
                "blockers": result["blockers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
