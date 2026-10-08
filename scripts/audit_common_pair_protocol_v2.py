"""Targeted shared-pair linkage audit over canonical strong matrix cells.

Version 2 of the selected-record image/pair/reference audit. It keeps the exact
v1 semantics (native ``--strong``/``--weak`` command linkage, metadata binding,
650 images / 5308 pairs, chronological source-gap linkage, exact equality of
left/right/labels/endpoints/gaps, frozen embedding bytes and weak cache
references across bindings) and widens the accepted cell set from the six
low-LR random head/tail cells to any unique canonical completed strong matrix
cell: ``(random|lookalike) x (head|tail|full) x lr{1e-06,1e-05} x seed{42,1,2}``
(36 canonical names). Coverage stays explicit: all counts, including 36 checked
cells, keep ``mechanism_complete`` and ``publication_ready`` false. This remains a
targeted selected-record audit, not a full native-ancestry reverification.

This module is intentionally self-contained: it never imports or mutates
``audit_common_pair_protocol_v1`` and reports under a new experiment type so
v1 bytes and globals are preserved.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs
from scripts.run_common_mechanism_v1 import validate_pairs

EXPERIMENT = "common-image-pair-linkage-audit-v2"
CELL = re.compile(r"(random|lookalike)_(head|tail|full)_lr(1e-06|1e-05)_s(42|1|2)\Z")
FIELDS = ("left", "right", "labels", "subject_a", "subject_b", "source_gap")
WEAK_MODELS = ("frozen", "tuned_seed42", "tuned_seed1", "tuned_seed2")
EXPECTED_CANONICAL_CELLS = 36
LIMITATION = (
    "targeted selected-record and exact-array linkage audit only over canonical "
    "strong matrix cells; partial counts remain partial and are not full "
    "ancestry hash, human identity clearance, training provenance, full "
    "mechanism or seed-population inference"
)


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def linked(path, records):
    record = file_record(path)
    if record not in records:
        raise ValueError("selected file not bound by exact native record")
    return record


def argument(native, flag):
    command = native["command"]
    if (not isinstance(command, list) or not all(isinstance(token, str) for token in command)
            or command.count(flag) != 1 or command.index(flag) + 1 >= len(command)
            or command[command.index(flag) + 1].startswith("--")):
        raise ValueError("one explicit parent argument required")
    return resolve(command[command.index(flag) + 1])


def records(paths, message):
    """Snapshot file records, turning a disappearance into a protocol error."""
    try:
        return [file_record(path) for path in paths]
    except (OSError, ValueError) as error:
        raise RuntimeError(message) from error


def audit(bindings):
    """Verify selected live records and exact arrays; do not claim all-input hashing."""
    cells, paths = set(), set()
    reference = frozen = weak_path = weak_records = metadata_record = None
    cache = PROJECT_ROOT / "data/external/fgnet_crops.npz"
    for binding in bindings:
        binding = resolve(binding)
        common = json.loads(binding.read_text(encoding="utf-8"))
        if (common["experiment"] != "common-image-index-mechanism-v1"
                or common["metrics"].get("execution_complete") is not True):
            raise ValueError("completed common-index native required")
        strong_path, current_weak = argument(common, "--strong"), argument(common, "--weak")
        linked(strong_path, common["inputs"])
        linked(current_weak, common["inputs"])
        strong = json.loads(strong_path.read_text(encoding="utf-8"))
        weak = json.loads(current_weak.read_text(encoding="utf-8"))
        p = strong["parameters"]
        if (strong["experiment"] != "adaface-fixed8-cuda-full-fgnet-roc-v2"
                or strong["metrics"].get("execution_complete") is not True
                or type(p.get("protocol_seed")) is not int or p["protocol_seed"] != 42
                or type(p.get("tolerance")) is not int or p["tolerance"] != 2
                or weak["experiment"] != "fgnet-endpoint-age-matched-error-breakdown"):
            raise ValueError("expected strong and weak parent protocols required")
        names = p["actual_cells"]
        if (not isinstance(names, list) or len(names) != 1 or not isinstance(names[0], str)
                or not CELL.fullmatch(names[0]) or names[0] in cells):
            raise ValueError("one unique canonical strong matrix cell per binding required")
        cell = names[0]
        if set(common["metrics"]["results"]["strong"]["diagnostics"]) != {cell}:
            raise ValueError("common cell not linked to strong parent")
        cells.add(cell)
        meta = linked(cache, strong["inputs"])
        linked(cache, common["inputs"])
        linked(cache, weak["inputs"])
        score_path = strong_path.parent / "private/scores.npz"
        linked(score_path, strong["outputs"])
        linked(score_path, common["inputs"])
        frozen_path = strong_path.parent / "private/frozen_embeddings.npz"
        frozen_record = linked(frozen_path, strong["outputs"])
        linked(frozen_path, common["inputs"])
        current_weak_records = []
        for model in WEAK_MODELS:
            path = current_weak.parent / f"private/private_embeddings_{model}.npz"
            record = linked(path, weak["outputs"])
            linked(path, common["inputs"])
            current_weak_records.append(record)
            paths.add(path)
        with np.load(cache, allow_pickle=False) as data:
            people, ages = data["subjects"], data["ages"]
        with np.load(score_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in FIELDS}
        if (people.shape != (650,) or ages.shape != (650,) or len(np.unique(people)) != 82
                or people.dtype.kind not in "iu" or ages.dtype.kind not in "iuf"
                or not np.isfinite(ages).all() or np.any(ages < 0)
                or len(arrays["labels"]) != 5308):
            raise ValueError("full shared image/pair coverage required")
        validate_pairs(arrays["left"], arrays["right"], arrays["labels"],
                       arrays["subject_a"], arrays["subject_b"], people)
        expected_gap = np.abs(ages[arrays["left"]] - ages[arrays["right"]])
        if not np.array_equal(arrays["source_gap"], expected_gap):
            raise ValueError("source gap not linked to bound chronological ages")
        frozen_key = (frozen_record["sha256"], frozen_record["bytes"])
        if reference is not None and (
                any(not np.array_equal(reference[key], arrays[key]) for key in FIELDS)
                or frozen != frozen_key or metadata_record != meta
                or weak_path != current_weak or weak_records != current_weak_records):
            raise ValueError("shared image-pair/metadata/frozen/weak reference differs")
        reference, frozen = arrays, frozen_key
        metadata_record, weak_path, weak_records = meta, current_weak, current_weak_records
        paths.update((binding, strong_path, current_weak, cache, score_path, frozen_path))
    if not cells:
        raise ValueError("at least one native binding required")
    return dict(execution_complete=True, publication_ready=False, mechanism_complete=False,
                checked_cells=sorted(cells), checked_cell_count=len(cells),
                expected_canonical_cell_count=EXPECTED_CANONICAL_CELLS, n_images=650,
                n_pairs=5308, n_recorded_persons=int(len(np.unique(people))),
                exact_pair_arrays_equal=True, frozen_embedding_bytes_equal=True,
                weak_embedding_bindings_equal=True,
                limitation=LIMITATION), paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    # Discover selected records, snapshot, then independently repeat against that snapshot.
    _, paths = audit(args.bindings)
    paths.update((Path(__file__).resolve(), PROJECT_ROOT / "scripts/run_common_mechanism_v1.py",
                  PROJECT_ROOT / "scripts/evaluate_oriented_cuda_v1.py",
                  PROJECT_ROOT / "src/age_gap/common/manifest.py",
                  PROJECT_ROOT / "src/age_gap/common/io.py"))
    inputs = sorted(paths)
    before = records(inputs, "selected protocol ancestry changed")
    payload, repeated_paths = audit(args.bindings)
    if not repeated_paths <= paths or before != records(inputs, "selected protocol ancestry changed"):
        raise RuntimeError("selected protocol ancestry changed")
    args.out.mkdir(parents=True)
    summary = args.out / "summary.json"
    target = args.out / "summary.manifest.json"
    try:
        summary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
        write_experiment_manifest(target, experiment=EXPERIMENT,
                                  parameters=dict(seed=None,
                                                  scope="targeted selected canonical cells"),
                                  metrics=payload, inputs=inputs, outputs=[summary])
        validate_written_inputs(target, before)
        if before != records(inputs, "selected inputs changed after manifest publication"):
            raise RuntimeError("selected inputs changed after manifest publication")
    except BaseException:
        # Remove only this invocation's newly written completion manifest.
        if target.exists():
            target.unlink()
        raise
    print(json.dumps(payload, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
