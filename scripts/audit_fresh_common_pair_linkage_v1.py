"""Selected-record fresh family linkage audit; not full ancestry or human clearance."""

from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record
from scripts.fresh_weak_common_binding_v1 import (
    PAIR_FIELDS,
    exact_pairs,
    read_vectors,
    strong_cell,
    weak_files,
)
from scripts.shared_byte_integrity_v2 import parse_manifest_strict


def resolve(value):
    path = Path(value)
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("selected path must remain inside project")
    return path


def linked(path, records):
    if file_record(path) not in records:
        raise ValueError("exact selected record not bound")


def parent(common, flag):
    command = common["command"]
    if (
        type(command) is not list
        or not all(type(token) is str for token in command)
        or command.count(flag) != 1
    ):
        raise ValueError("one literal parent argument required")
    index = command.index(flag) + 1
    if index >= len(command) or command[index].startswith("--"):
        raise ValueError("literal parent path missing")
    path = resolve(command[index])
    linked(path, common["inputs"])
    return path


def audit(binding):
    """One completed new-schema cell; compare every selected original frame/vector."""
    binding = resolve(binding)
    before_binding = file_record(binding)
    common, _, _ = parse_manifest_strict(binding.read_bytes())
    if (
        common["experiment"] != "common-image-index-mechanism-v2-fresh-weak"
        or common["metrics"].get("execution_complete") is not True
    ):
        raise ValueError("completed fresh-common binding required")
    strong_path, weak_path = parent(common, "--strong"), parent(common, "--weak")
    before_parents = [file_record(p) for p in (strong_path, weak_path)]
    strong, _, _ = parse_manifest_strict(strong_path.read_bytes())
    weak, _, _ = parse_manifest_strict(weak_path.read_bytes())
    cell = strong_cell(strong)
    files = weak_files(weak, weak_path)
    if (
        common["metrics"].get("strong_cells") != [cell]
        or common["metrics"].get("weak_keys") != list(files)
        or common["metrics"].get("weak_upstream_bindings") != weak["metrics"]["bindings"]
    ):
        raise ValueError("common summary not linked to literal parents")
    cache = resolve(PROJECT_ROOT / "data/external/fgnet_crops.npz")
    for native in (common, strong, weak):
        linked(cache, native["inputs"])
    paths = {binding, strong_path, weak_path, cache}
    pair_paths = [resolve(p.parent / "private/scores.npz") for p in (strong_path, weak_path)]
    selected_vectors = {
        "strong": {
            k: resolve(strong_path.parent / f"private/{k}_embeddings.npz") for k in ("frozen", cell)
        },
        "weak": {k: resolve(p) for k, p in files.items()},
    }
    for native, pair, vectors in (
        (strong, pair_paths[0], selected_vectors["strong"]),
        (weak, pair_paths[1], selected_vectors["weak"]),
    ):
        for path in (pair, *vectors.values()):
            linked(path, native["outputs"])
            linked(path, common["inputs"])
            paths.add(path)
    before = {path: file_record(path) for path in paths}
    with np.load(cache, allow_pickle=False) as data:
        people, ages = data["subjects"], data["ages"]
    frames = []
    for path in pair_paths:
        with np.load(path, allow_pickle=False) as data:
            frames.append({k: data[k] for k in PAIR_FIELDS})
    exact_pairs(*frames, people, ages)
    for vectors in selected_vectors.values():
        for path in vectors.values():
            read_vectors(path)
    if (
        before_binding != file_record(binding)
        or before_parents != [file_record(p) for p in (strong_path, weak_path)]
        or before != {path: file_record(path) for path in paths}
    ):
        raise ValueError("selected bytes changed during audit")
    return dict(
        exact_pair_arrays_equal=True,
        typed_vectors_checked=6,
        n_images=650,
        n_pairs=5308,
        checked_cells=[cell],
        fresh_weak_family=True,
        full_ancestry_reverified=False,
        numerical_probe_replay=False,
        human_identity_audit=False,
        mechanism_complete=False,
        publication_ready=False,
    ), paths
