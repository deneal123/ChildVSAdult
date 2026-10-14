"""Source-only contracts for prospective fresh weak/common diagnostics.

This is not independent native acceptance: the coordinator must first accept
real native producer/reader outputs. Old weak cache-key schemas do not apply.
"""

import ast
import re
from pathlib import Path

import numpy as np

WEAK_EXPERIMENT = "facenet-fixed8-validation-full-fgnet-cuda-v3"
WEAK_KEYS = ("frozen", "tuned_s1", "tuned_s2", "tuned_s42")
PAIR_FIELDS = ("left", "right", "labels", "subject_a", "subject_b", "source_gap")


def strong_cell(native):
    """One safe canonical cell and the explicit strong source protocol."""
    p, m = native["parameters"], native["metrics"]
    names = p["actual_cells"]
    if (
        native["experiment"] != "adaface-fixed8-cuda-full-fgnet-roc-v2"
        or m["execution_complete"] is not True
        or m["publication_ready"] is not False
        or type(names) is not list
        or len(names) != 1
        or type(names[0]) is not str
        or not re.fullmatch(
            r"(random|lookalike)_(head|tail|full)_lr(1e-06|1e-05)_s(42|1|2)", names[0]
        )
    ):
        raise ValueError("one completed canonical strong cell required")
    for key, expected in (("protocol_seed", 42), ("tolerance", 2)):
        if type(p[key]) is not int or p[key] != expected:
            raise ValueError("strong pair protocol differs")
    return names[0]


def weak_files(native, manifest_path):
    """Map by explicit key, never by the old 42-first ordering or old filenames."""
    p, m = native["parameters"], native["metrics"]
    if (
        native["experiment"] != WEAK_EXPERIMENT
        or m["execution_complete"] is not True
        or m["publication_ready"] is not False
        or p["native_experiment"] != "facenet-fixed8-validation-loss-v2"
        or p["device"] != "cuda"
        or p["image_inference"] is not True
        or p["evaluator_version"] != "weak-fixed8-evaluator-v3-root-e421"
    ):
        raise ValueError("completed fresh weak v3 source contract required")
    for key, expected in (
        ("protocol_seed", 42),
        ("tolerance", 2),
        ("n_boot", 2000),
        ("bootstrap_seed", 0),
    ):
        if type(p[key]) is not int or p[key] != expected:
            raise ValueError("fresh weak protocol/resampling differs")
    if (
        m["seeds"] != [1, 2, 42]
        or not all(type(s) is int for s in m["seeds"])
        or p["actual_cells"] != list(WEAK_KEYS[1:])
        or m["actual_cells"] != list(WEAK_KEYS[1:])
        or type(m["n_crops"]) is not int
        or m["n_crops"] != 650
    ):
        raise ValueError("all three fresh weak seeds/full pool required")
    bindings = m["bindings"]
    if (
        type(bindings) is not list
        or len(bindings) != 3
        or not all(type(value) is str and value.strip() for value in bindings)
        or len({Path(value).resolve() for value in bindings}) != 3
    ):
        raise ValueError("three unique upstream native bindings required")
    base = Path(manifest_path).parent / "private"
    return {key: base / f"embeddings_{key}.npz" for key in WEAK_KEYS}


def exact_pairs(strong, weak, people, ages):
    """Validate both bound chronological protocols and exact source-array equality."""
    if (
        people.shape != (650,)
        or ages.shape != (650,)
        or people.dtype.kind not in "iu"
        or ages.dtype.kind not in "iu"
        or (people < 0).any()
        or (ages < 0).any()
        or ages.max() > np.iinfo(np.int64).max
        or len(np.unique(people)) != 82
    ):
        raise ValueError("complete chronological FGNET metadata required")
    for source in (strong, weak):
        for key in PAIR_FIELDS:
            value = source[key]
            if value.shape != (5308,) or value.dtype.kind not in "iu":
                raise ValueError("typed full-pool pair arrays required")
        left, right, labels = (source[k] for k in ("left", "right", "labels"))
        if (
            (left < 0).any()
            or (right < 0).any()
            or (left >= 650).any()
            or (right >= 650).any()
            or set(np.unique(labels)) != {0, 1}
        ):
            raise ValueError("bounded indices and two pair classes required")
        if (
            not np.array_equal(source["subject_a"], people[left])
            or not np.array_equal(source["subject_b"], people[right])
            or not np.array_equal(labels, (people[left] == people[right]).astype(int))
            or not np.array_equal(
                source["source_gap"],
                np.abs(ages[left].astype(np.int64) - ages[right].astype(np.int64)),
            )
        ):
            raise ValueError("chronological endpoint/label linkage differs")
    if any(not np.array_equal(strong[key], weak[key]) for key in PAIR_FIELDS):
        raise ValueError("strong/fresh weak pair arrays differ")


def read_vectors(path):
    """Indices are integers, full ordered pool; vectors finite unit float512."""
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"indices", "embeddings"}:
            raise ValueError("new typed embedding layout required")
        indices, vectors = data["indices"], data["embeddings"]
    if (
        indices.dtype.kind not in "iu"
        or indices.shape != (650,)
        or not np.array_equal(indices, np.arange(650))
        or vectors.dtype.kind != "f"
        or vectors.shape != (650, 512)
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    ):
        raise ValueError("full finite unit embedding pool required")
    return vectors


def script_ancestry(script, root):
    """Literal local imports only, not dynamic or third-party runtime closure."""
    pending, found = [Path(script).resolve()], set()
    base = (root / "scripts").resolve()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError("local script dependency unavailable")
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = (
                    ["scripts." + item.name for item in node.names]
                    if node.module == "scripts"
                    else [node.module]
                )
            elif isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            for module in modules:
                if module.startswith("scripts."):
                    pending.append(root.joinpath(*module.split(".")).with_suffix(".py").resolve())
    package = base / "__init__.py"
    if package.is_file():
        found.add(package)
    return sorted(found)
