"""Render accepted completed v3 aggregate training histories: one standalone LaTeX table fragment
and one four-panel PDF figure. Presentation only, never inference.
Reads a hash-verified ``strong-native-training-trajectories-v3`` aggregate summary and its published
manifest, enforces declared membership from repeated ``--expected-cell`` flags (no globbing),
strict-parses the bytes (duplicate/non-finite/bool rejection), re-runs the immutable descriptive
reducer for the eight-epoch contract, and closes the whole physical input set with
``shared_byte_integrity_v2`` (aliases re-resolved before and after; every written summary/table/figure
byte re-verified after the manifest). It never infers validation loss, never treats the diagnostic AUC
peak as checkpoint selection, never forms a seed mean, and never makes a seed-population, mechanism,
confidence-interval or saturation claim. CPU-only; the real native run stays the root's action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.shared_byte_integrity_v2 import (
    ByteIntegrityError,
    audited_records,
    parse_manifest_strict,
    rehash_phase,
)
from scripts.strong_training_trajectory_v1 import summarize

EXPERIMENT = "strong-native-training-history-presentation-v1"
NATIVE_EXPERIMENT = "strong-native-training-trajectories-v3"
CELL_VERSION = "strong-training-trajectory-v3"
EPOCHS, MATRIX_CELLS = 8, 36
MEASUREMENTS = ("training_loss", "validation_auc", "mean_gradient_norm", "epoch_seconds")
NONNEGATIVE = ("training_loss", "mean_gradient_norm", "epoch_seconds")
NEGATIVES, SCOPES, LEARNING_RATES, SEEDS = (
    ("random", "lookalike"),
    ("head", "tail", "full"),
    (1e-6, 1e-5),
    (42, 1, 2),
)
RECORD_KEYS = frozenset({"path", "bytes", "sha256"})
PANELS = (
    ("training_loss", "Training loss"),
    ("validation_auc", "Validation ROC-AUC"),
    ("mean_gradient_norm", "Mean gradient norm"),
    ("epoch_seconds", "Epoch wall seconds"),
)
# Mandatory producing closure (own module plus helper/parser/manifest/io/settings); ``extra`` can only
# add. Fixed in code, always resolved/contained before any read.
MANDATORY_DEPENDENCIES = (
    "scripts/shared_byte_integrity_v2.py",
    "scripts/strong_training_trajectory_v1.py",
    "src/age_gap/common/manifest.py",
    "src/age_gap/common/io.py",
    "src/age_gap/settings/__init__.py",
    "src/age_gap/settings/config.py",
    "pyproject.toml",
    "uv.lock",
    "src/age_gap/settings/settings.toml",
)
LIMITATION = (
    "descriptive completed aggregate native histories only; the matrix is not established as "
    "complete; validation loss was never collected and is not inferred from validation AUC; the "
    "diagnostic AUC peak is not checkpoint selection; no seed mean, seed-population, mechanism, "
    "confidence-interval or saturation claim"
)
CLAIM_FLAGS = (
    "historical_loss_available",
    "checkpoint_reselection",
    "seed_population_claim",
    "mechanism_claim",
    "publication_clearance",
)


class RenderError(Exception):
    """One rejected declaration/contract, worded so private paths never leak."""

    def __init__(self, stage, subject, detail):
        super().__init__(f"{stage}: {subject}: {detail}")
        self.stage, self.subject, self.detail = stage, subject, detail


def _fail(stage, subject, detail):
    raise RenderError(stage, subject, detail)


def _require(condition, stage, subject, detail):
    if not condition:
        _fail(stage, subject, detail)


def _typed(value, kind, subject, field, stage="contract"):
    ok = (
        type(value) is int
        if kind == "int"
        else type(value) is str
        if kind == "str"
        else type(value) is dict
        if kind == "obj"
        else type(value) is not bool
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )
    if not ok:
        _fail(stage, subject, f"reason=type field={field} expected={kind}")
    return value


def _fields(mapping, specs, subject, stage="contract"):
    return [_typed(mapping.get(field), kind, subject, field, stage) for field, kind in specs]


def _canonical(path):
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def _exact(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_exact(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _relative_record(path, base):
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            size += len(block)
            digest.update(block)
    return {
        "path": Path(path).relative_to(base).as_posix(),
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _contained(alias, base, subject, stage):
    """Resolve a declared alias inside the root; reject escapes, escapes-on-reresolve and retargets."""
    declared = base / alias if not Path(alias).is_absolute() else Path(alias)
    if not declared.is_relative_to(base):
        _fail(stage, subject, "reason=outside-root")
    target = _canonical(declared)
    if target is None or not target.is_file() or not target.is_relative_to(base):
        _fail(stage, subject, "reason=missing-or-outside-root")
    if _canonical(declared) != target:
        _fail(stage, subject, "reason=alias-retargeted")
    return target


def _dependency_declarations(extra, *, base=None, own_module=None):
    root = Path(base) if base is not None else PROJECT_ROOT
    own = _canonical(own_module if own_module is not None else Path(__file__))
    if own is None or not own.is_file() or not own.is_relative_to(root):
        _fail("dependencies", "own-module", "reason=unresolvable")
    paths = [(Path(own_module if own_module is not None else __file__).absolute(), own)]
    for index, relative in enumerate((*MANDATORY_DEPENDENCIES, *(extra or ()))):
        alias = root / relative if not Path(relative).is_absolute() else Path(relative)
        paths.append((alias, _contained(alias, root, f"source[{index}]", "dependencies")))
    records, seen = [], set()
    for path, _target in paths:
        token = os.path.normcase(str(path))
        if token not in seen:
            seen.add(token)
            records.append(_relative_record(path, root))
    return records


def _alias_bytes(path, base, subject):
    target = _contained(path, base, subject, "preflight")
    try:
        return target.read_bytes(), target
    except (OSError, RuntimeError, ValueError):
        _fail("preflight", subject, "reason=unreadable")


def _validate_manifest(data, summary_rel):
    subject = "manifest"
    _require(type(data) is dict, "manifest", subject, "reason=manifest-shape")
    _require(
        type(data.get("schema_version")) is int and data["schema_version"] == 1,
        "manifest",
        subject,
        "reason=schema-version",
    )
    _require(
        data.get("experiment") == NATIVE_EXPERIMENT, "manifest", subject, "reason=wrong-experiment"
    )
    inputs, outputs = data.get("inputs"), data.get("outputs")
    _require(type(inputs) is list and bool(inputs), "manifest", subject, "reason=inputs-shape")
    _require(
        type(outputs) is list and len(outputs) == 1, "manifest", subject, "reason=outputs-shape"
    )
    leaves = set()
    for group, records in (("input", inputs), ("output", outputs)):
        for position, record in enumerate(records):
            item = f"{subject}.{group}[{position}]"
            _require(
                type(record) is dict and set(record) == RECORD_KEYS,
                "manifest",
                item,
                "reason=record-shape",
            )
            _fields(
                record, (("path", "str"), ("bytes", "int"), ("sha256", "str")), item, "manifest"
            )
            if group == "output":
                leaves.add(str(record["path"]).replace("\\", "/"))
    _require(leaves == {summary_rel}, "manifest", subject, "reason=summary-not-a-declared-output")
    _require(
        "parameters" in data and "metrics" in data,
        "manifest",
        subject,
        "reason=missing-parameters-or-metrics",
    )
    return data


def _parse_key(key, subject="membership"):
    if type(key) is not str or len(key.split("|")) != 4:
        _fail("membership", subject, "reason=cell-key-shape")
    negative, scope, lr, seed = key.split("|")
    if not (lr.startswith("lr") and seed.startswith("seed")):
        _fail("membership", subject, "reason=cell-key-spelling")
    try:
        learning_rate, seed_value = float(lr[2:]), int(seed[4:])
    except (TypeError, ValueError):
        _fail("membership", subject, "reason=cell-key-spelling")
    for name, ok in (
        ("negative", negative in NEGATIVES),
        ("trainable_scope", scope in SCOPES),
        ("learning_rate", learning_rate in LEARNING_RATES and f"lr{learning_rate:g}" == lr),
        ("seed", seed_value in SEEDS and f"seed{seed_value}" == seed),
    ):
        _require(ok, "membership", subject, f"reason=cell field={name}")
    return negative, scope, learning_rate, seed_value


def _membership(cells):
    if type(cells) not in (list, tuple) or not cells:
        _fail("membership", "expected", "reason=empty-or-wrong-type")
    declared = set()
    for index, raw in enumerate(cells):
        _parse_key(raw, f"expected[{index}]")
        if raw in declared:
            _fail("membership", f"expected[{index}]", "reason=duplicate-expected-cell")
        declared.add(raw)
    _require(len(declared) <= MATRIX_CELLS, "membership", "expected", "reason=exceeds-matrix")
    return declared


def _validate_cell(cell, key):
    subject, stage = f"cell[{key}]", "contract"
    _require(type(cell) is dict, stage, subject, "reason=cell-shape")
    _require(cell.get("version") == CELL_VERSION, stage, subject, "reason=cell-version")
    epochs, selected = _fields(cell, (("epochs", "int"), ("selected_epoch", "int")), subject, stage)
    _require(epochs == EPOCHS and selected == EPOCHS, stage, subject, "reason=epochs")
    _require(
        cell.get("checkpoint_selection") == "last_epoch",
        stage,
        subject,
        "reason=checkpoint_selection",
    )
    _require(
        "validation_loss" in cell and cell["validation_loss"] is None,
        stage,
        subject,
        "reason=fabricated-validation-loss",
    )
    for flag in CLAIM_FLAGS:
        _require(cell.get(flag) is False, stage, subject, f"reason=claim field={flag}")
    fields = _typed(cell.get("cell"), "obj", subject, "cell")
    negative, scope, learning_rate, seed = _fields(
        fields,
        (
            ("negative", "str"),
            ("trainable_scope", "str"),
            ("learning_rate", "num"),
            ("seed", "int"),
        ),
        subject,
        stage,
    )
    _require(
        f"{negative}|{scope}|lr{learning_rate:g}|seed{seed}" == key,
        stage,
        subject,
        "reason=cell-key-mismatch",
    )
    curves = _typed(cell.get("curves"), "obj", subject, "curves")
    _require(set(curves) == set(MEASUREMENTS), stage, subject, "reason=curve-set")
    values = {}
    for measurement in MEASUREMENTS:
        curve = _typed(curves.get(measurement), "obj", subject, measurement)
        raw = curve.get("values")
        _require(
            type(raw) is list and len(raw) == EPOCHS,
            "history",
            subject,
            f"reason=history-shape field={measurement}",
        )
        nums = [
            _typed(item, "num", subject, f"{measurement}.values[{i}]", "history")
            for i, item in enumerate(raw)
        ]
        derived = {
            "first": nums[0],
            "last": nums[-1],
            "change": nums[-1] - nums[0],
            "min": min(nums),
            "max": max(nums),
        }
        for name, expected in derived.items():
            _require(
                _typed(curve.get(name), "num", subject, f"{measurement}.{name}") == expected,
                "curves",
                subject,
                f"reason=derived-mismatch field={measurement}.{name}",
            )
        values[measurement] = nums
    _require(
        all(0.0 <= item <= 1.0 for item in values["validation_auc"]),
        "history",
        subject,
        "reason=auc-range",
    )
    for measurement in NONNEGATIVE:
        _require(
            all(item >= 0 for item in values[measurement]),
            "history",
            subject,
            f"reason=negative field={measurement}",
        )
    descriptive = summarize(
        [{"epoch": i + 1, **{m: values[m][i] for m in MEASUREMENTS}} for i in range(EPOCHS)],
        epochs=EPOCHS,
    )
    _require(
        _exact(cell.get("descriptive"), descriptive), stage, subject, "reason=descriptive-mismatch"
    )
    return {
        "key": key,
        "label": f"{negative}/{scope}/{learning_rate:g}/seed{seed}",
        "negative": negative,
        "trainable_scope": scope,
        "learning_rate": learning_rate,
        "seed": seed,
        "curves": {m: values[m] for m in MEASUREMENTS},
        "training_loss_first": values["training_loss"][0],
        "training_loss_last": values["training_loss"][-1],
        "validation_auc_first": values["validation_auc"][0],
        "validation_auc_last": values["validation_auc"][-1],
        "last_gradient_norm": values["mean_gradient_norm"][-1],
        "epoch_minutes_total": sum(values["epoch_seconds"]) / 60.0,
        "diagnostic_peak_auc_epoch": descriptive["diagnostic_peak_auc_epoch"],
    }


def _validate_summary(data, expected):
    subject = "summary"
    _require(type(data) is dict, "contract", subject, "reason=summary-shape")
    _require(
        data.get("execution_complete") is True, "contract", subject, "reason=execution-complete"
    )
    _require(
        data.get("publication_ready") is False, "contract", subject, "reason=publication-ready"
    )
    _require(
        "validation_loss" in data and data["validation_loss"] is None,
        "contract",
        subject,
        "reason=fabricated-validation-loss",
    )
    for flag in CLAIM_FLAGS:
        _require(data.get(flag) is False, "contract", subject, f"reason=claim field={flag}")
    cells = _typed(data.get("cells"), "obj", subject, "cells")
    _require(
        _typed(data.get("cell_count"), "int", subject, "cell_count") == len(cells),
        "contract",
        subject,
        "reason=cell-count",
    )
    _require(set(cells) == expected, "membership", subject, "reason=declared-membership-mismatch")
    return [_validate_cell(cells[key], key) for key in sorted(cells)]


def _payload(rows, expected):
    return {
        "version": EXPERIMENT,
        "execution_complete": True,
        "publication_ready": False,
        "source_experiment": NATIVE_EXPERIMENT,
        "source_cell_version": CELL_VERSION,
        "membership": {
            "declared_cells": sorted(expected),
            "declared_cell_count": len(expected),
            "matrix_cell_count": MATRIX_CELLS,
            "selection": "explicit repeated --expected-cell; no directory globbing",
        },
        "validation_loss": None,
        "validation_loss_status": "not collected; never inferred from validation AUC",
        "diagnostic_peak_is_checkpoint_selection": False,
        "seed_mean": None,
        "aggregate_seed_mean": False,
        "mechanism_claim": False,
        "seed_population_claim": False,
        "confidence_interval_claim": False,
        "saturation_claim": False,
        "rows": rows,
        "limitation": LIMITATION,
    }


def _table(payload):
    def label(row):
        return (
            f"{row['negative'].capitalize()} {row['trainable_scope']} "
            f"$10^{{{int(round(math.log10(row['learning_rate'])))}}}$ s{row['seed']}"
        )

    header = r"Checkpoint & Loss first & Loss last & AUC first & AUC last & Last grad.\ norm & Total epoch min. \\"
    body = [
        (r"{} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.2f} \\").format(
            label(row),
            row["training_loss_first"],
            row["training_loss_last"],
            row["validation_auc_first"],
            row["validation_auc_last"],
            row["last_gradient_norm"],
            row["epoch_minutes_total"],
        )
        for row in payload["rows"]
    ]
    return (
        "\n".join(
            (
                r"\begin{table*}[!t]",
                rf"\caption{{Completed aggregate training histories: {len(payload['rows'])} of {MATRIX_CELLS} "
                "declared matrix checkpoints, each conditional on its own last-epoch checkpoint. Curves are the "
                "recorded aggregate native metrics; validation loss was not collected and is never inferred from "
                "validation AUC. The diagnostic AUC peak is not checkpoint selection. No aggregate seed mean, no "
                "interval and no seed-population, mechanism or saturation claim; the matrix is not established as "
                "complete.}",
                r"\label{tab:strong-history-v1}",
                r"\centering\scriptsize",
                r"\begin{tabular}{@{}lrrrrrr@{}}",
                r"\toprule",
                header,
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
            )
        )
        + "\n"
    )


def _figure(payload, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, EPOCHS + 1))
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for axis, (measurement, title) in zip(axes.ravel(), PANELS, strict=True):
        for row in payload["rows"]:
            axis.plot(
                epochs,
                row["curves"][measurement],
                marker="o",
                markersize=3,
                linewidth=1.0,
                label=row["label"],
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.grid(alpha=0.3)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=7,
        frameon=False,
        title="negative / scope / learning rate / seed",
    )
    figure.text(
        0.5,
        0.005,
        "No validation loss; diagnostic peak is not checkpoint selection; no seed mean, "
        "seed-population, mechanism or saturation claim.",
        ha="center",
        va="bottom",
        fontsize=6,
    )
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    figure.savefig(
        path,
        format="pdf",
        metadata={
            "Title": "Strong fixed8 aggregate training histories",
            "Subject": "descriptive only; no seed mean, mechanism or saturation claim",
        },
    )
    plt.close(figure)


def _prepare_destination(destination, base):
    if destination.exists():
        raise FileExistsError("fresh destination required")
    parent = _canonical(destination.parent)
    if parent is None or not parent.is_dir() or not parent.is_relative_to(base):
        _fail("output", "destination", "reason=outside-root-or-missing-parent")
    expected = parent / destination.name
    if expected.exists():
        raise FileExistsError("fresh destination required")
    destination.mkdir(parents=False, exist_ok=False)
    actual = _canonical(expected)
    if actual is None or actual != expected or not actual.is_dir():
        _fail("output", "destination", "reason=retargeted-after-creation")
    return actual


def _guard_destination(alias, destination, base):
    if (
        _canonical(alias) != destination
        or _canonical(destination) != destination
        or not destination.is_relative_to(base)
        or not destination.is_dir()
    ):
        _fail("output", "destination", "reason=owned-output-retargeted")


def _metadata_record(alias, raw, base):
    alias = Path(alias)
    alias = alias if alias.is_absolute() else base / alias
    if not alias.is_relative_to(base):
        _fail("preflight", "metadata", "reason=outside-root")
    return {
        "path": alias.relative_to(base).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _expected_inputs(paths, native, deps, base, dep_base):
    expected, seen = [], set()
    for path in paths:
        resolved = _canonical(path)
        token = str(resolved).lower() if resolved else str(path).lower()
        if token in seen:
            _fail("publication", "inputs", "reason=duplicate-input-path")
        seen.add(token)
        snapshot = None
        for record_map, root in ((native, base), (deps, dep_base)):
            try:
                snapshot = record_map.get(
                    os.path.normcase(str(resolved.relative_to(root))).replace("\\", "/")
                )
            except (ValueError, AttributeError):
                snapshot = None
            if snapshot is not None:
                break
        if snapshot is None:
            _fail("publication", "inputs", "reason=input-not-in-phase1-snapshot")
        try:
            current = file_record(path)
        except (OSError, RuntimeError, ValueError):
            _fail("publication", "inputs", "reason=unreadable-declared-input")
        if (current["bytes"], current["sha256"]) != (snapshot["bytes"], snapshot["sha256"]):
            _fail("publication", "inputs", "reason=input-changed-after-phase1")
        expected.append(
            {"path": current["path"], "bytes": snapshot["bytes"], "sha256": snapshot["sha256"]}
        )
    return expected


def _verify_published(
    target,
    expected_inputs,
    expected_outputs,
    expected_parameters,
    payload,
    native,
    deps,
    base,
    dep_base,
    destination,
):
    rehash_phase(native, base)
    rehash_phase(deps, dep_base)
    if (
        _canonical(destination) != destination
        or not destination.is_relative_to(base)
        or not destination.is_dir()
    ):
        _fail("output", "destination", "reason=retargeted-before-or-after-write")
    try:
        data = parse_manifest_strict(Path(target).read_bytes())[0]
    except (OSError, RuntimeError, ValueError):
        _fail("publication", "manifest", "reason=unreadable-manifest")
    _require(type(data) is dict, "publication", "manifest", "reason=manifest-shape")
    _require(
        data.get("experiment") == EXPERIMENT, "publication", "manifest", "reason=wrong-experiment"
    )
    _require(
        type(data.get("schema_version")) is int and data["schema_version"] == 1,
        "publication",
        "manifest",
        "reason=schema-version",
    )
    for field, expected in (
        ("inputs", expected_inputs),
        ("outputs", expected_outputs),
        ("parameters", expected_parameters),
        ("metrics", payload),
    ):
        _require(
            _exact(data.get(field), expected),
            "publication",
            "manifest",
            f"reason={field}-changed-during-publication",
        )
    final = []
    for record in expected_outputs:
        path = Path(record["path"])
        try:
            final.append(file_record(path if path.is_absolute() else base / path))
        except (OSError, RuntimeError, ValueError):
            _fail("publication", "outputs", "reason=published-output-unreadable")
    _require(
        final == expected_outputs,
        "publication",
        "outputs",
        "reason=published-output-changed-after-write",
    )


def render(
    summary,
    manifest,
    expected_cells,
    out,
    *,
    root=PROJECT_ROOT,
    dependencies=None,
    dependency_root=None,
    own_module=None,
    figure_writer=None,
):
    """Validate, render and publish one fresh presentation directory."""
    base = Path(root).resolve()
    dep_base = Path(dependency_root).resolve() if dependency_root is not None else base
    expected = _membership(expected_cells)
    summary_bytes, summary_target = _alias_bytes(summary, base, "summary")
    manifest_bytes, manifest_target = _alias_bytes(manifest, base, "manifest")
    if Path(out).exists():
        raise FileExistsError("fresh destination required")
    manifest_data = _validate_manifest(
        parse_manifest_strict(manifest_bytes)[0], summary_target.relative_to(base).as_posix()
    )
    summary_data = parse_manifest_strict(summary_bytes)[0]
    _require(
        _exact(summary_data, manifest_data.get("metrics")),
        "publication",
        "summary",
        "reason=summary-differs-from-manifest-metrics",
    )
    dependencies = _dependency_declarations(dependencies, base=dep_base, own_module=own_module)
    # Bind exactly the bytes that were parsed, retaining caller alias spellings.
    # A later read must never retrospectively authenticate different metadata.
    native_declarations = [
        _metadata_record(manifest, manifest_bytes, base),
        _metadata_record(summary, summary_bytes, base),
    ]
    native_declarations += list(manifest_data["inputs"]) + list(manifest_data["outputs"])
    native_before, dep_before = (
        audited_records(native_declarations, base),
        audited_records(dependencies, dep_base),
    )
    payload = _payload(_validate_summary(summary_data, expected), expected)
    input_paths = sorted(
        {
            snapshot_base / key
            for snapshot, snapshot_base in ((native_before, base), (dep_before, dep_base))
            for key in snapshot
        },
        key=str,
    )
    expected_inputs = _expected_inputs(input_paths, native_before, dep_before, base, dep_base)
    destination = _prepare_destination(Path(out), base)
    output_alias = Path(out)
    summary_path, table_path = destination / "summary.json", destination / "history_table.tex"
    figure_path = destination / "history_figure.pdf"
    target = destination / "history.manifest.json"
    expected_parameters = dict(
        epochs=EPOCHS,
        checkpoint_selection="last_epoch",
        matrix_cell_count=MATRIX_CELLS,
        declared_cell_count=len(expected),
        figure="four-panel-pdf",
    )
    try:
        _guard_destination(output_alias, destination, base)
        summary_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        _guard_destination(output_alias, destination, base)
        table_path.write_text(_table(payload), encoding="utf-8")
        _guard_destination(output_alias, destination, base)
        (figure_writer or _figure)(payload, figure_path)
        _guard_destination(output_alias, destination, base)
        expected_outputs = [file_record(path) for path in (summary_path, table_path, figure_path)]
        _guard_destination(output_alias, destination, base)
        write_experiment_manifest(
            target,
            experiment=EXPERIMENT,
            parameters=expected_parameters,
            metrics=payload,
            inputs=input_paths,
            outputs=[summary_path, table_path, figure_path],
        )
        _guard_destination(output_alias, destination, base)
        _verify_published(
            target,
            expected_inputs,
            expected_outputs,
            expected_parameters,
            payload,
            native_before,
            dep_before,
            base,
            dep_base,
            destination,
        )
    except BaseException:
        # Retargeted output is not ours to mutate or clean up.
        try:
            _guard_destination(output_alias, destination, base)
            if (
                target.exists()
                and _canonical(target) == target
                and target.is_relative_to(destination)
            ):
                target.unlink()
            if summary_path.exists() and _canonical(summary_path) == summary_path:
                failed_payload = dict(payload, execution_complete=False)
                summary_path.write_text(
                    json.dumps(failed_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
                )
        except (RenderError, OSError):
            pass
        raise
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-cell", action="append", default=[], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    try:
        payload = render(args.summary, args.manifest, args.expected_cell, args.out, root=args.root)
    except FileExistsError:
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": "output",
                    "subject": "out",
                    "detail": "reason=fresh-destination-required",
                }
            )
        )
        return 1
    except ByteIntegrityError as error:
        print(
            json.dumps(
                {"ok": False, "stage": "integrity", "subject": "bytes", "detail": str(error)}
            )
        )
        return 1
    except RenderError as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": error.stage,
                    "subject": error.subject,
                    "detail": error.detail,
                }
            )
        )
        return 1
    except Exception:
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": "render",
                    "subject": "artifact",
                    "detail": "reason=unexpected-failure",
                }
            )
        )
        return 1
    print(json.dumps({"ok": True, "rows": len(payload["rows"]), "outputs": 3}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
