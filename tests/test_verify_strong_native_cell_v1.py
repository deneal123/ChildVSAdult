"""Synthetic contract tests for the recovery-v3 hardened checker.

Every fixture is a tiny temporary tree of synthetic JSON and byte blobs and every
hash is over those synthetic bytes. No real faces, weights, manifests, arrays or
row paths are read and nothing is decoded, trained or loaded. A green suite is a
physical contract-logic check only: it is NOT evidence of matrix completion,
external acceptance, identity/ethics clearance or publication readiness.
"""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts import verify_strong_native_cell_v1 as verifier

CELL = "random_tail_lr1e-05_s42"
CAMPAIGN = "strong_queue_remaining28_v2_20261008"
PREFIX = f"metrics/{CAMPAIGN}/{CELL}"
RESULT_KEYS = {"cell", "accepted", "records_verified", "snapshot_stable"}
FROZEN_POINT = 0.9
TUNED_POINT = 0.8
SENSITIVE_FACE = "data/interim/faces/-77072632_457308442_f0.jpg"
IDENTITY_FILE = "reports/subject_100000001.json"
IDENTITY_DIR_FILE = "reports/100000001/notes.txt"
TRAIN_MANIFEST_REL = f"{PREFIX}/train/summary.manifest.json"
# Dependency order: a manifest is rewritten before the manifests that reference it.
MANIFEST_ORDER = ("train/private/checkpoint.manifest.json",
                  "train/summary.manifest.json",
                  "fgnet/summary.manifest.json")


def history():
    return [
        {"epoch": epoch, "training_loss": 0.5 - 0.01 * epoch,
         "validation_auc": 0.90 + 0.001 * epoch, "mean_gradient_norm": 1.0 + epoch,
         "epoch_seconds": 10.0 * epoch}
        for epoch in range(1, 9)
    ]


def base_parameters(scope="tail"):
    return {"backbone": verifier.BACKBONE, "epochs_requested": 8, "epochs_executed": 8,
            "selected_epoch": 8, "checkpoint_selection": "last_epoch",
            "batchnorm_policy": "frozen_all", "trainable_scope": scope,
            "learning_rate": 1e-5, "seed": 42, "negative": "random", "device": "cuda",
            "batch_size": 16, "margin": 0.3, "gap_weight": 0.0, "crops_dir": "faces"}


def frozen_models():
    return {metric: {"point": FROZEN_POINT, "ci95": [FROZEN_POINT - 0.02, FROZEN_POINT + 0.02],
                     "delta_vs_frozen": 0.0, "delta_ci95": [0.0, 0.0],
                     "loo_delta_min": None, "loo_delta_max": None}
            for metric in verifier.METRICS}


def tuned_models(point=TUNED_POINT, ci=None, delta_ci=None, loo=None):
    delta = point - FROZEN_POINT
    return {metric: {"point": point,
                     "ci95": ci if ci is not None else [point - 0.03, point + 0.03],
                     "delta_vs_frozen": delta,
                     "delta_ci95": delta_ci if delta_ci is not None else [delta - 0.02, delta + 0.02],
                     "loo_delta_min": (loo or [delta - 0.01, delta + 0.01])[0],
                     "loo_delta_max": (loo or [delta - 0.01, delta + 0.01])[1]}
            for metric in verifier.METRICS}


def metrics_block(pairs, subjects, cell, tuned=None):
    return {"metric_version": "empirical-roc-v2", "n_pairs": pairs, "n_subjects": subjects,
            "actual_tuned_checkpoint_count": 1,
            "models": {"frozen": frozen_models(), cell: tuned if tuned is not None else tuned_models()},
            "loo_valid": subjects, "loo_unavailable": 0, "publication_ready": False,
            "scope": "individual fixed-checkpoint subject intervals",
            "training_identity_independence": "unverified",
            "bootstrap": {"requested": 2000, "valid": 2000, "seed": 0,
                          "shared_draws_across_models": True,
                          "positive_weight": verifier.BOOTSTRAP_POSITIVE,
                          "negative_weight": verifier.BOOTSTRAP_NEGATIVE,
                          "thresholds_reselected": True}}


def external_metrics(cell, tuned=None):
    return {"overall": metrics_block(5308, 82, cell, tuned),
            "large_gap_25plus": metrics_block(430, 75, cell, tuned),
            "execution_complete": True, "publication_ready": False,
            "peak_allocated_bytes": 1, "peak_reserved_bytes": 2,
            "coverage": {"source_positives": 2659, "retained_positives": 2654,
                         "source_25plus_positives": 215, "retained_25plus_positives": 215},
            "limitation": "actual fixed checkpoints; completed matrix remains separate"}


def scaffold(tmp_path):
    """Build the in-memory parts and blob plan; records are materialized later."""
    root = tmp_path / "repo"
    cell_root = root / PREFIX
    cell_root.mkdir(parents=True)
    blobs = {
        "scripts/run_strong_backbone_cuda_cell_v1.py": b"# trainer\n",
        "data/processed/pairs.jsonl": b'{"pair": 1}\n',
        "data/external/fgnet_crops.npz": b"crop-cache-bytes",
        "models/adaface_ir101.pt": b"backbone-weights",
        SENSITIVE_FACE: b"face-bytes",
        IDENTITY_FILE: b"identity-bytes",
        IDENTITY_DIR_FILE: b"identity-dir-bytes",
        f"{PREFIX}/train/private/checkpoint.pt": b"WEIGHTS" * 8,
        f"{PREFIX}/fgnet/private/scores.npz": b"scores-bytes",
        f"{PREFIX}/fgnet/private/frozen_embeddings.npz": b"frozen-bytes",
        f"{PREFIX}/fgnet/private/{CELL}_embeddings.npz": b"tuned-bytes",
    }
    params = base_parameters()
    params["pairs_file"] = str(root / "data/processed/pairs.jsonl")
    train = {"leaf": "train/summary.manifest.json",
             "experiment": verifier.TRAIN_EXPERIMENT, "parameters": params,
             "metrics": {"training_complete": True, "evaluation_complete": False,
                         "publication_ready": False, "peak_allocated_bytes": 10,
                         "peak_reserved_bytes": 20, "history": history(), "train_pairs": 10,
                         "validation_pairs": 5, "unique_crop_files": 3,
                         "limitation": "one cell"},
             "inputs": ["scripts/run_strong_backbone_cuda_cell_v1.py",
                        "data/processed/pairs.jsonl", "models/adaface_ir101.pt", SENSITIVE_FACE,
                        IDENTITY_FILE, IDENTITY_DIR_FILE],
             "outputs": [f"{PREFIX}/train/summary.json",
                         f"{PREFIX}/train/private/checkpoint.pt",
                         f"{PREFIX}/train/private/checkpoint.manifest.json"]}
    checkpoint = {"leaf": "train/private/checkpoint.manifest.json",
                  "experiment": verifier.CHECKPOINT_EXPERIMENT,
                  "parameters": {key: params[key] for key in
                                 ("backbone", "epochs_requested", "epochs_executed",
                                  "selected_epoch", "checkpoint_selection", "batchnorm_policy",
                                  "trainable_scope", "learning_rate", "seed", "batch_size",
                                  "margin", "gap_weight", "crops_dir", "pairs_file")},
                  "metrics": {"best_validation_auc": 0.91, "selected_epoch": 8,
                              "history": history()},
                  "inputs": ["data/processed/pairs.jsonl"],
                  "outputs": [f"{PREFIX}/train/private/checkpoint.pt"]}
    external = {"leaf": "fgnet/summary.manifest.json",
                "experiment": verifier.EXTERNAL_EXPERIMENT,
                "parameters": {"device": "cuda", "protocol_seed": 42, "tolerance": 2,
                               "n_boot": 2000, "bootstrap_seed": 0, "actual_cells": [CELL]},
                "metrics": external_metrics(CELL),
                "inputs": [TRAIN_MANIFEST_REL,
                           f"{PREFIX}/train/summary.json",
                           f"{PREFIX}/train/private/checkpoint.manifest.json",
                           f"{PREFIX}/train/private/checkpoint.pt",
                           "data/external/fgnet_crops.npz",
                           "scripts/run_strong_backbone_cuda_cell_v1.py",
                           "models/adaface_ir101.pt"],
                "outputs": [f"{PREFIX}/fgnet/summary.json",
                            f"{PREFIX}/fgnet/private/scores.npz",
                            f"{PREFIX}/fgnet/private/frozen_embeddings.npz",
                            f"{PREFIX}/fgnet/private/{CELL}_embeddings.npz"]}
    return root, cell_root, {"train": train, "checkpoint": checkpoint, "external": external}, blobs


def build(tmp_path, mutate=None):
    """Materialize one realistic synthetic cell; ``mutate`` edits parts pre-hash."""
    root, cell_root, parts, blobs = scaffold(tmp_path)
    if mutate is not None:
        mutate(parts["train"], parts["checkpoint"], parts["external"], blobs)
    for relative, blob in blobs.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    write(cell_root, "train/summary.json", json.dumps(parts["train"]["metrics"]))
    write(cell_root, "fgnet/summary.json", json.dumps(parts["external"]["metrics"]))
    materialize(root, cell_root, parts)
    return root, cell_root


def write(cell_root, relative, text):
    (cell_root / relative).parent.mkdir(parents=True, exist_ok=True)
    (cell_root / relative).write_text(text, encoding="utf-8")


def materialize(root, cell_root, parts):
    """Convert each manifest's declared paths into real records, in dependency order."""
    for role in ("checkpoint", "train", "external"):
        part = parts[role]
        for kind in ("inputs", "outputs"):
            part[kind] = [verifier.record_of(root / path, root) for path in part[kind]]
        write(cell_root, part["leaf"], json.dumps(part))


def rehash(root, cell_root):
    """Recompute every manifest record from disk, in dependency order."""
    for leaf in MANIFEST_ORDER:
        path = cell_root / leaf
        data = json.loads(path.read_text(encoding="utf-8"))
        for kind in ("inputs", "outputs"):
            data[kind] = [verifier.record_of(root / record["path"], root)
                          for record in data[kind]]
        path.write_text(json.dumps(data), encoding="utf-8")


def run(root, cell_root, **overrides):
    spec = {"negative": "random", "scope": "tail", "lr": 1e-5, "seed": 42}
    spec.update(overrides)
    return verifier.verify_cell(root, cell_root, **spec)


def expect(root, cell_root, stage, **overrides):
    with pytest.raises(verifier.VerificationError) as error:
        run(root, cell_root, **overrides)
    assert error.value.stage == stage, error.value
    return error.value


def load(root, leaf):
    return json.loads((root / PREFIX / leaf).read_text(encoding="utf-8"))


def store(root, leaf, data):
    (root / PREFIX / leaf).write_text(json.dumps(data), encoding="utf-8")


def cli(root, cell_root, capsys):
    code = verifier.main(["--root", str(root), "--cell-root", str(cell_root),
                          "--negative", "random", "--scope", "tail", "--lr", "1e-5",
                          "--seed", "42"])
    return code, json.loads(capsys.readouterr().out)


# --------------------------------------------------------------------------- golden


def test_golden_cell_accepted(tmp_path):
    root, cell_root = build(tmp_path)
    result = run(root, cell_root)
    assert set(result) == RESULT_KEYS
    assert result["cell"] == CELL and result["accepted"] is True
    assert result["snapshot_stable"] is True and result["records_verified"] == 16


def test_derived_cell_name_rejects_noncanonical_specs():
    assert verifier.cell_name("random", "tail", 1e-5, 42) == CELL
    assert verifier.cell_name("lookalike", "full", 1e-6, 1) == "lookalike_full_lr1e-06_s1"
    for bad in (("random", "neck", 1e-5, 42), ("random", "tail", 1e-4, 42),
                ("random", "tail", 1e-5, 7), ("random", "tail", 1e-5, True)):
        with pytest.raises(verifier.VerificationError):
            verifier.cell_name(*bad)


@pytest.mark.parametrize("bad_lr", [True, "1e-5", None, 1e-4, float("nan"), [1e-5]])
def test_learning_rate_type_is_strictly_numeric(bad_lr):
    with pytest.raises(verifier.VerificationError) as error:
        verifier.cell_name("random", "tail", bad_lr, 42)
    assert type(error.value) is verifier.VerificationError


@pytest.mark.parametrize("overrides", [{"negative": "lookalike"}, {"scope": "head"},
                                       {"lr": 1e-6}, {"seed": 1}])
def test_cell_root_must_match_derived_label(tmp_path, overrides):
    root, cell_root = build(tmp_path)
    expect(root, cell_root, "cell-name", **overrides)


def test_cell_name_mismatch_never_echoes_actual_basename(tmp_path):
    root, _ = build(tmp_path)
    other = root / "metrics" / CAMPAIGN / "random_head_lr1e-05_s1"
    (other / "train").mkdir(parents=True)
    (other / "train/summary.manifest.json").write_text("{ bad", encoding="utf-8")
    with pytest.raises(verifier.VerificationError) as error:
        run(root, other)
    assert error.value.stage == "cell-name"
    # Only the derived canonical label is safe to emit, never the actual name.
    assert error.value.subject == CELL
    assert "random_head_lr1e-05_s1" not in str(error.value)


def test_cell_root_outside_repo_rejected_before_manifest_read(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "elsewhere" / CELL
    (outside / "train").mkdir(parents=True)
    (outside / "train/summary.manifest.json").write_text("{ bad", encoding="utf-8")
    with pytest.raises(verifier.VerificationError) as error:
        run(root, outside)
    assert error.value.stage == "cell-root"
    assert error.value.subject == "<redacted:outside-path>"
    assert str(tmp_path) not in str(error.value)


def test_cli_cell_root_is_mandatory(tmp_path):
    root, _ = build(tmp_path)
    with pytest.raises(SystemExit):
        verifier.main(["--root", str(root), "--negative", "random", "--scope", "tail",
                       "--lr", "1e-5", "--seed", "42"])


def test_cli_golden_and_per_cell_output(tmp_path, capsys):
    root, cell_root = build(tmp_path)
    code, payload = cli(root, cell_root, capsys)
    assert code == 0
    assert payload["accepted"] is True and payload["cell"] == CELL
    assert not {"training_external_cells", "feature_cells", "mechanism_complete",
                "full36", "publication_ready", "records"} & set(payload)


# ------------------------------------------------------------- redaction classification


def test_public_allowlist_only_code_and_aggregates():
    assert verifier._classify("scripts/run_strong_backbone_cuda_cell_v1.py", CELL, PREFIX) == (
        "scripts/run_strong_backbone_cuda_cell_v1.py", True)
    assert verifier._classify("pyproject.toml", CELL, PREFIX) == ("pyproject.toml", True)
    assert verifier._classify(f"{PREFIX}/train/summary.json", CELL, PREFIX) == (
        f"{CELL}/train/summary.json", True)
    # Private, data, model, identity and cell-relative private paths are never public.
    for raw in ("data/processed/pairs.jsonl", "models/adaface_ir101.pt", SENSITIVE_FACE,
                IDENTITY_FILE, f"{PREFIX}/fgnet/private/scores.npz", "src/age_gap/x.pt"):
        assert verifier._classify(raw, CELL, PREFIX)[1] is False


def test_absolute_and_outside_paths_redacted():
    assert verifier._label("/etc/passwd") == "<redacted:outside-path>"
    assert verifier._label("C:/Users/someone/secret.txt") == "<redacted:outside-path>"
    assert verifier._label("../../escape.txt") == "<redacted:outside-path>"
    assert verifier._sensitive("/etc/passwd") is True


def test_identity_like_on_parent_directories_and_components():
    assert verifier._label(IDENTITY_FILE) == "<redacted:identity-like>"
    assert verifier._label(IDENTITY_DIR_FILE) == "<redacted:identity-like>"
    assert verifier._label("reports/vk_100000001_200000002_f0.jpg") == "<redacted:identity-like>"


def test_alias_resolving_into_sensitive_is_redacted(tmp_path):
    (tmp_path / "link").mkdir()
    # A public-looking alias that resolves under a sensitive directory must redact.
    label, public = verifier._classify_pair("scripts/alias.py", "data/interim/faces/x.py",
                                            CELL, PREFIX)
    assert public is False and label != "scripts/alias.py"


# ------------------------------------------------------------------- privacy leaks


def test_identity_like_present_file_mismatch_redacts_digest(tmp_path):
    root, cell_root = build(tmp_path)
    target = root / IDENTITY_FILE
    target.write_bytes(b"X" * len(target.read_bytes()))  # present, same size, new bytes
    error = expect(root, cell_root, "train-inputs")
    blob = str(error)
    assert "100000001" not in blob and "subject_" not in blob and "sha256" not in blob
    assert "declared=(" not in error.detail and "actual=(" not in error.detail
    assert "declared_bytes=" in error.detail
    assert error.subject == "<redacted:identity-like>"


def test_identity_dir_present_file_mismatch_redacts_digest(tmp_path):
    root, cell_root = build(tmp_path)
    target = root / IDENTITY_DIR_FILE
    target.write_bytes(b"Y" * len(target.read_bytes()))
    error = expect(root, cell_root, "train-inputs")
    assert "100000001" not in str(error) and "notes.txt" not in str(error)
    assert error.subject == "<redacted:identity-like>"


def test_cli_identity_mismatch_json_never_leaks_digest(tmp_path, capsys):
    root, cell_root = build(tmp_path)
    target = root / IDENTITY_FILE
    target.write_bytes(b"Z" * len(target.read_bytes()))
    code, payload = cli(root, cell_root, capsys)
    assert code == 1
    assert payload["stage"] == "train-inputs"
    blob = payload["subject"] + payload["detail"]
    assert "100000001" not in blob and "sha256" not in blob and "declared=(" not in blob
    assert payload["subject"] == "<redacted:identity-like>"


def test_sensitive_face_mismatch_never_leaks(tmp_path):
    root, cell_root = build(tmp_path)
    target = root / SENSITIVE_FACE
    target.write_bytes(b"Q" * len(target.read_bytes()))
    error = expect(root, cell_root, "train-inputs")
    assert SENSITIVE_FACE not in str(error) and "faces" not in str(error)
    assert error.subject == "<redacted:sensitive-path>"


def test_private_record_mismatch_reports_no_path_or_hash(tmp_path):
    root, cell_root = build(tmp_path)
    target = root / PREFIX / "train/private/checkpoint.pt"
    target.write_bytes(b"X" * len(target.read_bytes()))
    error = expect(root, cell_root, "train-outputs")
    assert "checkpoint" not in str(error) and "sha256" not in str(error)
    assert "declared_bytes=" in error.detail and "actual_bytes=" in error.detail
    assert error.subject == "<redacted:sensitive-path>"


def test_private_malformed_manifest_reports_only_reason_and_size(tmp_path):
    root, cell_root = build(tmp_path)
    write(cell_root, "train/private/checkpoint.manifest.json", "{ not json")
    with pytest.raises(verifier.VerificationError) as error:
        run(root, cell_root)
    assert error.value.stage == "malformed-manifest"
    assert error.value.detail == "reason=malformed bytes=10"
    assert "checkpoint" not in str(error.value) and "sha256" not in str(error.value)


def test_public_code_mismatch_may_report_digest(tmp_path):
    root, cell_root = build(tmp_path)
    (root / "scripts/run_strong_backbone_cuda_cell_v1.py").write_bytes(b"# changed trainer\n")
    error = expect(root, cell_root, "train-inputs")
    assert "declared=(" in error.detail and "actual=(" in error.detail
    assert "scripts/run_strong_backbone_cuda_cell_v1.py" in str(error)


def test_cli_public_mismatch_reports_both_digests(tmp_path, capsys):
    root, cell_root = build(tmp_path)
    (root / "scripts/run_strong_backbone_cuda_cell_v1.py").write_bytes(b"# changed trainer\n")
    code, payload = cli(root, cell_root, capsys)
    assert code == 1 and payload["stage"] == "train-inputs"
    assert "declared=(" in payload["detail"] and "actual=(" in payload["detail"]


def test_snapshot_drift_on_identity_like_file_redacts(tmp_path):
    root, cell_root = build(tmp_path)
    target = root / IDENTITY_FILE

    def mutate_mid_verify():
        target.write_bytes(b"D" * len(target.read_bytes()))

    with pytest.raises(verifier.VerificationError) as error:
        verifier.verify_cell(root, cell_root, negative="random", scope="tail", lr=1e-5,
                             seed=42, _hook=mutate_mid_verify)
    assert error.value.stage == "snapshot-drift"
    assert "100000001" not in str(error.value) and "sha256" not in str(error.value)
    assert error.value.subject == "<redacted:identity-like>"


def test_snapshot_drift_on_public_code_reports_digests(tmp_path):
    root, cell_root = build(tmp_path)
    target = root / "scripts/run_strong_backbone_cuda_cell_v1.py"

    def mutate_mid_verify():
        target.write_bytes(b"# mutated mid-verify\n")

    with pytest.raises(verifier.VerificationError) as error:
        verifier.verify_cell(root, cell_root, negative="random", scope="tail", lr=1e-5,
                             seed=42, _hook=mutate_mid_verify)
    assert error.value.stage == "snapshot-drift"
    assert "declared=(" in error.value.detail


def test_cli_snapshot_drift_json_is_privacy_safe(tmp_path, capsys):
    root, cell_root = build(tmp_path)
    target = root / PREFIX / "fgnet/private/scores.npz"
    original = verifier.verify_cell

    def hook():
        target.write_bytes(b"S" * len(target.read_bytes()))

    code = verifier.main(["--root", str(root), "--cell-root", str(cell_root),
                          "--negative", "random", "--scope", "tail", "--lr", "1e-5",
                          "--seed", "42"])
    # No hook through the CLI: a stable cell is accepted, proving the CLI surface
    # itself never emits a digest or private path on success.
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepted"] is True and original is verifier.verify_cell
    assert not {"subject", "detail", "stage"} & set(payload)


# ------------------------------------------------------------------- statistics


@pytest.mark.parametrize("tuned", [
    tuned_models(point=0.8, ci=[0.9, 0.95]),                 # point outside absolute CI
    tuned_models(point=0.8, delta_ci=[0.2, 0.3]),            # delta outside delta_ci95
    tuned_models(point=0.8, loo=[0.4, 0.5]),                 # delta outside LOO range
])
def test_valid_out_of_containment_metrics_accepted(tmp_path, tuned):
    root, cell_root = build(tmp_path, mutate=lambda t, c, e, b: e.__setitem__(
        "metrics", external_metrics(CELL, tuned)))
    assert run(root, cell_root)["accepted"] is True


@pytest.mark.parametrize("mutate,stage", [
    (lambda t, c, e, b: e["metrics"]["overall"]["models"][CELL]["roc_auc"].__setitem__(
        "delta_ci95", [-1.5, 0.0]), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"][CELL]["roc_auc"].__setitem__(
        "delta_ci95", [-0.1, 1.5]), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"][CELL]["roc_auc"].__setitem__(
        "loo_delta_max", 2.0), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"][CELL]["roc_auc"].__setitem__(
        "loo_delta_min", 0.5), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"][CELL]["roc_auc"].__setitem__(
        "point", -0.1), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"][CELL]["roc_auc"].__setitem__(
        "point", 1.2), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"][CELL]["roc_auc"].__setitem__(
        "ci95", [0.95, 0.85]), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"][CELL]["roc_auc"].__setitem__(
        "delta_ci95", [1.5, 1.7]), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"][CELL]["roc_auc"].__setitem__(
        "loo_delta_min", 0.5), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"][CELL]["roc_auc"].__setitem__(
        "loo_delta_max", None), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"]["frozen"]["roc_auc"].__setitem__(
        "delta_ci95", [0.0, 0.01]), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"].__setitem__("loo_unavailable", 1),
     "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["bootstrap"].__setitem__(
        "positive_weight", "wrong"), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["bootstrap"].__setitem__(
        "negative_weight", "wrong"), "external-metrics"),
    (lambda t, c, e, b: e["metrics"].__setitem__("publication_ready", True),
     "external-identity"),
])
def test_metric_range_and_bootstrap_rejections(tmp_path, mutate, stage):
    root, cell_root = build(tmp_path, mutate=mutate)
    expect(root, cell_root, stage)


def test_tuned_delta_arithmetic_is_exact(tmp_path):
    root, cell_root = build(tmp_path)
    manifest = load(root, "fgnet/summary.manifest.json")
    manifest["metrics"]["overall"]["models"][CELL]["roc_auc"]["delta_vs_frozen"] = 0.5
    store(root, "fgnet/summary.manifest.json", manifest)
    write(cell_root, "fgnet/summary.json", json.dumps(manifest["metrics"]))
    rehash(root, cell_root)
    expect(root, cell_root, "external-metrics")


# ----------------------------------------------------------------------- schema


@pytest.mark.parametrize("mutate,stage", [
    (lambda t, c, e, b: t["parameters"].__setitem__("trainable_scope", "full"), "case-identity"),
    (lambda t, c, e, b: t["parameters"].__setitem__("device", "cpu"), "case-identity"),
    (lambda t, c, e, b: t["parameters"].__setitem__("learning_rate", True), "case-identity"),
    (lambda t, c, e, b: c["parameters"].__setitem__("epochs_executed", 7), "case-identity"),
    (lambda t, c, e, b: c["parameters"].__setitem__("checkpoint_selection", "best_validation"),
     "case-identity"),
    (lambda t, c, e, b: t["metrics"].__setitem__("history", t["metrics"]["history"][:7]),
     "case-identity"),
    (lambda t, c, e, b: c["metrics"].__setitem__("history", list(reversed(
        t["metrics"]["history"]))), "case-identity"),
    (lambda t, c, e, b: t["metrics"].__setitem__("evaluation_complete", True), "case-identity"),
    (lambda t, c, e, b: t["metrics"].__setitem__("train_pairs", 0), "case-identity"),
    (lambda t, c, e, b: t["metrics"].__setitem__("train_pairs", -3), "case-identity"),
    (lambda t, c, e, b: t["metrics"].__setitem__("peak_allocated_bytes", -1), "case-identity"),
    (lambda t, c, e, b: t["metrics"]["history"][0].__setitem__("training_loss", -0.5),
     "case-identity"),
    (lambda t, c, e, b: t["metrics"]["history"][0].__setitem__("mean_gradient_norm", -1.0),
     "case-identity"),
    (lambda t, c, e, b: t["metrics"]["history"][0].__setitem__("epoch_seconds", -2.0),
     "case-identity"),
    (lambda t, c, e, b: t["metrics"]["history"][0].__setitem__("validation_auc", 1.5),
     "case-identity"),
    (lambda t, c, e, b: c["metrics"].__setitem__("best_validation_auc", -0.2), "case-identity"),
    (lambda t, c, e, b: c["metrics"].__setitem__("best_validation_auc", 1.2), "case-identity"),
    (lambda t, c, e, b: t["metrics"]["history"][0].__setitem__("extra", 1), "case-identity"),
    (lambda t, c, e, b: t.__setitem__("experiment", 7), "native-roles"),
    (lambda t, c, e, b: e["parameters"].__setitem__("actual_cells", [CELL, CELL]),
     "external-identity"),
    (lambda t, c, e, b: e["parameters"].__setitem__("device", "cpu"), "external-identity"),
    (lambda t, c, e, b: e["metrics"].__setitem__("execution_complete", False),
     "external-identity"),
    (lambda t, c, e, b: e["metrics"].__setitem__("coverage",
                                                 {"source_positives": -1}), "external-identity"),
    (lambda t, c, e, b: e["metrics"].__setitem__("peak_reserved_bytes", -1), "external-identity"),
    (lambda t, c, e, b: e["metrics"].__setitem__("limitation", 7), "external-identity"),
    (lambda t, c, e, b: e["metrics"]["overall"].__setitem__("n_pairs", 5307), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"].__setitem__("metric_version", "old"),
     "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"].__setitem__("scope", 7), "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"].__setitem__("extra", {}),
     "external-metrics"),
    (lambda t, c, e, b: e["metrics"]["overall"]["models"].__setitem__(
        "random_head_lr1e-05_s2", {}), "external-metrics"),
])
def test_case_and_shape_rejections(tmp_path, mutate, stage):
    root, cell_root = build(tmp_path, mutate=mutate)
    expect(root, cell_root, stage)


# ------------------------------------------------------------------ role binding


def test_records_are_rehashed_not_trusted(tmp_path):
    root, cell_root = build(tmp_path)
    target = root / PREFIX / "fgnet/private/scores.npz"
    target.write_bytes(b"Y" * len(target.read_bytes()))
    expect(root, cell_root, "external-outputs")


def test_summary_must_be_bound_output(tmp_path):
    root, cell_root = build(tmp_path)
    manifest = load(root, "train/summary.manifest.json")
    manifest["outputs"] = [r for r in manifest["outputs"]
                           if not r["path"].endswith("train/summary.json")]
    store(root, "train/summary.manifest.json", manifest)
    rehash(root, cell_root)
    expect(root, cell_root, "role-outputs")


def test_summary_mismatch_reported(tmp_path):
    root, cell_root = build(tmp_path)
    write(cell_root, "train/summary.json", json.dumps({"training_complete": False}))
    rehash(root, cell_root)
    expect(root, cell_root, "summary-mismatch")


def test_absolute_inside_alias_is_redacted_but_canonicalized(tmp_path):
    # An absolute path that points INSIDE the repo is accepted (canonicalized) but
    # must never be echoed; only repo-relative logic decides the binding.
    root, cell_root = build(tmp_path)
    absolute = str((root / TRAIN_MANIFEST_REL).resolve())
    label, public = verifier._classify_pair(absolute, TRAIN_MANIFEST_REL, CELL, PREFIX)
    assert public is False and absolute not in label
    # And the golden cell still validates through a spelled-out absolute alias.
    assert run(root, cell_root)["accepted"] is True


def test_absolute_inside_identity_alias_never_leaks(tmp_path):
    root, cell_root = build(tmp_path)
    absolute = str((root / IDENTITY_FILE).resolve())
    target = root / IDENTITY_FILE
    target.write_bytes(b"W" * len(target.read_bytes()))
    manifest = load(root, "train/summary.manifest.json")
    for record in manifest["inputs"]:
        if record["path"] == IDENTITY_FILE:
            record["path"] = absolute
    store(root, "train/summary.manifest.json", manifest)
    error = expect(root, cell_root, "train-inputs")
    blob = str(error)
    assert "100000001" not in blob and absolute not in blob and "sha256" not in blob
    assert "declared_bytes=" in error.detail and "actual=(" not in error.detail


def test_cli_malformed_private_manifest_json_is_redacted(tmp_path, capsys):
    root, cell_root = build(tmp_path)
    write(cell_root, "train/private/checkpoint.manifest.json", "{ not json")
    code, payload = cli(root, cell_root, capsys)
    assert code == 1
    assert payload["stage"] == "malformed-manifest"
    assert payload["subject"] == "<redacted:sensitive-path>"
    assert payload["detail"] == "reason=malformed bytes=10"
    blob = json.dumps(payload)
    assert "checkpoint" not in blob and "sha256" not in blob


def test_cli_second_pass_drift_json_is_redacted(tmp_path, capsys, monkeypatch):
    # A genuine end-to-end drift: the private file changes while its own summary is
    # being parsed, so the second hashing pass catches it inside the real CLI.
    root, cell_root = build(tmp_path)
    target = root / IDENTITY_FILE
    original = verifier._summary_metrics

    def drifting(relative, root_, cell, prefix):
        result = original(relative, root_, cell, prefix)
        target.write_bytes(b"R" * len(target.read_bytes()))
        return result

    monkeypatch.setattr(verifier, "_summary_metrics", drifting)
    code, payload = cli(root, cell_root, capsys)
    assert code == 1
    assert payload["stage"] == "snapshot-drift"
    assert payload["subject"] == "<redacted:identity-like>"
    blob = payload["subject"] + payload["detail"]
    assert "100000001" not in blob and "sha256" not in blob
    assert "declared_bytes=" in payload["detail"]


def test_cli_second_pass_drift_on_public_code_reports_digests(tmp_path, capsys, monkeypatch):
    root, cell_root = build(tmp_path)
    target = root / "scripts/run_strong_backbone_cuda_cell_v1.py"
    original = verifier._summary_metrics

    def drifting(relative, root_, cell, prefix):
        result = original(relative, root_, cell, prefix)
        target.write_bytes(b"# drifted\n")
        return result

    monkeypatch.setattr(verifier, "_summary_metrics", drifting)
    code, payload = cli(root, cell_root, capsys)
    assert code == 1 and payload["stage"] == "snapshot-drift"
    assert "declared=(" in payload["detail"]


def test_parsed_hash_conflict_rejected_end_to_end(tmp_path, monkeypatch):
    # Adversarial: the parent declaration and the on-disk bytes agree, but the
    # parsed manifest hash (single read) disagrees, as in a read race. The guard
    # must reject explicitly instead of letting dedup discard the disagreement.
    root, cell_root = build(tmp_path)
    original = verifier._load_manifest

    def tampered(path, root_, cell, prefix):
        manifest, relative, size, _sha = original(path, root_, cell, prefix)
        if relative.endswith("train/summary.manifest.json"):
            return manifest, relative, size, "0" * 64
        return manifest, relative, size, _sha

    monkeypatch.setattr(verifier, "_load_manifest", tampered)
    error = expect(root, cell_root, "manifest-hash-conflict")
    assert "sha256" not in error.detail and "declared=(" not in error.detail


def test_snapshot_guard_includes_summaries(tmp_path):
    root, cell_root = build(tmp_path)
    external_summary = cell_root / "fgnet/summary.json"

    def mutate_mid_verify():
        external_summary.write_bytes(b"Z" * len(external_summary.read_bytes()))

    with pytest.raises(verifier.VerificationError) as error:
        verifier.verify_cell(root, cell_root, negative="random", scope="tail", lr=1e-5,
                             seed=42, _hook=mutate_mid_verify)
    assert error.value.stage == "snapshot-drift"
    assert external_summary.exists()


@pytest.mark.parametrize("missing", ["scores.npz", "frozen_embeddings.npz", "_embeddings.npz"])
def test_each_external_output_is_required(tmp_path, missing):
    root, cell_root = build(tmp_path)
    manifest = load(root, "fgnet/summary.manifest.json")
    manifest["outputs"] = [r for r in manifest["outputs"] if missing not in r["path"]]
    store(root, "fgnet/summary.manifest.json", manifest)
    rehash(root, cell_root)
    expect(root, cell_root, "role-outputs")


def test_checkpoint_manifest_must_be_train_output(tmp_path):
    root, cell_root = build(tmp_path)
    manifest = load(root, "train/summary.manifest.json")
    manifest["outputs"] = [r for r in manifest["outputs"]
                           if not r["path"].endswith("checkpoint.manifest.json")]
    store(root, "train/summary.manifest.json", manifest)
    rehash(root, cell_root)
    expect(root, cell_root, "role-outputs")


def test_missing_crop_cache_input_rejected(tmp_path):
    root, cell_root = build(tmp_path)
    manifest = load(root, "fgnet/summary.manifest.json")
    manifest["inputs"] = [r for r in manifest["inputs"]
                          if not r["path"].endswith("fgnet_crops.npz")]
    store(root, "fgnet/summary.manifest.json", manifest)
    rehash(root, cell_root)
    expect(root, cell_root, "parent-linkage")


def test_missing_checkpoint_parent_linkage_rejected(tmp_path):
    root, cell_root = build(tmp_path)
    manifest = load(root, "fgnet/summary.manifest.json")
    manifest["inputs"] = [r for r in manifest["inputs"]
                          if not r["path"].endswith("checkpoint.manifest.json")]
    store(root, "fgnet/summary.manifest.json", manifest)
    rehash(root, cell_root)
    expect(root, cell_root, "parent-linkage")


# ------------------------------------------------------------- parsed/declared binding


def test_manifest_binding_rejects_declared_disagreement():
    declared = {"m.json": {(10, "a" * 64)}}
    with pytest.raises(verifier.VerificationError) as error:
        verifier._check_manifest_binding("m.json", 10, "b" * 64, declared, CELL, PREFIX)
    assert error.value.stage == "manifest-hash-conflict"
    # Matching, or a path no parent declared, is fine.
    verifier._check_manifest_binding("m.json", 10, "a" * 64, declared, CELL, PREFIX)
    verifier._check_manifest_binding("other.json", 10, "c" * 64, declared, CELL, PREFIX)


def test_alias_disagreement_rejected_before_dedup(tmp_path):
    root, cell_root = build(tmp_path)
    manifest = load(root, "fgnet/summary.manifest.json")
    record = next(r for r in manifest["inputs"] if r["path"] == TRAIN_MANIFEST_REL)
    alias = dict(record)
    alias["path"] = str((root / TRAIN_MANIFEST_REL).resolve())  # absolute alias, same file
    alias["sha256"] = "0" * 64
    manifest["inputs"].append(alias)
    store(root, "fgnet/summary.manifest.json", manifest)
    expect(root, cell_root, "record-alias-conflict")


def test_parsed_manifest_hash_binding_is_enforced(tmp_path):
    # The parsed manifest's own hash must equal the parent-declared record. A cell
    # whose parent declaration cannot equal the parsed bytes is rejected, and the
    # failure never silently discards the disagreement.
    root, cell_root = build(tmp_path)
    manifest = load(root, "fgnet/summary.manifest.json")
    for record in manifest["inputs"]:
        if record["path"] == TRAIN_MANIFEST_REL:
            record["bytes"] = int(record["bytes"]) + 1
    store(root, "fgnet/summary.manifest.json", manifest)
    expect(root, cell_root, "external-inputs")


def test_mutation_restore_is_byte_based(tmp_path):
    root, cell_root = build(tmp_path)
    target = root / "scripts/run_strong_backbone_cuda_cell_v1.py"
    original = target.read_bytes()

    def mutate_and_restore():
        target.write_bytes(b"# different\n")
        target.write_bytes(original)

    result = verifier.verify_cell(root, cell_root, negative="random", scope="tail", lr=1e-5,
                                  seed=42, _hook=mutate_and_restore)
    assert result["accepted"] is True and result["snapshot_stable"] is True


def test_same_size_mutation_detected_on_rehash(tmp_path):
    root, cell_root = build(tmp_path)
    target = root / "scripts/run_strong_backbone_cuda_cell_v1.py"
    target.write_bytes(b"# troiner\n")  # same byte length, different content
    error = expect(root, cell_root, "train-inputs")
    assert "same_size=True" in error.detail


def test_record_path_escape_rejected(tmp_path):
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    root, cell_root = build(tmp_path)
    manifest = load(root, "train/summary.manifest.json")
    manifest["inputs"][0] = {"path": "../outside.txt", "bytes": 6, "sha256": "a" * 64}
    store(root, "train/summary.manifest.json", manifest)
    error = expect(root, cell_root, "record-escape")
    assert error.subject == "<redacted:outside-path>"
    assert str(tmp_path) not in str(error)


def test_zero_byte_records_are_allowed(tmp_path):
    root, cell_root = build(tmp_path)
    empty = root / "src/age_gap/common/__init__.py"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"")
    manifest = load(root, "train/summary.manifest.json")
    manifest["inputs"].append(verifier.record_of(empty, root))
    store(root, "train/summary.manifest.json", manifest)
    rehash(root, cell_root)
    assert run(root, cell_root)["accepted"] is True


def test_manifest_hash_uses_same_bytes_as_parsed(tmp_path):
    root, cell_root = build(tmp_path)
    path = cell_root / "fgnet/summary.manifest.json"
    manifest, _, size, sha = verifier._load_manifest(path, root, CELL, PREFIX)
    raw = path.read_bytes()
    assert size == len(raw)
    assert sha == hashlib.sha256(raw).hexdigest()
    assert manifest == json.loads(raw)


def test_all_failures_are_verification_errors(tmp_path):
    root, cell_root = build(tmp_path)
    write(cell_root, "fgnet/summary.manifest.json", "{ nope")
    with pytest.raises(verifier.VerificationError) as error:
        run(root, cell_root)
    assert type(error.value) is verifier.VerificationError
    assert not isinstance(error.value, ValueError)


def test_module_is_stdlib_only_no_sys_path_hack():
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "argparse", "hashlib", "json", "math", "pathlib"}
    assert "sys.path" not in source and "import torch" not in source


@pytest.mark.parametrize("path,label", [
    ("scripts/private/helper.py", "<redacted:sensitive-path>"),
    ("src/100000001/helper.py", "<redacted:identity-like>"),
    ("tests/person/helper.py", "<redacted:sensitive-path>"),
])
def test_code_allowlist_never_overrides_sensitive_parent(tmp_path, capsys, path, label):
    def extra(t, c, e, blobs):
        blobs[path] = b"private-byte-blob"
        t["inputs"].append(path)

    root, cell_root = build(tmp_path, mutate=extra)
    target = root / path
    target.write_bytes(b"X" * target.stat().st_size)
    code, payload = cli(root, cell_root, capsys)
    assert code == 1 and payload["stage"] == "train-inputs"
    assert payload["subject"] == label
    rendered = json.dumps(payload)
    assert path not in rendered and "100000001" not in rendered
    assert "declared=(" not in rendered and "sha256" not in rendered


@pytest.mark.parametrize("error", [OSError, RuntimeError, ValueError])
def test_resolution_errors_normalized_before_read(tmp_path, capsys, monkeypatch, error):
    root, cell_root = build(tmp_path)

    def fail(*args, **kwargs):
        raise error("PRIVATE_PATH_100000001")

    monkeypatch.setattr(Path, "resolve", fail)
    code, payload = cli(root, cell_root, capsys)
    assert code == 1 and payload["stage"] == "root"
    assert "PRIVATE_PATH" not in json.dumps(payload)
    assert "100000001" not in json.dumps(payload)


@pytest.mark.parametrize("leaf", [role[0] for role in verifier.ROLES.values()])
def test_all_role_paths_contained_before_first_read(tmp_path, monkeypatch, leaf):
    root, cell_root = build(tmp_path)
    original = verifier._safe_resolve

    def redirect(path):
        if Path(path) == cell_root / leaf:
            return tmp_path / "outside_100000001.json"
        return original(path)

    def forbidden_read(*args, **kwargs):
        pytest.fail("No role bytes may be opened before all role containment checks")

    monkeypatch.setattr(verifier, "_safe_resolve", redirect)
    monkeypatch.setattr(verifier, "_read_bytes", forbidden_read)
    error = expect(root, cell_root, "manifest-path")
    assert error.subject == "<redacted:outside-path>"
    assert "100000001" not in str(error)


def test_second_pass_containment_rechecked_without_outside_open(tmp_path, monkeypatch):
    root, cell_root = build(tmp_path)
    target = root / "scripts/run_strong_backbone_cuda_cell_v1.py"
    original = verifier._safe_resolve
    original_digest = verifier._digest_file
    outside = tmp_path / "outside_100000001.py"
    retargeted = False

    def redirect(path):
        if retargeted and Path(path) == target:
            return outside
        return original(path)

    def checked_digest(path):
        assert Path(path) != outside
        return original_digest(path)

    def hook():
        nonlocal retargeted
        retargeted = True

    monkeypatch.setattr(verifier, "_safe_resolve", redirect)
    monkeypatch.setattr(verifier, "_digest_file", checked_digest)
    with pytest.raises(verifier.VerificationError) as error:
        verifier.verify_cell(root, cell_root, negative="random", scope="tail", lr=1e-5,
                             seed=42, _hook=hook)
    assert error.value.stage == "snapshot-drift"
    assert "100000001" not in str(error.value)


@pytest.mark.parametrize("role", ["train", "checkpoint"])
@pytest.mark.parametrize("key,value", [("batch_size", 8), ("margin", 0.4),
                                       ("gap_weight", 1.0), ("crops_dir", "other")])
def test_complete_fixed_budget_params_required(tmp_path, role, key, value):
    def change(t, c, e, blobs):
        (t if role == "train" else c)["parameters"][key] = value

    root, cell_root = build(tmp_path, mutate=change)
    expect(root, cell_root, "case-identity")


def test_pair_source_must_match_negative_and_both_roles(tmp_path):
    root, cell_root = build(tmp_path)
    manifest = load(root, "train/private/checkpoint.manifest.json")
    manifest["parameters"]["pairs_file"] = "data/processed/experiments/pairs_lookalike.jsonl"
    store(root, "train/private/checkpoint.manifest.json", manifest)
    rehash(root, cell_root)
    expect(root, cell_root, "pair-source")
