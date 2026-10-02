"""Synthetic tests for the candidate comparator FG-NET re-evaluation script.

No GPU, no real FG-NET data and no real checkpoints are required: the cached-crop loader, the
common-protocol checkpoint loader and the frozen backbone are replaced with tiny stand-ins.
The tests exercise the *reused* corrected utilities (matched pair loader, subject bootstrap,
LOSO, metric groups) through the script's own code paths.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from age_gap.evaluation import subject_bootstrap
from scripts import reevaluate_comparators_fgnet as candidate

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


class RowCounterModel(torch.nn.Module):
    """Deterministic embedder that records how many crop rows it ever encoded."""

    def __init__(self, dimension: int = 8) -> None:
        super().__init__()
        self.dimension = dimension
        self.rows_seen = 0
        self.calls = 0

    def preprocess(self, image: np.ndarray, bgr: bool = True) -> np.ndarray:
        array = image.astype(np.float32)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        flat = array.reshape(-1)
        vector = np.zeros(self.dimension, dtype=np.float32)
        vector[: min(self.dimension, flat.size)] = flat[: self.dimension]
        return vector

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.rows_seen += int(batch.shape[0])
        return torch.nn.functional.normalize(batch, dim=-1)


def _crop(value: int, size: int = 4) -> np.ndarray:
    base = np.arange(size * size * 3, dtype=np.int64).reshape(size, size, 3)
    return ((base + value) % 256).astype(np.uint8)


def _synthetic_matched_data(n_subjects: int = 20):
    """Balanced endpoint-age-matched style pairs, all in the 25+ stratum."""
    images_a: list[np.ndarray] = []
    images_b: list[np.ndarray] = []
    labels: list[int] = []
    subject_a: list[int] = []
    subject_b: list[int] = []
    for subject in range(n_subjects):
        left = _crop(2 * subject + 1)
        right = _crop(2 * subject + 2)
        # positive pair within the subject
        images_a.append(left)
        images_b.append(right)
        labels.append(1)
        subject_a.append(subject)
        subject_b.append(subject)
        # matched negative pair with a different subject
        other = (subject + 1) % n_subjects
        images_a.append(left)
        images_b.append(_crop(2 * other + 2))
        labels.append(0)
        subject_a.append(subject)
        subject_b.append(other)
    n_pairs = len(labels)
    metadata = {
        "protocol": "endpoint_age_matched",
        "subject_a": np.asarray(subject_a, dtype=np.int64),
        "subject_b": np.asarray(subject_b, dtype=np.int64),
        "stratum_age_gap": np.full(n_pairs, 30, dtype=np.int64),
        "age_a": np.full(n_pairs, 5, dtype=np.int64),
        "age_b": np.full(n_pairs, 35, dtype=np.int64),
        "observed_age_gap": np.full(n_pairs, 30, dtype=np.int64),
        "n_positive_source": n_subjects,
        "n_positive_retained": n_subjects,
        "n_positive_unmatched": 0,
        "positive_coverage": 1.0,
        "n_large_gap_positive_source": n_subjects,
        "n_large_gap_positive_retained": n_subjects,
        "n_large_gap_positive_unmatched": 0,
        "large_gap_positive_coverage": 1.0,
        "negative_endpoint_match_error": np.zeros(n_subjects, dtype=np.int64),
    }
    return images_a, images_b, np.asarray(labels, dtype=np.int64), metadata


# --------------------------------------------------------------------------------------
# CPU / pooling / embedding
# --------------------------------------------------------------------------------------


def test_configure_cpu_threads_caps_at_two(monkeypatch) -> None:
    recorded: dict[str, int] = {}
    monkeypatch.setattr(torch, "set_num_threads", lambda value: recorded.setdefault("threads", value))
    monkeypatch.setattr(candidate.torch, "set_num_interop_threads", lambda value: None)
    device = candidate.configure_cpu_threads()  # default limit is MAX_THREADS == 2
    assert device == "cpu"
    assert recorded["threads"] == 2


def test_build_unique_pool_deduplicates_repeated_crops() -> None:
    images_a = [_crop(1), _crop(2)]
    images_b = [_crop(2), _crop(3)]
    unique, (index_a, index_b) = candidate.build_unique_pool(images_a, images_b)
    assert len(unique) == 3  # crop(2) shared between a and b
    assert index_a.tolist() == [0, 1]
    assert index_b.tolist() == [1, 2]
    assert candidate._crop_digest(_crop(2)) == candidate._crop_digest(images_b[0])


def test_embed_unique_crops_encodes_each_unique_crop_once() -> None:
    unique = [_crop(1), _crop(2), _crop(3), _crop(3)]
    model = RowCounterModel()
    embeddings = candidate.embed_unique_crops(model, unique, "cpu")
    assert embeddings.shape == (4, model.dimension)
    assert model.rows_seen == len(unique)
    assert model.calls == 1
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_pair_scores_match_manual_cosine() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=float)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = candidate.pair_scores_from_embeddings(
        embeddings, np.asarray([0, 1]), np.asarray([1, 2])
    )
    manual = np.asarray(
        [embeddings[0] @ embeddings[1], embeddings[1] @ embeddings[2]]
    )
    assert np.allclose(scores, manual)


def test_score_unique_crops_uses_each_unique_crop_once() -> None:
    unique, (index_a, index_b) = candidate.build_unique_pool(
        [_crop(0), _crop(0)], [_crop(1), _crop(0)]
    )
    model = RowCounterModel()
    scores, embeddings = candidate.score_unique_crops(model, unique, index_a, index_b, "cpu")
    assert model.rows_seen == len(unique) == 2
    assert scores.shape == (2,)
    assert embeddings.shape[0] == len(unique)


# --------------------------------------------------------------------------------------
# Metric groups / paired uncertainty reuse
# --------------------------------------------------------------------------------------


def test_metric_groups_reuse_balanced_strata() -> None:
    labels = np.asarray([1, 0, 1, 0])
    scores = np.asarray([0.9, 0.1, 0.8, 0.2])
    strata = np.asarray([30, 30, 30, 30])
    groups = candidate._metric_groups(scores, labels, strata)
    assert set(groups) == {"overall", "large_gap_25_plus"}
    assert groups["overall"]["roc_auc"] == pytest.approx(1.0)
    assert groups["large_gap_25_plus"]["n_pos"] == 2.0


def test_metric_groups_reject_imbalance() -> None:
    labels = np.asarray([1, 1, 0, 0])
    scores = np.asarray([0.9, 0.8, 0.1, 0.2])
    strata = np.asarray([30, 30, 30, 10])  # 25+ stratum has one negative, no positive
    with pytest.raises(ValueError, match="balanced"):
        candidate._metric_groups(scores, labels, strata)


def test_paired_ci_and_loso_reuse_real_bootstrap() -> None:
    labels = np.tile(np.asarray([1, 0]), 4)
    subject_a = np.tile(np.asarray([0, 0, 1, 1]), 2)
    subject_b = np.tile(np.asarray([0, 2, 1, 3]), 2)
    frozen = np.asarray([0.3, 0.6, 0.55, 0.45, 0.5, 0.4, 0.52, 0.38])
    # Separate the classes so the tuned AUC strictly improves on a non-perfect frozen AUC.
    tuned = frozen + np.where(labels == 1, 0.3, -0.3)
    interval = subject_bootstrap.paired_subject_bootstrap_auc(
        frozen, tuned, labels, subject_a, subject_b, n_boot=100, seed=3
    )
    loso = subject_bootstrap.leave_one_subject_out_auc(
        frozen, tuned, labels, subject_a, subject_b
    )
    assert interval.delta_auc > 0
    assert interval.n_valid_resamples <= interval.n_requested_resamples == 100
    assert loso.n_subjects == 4


# --------------------------------------------------------------------------------------
# Checkpoint loading and provenance
# --------------------------------------------------------------------------------------


def test_load_common_checkpoint_mtlface_requires_external_backbone_name(monkeypatch, tmp_path) -> None:
    dimension = 4
    created: dict[str, object] = {}

    class StubMTLFace(torch.nn.Module):
        def __init__(self, backbone, dim):
            super().__init__()
            self.backbone = backbone
            self.separation = torch.nn.Identity()
            created["dimension"] = dim

    monkeypatch.setattr(candidate, "make_backbone", lambda name, pretrained=False: RowCounterModel(dimension))
    monkeypatch.setattr(candidate, "MTLFaceCommonModel", StubMTLFace)
    state = {
        "format": "mtlface-common-v1",
        "backbone": {},  # shadowed: the state dict overwrote the backbone name key
        "dimension": dimension,
        "separation": {},
    }
    checkpoint = tmp_path / "mtlface.pt"
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="cannot determine backbone name"):
        candidate.load_common_checkpoint(checkpoint, "cpu")

    model, info = candidate.load_common_checkpoint(
        checkpoint, "cpu", backbone_name="arcface_r50_casia"
    )
    assert info["format"] == "mtlface-common-v1"
    assert info["backbone"] == "arcface_r50_casia"
    assert created["dimension"] == dimension
    assert isinstance(model, StubMTLFace)


def test_load_common_checkpoint_cacon_and_rejects_unknown(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        candidate, "make_backbone", lambda name, pretrained=False: RowCounterModel(4)
    )
    checkpoint = tmp_path / "cacon.pt"
    torch.save(
        {"format": "cacon-common-v1", "backbone": "arcface_r50_casia", "state_dict": {}},
        checkpoint,
    )
    model, info = candidate.load_common_checkpoint(checkpoint, "cpu")
    assert info["backbone"] == "arcface_r50_casia"

    bad = tmp_path / "bad.pt"
    torch.save({"format": "unknown-v1", "backbone": "arcface_r50_casia"}, bad)
    with pytest.raises(ValueError, match="unsupported common-protocol format"):
        candidate.load_common_checkpoint(bad, "cpu")


def test_load_common_checkpoint_rejects_contradiction(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        candidate, "make_backbone", lambda name, pretrained=False: RowCounterModel(4)
    )
    checkpoint = tmp_path / "cacon.pt"
    torch.save(
        {"format": "cacon-common-v1", "backbone": "cacon_name", "state_dict": {}}, checkpoint
    )
    with pytest.raises(ValueError, match="contradicts"):
        candidate.load_common_checkpoint(checkpoint, "cpu", backbone_name="other_name")


def test_training_provenance_reports_missing_and_matching_manifest(tmp_path) -> None:
    checkpoint = tmp_path / "run.pt"
    checkpoint.write_bytes(b"weights")
    record = {
        "run_id": "cacon_e8_s42",
        "seed": 42,
        "checkpoint": checkpoint,
        "manifest_path": tmp_path / "run.manifest.json",
    }
    legacy = candidate.training_provenance(record)
    assert legacy["provenance_status"].startswith("legacy")
    assert legacy["manifest_checkpoint_match"] is None

    from age_gap.common.manifest import sha256_file

    record["manifest_path"].write_text(
        json.dumps(
            {
                "experiment": "cacon-common-protocol",
                "outputs": [
                    {"path": f"models/{checkpoint.name}", "sha256": sha256_file(checkpoint)}
                ],
            }
        ),
        encoding="utf-8",
    )
    verified = candidate.training_provenance(record)
    assert verified["manifest_checkpoint_match"] is True
    assert verified["provenance_status"].startswith("verified")

    record["manifest_path"].write_text(
        json.dumps(
            {
                "experiment": "cacon-common-protocol",
                "outputs": [{"path": f"models/{checkpoint.name}", "sha256": "stale"}],
            }
        ),
        encoding="utf-8",
    )
    stale = candidate.training_provenance(record)
    assert stale["manifest_checkpoint_match"] is False
    assert "does not bind" in stale["provenance_status"]


# --------------------------------------------------------------------------------------
# Discovery and aggregation
# --------------------------------------------------------------------------------------


def test_discover_common_protocol_runs_requires_checkpoints(tmp_path) -> None:
    result_dir = tmp_path / "metrics"
    models_dir = tmp_path / "models"
    result_dir.mkdir()
    models_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        candidate.discover_common_protocol_runs(
            result_dir, models_dir, backbone="b", epochs=1, seeds=(42,)
        )
    for method in candidate.METHODS:
        (result_dir / f"{method}_b_e1_s42.json").write_text("{}", encoding="utf-8")
        (models_dir / f"{method}_b_e1_s42.pt").write_bytes(b"ckpt")
    runs = candidate.discover_common_protocol_runs(
        result_dir, models_dir, backbone="b", epochs=1, seeds=(42,)
    )
    assert set(runs) == set(candidate.METHODS)
    assert all(row["seed"] == 42 for rows in runs.values() for row in rows)


def test_cross_method_rows_and_aggregate_are_paired_per_seed() -> None:
    meta = {
        "labels": np.tile(np.asarray([1, 0]), 4),
        "subject_a": np.tile(np.asarray([0, 0, 1, 1]), 2),
        "subject_b": np.tile(np.asarray([0, 2, 1, 3]), 2),
        "stratum_masks": {
            "overall": np.ones(8, dtype=bool),
            "large_gap_25plus": np.ones(8, dtype=bool),
        },
    }
    per_seed = {
        42: {
            "mtlface": np.asarray([0.5, 0.4, 0.55, 0.45, 0.52, 0.38, 0.57, 0.47]),
            "cacon": np.asarray([0.6, 0.3, 0.65, 0.35, 0.62, 0.28, 0.67, 0.37]),
        }
    }
    rows = candidate.cross_method_rows(
        per_seed, meta, bootstrap_seed=0, n_boot=50
    )
    assert {row["stratum"] for row in rows} == set(candidate.STRATA)
    assert all(row["a"] == candidate.METHOD_DISPLAY["mtlface"] for row in rows)
    aggregate = candidate.aggregate_cross_method(rows)
    assert set(aggregate) == set(candidate.STRATA)
    assert aggregate["overall"]["delta_auc"]["n_seeds"] == 1


def test_aggregate_seed_rows_reports_std_across_seeds() -> None:
    def _row(seed, roc, delta):
        metrics = {
            stratum: dict.fromkeys(candidate.PUBLIC_METRICS, roc)
            for stratum in candidate.STRATA
        }
        paired = {
            stratum: {"delta_auc": delta} for stratum in candidate.STRATA
        }
        return {"seed": seed, "metrics": metrics, "paired_vs_frozen": paired}

    aggregate = candidate.aggregate_seed_rows([_row(1, 0.8, 0.01), _row(2, 0.9, 0.03)])
    assert aggregate["overall"]["roc_auc"]["mean"] == pytest.approx(0.85)
    assert aggregate["overall"]["roc_auc"]["n_seeds"] == 2
    assert aggregate["large_gap_25plus"]["delta_auc_vs_frozen"]["mean"] == pytest.approx(0.02)


# --------------------------------------------------------------------------------------
# Public payload hygiene and private cache
# --------------------------------------------------------------------------------------


def test_build_public_payload_contains_no_raw_scores_or_subjects() -> None:
    payload = candidate.build_public_payload(
        pair_counts={"total": 4, "positive": 2, "negative": 2,
                     "large_gap_total": 4, "large_gap_positive": 2, "large_gap_negative": 2},
        coverage={"positive_fraction": 1.0, "unique_crops_embedded": 8},
        negative_seed=42,
        endpoint_age_tolerance=2,
        bootstrap_seed=0,
        n_boot=100,
        frozen_metrics={"overall": {"roc_auc": 0.8}},
        method_rows={
            "mtlface": [
                {
                    "seed": 42,
                    "metrics": {
                        s: dict.fromkeys(candidate.PUBLIC_METRICS, 0.9)
                        for s in candidate.STRATA
                    },
                    "paired_vs_frozen": {s: {"delta_auc": 0.05} for s in candidate.STRATA},
                }
            ]
        },
        cross_rows=[],
        legacy_note="legacy note",
    )
    serialised = json.dumps(payload, allow_nan=False)
    assert "subject_a" not in serialised
    assert "images" not in serialised
    assert "raw_scores" not in serialised
    assert payload["frozen_common_backbone"]["backbone"] == candidate.FROZEN_BACKBONE
    assert payload["provenance_notes"]["legacy_protocol_note"] == "legacy note"


def test_save_embedding_cache_writes_private_npz(tmp_path) -> None:
    embeddings = {"frozen_arcface_r50_casia": np.eye(2, dtype=float)}
    path = candidate.save_embedding_cache(tmp_path / "cache", embeddings, [_crop(0), _crop(1)])
    assert path.is_file()
    loaded = np.load(path, allow_pickle=False)
    assert "frozen_arcface_r50_casia" in loaded.files
    assert "crop_sha256" in loaded.files


# --------------------------------------------------------------------------------------
# End-to-end orchestration with tiny stand-ins (no data, no GPU)
# --------------------------------------------------------------------------------------


def test_reevaluate_end_to_end_writes_public_aggregate(monkeypatch, tmp_path) -> None:
    images_a, images_b, labels, metadata = _synthetic_matched_data()
    monkeypatch.setattr(
        candidate,
        "load_matched_fgnet_pairs",
        lambda *a, **k: (images_a, images_b, labels, metadata),
    )
    monkeypatch.setattr(
        candidate, "make_backbone", lambda name, pretrained=True: RowCounterModel(8)
    )
    monkeypatch.setattr(
        candidate,
        "load_common_checkpoint",
        lambda checkpoint, device, **k: (RowCounterModel(8), {"format": "x", "backbone": "b"}),
    )
    runs = {}
    for method in candidate.METHODS:
        ckpt = tmp_path / f"{method}.pt"
        ckpt.write_bytes(b"w")
        runs[method] = [
            {
                "method": method,
                "seed": seed,
                "run_id": f"{method}_s{seed}",
                "checkpoint": ckpt,
                "manifest_path": tmp_path / f"{method}_s{seed}.manifest.json",
            }
            for seed in candidate.DEFAULT_SEEDS
        ]
    monkeypatch.setattr(candidate, "discover_common_protocol_runs", lambda *a, **k: runs)
    monkeypatch.setattr(candidate, "data_path", lambda *a, **k: tmp_path / "missing")
    captured: dict = {}
    monkeypatch.setattr(
        candidate,
        "write_experiment_manifest",
        lambda path, **kwargs: captured.update({"path": path, **kwargs}),
    )

    output = tmp_path / "out.json"
    saved = candidate.reevaluate(
        result_dir=tmp_path,
        models_dir=tmp_path,
        output=output,
        n_boot=20,
        embeddings_dir=tmp_path / "emb",
    )
    assert saved == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["device"] == "cpu"
    assert payload["torch_threads"] == candidate.MAX_THREADS
    for display in (candidate.METHOD_DISPLAY["mtlface"], candidate.METHOD_DISPLAY["cacon"]):
        assert display in payload["comparators"]
        assert len(payload["comparators"][display]["runs"]) == 3
        assert payload["comparators"][display]["aggregate_across_seeds"]["overall"]["roc_auc"]["n_seeds"] == 3
    assert payload["pair_counts"]["positive"] == 20
    assert payload["pair_counts"]["negative"] == 20
    assert (tmp_path / "emb" / "fgnet_comparator_embeddings.npz").is_file()
    assert captured["experiment"] == candidate.EXPERIMENT
    # public aggregate must not leak per-pair score arrays or subject identifiers
    def _keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from _keys(value)
        elif isinstance(node, list):
            for value in node:
                yield from _keys(value)

    keys = set(_keys(payload))
    assert not {"scores", "subject_a", "subject_b", "images_a", "images_b"} & keys


def test_reevaluate_refuses_to_overwrite(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        candidate.reevaluate(
            result_dir=tmp_path, models_dir=tmp_path, output=output, overwrite=False
        )
