"""Synthetic integration tests for :mod:`run_mtlface_campaign_v2`.

These tests inject tiny models/datasets and real local prerequisite manifests so
the campaign's *control logic* is exercised without the 175 MB backbone or any
real crop. They prove: per-seed continuing generators, per-seed checkpoints,
fail-closed prerequisites and nonfinite metrics, no overwrite, and that a
partial/failed cell is never credited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from scripts import run_mtlface_campaign_v2 as runner
from scripts.mtlface_epoch_v2 import run_recognition_epoch
from scripts.mtlface_training_v2 import CosFaceHead
from scripts.prepare_sota_face_list import POLICY

EMBEDDING = 4


# --------------------------------------------------------------------------- #
# tiny fakes
# --------------------------------------------------------------------------- #
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 2 * 2, EMBEDDING)

    def set_batchnorm_policy(self, policy="frozen_all"):
        if policy != "frozen_all":
            raise ValueError("frozen_all only")

    def losses(self, images, identities, ages, identity_head):
        embeddings = self.fc(images.flatten(1))
        identity = F.cross_entropy(identity_head(embeddings, identities), identities)
        return dict(identity=identity, total=identity)

    def preprocess(self, value):
        return value


class TinyDataset(Dataset):
    def __init__(self, classes=2, size=4):
        generator = torch.Generator().manual_seed(0)
        self.images = torch.randn(size, 3, 2, 2, generator=generator)
        self.identities = torch.tensor([index % classes for index in range(size)])
        self.ages = torch.full((size,), -1, dtype=torch.int64)
        self.n_classes = classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.identities[index], self.ages[index]


def tiny_factory(classes, captured):
    def factory(weights, classes_, *, seed, scope):
        captured.append(seed)
        torch.manual_seed(seed)
        model = TinyModel()
        head = CosFaceHead(EMBEDDING, classes_)
        metadata = dict(
            backbone="arcface_r50_casia",
            scope=scope,
            seed=seed,
            channels=3,
            spatial_size=2,
            embedding_dimension=EMBEDDING,
            cosface_scale=64.0,
            cosface_margin=0.35,
            batchnorm_policy="frozen_all",
            full_joint_fas=False,
            weights=runner.file_record(weights),
        )
        return model, head, metadata

    return factory


def tiny_dataset_factory(classes):
    return lambda model, faces: (
        TinyDataset(classes),
        json.loads(Path(faces).read_text(encoding="utf-8")),
    )


def real_epoch_runner():
    return run_recognition_epoch


# --------------------------------------------------------------------------- #
# prerequisite fixtures
# --------------------------------------------------------------------------- #
def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def build_prereqs(
    tmp_path,
    *,
    identity_classes=2,
    contract_overrides=None,
    init_flags=None,
    smoke_flags=None,
    smoke_binds_init=True,
    smoke_binds_faces=True,
):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"synthetic-weights")
    weight_record = runner.file_record(weights)
    contract = dict(
        backbone="arcface_r50_casia",
        scope="head",
        channels=3,
        spatial_size=2,
        embedding_dimension=EMBEDDING,
        cosface_scale=64.0,
        cosface_margin=0.35,
        batchnorm_policy="frozen_all",
        full_joint_fas=False,
        weights=weight_record,
    )
    contract.update(contract_overrides or {})

    faces_jsonl = tmp_path / "faces.jsonl"
    faces_jsonl.write_text("", encoding="utf-8")
    faces_path = _write_json(
        tmp_path / "faces.manifest.json",
        dict(
            experiment=runner.FACES_EXPERIMENT,
            parameters=POLICY,
            metrics=dict(
                preparation_complete=True,
                retained_crops_decoded=True,
                counts=dict(retained_people=identity_classes, retained_images=4),
            ),
            inputs=[],
            outputs=[runner.file_record(faces_jsonl)],
        ),
    )

    init_summary = tmp_path / "init.summary.json"
    init_summary.write_text("{}", encoding="utf-8")
    init_metrics = dict(
        initialization_complete=True,
        training_complete=False,
        identity_classes=identity_classes,
        initialization=contract,
        full_joint_fas=False,
    )
    init_metrics.update(init_flags or {})
    initialization = _write_json(
        tmp_path / "init.manifest.json",
        dict(
            experiment=runner.INITIALIZATION_EXPERIMENT,
            parameters=dict(scope="head"),
            metrics=init_metrics,
            inputs=[weight_record],
            outputs=[runner.file_record(init_summary)],
        ),
    )

    smoke_inputs = [weight_record]
    if smoke_binds_init:
        smoke_inputs.append(runner.file_record(initialization))
    if smoke_binds_faces:
        smoke_inputs.append(runner.file_record(faces_path))
    smoke_metrics = dict(
        real_crop_smoke_complete=True, training_complete=False, full_joint_fas=False
    )
    smoke_metrics.update(smoke_flags or {})
    smoke = _write_json(
        tmp_path / "smoke.manifest.json",
        dict(
            experiment=runner.SMOKE_EXPERIMENT,
            parameters=dict(scope="head", device="cpu", optimizer="SGD"),
            metrics=smoke_metrics,
            inputs=smoke_inputs,
            outputs=[],
        ),
    )
    return dict(weights=weights, faces=faces_path, initialization=initialization, smoke=smoke)


def run(tmp_path, prereqs, **overrides):
    captured: list[int] = []
    classes = overrides.pop("classes", 2)
    options = dict(
        faces=prereqs["faces"],
        initialization=prereqs["initialization"],
        smoke=prereqs["smoke"],
        weights=prereqs["weights"],
        out=tmp_path / "out",
        seeds=[42, 1],
        epochs=2,
        batch_size=2,
        model_factory=tiny_factory(classes, captured),
        dataset_factory=tiny_dataset_factory(classes),
        epoch_runner=real_epoch_runner(),
        extra_sources=[Path(runner.__file__)],
        command=["test"],
    )
    options.update(overrides)
    return runner.run_campaign(**options), captured


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_campaign_writes_seed_cells_and_truthful_flags(tmp_path):
    prereqs = build_prereqs(tmp_path)
    result, captured = run(tmp_path, prereqs)
    out = tmp_path / "out"
    assert result["campaign_complete"] is True and result["completed_seeds"] == [42, 1]
    assert set(captured) == {42, 1} and captured.count(1) == 1
    for seed in (42, 1):
        checkpoint = out / "private" / f"seed_{seed}_last.pt"
        history = json.loads((out / "private" / f"seed_{seed}_history.json").read_text())
        assert checkpoint.exists() and len(history) == 2
        assert [row["epoch"] for row in history] == [1, 2]
        native = json.loads((out / "private" / f"seed_{seed}.manifest.json").read_text())
        metrics = native["metrics"]
        assert metrics["training_complete"] is True
        assert metrics["epochs_executed"] == metrics["epochs_requested"] == 2
        assert metrics["selected_epoch"] == 2
        for key in runner.UNSUPPORTED_FLAGS:
            assert metrics[key] is False
        assert native["parameters"]["seed"] == seed
    summary = json.loads((out / "summary.json").read_text())
    assert summary["training_complete"] is True and summary["epochs_executed"] == 2
    for key in runner.UNSUPPORTED_FLAGS:
        assert summary[key] is False
    manifest = json.loads((out / "summary.manifest.json").read_text())
    assert manifest["experiment"] == runner.CAMPAIGN_EXPERIMENT
    assert manifest["parameters"]["checkpoint_selection"] == "last_epoch"
    assert {cell["seed"] for cell in manifest["metrics"]["cells"]} == {42, 1}
    assert {cell["seed"] for cell in summary["cells"]} == {42, 1}
    assert manifest["parameters"]["implementation_mode"] == "injected-test"
    for cell in summary["cells"]:
        assert cell["checkpoint"] in manifest["outputs"]
        assert cell["history"] in manifest["outputs"]
    # native source binding: weights + all three prerequisite manifests
    bound = {record["path"] for record in manifest["inputs"]}
    for path in (prereqs["weights"], prereqs["faces"], prereqs["initialization"], prereqs["smoke"]):
        assert runner.file_record(path)["path"] in bound
    assert runner.file_record(Path(runner.__file__))["path"] in bound
    assert runner.file_record(Path(__file__))["path"] in bound
    assert runner.file_record(runner.PROJECT_ROOT / "pyproject.toml")["path"] in bound


def test_seeded_heads_differ_and_generator_is_continuing_per_seed(tmp_path):
    prereqs = build_prereqs(tmp_path)
    kept: list[torch.Generator] = []

    def recording_runner(model, head, optimizer, dataset, *, batch_size, generator, device):
        kept.append(generator)
        return run_recognition_epoch(
            model,
            head,
            optimizer,
            dataset,
            batch_size=batch_size,
            generator=generator,
            device=device,
        )

    run(tmp_path, prereqs, epoch_runner=recording_runner)
    # Order: seed42 e1, seed42 e2, seed1 e1, seed1 e2 -> continuing per seed.
    assert len(kept) == 4
    assert kept[0] is kept[1] and kept[2] is kept[3] and kept[0] is not kept[2]
    out = tmp_path / "out"
    first = torch.load(out / "private" / "seed_42_last.pt", weights_only=True)
    second = torch.load(out / "private" / "seed_1_last.pt", weights_only=True)
    assert not torch.equal(first["identity_head"]["weight"], second["identity_head"]["weight"])
    assert not torch.equal(first["model"]["fc.weight"], second["model"]["fc.weight"])


# --------------------------------------------------------------------------- #
# fail-closed prerequisites
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "changes,match",
    [
        (dict(smoke_binds_init=False), "initialization prerequisite"),
        (dict(smoke_binds_faces=False), "face-list prerequisite"),
    ],
)
def test_missing_prerequisite_source_binding_rejected(tmp_path, changes, match):
    prereqs = build_prereqs(tmp_path, **changes)
    with pytest.raises(ValueError, match=match):
        run(tmp_path, prereqs)


def test_changed_prerequisite_bytes_rejected(tmp_path):
    prereqs = build_prereqs(tmp_path)
    prereqs["weights"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        run(tmp_path, prereqs)


def test_weight_contract_mismatch_rejected(tmp_path):
    other = tmp_path / "other.bin"
    other.write_bytes(b"other")
    prereqs = build_prereqs(tmp_path, contract_overrides={"weights": runner.file_record(other)})
    with pytest.raises(ValueError, match="weight contract mismatch"):
        run(tmp_path, prereqs)


def test_already_trained_prerequisite_rejected(tmp_path):
    prereqs = build_prereqs(tmp_path, init_flags={"training_complete": True})
    with pytest.raises(ValueError, match="must not claim training"):
        run(tmp_path, prereqs)


def test_full_joint_claim_in_prerequisite_rejected(tmp_path):
    prereqs = build_prereqs(tmp_path, init_flags={"full_joint_fas": True})
    with pytest.raises(ValueError, match="unsupported claim"):
        run(tmp_path, prereqs)


# --------------------------------------------------------------------------- #
# fail-closed numerics and source stability
# --------------------------------------------------------------------------- #
def test_nonfinite_metrics_abort_cell(tmp_path):
    prereqs = build_prereqs(tmp_path)

    def nan_runner(model, head, optimizer, dataset, *, batch_size, generator, device):
        return dict(
            samples=4,
            batches=2,
            sample_weighted_means={"total": float("nan"), "gradient_norm": 1.0},
        )

    with pytest.raises(FloatingPointError, match="nonfinite metric"):
        run(tmp_path, prereqs, epoch_runner=nan_runner)
    assert not (tmp_path / "out" / "private" / "seed_42_last.pt").exists()
    assert not (tmp_path / "out" / "summary.manifest.json").exists()


def test_nonfinite_snapshot_aborts_cell(tmp_path):
    prereqs = build_prereqs(tmp_path)

    def corrupting_runner(model, head, optimizer, dataset, *, batch_size, generator, device):
        result = run_recognition_epoch(
            model,
            head,
            optimizer,
            dataset,
            batch_size=batch_size,
            generator=generator,
            device=device,
        )
        with torch.no_grad():
            model.fc.weight[0, 0] = float("inf")
        return result

    with pytest.raises(FloatingPointError, match="nonfinite snapshot"):
        run(tmp_path, prereqs, epochs=1, epoch_runner=corrupting_runner)
    assert not (tmp_path / "out" / "private" / "seed_42_last.pt").exists()


def test_source_change_mid_run_fails_closed(tmp_path):
    prereqs = build_prereqs(tmp_path)
    calls = {"count": 0}

    def mutating_runner(model, head, optimizer, dataset, *, batch_size, generator, device):
        result = run_recognition_epoch(
            model,
            head,
            optimizer,
            dataset,
            batch_size=batch_size,
            generator=generator,
            device=device,
        )
        calls["count"] += 1
        if calls["count"] == 2:  # last epoch of the first seed
            prereqs["weights"].write_bytes(b"changed-during-run")
        return result

    with pytest.raises(RuntimeError, match="sources changed"):
        run(tmp_path, prereqs, epoch_runner=mutating_runner)
    assert not (tmp_path / "out" / "private" / "seed_42_last.pt").exists()
    assert not (tmp_path / "out" / "summary.manifest.json").exists()


def test_error_in_later_cell_not_credited(tmp_path):
    prereqs = build_prereqs(tmp_path)
    calls = {"count": 0}

    def failing_runner(model, head, optimizer, dataset, *, batch_size, generator, device):
        calls["count"] += 1
        if calls["count"] == 3:  # first epoch of the second seed
            raise RuntimeError("synthetic epoch failure")
        return run_recognition_epoch(
            model,
            head,
            optimizer,
            dataset,
            batch_size=batch_size,
            generator=generator,
            device=device,
        )

    with pytest.raises(RuntimeError, match="synthetic epoch failure"):
        run(tmp_path, prereqs, epoch_runner=failing_runner)
    out = tmp_path / "out"
    assert (out / "private" / "seed_42.manifest.json").exists()  # completed cell stays
    assert not (out / "private" / "seed_1.manifest.json").exists()  # failed cell absent
    assert not (out / "private" / "seed_1_last.pt").exists()
    assert not (out / "summary.manifest.json").exists()


def test_fresh_output_refused_no_overwrite(tmp_path):
    prereqs = build_prereqs(tmp_path)
    run(tmp_path, prereqs)
    checkpoint = tmp_path / "out" / "private" / "seed_42_last.pt"
    original = checkpoint.read_bytes()
    with pytest.raises(FileExistsError, match="fresh output"):
        run(tmp_path, prereqs)
    assert checkpoint.read_bytes() == original


# --------------------------------------------------------------------------- #
# protocol / CLI contracts
# --------------------------------------------------------------------------- #
def test_protocol_is_explicit_and_seed_independent():
    assert runner.PROTOCOL["checkpoint_selection"] == "last_epoch"
    assert runner.PROTOCOL["optimizer"] == "SGD" and runner.PROTOCOL["device"] == "cpu"
    assert runner.PROTOCOL["threads"] == 1
    assert runner.PROTOCOL["epochs"] >= 1 and list(runner.PROTOCOL["seeds"]) == [42, 1, 2]


def test_seeds_from_parses_and_rejects_bad_input():
    assert runner.seeds_from("42, 1,2") == [42, 1, 2]
    with pytest.raises(argparse.ArgumentTypeError):
        runner.seeds_from("")
    with pytest.raises(argparse.ArgumentTypeError):
        runner.seeds_from("-1")


def test_cli_requires_every_prerequisite(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_mtlface_campaign_v2.py"])
    with pytest.raises(SystemExit):
        runner.main()


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"samples": 3, "batches": 2, "sample_weighted_means": {"total": 1, "gradient_norm": 1}},
        {"samples": 4, "batches": 1, "sample_weighted_means": {"total": 1, "gradient_norm": 1}},
        {"samples": 4, "batches": 2, "sample_weighted_means": {"total": 1}},
        {"samples": 4, "batches": 2, "sample_weighted_means": {"total": "1", "gradient_norm": 1}},
        {"samples": 4, "batches": 2, "sample_weighted_means": {"total": True, "gradient_norm": 1}},
        {"samples": 4, "batches": 2, "sample_weighted_means": {"total": 1, "gradient_norm": -1}},
    ],
)
def test_incomplete_or_malformed_epoch_not_credited(tmp_path, result):
    prereqs = build_prereqs(tmp_path)
    with pytest.raises(ValueError):
        run(tmp_path, prereqs, epoch_runner=lambda *args, **kwargs: result)
    assert not (tmp_path / "out/private/seed_42.manifest.json").exists()
    assert not (tmp_path / "out/summary.manifest.json").exists()


def test_tampered_previous_checkpoint_prevents_campaign_completion(tmp_path):
    prereqs = build_prereqs(tmp_path)
    calls = 0

    def epoch(*args, **kwargs):
        nonlocal calls
        result = run_recognition_epoch(*args, **kwargs)
        calls += 1
        if calls == 3:
            (tmp_path / "out/private/seed_42_last.pt").write_bytes(b"tampered")
        return result

    with pytest.raises(ValueError, match="hash mismatch"):
        run(tmp_path, prereqs, epoch_runner=epoch)
    assert not (tmp_path / "out/summary.manifest.json").exists()


def test_wrong_dataset_native_not_credited(tmp_path):
    prereqs = build_prereqs(tmp_path)
    with pytest.raises(ValueError, match="dataset classifier/count/native"):
        run(tmp_path, prereqs, dataset_factory=lambda *args: (TinyDataset(), {}))
    assert not (tmp_path / "out/private/seed_42.manifest.json").exists()


def test_bool_learning_rate_rejected(tmp_path):
    prereqs = build_prereqs(tmp_path)
    with pytest.raises(ValueError, match="learning rate"):
        run(tmp_path, prereqs, learning_rate=True)


def test_preflight_binds_sources_without_building_or_training(tmp_path):
    prereqs = build_prereqs(tmp_path)

    def refuse(*args, **kwargs):
        pytest.fail("preflight must not build/train models or decode crops")

    result, _ = run(
        tmp_path,
        prereqs,
        preflight_only=True,
        model_factory=refuse,
        dataset_factory=refuse,
        epoch_runner=refuse,
    )
    assert result["preflight_complete"] is True
    assert result["training_complete"] is False
    assert not (tmp_path / "out/private").exists()
    native = runner.verified(tmp_path / "out/summary.manifest.json", runner.PREFLIGHT_EXPERIMENT)
    assert native["parameters"]["epochs"] == 2
    assert native["parameters"]["preflight_only"] is True
