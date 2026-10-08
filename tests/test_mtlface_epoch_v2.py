import copy
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from age_gap.common.manifest import file_record
from scripts.mtlface_epoch_v2 import load_bound_dataset, run_recognition_epoch, validate_face_rows
from scripts.prepare_sota_face_list import POLICY
from tests.test_mtlface_training_v2 import joint_model_and_batch


def fixture_rows():
    rows = [
        dict(
            face_id="a",
            person_id="p",
            identity=0,
            crop_path="a.jpg",
            age=0,
            age_status="explicit",
            age_candidates=[0],
        ),
        dict(
            face_id="b",
            person_id="p",
            identity=0,
            crop_path="b.jpg",
            age=None,
            age_status="conflict_masked",
            age_candidates=[10, 11],
        ),
    ]
    counts = dict(
        retained_images=2,
        retained_people=1,
        retained_age_explicit=1,
        retained_age_conflict_masked=1,
    )
    return rows, counts, {Path("a.jpg").resolve(), Path("b.jpg").resolve()}


def test_face_list_validation_preserves_zero_and_masked_conflict():
    records = validate_face_rows(*fixture_rows())
    assert records[0].age == 0 and records[1].age is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity", 1),
        ("face_id", "a"),
        ("crop_path", "a.jpg"),
        ("age", 10),
        ("age_candidates", [11, 10]),
        ("age_status", "explicit"),
    ],
)
def test_hash_correct_but_inconsistent_face_rows_refused(field, value):
    rows, counts, bound = fixture_rows()
    rows[1][field] = value
    with pytest.raises(ValueError):
        validate_face_rows(rows, counts, bound)


def test_unbound_crop_refused():
    rows, counts, _ = fixture_rows()
    with pytest.raises(ValueError, match="not bound"):
        validate_face_rows(rows, counts, set())


def test_epoch_covers_partial_last_batch_and_reproduces_seeded_execution():
    model, head, batch = joint_model_and_batch()
    batch = tuple(torch.cat((value, value[:1]), dim=0) for value in batch)
    dataset = TensorDataset(*batch)
    clone, clone_head = copy.deepcopy(model), copy.deepcopy(head)
    optimizer = torch.optim.SGD([*model.parameters(), *head.parameters()], lr=1e-5)
    clone_optimizer = torch.optim.SGD([*clone.parameters(), *clone_head.parameters()], lr=1e-5)
    result = run_recognition_epoch(
        model,
        head,
        optimizer,
        dataset,
        batch_size=2,
        generator=torch.Generator().manual_seed(42),
        device="cpu",
    )
    repeated = run_recognition_epoch(
        clone,
        clone_head,
        clone_optimizer,
        dataset,
        batch_size=2,
        generator=torch.Generator().manual_seed(42),
        device="cpu",
    )
    assert result == repeated and result["samples"] == 5 and result["batches"] == 3
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, clone.state_dict()[key])


def test_epoch_requires_explicit_rng():
    model, head, batch = joint_model_and_batch()
    optimizer = torch.optim.SGD([*model.parameters(), *head.parameters()], lr=1e-5)
    with pytest.raises(ValueError, match="Generator"):
        run_recognition_epoch(
            model,
            head,
            optimizer,
            TensorDataset(*batch),
            batch_size=2,
            generator=None,
            device="cpu",
        )


def native_fixture(tmp_path):
    rows, counts, _ = fixture_rows()
    for row in rows:
        path = tmp_path / (row["face_id"] + ".jpg")
        path.write_bytes(b"hash-only fixture; no image decoding test")
        row["crop_path"] = str(path)
    private = tmp_path / "private"
    private.mkdir()
    faces = private / "faces.jsonl"
    faces.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    native = dict(
        experiment="sota-recognition-face-list",
        parameters=POLICY.copy(),
        metrics=dict(preparation_complete=True, retained_crops_decoded=True, counts=counts),
        inputs=[file_record(Path(r["crop_path"])) for r in rows],
        outputs=[file_record(faces)],
    )
    manifest = tmp_path / "summary.manifest.json"
    manifest.write_text(json.dumps(native), encoding="utf-8")
    return manifest, native, faces


def test_bound_loader_accepts_hash_valid_consistent_list(tmp_path):
    manifest, _, _ = native_fixture(tmp_path)
    dataset, _ = load_bound_dataset(manifest, None)
    assert len(dataset) == 2 and dataset.n_classes == 1


def test_bound_loader_rejects_changed_crop_bytes(tmp_path):
    manifest, native, _ = native_fixture(tmp_path)
    Path(native["inputs"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_bound_dataset(manifest, None)


def test_bound_loader_rejects_unbound_output_even_with_valid_rows(tmp_path):
    manifest, native, _ = native_fixture(tmp_path)
    native["outputs"] = []
    manifest.write_text(json.dumps(native), encoding="utf-8")
    with pytest.raises(ValueError, match="output not bound"):
        load_bound_dataset(manifest, None)


def test_bound_loader_rejects_rehashed_but_inconsistent_age_policy(tmp_path):
    manifest, native, faces = native_fixture(tmp_path)
    rows = [json.loads(line) for line in faces.read_text(encoding="utf-8").splitlines()]
    rows[1]["age"] = 10
    faces.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    native["outputs"] = [file_record(faces)]
    manifest.write_text(json.dumps(native), encoding="utf-8")
    with pytest.raises(ValueError, match="age conflict"):
        load_bound_dataset(manifest, None)
