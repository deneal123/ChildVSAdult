import copy

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from age_gap.common.manifest import file_record
from scripts.cacon_dataset_v2 import ThreeViewDataset, ThreeViewRecord


def fixture(tmp_path, seed=42):
    source, generated = tmp_path / "source.png", tmp_path / "generated.png"
    image = np.random.default_rng(7).integers(0, 256, (23, 31, 3), dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    assert cv2.imwrite(str(generated), np.full_like(image, 37))
    row = ThreeViewRecord("face", file_record(source), file_record(generated))
    dataset = ThreeViewDataset(
        [row],
        lambda image: image.transpose(2, 0, 1).astype(np.float32) / 255,
        np.random.default_rng(seed),
    )
    return dataset, source, generated


def test_dataset_rng_advances_but_new_seed_reproduces_first_item(tmp_path):
    dataset, _, _ = fixture(tmp_path)
    state = copy.deepcopy(dataset.rng.bit_generator.state)
    first = dataset[0]
    second = dataset[0]
    assert state != dataset.rng.bit_generator.state
    assert not torch.equal(first[0], second[0])
    recreated = ThreeViewDataset(dataset.records, dataset.preprocess, np.random.default_rng(42))
    for actual, expected in zip(first, recreated[0], strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(first[2], torch.full((3, 23, 31), 37 / 255))
    assert dataset.generator_provenance_verified is False


def test_repeated_loader_epochs_do_not_reset_rng(tmp_path):
    dataset, _, _ = fixture(tmp_path)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    first, second = next(iter(loader)), next(iter(loader))
    assert first[0].shape == (1, 3, 23, 31)
    assert not torch.equal(first[0], second[0])


def test_mutated_cache_refused_before_augmentation(tmp_path):
    dataset, _, generated = fixture(tmp_path)
    assert cv2.imwrite(str(generated), np.zeros((23, 31, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="hash"):
        dataset[0]


def test_worker_rng_duplication_refused(tmp_path, monkeypatch):
    dataset, _, _ = fixture(tmp_path)
    monkeypatch.setattr("scripts.cacon_dataset_v2.get_worker_info", lambda: object())
    with pytest.raises(ValueError, match="num_workers=0"):
        dataset[0]


def test_source_path_cannot_be_used_as_generated_view(tmp_path):
    dataset, _, _ = fixture(tmp_path)
    row = dataset.records[0]
    with pytest.raises(ValueError, match="reuse source"):
        ThreeViewDataset(
            [ThreeViewRecord("face", row.source_record, row.source_record)],
            dataset.preprocess,
            np.random.default_rng(42),
        )


def test_duplicate_faces_refused(tmp_path):
    dataset, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        ThreeViewDataset([dataset.records[0]] * 2, dataset.preprocess, np.random.default_rng(42))


def test_bad_preprocessing_refused(tmp_path):
    dataset, _, _ = fixture(tmp_path)
    dataset.preprocess = lambda image: image.transpose(2, 0, 1).astype(np.float32) * float("nan")
    with pytest.raises(ValueError, match="finite"):
        dataset[0]


def test_hash_valid_but_wrong_synthesized_dimensions_refused(tmp_path):
    dataset, _, generated = fixture(tmp_path)
    assert cv2.imwrite(str(generated), np.zeros((2, 2, 3), dtype=np.uint8))
    original = dataset.records[0]
    replaced = ThreeViewRecord(original.face_id, original.source_record, file_record(generated))
    checked = ThreeViewDataset([replaced], dataset.preprocess, np.random.default_rng(42))
    state = copy.deepcopy(checked.rng.bit_generator.state)
    with pytest.raises(ValueError, match="dimensions"):
        checked[0]
    assert state == checked.rng.bit_generator.state


def test_untyped_record_refused(tmp_path):
    with pytest.raises(ValueError, match="ThreeViewRecord"):
        ThreeViewDataset(["not a record"], None, np.random.default_rng(42))
