import hashlib

import cv2
import numpy as np
import pytest
import torch

from scripts.mtlface_target_stream_v3 import TargetAgeStream
from scripts.mtlface_training_v2 import FaceRecord, RecognitionDataset
from scripts.prepare_mtlface_target_stream_v3 import prepare


def fixture(tmp_path, seed=42):
    rows, bindings = [], {}
    for i, age in enumerate((0, 11, 21, 31, 41, 51, 61, None)):
        path = tmp_path / f"{i}.jpg"
        assert cv2.imwrite(str(path), np.full((4, 4, 3), i, dtype=np.uint8))
        data = path.read_bytes()
        bindings[path.resolve()] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        rows.append(FaceRecord(path, i, age))
    dataset = RecognitionDataset(rows, lambda image: image.transpose(2, 0, 1).astype(np.float32))
    return TargetAgeStream(dataset, torch.Generator().manual_seed(seed), bindings)


def test_continuing_rng_and_ledger(tmp_path):
    first = fixture(tmp_path)
    second = fixture(tmp_path)
    for _ in range(2):
        a, ga = first.draw_indices(80)
        b, gb = second.draw_indices(80)
        assert a == b and torch.equal(ga, gb)
    ledger = first.ledger()
    assert ledger == second.ledger()
    assert ledger["batch_partition"] == [80, 80]
    assert sum(ledger["sampled_row_draws_by_group"]) == 160
    assert sum(ledger["successfully_decoded_views_by_group"]) == 0
    assert ledger["pool_rows_by_group"] == [1] * 7
    assert ledger["excluded_missing_or_conflict_rows"] == 1
    assert ledger["repeated_row_draws"] == 153
    assert ledger["rng_current_sha256"] != ledger["rng_initial_sha256"]
    assert not ledger["training_complete"]


def test_decode_counts_and_shapes(tmp_path):
    stream = fixture(tmp_path)
    images, groups = stream.draw_images(40)
    assert images.shape == (40, 3, 4, 4)
    assert images.is_floating_point()
    assert groups.dtype == torch.int64
    assert sum(stream.ledger()["successfully_decoded_views_by_group"]) == 40


def test_changed_crop_refused_rng_not_rolled_back(tmp_path):
    stream = fixture(tmp_path)
    for row in stream.records:
        row.path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="native binding"):
        stream.draw_images(5)
    ledger = stream.ledger()
    assert sum(ledger["sampled_row_draws_by_group"]) == 5
    assert sum(ledger["successfully_decoded_views_by_group"]) == 0
    assert ledger["failed_decode_requests"] == 1


@pytest.mark.parametrize("count", [0, -1, True, 1.2])
def test_bad_count_does_not_advance_rng(tmp_path, count):
    stream = fixture(tmp_path)
    before = stream.ledger()
    with pytest.raises(ValueError):
        stream.draw_indices(count)
    assert stream.ledger() == before


def test_missing_group_refused(tmp_path):
    stream = fixture(tmp_path)
    dataset = RecognitionDataset(stream.records[:6], stream.preprocess)
    bindings = {p: dict(bytes=n, sha256=h) for p, (n, h) in stream.crop_records.items()}
    with pytest.raises(ValueError, match="all seven"):
        TargetAgeStream(dataset, torch.Generator(), bindings)


def test_preprocess_mutation_refused(tmp_path):
    stream = fixture(tmp_path)
    stream.dataset.preprocess = lambda x: x
    with pytest.raises(RuntimeError, match="changed"):
        stream.draw_indices(1)


def test_worker_clone_refused(tmp_path, monkeypatch):
    stream = fixture(tmp_path)
    monkeypatch.setattr("scripts.mtlface_target_stream_v3.get_worker_info", lambda: object())
    with pytest.raises(RuntimeError, match="cloned RNG"):
        stream.draw_indices(1)


def test_seeds_produce_different_streams(tmp_path):
    first = fixture(tmp_path, 42)
    second = fixture(tmp_path, 1)
    assert first.draw_indices(80)[0] != second.draw_indices(80)[0]


def test_partial_decode_failure_counts_completed_views(tmp_path, monkeypatch):
    stream = fixture(tmp_path)
    expected = fixture(tmp_path).draw_indices(5)[0]
    assert expected[0] != expected[1]
    # First view succeeds, second fails, even though all five RNG draws occurred.
    stream.records[expected[1]].path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="native binding"):
        stream.draw_images(5)
    ledger = stream.ledger()
    assert sum(ledger["sampled_row_draws_by_group"]) == 5
    assert sum(ledger["successfully_decoded_views_by_group"]) == 1
    assert ledger["failed_decode_requests"] == 1


@pytest.mark.parametrize("seeds,batches,size", [([1, 1], 1, 1), ([-1], 1, 1), ([True], 1, 1), ([1], 0, 1)])
def test_preparation_bad_protocol_refused_before_loading(tmp_path, seeds, batches, size):
    with pytest.raises(ValueError):
        prepare(tmp_path / "absent.json", tmp_path / "out", seeds, batches, size)


def test_preparation_existing_output_refused(tmp_path):
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "absent.json", tmp_path, [42], 1, 1)


def test_preparation_native_nested_counts(tmp_path, monkeypatch):
    import scripts.prepare_mtlface_target_stream_v3 as module

    stream = fixture(tmp_path)
    native = dict(
        inputs=[dict(path=str(p), bytes=n, sha256=h) for p, (n, h) in stream.crop_records.items()],
        outputs=[],
        metrics=dict(counts=dict(retained_age_missing=1, retained_age_conflict_masked=0)),
    )
    monkeypatch.setattr(module, "load_bound_dataset", lambda *a: (stream.dataset, native))
    monkeypatch.setattr(module, "file_record", lambda path: dict(path=str(path)))
    captured = {}
    monkeypatch.setattr(module, "write_experiment_manifest", lambda path, **kw: captured.update(kw))
    metrics = prepare(tmp_path / "faces.json", tmp_path / "output", [42, 1, 2], 2, 64)
    assert metrics["missing_age_rows"] == 1
    assert metrics["conflict_masked_age_rows"] == 0
    assert len(metrics["cells"]) == 3
    assert captured["metrics"] == metrics
    assert not metrics["training_complete"]
    for cell in metrics["cells"]:
        assert sum(cell["ledger"]["sampled_row_draws_by_group"]) == 128
        assert sum(cell["ledger"]["successfully_decoded_views_by_group"]) == 0
