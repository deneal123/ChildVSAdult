from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.build_facenet_trunk_cache_v1 import collect_pool, pixel_digest, state_digest


class FakeDataset:
    def __init__(self, split, *, pairs_file, crops_dir):
        assert crops_dir == "faces"
        count = 1200 if split == "train" else 9158
        marker = str(pairs_file) if split == "train" else "shared"
        self._items = [(Path(marker + "a.jpg"), Path(marker + "b.jpg"), 1, 2, 1.0)] * count

    def __len__(self):
        return len(self._items)


def test_both_arms_train_val_union_without_test():
    paths, counts = collect_pool({"low": "low", "cross": "cross"}, FakeDataset)
    assert len(paths) == 6
    assert counts == {arm: {"train": 1200, "val": 9158} for arm in ("low", "cross")}


def test_changed_heldout_rejected():
    class Changed(FakeDataset):
        def __init__(self, split, **kwargs):
            super().__init__(split, **kwargs)
            if split == "val" and kwargs["pairs_file"] == "cross":
                self._items[-1] = (*self._items[-1][:2], 0, 2, 1.0)

    with pytest.raises(ValueError, match="held-out"):
        collect_pool({"low": "low", "cross": "cross"}, Changed)


def test_missing_source_rows_rejected():
    class Missing(FakeDataset):
        def __init__(self, split, **kwargs):
            super().__init__(split, **kwargs)
            self._items.pop()

    with pytest.raises(ValueError, match="coverage"):
        collect_pool({"low": "low", "cross": "cross"}, Missing)


def test_wrong_arms_rejected():
    with pytest.raises(ValueError, match="both"):
        collect_pool({"low": "low"}, FakeDataset)


def test_pixel_digest_layout_independent():
    image = np.arange(3 * 160 * 160, dtype=np.float32).reshape(3, 160, 160)
    assert pixel_digest(image) == pixel_digest(np.asfortranarray(image))
    assert pixel_digest(image) != pixel_digest(image + 1)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((3, 160, 160)),
        np.zeros((160, 160, 3), dtype=np.float32),
        np.full((3, 160, 160), np.nan, dtype=np.float32),
    ],
)
def test_invalid_pixels(image):
    with pytest.raises(ValueError):
        pixel_digest(image)


def test_state_digest_tracks_buffers_and_parameters():
    model = torch.nn.BatchNorm1d(2)
    before = state_digest(model)
    model.running_mean.add_(1)
    assert state_digest(model) != before
    second = state_digest(model)
    with torch.no_grad():
        model.weight.add_(1)
    assert state_digest(model) != second
