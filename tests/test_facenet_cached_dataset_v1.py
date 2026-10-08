from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.facenet_cached_dataset_v1 import CachedPairDataset


def fixture():
    native = SimpleNamespace(
        _items=[(Path("a"), Path("b"), i % 2, i, 1.0) for i in range(19)],
        identity_groups=[None] * 19,
    )
    features = np.ones((2, 1792, 1, 1), dtype=np.float32)
    return native, features, {"a": 0, "b": 1}


def test_metadata_copy_and_bank_immutability():
    native, features, rows = fixture()
    cached = CachedPairDataset(native, features, rows)
    assert cached.labels == [i % 2 for i in range(19)]
    assert cached.gaps == list(range(19))
    cached[0][0].zero_()
    assert features.min() == 1
    assert cached._items == native._items


def test_full_epoch_loader_order_and_rng():
    native, features, rows = fixture()
    cached = CachedPairDataset(native, features, rows)

    class Pixels(Dataset):
        def __len__(self):
            return len(native._items)

        def __getitem__(self, index):
            return torch.zeros(3, 160, 160), torch.tensor(index)

    def collect(dataset):
        torch.manual_seed(42)
        loader = DataLoader(dataset, batch_size=6, shuffle=True)
        output = []
        for _ in range(10):
            output.append([batch[-1].tolist() for batch in loader])
        return output, torch.get_rng_state()

    # Age gaps encode row indices without changing cached feature payload shape.
    class Indexed(Dataset):
        def __len__(self):
            return len(cached)

        def __getitem__(self, index):
            values = cached[index]
            return values[0], torch.tensor(index)

    reference, rng = collect(Pixels())
    result, cached_rng = collect(Indexed())
    assert result == reference
    assert torch.equal(rng, cached_rng)
    assert all(len(epoch[-1]) == 1 for epoch in result)


def test_missing_crop_rejected():
    native, features, _ = fixture()
    with pytest.raises(ValueError, match="absent"):
        CachedPairDataset(native, features, {"a": 0})


def test_invalid_feature_rejected():
    native, features, rows = fixture()
    cached = CachedPairDataset(native, features, rows)
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        cached[0]
