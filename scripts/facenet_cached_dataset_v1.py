"""Unqualified per-crop feature dataset for frozen-trunk trajectory checks."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedPairDataset(Dataset):
    """Preserve native row metadata; never merge rows by pixel/identity similarity."""

    def __init__(self, native, features, crop_rows):
        if features.dtype != np.float32 or features.ndim != 4:
            raise ValueError("float32 feature bank required")
        if features.shape[1:] != (1792, 1, 1):
            raise ValueError("native pre-Dropout feature shape required")
        self._items = list(native._items)
        self._identity_groups = list(native.identity_groups)
        if len(self._identity_groups) != len(self._items):
            raise ValueError("identity metadata misaligned")
        self.features = features
        self.rows = {}
        for path, row in crop_rows.items():
            key = Path(path).resolve()
            if key in self.rows or type(row) is not int or not 0 <= row < len(features):
                raise ValueError("invalid or ambiguous feature index")
            self.rows[key] = row
        for item in self._items:
            for path in item[:2]:
                if Path(path).resolve() not in self.rows:
                    raise ValueError("native crop absent from bank")

    def __len__(self):
        return len(self._items)

    @property
    def labels(self):
        return [item[2] for item in self._items]

    @property
    def gaps(self):
        return [item[3] for item in self._items]

    @property
    def identity_groups(self):
        return list(self._identity_groups)

    def __getitem__(self, index):
        a, b, label, _gap, weight = self._items[index]
        tensors = []
        for path in (a, b):
            # Copy avoids a writable torch view into an immutable mmap bank.
            array = np.array(self.features[self.rows[Path(path).resolve()]], copy=True)
            if not np.isfinite(array).all():
                raise ValueError("nonfinite cached feature")
            tensors.append(torch.from_numpy(array))
        return (*tensors, torch.tensor(float(label)), torch.tensor(float(weight)))
