"""Bound three-view input component; not a synthesized-cache producer.

Byte integrity is checked, not generator fidelity or human identity truth.
Use num_workers=0: the continuing NumPy RNG is owned by this dataset, and
must not be duplicated across worker processes or reseeded per epoch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record
from scripts.cacon_views_v2 import DEFAULT_VIEW_POLICY, ViewPolicy, three_views


@dataclass(frozen=True)
class ThreeViewRecord:
    face_id: str
    source_record: dict
    generated_record: dict


def _verified_image(record):
    path = PROJECT_ROOT / record["path"]
    if file_record(path) != record:
        raise ValueError("three-view cache hash mismatch")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("three-view cache image is unreadable")
    # Recheck after decoding, before admitting it to augmentation.
    if file_record(path) != record:
        raise ValueError("three-view cache changed during decode")
    return image


class ThreeViewDataset(Dataset):
    def __init__(self, records, preprocess, rng, *, policy=DEFAULT_VIEW_POLICY):
        if not isinstance(rng, np.random.Generator) or not isinstance(policy, ViewPolicy):
            raise ValueError("explicit continuing NumPy RNG and ViewPolicy required")
        self.records = tuple(copy.deepcopy(tuple(records)))
        if not all(isinstance(row, ThreeViewRecord) for row in self.records):
            raise ValueError("ThreeViewRecord items required")
        if not self.records or len({row.face_id for row in self.records}) != len(self.records):
            raise ValueError("nonempty unique face records required")
        source_paths, generated_paths = set(), set()
        for row in self.records:
            if not isinstance(row.face_id, str) or not row.face_id:
                raise ValueError("nonempty face ID required")
            source, generated = row.source_record, row.generated_record
            for record in (source, generated):
                if (
                    set(record) != {"path", "bytes", "sha256"}
                    or file_record(PROJECT_ROOT / record["path"]) != record
                ):
                    raise ValueError("valid bound source/generated file record required")
            source_path = (PROJECT_ROOT / source["path"]).resolve()
            generated_path = (PROJECT_ROOT / generated["path"]).resolve()
            if source_path in source_paths or generated_path in generated_paths:
                raise ValueError("duplicate source or synthesized cache path")
            source_paths.add(source_path)
            generated_paths.add(generated_path)
        if source_paths & generated_paths:
            raise ValueError("synthesized cache must not reuse source file paths")
        self.preprocess, self.rng, self.policy = preprocess, rng, policy
        self.generator_provenance_verified = False

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if get_worker_info() is not None:
            raise ValueError("num_workers=0 required; shared RNG worker duplication refused")
        row = self.records[index]
        source = _verified_image(row.source_record)
        generated = _verified_image(row.generated_record)
        images, _ = three_views(source, generated, self.rng, policy=self.policy)
        tensors = []
        for image in images:
            tensor = torch.from_numpy(np.asarray(self.preprocess(image)))
            if (
                tensor.ndim != 3
                or tensor.shape[0] != 3
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("finite CHW three-channel preprocessing required")
            tensors.append(tensor)
        if len({tuple(t.shape) for t in tensors}) != 1:
            raise ValueError("equal preprocessed view dimensions required")
        return tuple(tensors)
