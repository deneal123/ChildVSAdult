"""Continuing CPU RNG target sampling; not a training or scientific result.

Uniform group then uniform record, with replacement. Unlike the pinned author's
dataset this does not reset a global RNG for each item. Batch partition is part
of the protocol. The caller must bind the dataset and native crop records before
construction; the stream additionally verifies each crop actually decoded.
"""

from __future__ import annotations

import hashlib

import torch
from torch.utils.data import get_worker_info

from scripts.mtlface_components_v2 import age_group
from scripts.mtlface_training_v2 import RecognitionDataset


def _positive_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("positive integer draw count required")


def _rng_hash(generator):
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


class TargetAgeStream:
    def __init__(self, dataset, generator, crop_records):
        if not isinstance(dataset, RecognitionDataset):
            raise TypeError("validated RecognitionDataset required")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise TypeError("caller-owned CPU generator required")
        self.dataset = dataset
        self.records = dataset.records
        self.preprocess = dataset.preprocess
        self.generator = generator
        self.initial_rng_hash = _rng_hash(generator)
        self.pools = [[] for _ in range(7)]
        self.crop_records = {}
        for index, row in enumerate(self.records):
            path = row.path.resolve()
            binding = crop_records.get(path)
            if (
                not isinstance(binding, dict)
                or isinstance(binding.get("bytes"), bool)
                or not isinstance(binding.get("bytes"), int)
                or binding["bytes"] <= 0
                or not isinstance(binding.get("sha256"), str)
                or len(binding["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in binding["sha256"])
            ):
                raise ValueError("native size/SHA256 binding required for every crop")
            self.crop_records[path] = (binding["bytes"], binding["sha256"])
            if row.age is not None:
                self.pools[age_group(row.age)].append(index)
        if any(not pool for pool in self.pools):
            raise ValueError("all seven target groups required; no silent redraw")
        self.sampled = [0] * 7
        self.decoded = [0] * 7
        self.unique = [set() for _ in range(7)]
        self.requests = []
        self.failed_decode_requests = 0

    def _guard(self):
        if get_worker_info() is not None:
            raise RuntimeError("target stream requires num_workers=0; cloned RNG refused")
        if self.dataset.records is not self.records or self.dataset.preprocess is not self.preprocess:
            raise RuntimeError("dataset records/preprocessing changed")

    def draw_indices(self, count):
        self._guard()
        _positive_int(count)
        groups = torch.randint(7, (count,), generator=self.generator)
        indices = []
        for group in groups.tolist():
            pool = self.pools[group]
            index = pool[torch.randint(len(pool), (), generator=self.generator).item()]
            indices.append(index)
            self.sampled[group] += 1
            self.unique[group].add(index)
        self.requests.append(count)
        return tuple(indices), groups

    def _verify(self, path):
        size, checksum = self.crop_records[path]
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != checksum:
            raise RuntimeError("selected crop changed from native binding")

    def draw_images(self, count):
        indices, groups = self.draw_indices(count)
        images = []
        try:
            for index, group in zip(indices, groups.tolist(), strict=True):
                row = self.records[index]
                path = row.path.resolve()
                self._verify(path)
                image, identity, age = self.dataset[index]
                self._verify(path)
                self._guard()
                if identity != row.identity or age != row.age or age_group(age) != group:
                    raise RuntimeError("decoded metadata changed")
                self.decoded[group] += 1
                images.append(image)
            batch = torch.stack(images)
        except Exception:
            self.failed_decode_requests += 1
            # RNG and successfully decoded views remain consumed, even on partial failure.
            raise
        return batch, groups

    def ledger(self):
        self._guard()
        eligible = sum(map(len, self.pools))
        return {
            "policy": "uniform-group-then-uniform-row-with-replacement",
            "rng_initial_sha256": self.initial_rng_hash,
            "rng_current_sha256": _rng_hash(self.generator),
            "batch_partition": list(self.requests),
            "eligible_rows": eligible,
            "excluded_missing_or_conflict_rows": len(self.records) - eligible,
            "eligible_recorded_identities": len(
                {self.records[i].identity for pool in self.pools for i in pool}
            ),
            "pool_rows_by_group": list(map(len, self.pools)),
            "sampled_row_draws_by_group": list(self.sampled),
            "unique_sampled_rows_by_group": list(map(len, self.unique)),
            "repeated_row_draws": sum(self.sampled) - sum(map(len, self.unique)),
            "successfully_decoded_views_by_group": list(self.decoded),
            "failed_decode_requests": self.failed_decode_requests,
            "training_complete": False,
            "scientific_evaluation_complete": False,
        }
