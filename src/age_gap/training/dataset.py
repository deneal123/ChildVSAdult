"""Датасет пар поверх предрассчитанных baseline-эмбеддингов (MVP-3/age-supervised).

Adapter обучается на кэшированных ArcFace-эмбеддингах, поэтому энкодер во время обучения
не запускается. Каждый пример — (emb_a, emb_b, label, weight, age_gap):

- weight: age-supervised вес — позитивы с большим возрастным разрывом весят больше, чтобы
  обучение целенаправленно тянуло вместе именно кросс-возрастные пары (северная звезда —
  TAR@FAR на age_gap > 15), а не разменивало их на look-alike негативы;
- age_gap: возрастной разрыв пары (или -1, если неизвестен) — для бакетной валидации.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import Pair
from age_gap.models.embeddings import load_embeddings

log = get_logger(__name__)

# Нормировочный разрыв: при age_gap >= GAP_NORM вес позитива достигает (1 + gap_weight).
GAP_NORM = 30.0


def _pair_weight(label: int, age_gap: int | None, gap_weight: float) -> float:
    """Вес пары: позитив с разрывом g -> 1 + gap_weight * min(g, GAP_NORM)/GAP_NORM; иначе 1."""
    if label == 1 and age_gap is not None:
        return 1.0 + gap_weight * min(float(age_gap), GAP_NORM) / GAP_NORM
    return 1.0


class PairEmbeddingDataset(Dataset):
    """Пары эмбеддингов с метками, age-gap-весами и разрывом для бакетной валидации."""

    def __init__(
        self,
        pairs_file: str | None = None,
        embeddings_file: str | None = None,
        split: str | None = "train",
        gap_weight: float = 0.0,
    ) -> None:
        pairs_file = pairs_file or str(data_path("data_dir", "processed", "pairs.jsonl"))
        embeddings = load_embeddings(embeddings_file)

        self._a: list[np.ndarray] = []
        self._b: list[np.ndarray] = []
        self._y: list[int] = []
        self._w: list[float] = []
        self._gap: list[int] = []
        skipped = 0
        for row in read_jsonl(pairs_file):
            p = Pair.from_dict(row)
            if split is not None and p.split != split:
                continue
            ea, eb = embeddings.get(p.face_a), embeddings.get(p.face_b)
            if ea is None or eb is None:
                skipped += 1
                continue
            self._a.append(ea)
            self._b.append(eb)
            self._y.append(p.label)
            self._w.append(_pair_weight(p.label, p.age_gap, gap_weight))
            self._gap.append(p.age_gap if p.age_gap is not None else -1)
        log.info(
            "Датасет(split=%s): пар=%d (skipped=%d, gap_weight=%.1f)",
            split,
            len(self._y),
            skipped,
            gap_weight,
        )

    def __len__(self) -> int:
        return len(self._y)

    @property
    def gaps(self) -> list[int]:
        return self._gap

    @property
    def labels(self) -> list[int]:
        return self._y

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(np.asarray(self._a[idx], dtype=np.float32)),
            torch.from_numpy(np.asarray(self._b[idx], dtype=np.float32)),
            torch.tensor(float(self._y[idx]), dtype=torch.float32),
            torch.tensor(self._w[idx], dtype=torch.float32),
        )
