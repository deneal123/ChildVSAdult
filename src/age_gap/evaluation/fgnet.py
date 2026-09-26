"""Внешний cross-age бенчмарк FG-NET (другой домен, чем VK — для чистой атрибуции).

FG-NET: 1002 фото, 82 субъекта, возраст 0–69, разрывы до ~45 лет. Имена файлов кодируют
субъекта и возраст: ``001A02.JPG`` = субъект 001, возраст 2. Лица детектируются и
выравниваются нашим InsightFace-пайплайном (как и VK-кропы) и кэшируются в .npz.

Пары: позитивы — внутри субъекта (с известным age_gap); негативы — между субъектами
(сбалансировано). Метрики — overall/large-gap ROC-AUC + 10-fold accuracy.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch

from age_gap.common.io import data_path, resolve_path
from age_gap.common.logging import get_logger
from age_gap.evaluation.benchmark_external import accuracy_10fold, pair_scores
from age_gap.evaluation.metrics import roc_auc, tar_at_far

log = get_logger(__name__)

_NAME_RE = re.compile(r"(\d+)A(\d+)", re.IGNORECASE)  # 001A02 -> subject 001, age 02
PairProtocol = Literal["endpoint_age_matched", "legacy_random"]


def _match_negatives(
    subjects: np.ndarray,
    ages: np.ndarray,
    positives: list[tuple[int, int, int]],
    *,
    tolerance: int,
    seed: int,
) -> list[tuple[int, int, int, int, int, int]]:
    """Match at most one unique negative to each positive.

    Return (positive_index, left, right, observed_gap, endpoint_error, source_positive_gap).
    A negative retains one endpoint and replaces the other with a different identity. The
    replacement must be within ``tolerance`` years and preserve the positive's exact age gap.
    """
    if tolerance < 0:
        raise ValueError("endpoint_age_tolerance must be non-negative")

    age_indices: dict[int, list[int]] = {}
    for index, age in enumerate(ages.tolist()):
        age_indices.setdefault(int(age), []).append(index)

    candidate_lists: list[list[tuple[int, int, int, int]]] = []
    rng = np.random.default_rng(seed)
    for pos_left, pos_right, pos_gap in positives:
        age_left, age_right = int(ages[pos_left]), int(ages[pos_right])
        candidates: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        right_options = [
            index
            for age in range(age_right - tolerance, age_right + tolerance + 1)
            for index in age_indices.get(age, [])
        ]
        for right in right_options:
            gap = abs(age_left - int(ages[right]))
            if subjects[pos_left] != subjects[right] and gap == pos_gap:
                key = (min(pos_left, right), max(pos_left, right))
                candidates[key] = (pos_left, right, gap, abs(int(ages[right]) - age_right))

        left_options = [
            index
            for age in range(age_left - tolerance, age_left + tolerance + 1)
            for index in age_indices.get(age, [])
        ]
        for left in left_options:
            gap = abs(int(ages[left]) - age_right)
            if subjects[left] != subjects[pos_right] and gap == pos_gap:
                key = (min(left, pos_right), max(left, pos_right))
                candidates[key] = (
                    left,
                    pos_right,
                    gap,
                    abs(int(ages[left]) - age_left),
                )
        options = list(candidates.values())
        rng.shuffle(options)
        candidate_lists.append(options)

    # Maximum bipartite matching keeps each negative pair unique and retains as many
    # positives as possible. The seeded candidate order makes the chosen matching repeatable.
    owners: dict[tuple[int, int], int] = {}
    assigned: dict[int, tuple[int, int, int, int]] = {}

    def augment(positive_index: int, visited: set[tuple[int, int]]) -> bool:
        for left, right, gap, error in candidate_lists[positive_index]:
            key = (min(left, right), max(left, right))
            if key in visited:
                continue
            visited.add(key)
            previous_owner = owners.get(key)
            if previous_owner is None or augment(previous_owner, visited):
                owners[key] = positive_index
                assigned[positive_index] = (left, right, gap, error)
                return True
        return False

    order = sorted(range(len(positives)), key=lambda index: len(candidate_lists[index]))
    for positive_index in order:
        augment(positive_index, set())
    return [
        (index, *assigned[index], positives[index][2])
        for index in range(len(positives))
        if index in assigned
    ]


def _age_gap_diagnostics(labels: np.ndarray, gaps: np.ndarray) -> dict[str, float]:
    positive = gaps[labels == 1].astype(np.float64)
    negative = gaps[labels == 0].astype(np.float64)
    if not len(positive) or not len(negative):
        return {
            "positive_age_gap_mean": float(np.mean(positive)) if len(positive) else float("nan"),
            "negative_age_gap_mean": float(np.mean(negative)) if len(negative) else float("nan"),
            "age_gap_mean_abs_difference": float("nan"),
            "age_gap_ks_statistic": float("nan"),
        }
    values = np.sort(np.concatenate([positive, negative]))
    pos_cdf = np.searchsorted(np.sort(positive), values, side="right") / len(positive)
    neg_cdf = np.searchsorted(np.sort(negative), values, side="right") / len(negative)
    return {
        "positive_age_gap_mean": float(positive.mean()),
        "negative_age_gap_mean": float(negative.mean()),
        "age_gap_mean_abs_difference": float(abs(positive.mean() - negative.mean())),
        "age_gap_ks_statistic": float(np.max(np.abs(pos_cdf - neg_cdf))),
    }


def _ext_dir() -> Path:
    return resolve_path(str(data_path("data_dir", "external")))


def prepare_crops(zip_path: Path | str | None = None, cache: Path | str | None = None) -> Path:
    """Детектировать+выровнять лица FG-NET и закэшировать в .npz. Возвращает путь кэша."""
    zip_path = Path(zip_path) if zip_path else _ext_dir() / "FGNET.zip"
    cache = Path(cache) if cache else _ext_dir() / "fgnet_crops.npz"
    if cache.exists():
        log.info("FG-NET кэш уже есть: %s", cache)
        return cache

    from age_gap.preprocessing.crop_align import align_face
    from age_gap.preprocessing.detect import FaceDetector

    # Детекция на CPU: onnxruntime-gpu (cuDNN12) и torch cu132 (cuDNN13) конфликтуют в одном
    # процессе (WinError 127). facenet-эмбеддинги при этом считаются на GPU.
    detector = FaceDetector(device="cpu")
    crops: list[np.ndarray] = []
    subjects: list[int] = []
    ages: list[int] = []
    skipped = 0

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        for name in names:
            m = _NAME_RE.search(Path(name).stem)
            if not m:
                continue
            buf = np.frombuffer(z.read(name), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                continue
            try:
                faces = detector.detect(img)
            except Exception as exc:  # noqa: BLE001
                log.warning("FG-NET %s: ошибка детекции %s", name, exc)
                skipped += 1
                continue
            if not faces:
                skipped += 1
                continue
            target = max(faces, key=lambda f: f.area)
            crop = align_face(img, target.kps)  # 112x112 BGR
            crops.append(crop)
            subjects.append(int(m.group(1)))
            ages.append(int(m.group(2)))

    arr = np.stack(crops).astype(np.uint8)
    np.savez(
        cache,
        crops=arr,
        subjects=np.asarray(subjects, np.int64),
        ages=np.asarray(ages, np.int64),
    )
    log.info("FG-NET: лиц %d (skipped %d) -> %s", len(crops), skipped, cache)
    return cache


def load_pairs(
    cache: Path | str | None = None,
    neg_per_pos: float = 1.0,
    seed: int = 42,
    *,
    protocol: PairProtocol = "legacy_random",
    endpoint_age_tolerance: int = 2,
    return_metadata: bool = False,
) -> tuple[list, list, np.ndarray, np.ndarray] | tuple[
    list, list, np.ndarray, np.ndarray, dict[str, object]
]:
    """Build FG-NET pairs; optionally return endpoint IDs/ages for subject-level analysis.

    ``endpoint_age_matched`` pairs negative endpoints within the configured age tolerance
    of a positive pair's corresponding endpoints. ``legacy_random`` reproduces the former
    random cross-identity negative sampling and its -1 negative age-gap sentinel.
    """
    if protocol not in ("endpoint_age_matched", "legacy_random"):
        raise ValueError(f"Unknown FG-NET pair protocol: {protocol}")
    if neg_per_pos < 0:
        raise ValueError("neg_per_pos must be non-negative")
    cache = Path(cache) if cache else _ext_dir() / "fgnet_crops.npz"
    data = np.load(cache, allow_pickle=False)
    crops, subjects, ages = data["crops"], data["subjects"], data["ages"]
    rng = np.random.default_rng(seed)

    by_subject: dict[int, list[int]] = {}
    for i, s in enumerate(subjects.tolist()):
        by_subject.setdefault(s, []).append(i)

    pos: list[tuple[int, int, int]] = []  # (i, j, age_gap)
    for idxs in by_subject.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                pos.append((ia, ib, int(abs(int(ages[ia]) - int(ages[ib])))))

    n_positive_source = len(pos)
    n_large_gap_source = sum(gap >= 25 for _, _, gap in pos)
    n_neg = int(len(pos) * neg_per_pos)
    neg: list[tuple[int, int, int]] = []
    endpoint_errors: list[int] = []
    source_positive_gaps: list[int] = []
    if protocol == "endpoint_age_matched":
        if neg_per_pos != 1.0:
            raise ValueError("endpoint_age_matched currently requires neg_per_pos=1.0")
        matched = _match_negatives(
            subjects, ages, pos, tolerance=endpoint_age_tolerance, seed=seed
        )
        retained_positive_indices = [positive_index for positive_index, *_ in matched]
        pos = [pos[index] for index in retained_positive_indices]
        neg = [(left, right, gap) for _, left, right, gap, _, _ in matched]
        endpoint_errors = [error for _, _, _, _, error, _ in matched]
        source_positive_gaps = [gap for _, _, _, _, _, gap in matched]
    else:
        seen: set[tuple[int, int]] = set()
        attempts = 0
        n = len(subjects)
        while len(neg) < n_neg and attempts < n_neg * 50 + 1000:
            attempts += 1
            i, j = int(rng.integers(n)), int(rng.integers(n))
            if subjects[i] == subjects[j]:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            neg.append((i, j, -1))

    a_idx = [p[0] for p in pos] + [q[0] for q in neg]
    b_idx = [p[1] for p in pos] + [q[1] for q in neg]
    issame = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    gaps = np.asarray([p[2] for p in pos] + [q[2] for q in neg], dtype=np.int64)

    images_a = [crops[i] for i in a_idx]
    images_b = [crops[i] for i in b_idx]
    log.info("FG-NET пар: pos=%d, neg=%d", len(pos), len(neg))
    if not return_metadata:
        return images_a, images_b, issame, gaps
    metadata: dict[str, object] = {
        "protocol": protocol,
        "seed": seed,
        "endpoint_age_tolerance": endpoint_age_tolerance if protocol == "endpoint_age_matched" else None,
        "n_positive_source": n_positive_source,
        "n_positive_retained": len(pos),
        "n_positive_unmatched": n_positive_source - len(pos),
        "positive_coverage": len(pos) / n_positive_source if n_positive_source else 0.0,
        "n_large_gap_positive_source": n_large_gap_source,
        "n_large_gap_positive_retained": sum(gap >= 25 for _, _, gap in pos),
        "n_large_gap_positive_unmatched": n_large_gap_source
        - sum(gap >= 25 for _, _, gap in pos),
        "large_gap_positive_coverage": (
            sum(gap >= 25 for _, _, gap in pos) / n_large_gap_source
            if n_large_gap_source
            else 0.0
        ),
        "subject_a": subjects[a_idx].astype(np.int64),
        "subject_b": subjects[b_idx].astype(np.int64),
        "age_a": ages[a_idx].astype(np.int64),
        "age_b": ages[b_idx].astype(np.int64),
        "observed_age_gap": np.abs(ages[a_idx] - ages[b_idx]).astype(np.int64),
        "stratum_age_gap": np.asarray(
            [gap for _, _, gap in pos] + source_positive_gaps
            if protocol == "endpoint_age_matched"
            else [gap for _, _, gap in pos] + [-1] * len(neg),
            dtype=np.int64,
        ),
        "negative_endpoint_match_error": np.asarray(endpoint_errors, dtype=np.int64),
    }
    return images_a, images_b, issame, gaps, metadata


def evaluate(
    backbone: torch.nn.Module,
    device: str,
    large_gap_threshold: int = 25,
    *,
    protocol: PairProtocol = "legacy_random",
    endpoint_age_tolerance: int = 2,
    seed: int = 42,
) -> dict[str, float | str]:
    images_a, images_b, issame, gaps, metadata = load_pairs(
        seed=seed,
        protocol=protocol,
        endpoint_age_tolerance=endpoint_age_tolerance,
        return_metadata=True,
    )
    scores = pair_scores(backbone, images_a, images_b, device, rgb=False)
    stratum_gaps = np.asarray(metadata["stratum_age_gap"], dtype=np.int64)
    if protocol == "endpoint_age_matched":
        mask = ((issame == 0) & (stratum_gaps >= large_gap_threshold)) | (
            (issame == 1) & (stratum_gaps >= large_gap_threshold)
        )
    else:
        mask = (issame == 0) | ((issame == 1) & (gaps >= large_gap_threshold))
    diagnostic_gaps = np.asarray(metadata["observed_age_gap"], dtype=np.int64)
    result: dict[str, float | str] = {
        "pair_protocol": protocol,
        "n_pairs": float(len(issame)),
        "accuracy_10fold": accuracy_10fold(scores, issame),
        "roc_auc": roc_auc(scores, issame),
        "large_gap_auc": roc_auc(scores[mask], issame[mask]) if mask.any() else float("nan"),
        "tar@far=0.01": tar_at_far(scores, issame, 0.01),
    }
    result.update(_age_gap_diagnostics(issame, diagnostic_gaps))
    if protocol == "endpoint_age_matched":
        errors = np.asarray(metadata["negative_endpoint_match_error"], dtype=np.int64)
        result["negative_endpoint_match_max_error"] = float(errors.max(initial=0))
        result["positive_coverage"] = float(metadata["positive_coverage"])
        result["positive_unmatched"] = float(metadata["n_positive_unmatched"])
        result["large_gap_positive_coverage"] = float(metadata["large_gap_positive_coverage"])
        result["large_gap_positive_unmatched"] = float(metadata["n_large_gap_positive_unmatched"])
    return result
