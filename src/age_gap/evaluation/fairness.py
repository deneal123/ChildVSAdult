"""Демографический аудит: одинаков ли прирост +pairs по полу/возрасту (Q1, FR-этика).

Стандартная постановка fairness для распознавания лиц — ошибки по *видимым* (apparent)
демографическим атрибутам. insightface ``genderage`` (buffalo_l) на наших выровненных кропах
даёт кажущийся пол и возраст на КАЖДОЕ лицо. Стратифицируем по атрибутам якоря пары
(``face_a``) — он есть и у позитивов, и у негативов, значит в каждой страте есть оба класса и
ROC-AUC определён. Сравниваем frozen vs дообученный на наших парах backbone в каждой страте:
важно не только что прирост есть, но что он не достаётся одной группе за счёт другой.

genderage запускается на CPU: в одном процессе с torch-CUDA провайдер CUDA для onnxruntime
конфликтует по cuDNN (см. Observations), а CPU-провайдер этого избегает; модель крошечная.
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import torch

from age_gap.common.io import append_jsonl, data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import Pair
from age_gap.evaluation.metrics import roc_auc
from age_gap.training.finetune import _bb_prep, _crop_path

log = get_logger(__name__)

# insightface genderage: argmax(pred[:2]) -> 0=female, 1=male.
_GENDER = {0: "F", 1: "M"}


def _genderage_path() -> str:
    root = os.path.expanduser("~/.insightface/models")
    return os.path.join(root, "buffalo_l", "genderage.onnx")


class GenderEstimator:
    """Обёртка над insightface genderage (apparent пол+возраст по выровненному кропу)."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model: Any = None

    def _ensure(self) -> Any:
        if self._model is None:
            from insightface import model_zoo

            from age_gap.common.device import onnx_ctx_id, onnx_providers

            providers = onnx_providers(self.device)
            ctx_id = onnx_ctx_id(self.device)
            log.info("Загрузка genderage (ctx_id=%d, providers=%s)", ctx_id, providers)
            model = model_zoo.get_model(_genderage_path(), providers=providers)
            model.prepare(ctx_id=ctx_id)
            self._model = model
        return self._model

    def predict(self, crop_bgr: np.ndarray) -> tuple[int, int]:
        """(gender, age): gender 0=female/1=male, age — целое. bbox = весь кроп."""
        from insightface.app.common import Face

        model = self._ensure()
        h, w = crop_bgr.shape[:2]
        face = Face(bbox=np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32), det_score=1.0)
        gender, age = model.get(crop_bgr, face)
        return int(gender), int(age)


def compute_face_attributes(
    face_ids: list[str],
    out_file: str | None = None,
    crops_dir: str = "faces",
    device: str = "cpu",
) -> dict[str, tuple[str, int]]:
    """Apparent пол+возраст для каждого лица (резюмируемо: считаем только недостающие).

    Возвращает ``{face_id: (gender_str, age_est)}``.
    """
    out_file = out_file or str(data_path("data_dir", "interim", "face_genderage.jsonl"))
    done: dict[str, dict[str, Any]] = {r["face_id"]: r for r in read_jsonl(out_file)}
    todo = [f for f in face_ids if f not in done]
    log.info(
        "genderage: всего лиц=%d, уже посчитано=%d, осталось=%d",
        len(face_ids),
        len(done),
        len(todo),
    )

    if todo:
        est = GenderEstimator(device=device)
        for i, fid in enumerate(todo, 1):
            img = cv2.imread(str(_crop_path(fid, crops_dir)))
            if img is None:
                continue
            g, a = est.predict(img)
            rec = {"face_id": fid, "gender": g, "age_est": a}
            append_jsonl(out_file, rec)
            done[fid] = rec
            if i % 2000 == 0:
                log.info("genderage: %d/%d", i, len(todo))

    return {fid: (_GENDER.get(r["gender"], "?"), int(r["age_est"])) for fid, r in done.items()}


def _age_band(age: int) -> str:
    if age < 18:
        return "0-17"
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    return "45+"


def _encode_faces(
    backbone: torch.nn.Module, device: str, face_ids: list[str], batch_size: int = 128
) -> dict[str, np.ndarray]:
    """Прогнать уникальные кропы через backbone один раз -> {face_id: embedding}."""
    prep = _bb_prep(backbone)
    crops_dir = getattr(backbone, "crops_dir", "faces")
    backbone.eval()
    embs: dict[str, np.ndarray] = {}
    batch_ids: list[str] = []
    batch_t: list[torch.Tensor] = []

    def flush() -> None:
        if not batch_t:
            return
        x = torch.stack(batch_t).to(device)
        with torch.no_grad():
            z = backbone(x).cpu().numpy()
        for fid, vec in zip(batch_ids, z, strict=True):
            embs[fid] = vec
        batch_ids.clear()
        batch_t.clear()

    for fid in face_ids:
        img = cv2.imread(str(_crop_path(fid, crops_dir)))
        if img is None:
            continue
        batch_t.append(torch.from_numpy(prep(img)))
        batch_ids.append(fid)
        if len(batch_t) >= batch_size:
            flush()
    flush()
    return embs


def _stratum_auc(
    rows: list[tuple[float, int]], min_pos: int = 20, min_neg: int = 20
) -> tuple[float, int, int]:
    """ROC-AUC по списку (score, label); nan если позитивов/негативов слишком мало."""
    if not rows:
        return float("nan"), 0, 0
    s = np.asarray([r[0] for r in rows], dtype=float)
    y = np.asarray([r[1] for r in rows], dtype=int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos < min_pos or n_neg < min_neg:
        return float("nan"), n_pos, n_neg
    return roc_auc(s, y), n_pos, n_neg


def _paired_gain_ci(
    items: list[tuple[float, float, int]], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap-CI прироста (tuned−frozen): ресэмплинг ОДНИХ И ТЕХ ЖЕ пар для обеих моделей.

    Парный ресэмплинг учитывает корреляцию frozen/tuned-скоров (CI у́же, чем у независимого).
    """
    if len(items) < 40:
        return float("nan"), float("nan")
    sf = np.asarray([it[0] for it in items], dtype=float)
    st = np.asarray([it[1] for it in items], dtype=float)
    y = np.asarray([it[2] for it in items], dtype=int)
    n = len(items)
    rng = np.random.default_rng(seed)
    gains: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yi = y[idx]
        if (yi == 1).sum() < 2 or (yi == 0).sum() < 2:
            continue
        gains.append(roc_auc(st[idx], yi) - roc_auc(sf[idx], yi))
    if not gains:
        return float("nan"), float("nan")
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def stratified_audit(
    frozen: torch.nn.Module,
    tuned: torch.nn.Module,
    device: str,
    attrs: dict[str, tuple[str, int]],
    split: str = "test",
    pairs_file: str | None = None,
) -> dict[str, dict[str, Any]]:
    """frozen vs tuned: ROC-AUC по стратам пол/возраст якоря (face_a) + прирост.

    Возвращает ``{stratum: {n_pos, n_neg, frozen, tuned, gain}}`` (плюс ключ ``overall``).
    """
    pairs_file = pairs_file or str(data_path("data_dir", "processed", "pairs.jsonl"))
    pairs = [Pair.from_dict(r) for r in read_jsonl(pairs_file) if r.get("split") == split]

    # Уникальные лица, у которых есть кроп (по каждому backbone отдельно — crops_dir может отличаться).
    need = sorted({p.face_a for p in pairs} | {p.face_b for p in pairs})
    log.info("Аудит split=%s: пар=%d, уникальных лиц=%d", split, len(pairs), len(need))
    emb_f = _encode_faces(frozen, device, need)
    emb_t = _encode_faces(tuned, device, need)

    # Страты: gender:<F|M>, age:<band> по атрибутам face_a; пара учитывается, если оба эмбеддинга есть.
    buckets: dict[
        str, list[tuple[float, float, int]]
    ] = {}  # stratum -> [(score_f, score_t, label)]

    def add(stratum: str, sf: float, st: float, label: int) -> None:
        buckets.setdefault(stratum, []).append((sf, st, label))

    for p in pairs:
        if p.face_a not in emb_f or p.face_b not in emb_f:
            continue
        if p.face_a not in emb_t or p.face_b not in emb_t:
            continue
        sf = float(emb_f[p.face_a] @ emb_f[p.face_b])
        st = float(emb_t[p.face_a] @ emb_t[p.face_b])
        add("overall", sf, st, p.label)
        a = attrs.get(p.face_a)
        if a is not None:
            add(f"gender:{a[0]}", sf, st, p.label)
            add(f"age:{_age_band(a[1])}", sf, st, p.label)

    result: dict[str, dict[str, Any]] = {}
    for stratum, items in buckets.items():
        af, np_, nn_ = _stratum_auc([(sf, y) for sf, _st, y in items])
        at, _, _ = _stratum_auc([(st, y) for _sf, st, y in items])
        lo, hi = _paired_gain_ci(items)  # 95% CI прироста (C)
        result[stratum] = {
            "n_pos": np_,
            "n_neg": nn_,
            "frozen": af,
            "tuned": at,
            "gain": at - af,
            "gain_lo": lo,
            "gain_hi": hi,
        }
    return result


def print_audit(result: dict[str, dict[str, Any]]) -> None:
    order = ["overall", "gender:F", "gender:M", "age:0-17", "age:18-29", "age:30-44", "age:45+"]
    keys = [k for k in order if k in result] + [k for k in result if k not in order]
    print(
        f"\n{'stratum':<14}{'n_pos':>8}{'n_neg':>8}{'frozen':>10}{'tuned':>10}"
        f"{'gain':>9}{'gain 95% CI':>20}"
    )
    for k in keys:
        r = result[k]
        ci = f"[{r['gain_lo']:+.4f},{r['gain_hi']:+.4f}]"
        print(
            f"{k:<14}{r['n_pos']:>8}{r['n_neg']:>8}"
            f"{r['frozen']:>10.4f}{r['tuned']:>10.4f}{r['gain']:>+9.4f}{ci:>20}"
        )
