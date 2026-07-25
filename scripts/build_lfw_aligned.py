"""Починка LFW: выровненные кропы вместо sklearn-изображений.

    uv run python scripts/build_lfw_aligned.py

БАГ. LFW грузился через sklearn ``fetch_lfw_pairs`` — это funneled-изображения БЕЗ 5-точечного
выравнивания, тогда как AgeDB-30/CALFW приходят из insightface ``.bin`` уже выровненными. ArcFace
и AdaFace требуют точного выравнивания, поэтому на невыровненном LFW они рушатся:

    ArcFace r100 (замороженный):  LFW 0.8298  |  AgeDB-30 0.9835  |  CALFW 0.9613

AgeDB-30 = 98.35% — ровно опубликованное значение, т.е. веса и препроцессинг .bin корректны. Но
AgeDB-30 СЛОЖНЕЕ LFW, а LFW у нас 83% при опубликованных ~99.8%. Это невозможно; значит сломан
именно путь загрузки LFW.

ПОЧЕМУ ЭТО ВАЖНО ДЛЯ СТАТЬИ. LFW — наш контроль забывания, и заголовочное «ценой −0.019 точности
на LFW» посчитано на этом сломанном пути. Число нужно пересчитать.

Вторая часть того же бага: sklearn по умолчанию отдаёт не полный кадр 250x250, а ТЕСНЫЙ кроп
125x94 (параметр slice_). То есть бэкбоны получали лица и невыровненные, И обрезанные — вдвойне
чужое распределение. Здесь запрашивается slice_=None.

ЧИНИМ БЕЗ СКАЧИВАНИЯ .bin: прогоняем полные кадры через тот же пайплайн, что и наши кропы
(InsightFace RetinaFace -> 5 точек -> norm_crop 112px). Если лицо не найдено, берём центральный
кроп и считаем такие случаи (они не должны быть массовыми).

Пишет data/external/lfw_aligned.npz (a, b, issame) — его подхватывает load_lfw().
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from age_gap.common.io import data_path
from age_gap.common.logging import get_logger
from age_gap.preprocessing.crop_align import align_face
from age_gap.preprocessing.detect import FaceDetector

log = get_logger(__name__)


def _align_one(det: FaceDetector, img_rgb: np.ndarray, size: int = 112) -> tuple[np.ndarray, bool]:
    """RGB -> выровненный BGR-кроп 112px. Второй элемент: удалось ли найти лицо."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    faces = det.detect(bgr)
    if faces:
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return align_face(bgr, f.kps, image_size=size), True
    h, w = bgr.shape[:2]                       # запасной путь: центральный квадрат
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(bgr[y0:y0 + s, x0:x0 + s], (size, size)), False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="10_folds")
    ap.add_argument("--size", type=int, default=112)
    args = ap.parse_args()

    from sklearn.datasets import fetch_lfw_pairs

    # slice_=None ОБЯЗАТЕЛЕН: по умолчанию sklearn отдаёт тесный кроп 125x94, на котором
    # детектор почти всегда промахивается (99.6% без детекта). Полный кадр — 250x250.
    data = fetch_lfw_pairs(subset=args.subset, color=True, resize=1.0, slice_=None)
    pairs = data.pairs
    if pairs.max() <= 1.0 + 1e-6:
        pairs = pairs * 255.0
    pairs = pairs.astype(np.uint8)
    issame = data.target.astype(np.int64)
    log.info("LFW(%s): пар=%d (pos=%d), кадр %s",
             args.subset, len(issame), int(issame.sum()), pairs.shape[2:4])

    det = FaceDetector()
    A, B, miss = [], [], 0
    for i in range(pairs.shape[0]):
        a, oka = _align_one(det, pairs[i, 0], args.size)
        b, okb = _align_one(det, pairs[i, 1], args.size)
        A.append(a)
        B.append(b)
        miss += (not oka) + (not okb)
        if (i + 1) % 500 == 0:
            log.info("выровнено %d/%d (без детекта: %d)", i + 1, pairs.shape[0], miss)

    dst = data_path("data_dir", "external", "lfw_aligned.npz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # miss_rate пишем в сам кеш: load_lfw() отказывается использовать кеш с высокой
    # долей промахов детектора (так первый, сломанный прогон не сможет тихо утечь в метрики)
    np.savez_compressed(dst, a=np.stack(A), b=np.stack(B), issame=issame,
                        miss_rate=np.float64(miss / (2 * pairs.shape[0])))
    total = 2 * pairs.shape[0]
    log.info("готово: %s | без детекта %d/%d (%.1f%%)", dst, miss, total, 100 * miss / total)
    print(f"OK: {dst} (без детекта {miss}/{total})")


if __name__ == "__main__":
    main()
