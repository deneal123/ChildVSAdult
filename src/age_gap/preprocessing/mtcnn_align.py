"""MTCNN-выравнивание лиц для facenet (его «родной» препроцессинг).

InceptionResnetV1 (facenet-pytorch) обучался на лицах, выровненных MTCNN (160px), а наши
основные кропы выровнены по 5-точечному шаблону ArcFace (112px). Это рассогласование
выравнивания — confound в эксперименте с facenet. Здесь строим параллельный набор кропов
``data/interim/faces_mtcnn/{face_id}.jpg``, выровненных MTCNN из ИСХОДНЫХ изображений.

Каждому usable-лицу из ``faces.jsonl`` сопоставляем то же лицо на оригинале по макс-IoU bbox
(на случай нескольких лиц), извлекаем выровненный 160-кроп. Резюмируемо: уже готовые кропы
пропускаются.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.common.schemas import FaceCrop, RawPost

log = get_logger(__name__)


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-9))


def _photo_paths() -> dict[str, str]:
    """photo_id -> локальный путь к исходному изображению (из posts.jsonl)."""
    out: dict[str, str] = {}
    for row in read_jsonl(data_path("data_dir", "raw", "posts.jsonl")):
        post = RawPost.from_dict(row)
        for photo in post.photos:
            if photo.local_path:
                out[photo.photo_id] = photo.local_path
    return out


def build_mtcnn_crops(
    faces_file: Path | None = None,
    out_dir: Path | None = None,
    image_size: int = 160,
    margin: int = 0,
    device: str = "cpu",
) -> int:
    """Построить MTCNN-кропы для всех usable-лиц. Возвращает число новых кропов.

    device="cpu" по умолчанию — чтобы не конкурировать за GPU с обучением; MTCNN на CPU
    приемлемо по скорости для разовой пересборки.
    """
    from facenet_pytorch import MTCNN, extract_face

    faces_file = faces_file or data_path("data_dir", "interim", "faces.jsonl")
    out_dir = out_dir or data_path("data_dir", "interim", "faces_mtcnn")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    photo_paths = _photo_paths()
    mtcnn = MTCNN(
        image_size=image_size, margin=margin, keep_all=True, post_process=False, device=device
    )

    new = 0
    skipped_exist = 0
    no_face = 0
    for row in read_jsonl(faces_file):
        fc = FaceCrop.from_dict(row)
        if not fc.is_usable:
            continue
        dest = Path(out_dir) / f"{fc.face_id}.jpg"
        if dest.exists():
            skipped_exist += 1
            continue
        local = photo_paths.get(fc.photo_id)
        if not local:
            continue
        img_bgr = cv2.imread(str(resolve_path(local)))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        boxes, _probs = mtcnn.detect(pil)
        if boxes is None or len(boxes) == 0:
            no_face += 1
            continue
        # Сопоставляем то же лицо, что выбрал наш пайплайн (макс-IoU к сохранённому bbox);
        # если bbox пуст — берём самый крупный бокс MTCNN.
        if fc.bbox and any(fc.bbox):
            box = max(boxes, key=lambda b: _iou(list(b), fc.bbox))
        else:
            box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        face = extract_face(pil, list(box), image_size=image_size, margin=margin)
        arr = face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()  # RGB uint8 160x160
        cv2.imwrite(str(dest), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        new += 1
        if new % 500 == 0:
            log.info("MTCNN-кропы: готово %d (no_face=%d)", new, no_face)

    log.info(
        "MTCNN-выравнивание: новых %d, уже было %d, без лица %d -> %s",
        new,
        skipped_exist,
        no_face,
        out_dir,
    )
    return new
