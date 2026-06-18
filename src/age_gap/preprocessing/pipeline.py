"""Оркестрация препроцессинга: посты -> кропы лиц + метаданные.

Поток (SKILL §11):
    posts.jsonl -> загрузка изображения -> детекция -> выбор целевого лица ->
    кроп+выравнивание -> скоринг качества -> FaceCrop (faces.jsonl)

Каждое решение об usable/reject логируется с причиной.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from age_gap.common.io import PROJECT_ROOT, data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.common.schemas import FaceCrop, RawPost
from age_gap.preprocessing.crop_align import align_face, save_crop
from age_gap.preprocessing.detect import DetectedFace, FaceDetector
from age_gap.preprocessing.quality import assess

log = get_logger(__name__)


def _select_target(faces: list[DetectedFace]) -> DetectedFace | None:
    """Выбрать целевое лицо — самое крупное по площади bbox."""
    if not faces:
        return None
    return max(faces, key=lambda f: f.area)


def process_photo(
    detector: FaceDetector,
    image_path: Path,
    photo_id: str,
    faces_dir: Path,
) -> FaceCrop:
    face_id = f"{photo_id}_f0"
    image = cv2.imread(str(image_path))
    if image is None:
        log.warning("Не удалось прочитать изображение %s (reject=unreadable_image)", image_path)
        return FaceCrop(
            face_id=face_id, photo_id=photo_id, is_usable=False, reject_reason="unreadable_image"
        )

    try:
        detected = detector.detect(image)
    except Exception as exc:  # noqa: BLE001 — сбой инференса на одном кадре не должен валить прогон
        log.warning("Фото %s: ошибка детекции (%s) -> reject=detect_error", photo_id, exc)
        return FaceCrop(
            face_id=face_id, photo_id=photo_id, is_usable=False, reject_reason="detect_error"
        )
    target = _select_target(detected)
    num_faces = len(detected)

    if target is None:
        log.info("Фото %s: лицо не найдено (reject=no_face_detected)", photo_id)
        return FaceCrop(
            face_id=face_id,
            photo_id=photo_id,
            num_faces_in_image=0,
            is_usable=False,
            reject_reason="no_face_detected",
        )

    # Кроп по bbox для оценки качества (резкость именно области лица).
    x1, y1, x2, y2 = (int(round(v)) for v in target.bbox)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_region = image[y1:y2, x1:x2]
    face_size = min(target.width, target.height)

    q = assess(
        face_region, face_size=face_size, det_score=target.det_score, num_faces_in_image=num_faces
    )

    crop_path: str | None = None
    if q.is_usable:
        aligned = align_face(image, target.kps)
        dest = faces_dir / f"{face_id}.jpg"
        save_crop(aligned, dest)
        crop_path = str(dest.relative_to(PROJECT_ROOT))

    if not q.is_usable:
        log.info("Фото %s: reject=%s (faces=%d)", photo_id, q.reject_reason, num_faces)

    return FaceCrop(
        face_id=face_id,
        photo_id=photo_id,
        bbox=target.bbox,
        landmarks=target.kps,
        face_crop_path=crop_path,
        face_quality_score=q.quality_score,
        det_score=round(target.det_score, 4),
        num_faces_in_image=num_faces,
        is_usable=q.is_usable,
        reject_reason=q.reject_reason,
    )


def run(
    posts_file: Path | None = None,
    faces_out: Path | None = None,
    faces_dir: Path | None = None,
    device: str | None = None,
) -> int:
    """Инкрементально и возобновляемо: каждая запись пишется сразу (JSONL append).

    Уже обработанные лица (по face_id из существующего faces.jsonl) пропускаются — длинный
    прогон можно безопасно перезапускать, он догонит остаток (устойчиво к обрывам на >10 мин).
    Возвращает число новых обработанных лиц.
    """
    posts_file = posts_file or data_path("data_dir", "raw", "posts.jsonl")
    faces_out = faces_out or data_path("data_dir", "interim", "faces.jsonl")
    faces_dir = faces_dir or data_path("data_dir", "interim", "faces")

    # Уже обработанные face_id (резюме).
    processed: set[str] = {row.get("face_id", "") for row in read_jsonl(faces_out)}
    if processed:
        log.info("Резюме: уже обработано %d лиц, продолжаем", len(processed))

    detector = FaceDetector(device=device)
    Path(faces_out).parent.mkdir(parents=True, exist_ok=True)
    new = 0
    usable = 0
    consecutive_errors = 0
    aborted = False
    with open(faces_out, "a", encoding="utf-8") as f:
        for row in read_jsonl(posts_file):
            post = RawPost.from_dict(row)
            for photo in post.photos:
                if not photo.local_path:
                    continue
                if f"{photo.photo_id}_f0" in processed:
                    continue
                image_path = resolve_path(photo.local_path)
                crop = process_photo(detector, image_path, photo.photo_id, Path(faces_dir))

                # Несколько detect_error подряд = вероятно умер CUDA-контекст: прерываем
                # прогон (прогресс сохранён построчно), чтобы свежий перезапуск догнал остаток
                # и не пометил весь хвост как ошибочный.
                if crop.reject_reason == "detect_error":
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        log.error(
                            "3 detect_error подряд — вероятно сбой GPU/контекста. Прерываю "
                            "(резюмируемо: перезапустите preprocess, лучше --device cpu)."
                        )
                        aborted = True
                        break
                    continue  # не пишем ошибочный кадр — повторим при следующем запуске
                consecutive_errors = 0

                f.write(json.dumps(crop.to_dict(), ensure_ascii=False) + "\n")
                f.flush()
                new += 1
                usable += int(crop.is_usable)
            if aborted:
                break

    log.info("Обработано новых лиц: %d (usable среди новых=%d) -> %s", new, usable, faces_out)
    return new
