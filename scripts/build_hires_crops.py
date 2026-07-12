"""Свободные hi-res кропы взрослых лиц из ОРИГИНАЛОВ (для beauty-модели).

Сейчас на диске только 112px identity-выровненные кропы — эстетика убита. Оригиналы целы
(data*/raw/images/), а bbox лежит в faces.jsonl в координатах оригинала, так что кропы с
полями регенерируются без докачки и без повторной детекции.

    uv run python scripts/build_hires_crops.py                     # then/now
    ENV_FOR_DYNACONF=natural uv run python scripts/build_hires_crops.py

Резюмируемо (готовые файлы пропускаются). Гардрейл: только взрослые (age>=18) — детские лица
не кропаются вовсе. Пишет data*/interim/faces_hires/{face_id}.jpg (в .gitignore).
"""

from __future__ import annotations

import argparse

import cv2

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.common.schemas import RawPost
from age_gap.engagement.features import ADULT_MIN_AGE
from age_gap.preprocessing.crop_align import margin_crop

log = get_logger(__name__)


def _photo_paths() -> dict[str, str]:
    out: dict[str, str] = {}
    for row in read_jsonl(data_path("data_dir", "raw", "posts.jsonl")):
        for photo in RawPost.from_dict(row).photos:
            if photo.local_path:
                out[photo.photo_id] = photo.local_path
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin", type=float, default=0.4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out-suffix", default="", help="суффикс папки: faces_hires<suffix> "
                                                     "(для мульти-масштабных кропов)")
    ap.add_argument("--face-list", default=None,
                    help="файл со списком face_id (по одному в строке): кропать ТОЛЬКО их")
    args = ap.parse_args()

    only = None
    if args.face_list:
        only = {ln.strip() for ln in resolve_path(args.face_list).read_text(encoding="utf-8").splitlines()
                if ln.strip()}
        log.info("Точечный режим: только %d указанных лиц", len(only))

    out_dir = data_path("data_dir", "interim", f"faces_hires{args.out_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    photo_paths = _photo_paths()
    ages = {r["face_id"]: r for r in read_jsonl(data_path("data_dir", "interim", "face_genderage.jsonl"))}

    new = skipped = no_img = child = 0
    for f in read_jsonl(data_path("data_dir", "interim", "faces.jsonl")):
        if not f.get("is_usable") or not f.get("bbox"):
            continue
        if only is not None and f["face_id"] not in only:
            continue
        ga = ages.get(f["face_id"])
        if ga is None or float(ga["age_est"]) < ADULT_MIN_AGE:  # guardrail: только взрослые
            child += 1
            continue
        dest = out_dir / f"{f['face_id']}.jpg"
        if dest.exists():
            skipped += 1
            continue
        local = photo_paths.get(f.get("photo_id"))
        if not local:
            continue
        img = cv2.imread(str(resolve_path(local)))
        if img is None:
            no_img += 1
            continue
        crop = margin_crop(img, f["bbox"], margin=args.margin, size=args.size)
        cv2.imwrite(str(dest), crop)
        new += 1
        if new % 2000 == 0:
            log.info("hi-res кропы: готово %d (skip=%d, no_img=%d)", new, skipped, no_img)

    log.info("Готово: новых %d, было %d, без оригинала %d, детских пропущено %d -> %s",
             new, skipped, no_img, child, out_dir)
    print(f"OK: faces_hires new={new} skipped={skipped} no_img={no_img} -> {out_dir}")


if __name__ == "__main__":
    main()
