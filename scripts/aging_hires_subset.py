"""#4 Native-resolution aging sanity check (subset).

FRANAging internally upscales our stored 112px crops to 512px before re-aging, so the §5.2
"synthetic hurts" result might be a low-resolution artifact. This script re-crops a TRAIN
subset at native 512px directly from the raw images (insightface norm_crop on stored
landmarks), then trains synthetic-FRAN from native-512 source vs upscaled-112 source on the
SAME faces. If both stay weak vs frozen, the finding is not a resolution artifact.

Does NOT touch pairs.jsonl / the split (uses identity_groups + group_splits + face crops).

    uv run python scripts/aging_hires_subset.py --max-persons 150 --epochs 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from insightface.utils.face_align import norm_crop

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup
from age_gap.evaluation.external_suite import eval_all, print_table
from age_gap.models.aging import FRANAging
from age_gap.models.facenet import FaceNetBackbone
from age_gap.training.finetune import _crop_path, load_finetuned
from age_gap.training.synthetic import train_synthetic

log = get_logger(__name__)
HIRES_DIR = "faces_hires512"


def _raw_index() -> dict[str, Path]:
    base = Path(resolve_path(str(data_path("data_dir", "raw", "images"))))
    idx: dict[str, Path] = {}
    for p in base.rglob("*.jpg"):
        idx.setdefault(p.stem, p)
    return idx


def _recrop_subset(max_persons: int, size: int) -> set[str]:
    gsplit = {
        r["identity_group_id"]: r["split"]
        for r in read_jsonl(data_path("splits_dir", "group_splits.jsonl"))
    }
    meta = {r["face_id"]: r for r in read_jsonl(data_path("data_dir", "interim", "faces.jsonl"))}
    raw = _raw_index()
    out_dir = Path(resolve_path(str(data_path("data_dir", "interim", HIRES_DIR))))
    out_dir.mkdir(parents=True, exist_ok=True)

    subset: set[str] = set()
    npersons = 0
    for row in read_jsonl(data_path("data_dir", "processed", "identity_groups.jsonl")):
        g = IdentityGroup.from_dict(row)
        if gsplit.get(g.identity_group_id) != "train":
            continue
        person_faces: list[str] = []
        for fid in g.faces:
            if not _crop_path(fid, "faces").exists():
                continue
            m = meta.get(fid)
            if m is None:
                continue
            lmk = np.asarray(m.get("landmarks") or [], dtype=np.float32)
            if lmk.shape != (5, 2):
                continue
            rp = raw.get(m["photo_id"])
            if rp is None:
                continue
            img = cv2.imread(str(rp))
            if img is None:
                continue
            cv2.imwrite(str(out_dir / f"{fid}.jpg"), norm_crop(img, lmk, image_size=size))
            person_faces.append(fid)
        if len(person_faces) >= 2:
            subset.update(person_faces)
            npersons += 1
            if npersons >= max_persons:
                break
    log.info("hires subset: persons=%d faces=%d -> %s", npersons, len(subset), out_dir)
    return subset


def main() -> None:
    ap = argparse.ArgumentParser(description="Native-res aging sanity check (subset)")
    ap.add_argument("--max-persons", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    device = torch_device()
    models = data_path("models_dir")
    subset = _recrop_subset(args.max_persons, args.size)
    if len(subset) < 50:
        raise RuntimeError(f"subset too small ({len(subset)}); raise --max-persons")

    ck512 = train_synthetic(
        epochs=args.epochs, trainable_scope="head", aging=FRANAging(input_size=args.size),
        crops_dir=HIRES_DIR, face_ids=subset, ckpt_out=Path(str(models / "abl_syn_fran_hires.pt")),
    )
    ck112 = train_synthetic(
        epochs=args.epochs, trainable_scope="head", aging=FRANAging(input_size=args.size),
        crops_dir="faces", face_ids=subset, ckpt_out=Path(str(models / "abl_syn_fran_lores.pt")),
    )

    results = {
        "frozen": eval_all(FaceNetBackbone(pretrained="casia-webface").to(device).eval(), device),
        "+syn_fran_112(subset)": eval_all(load_finetuned(ck112, device), device),
        "+syn_fran_512(subset)": eval_all(load_finetuned(ck512, device), device),
    }
    order = list(results)
    print_table(results, order)
    dst = resolve_path("docs", "aging_hires.json")
    dst.write_text(json.dumps({"n_faces": len(subset), "results": results}, indent=2), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
