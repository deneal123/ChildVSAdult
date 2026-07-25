"""Насыщен ли наш бенчмарк? Разложение AUC по ТИПУ НЕГАТИВА.

    uv run python scripts/hardneg_probe.py --backbones adaface_ir101 facenet

Замороженный AdaFace IR-101 даёт our.25+ = 0.9675 — почти потолок, тогда как весь наш пайплайн
(дообученный FaceNet) даёт 0.838. Отсюда вывод рецензента про практическую значимость.

Гипотеза о причине: сложность нашего теста живёт в ПОЗИТИВАХ (тот же человек через годы), а
негативы бесплатные — состав теста 4412 negative_cross_group (`easy`, случайные разные люди) +
695 negative_age_controlled (`medium`); негативов `hard` нет вовсе. Отличить человека от
СЛУЧАЙНОГО импостора современная модель умеет тривиально, поэтому AUC упирается в потолок.

Проверяем дёшево: считаем скоры один раз и пересчитываем AUC отдельно для каждого типа негатива
(позитивы одни и те же). Если на age-controlled негативах AUC заметно падает — сложность
действительно в негативах, и трудный бенчмарк вернёт запас, где наши данные снова смогут помогать.
Если не падает — насыщение настоящее, и направление закрыто.

Пишет metrics/hardneg_probe.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.evaluation.metrics import roc_auc
from age_gap.models.backbones import make_backbone
from age_gap.training.finetune import ImagePairDataset, _bb_prep

log = get_logger(__name__)


@torch.no_grad()
def _scores(backbone, device, batch=64):
    ds = ImagePairDataset(split="test", preprocess=_bb_prep(backbone),
                          crops_dir=getattr(backbone, "crops_dir", "faces"))
    s = []
    for ta, tb, _y, _w in DataLoader(ds, batch_size=batch):
        za, zb = backbone(ta.to(device)), backbone(tb.to(device))
        s.extend((za * zb).sum(-1).cpu().tolist())
    return np.asarray(s), np.asarray(ds.labels), np.asarray(ds.gaps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["adaface_ir101", "facenet"])
    ap.add_argument("--crops", default="faces")
    args = ap.parse_args()

    device = torch_device()
    # тип негатива в том же порядке, в каком ImagePairDataset читает pairs.jsonl
    ptype = [r["pair_type"] for r in read_jsonl(data_path("data_dir", "processed", "pairs.jsonl"))
             if r.get("split") == "test"]
    ptype = np.asarray(ptype)

    out: dict[str, dict[str, float]] = {}
    for name in args.backbones:
        bb = make_backbone(name, pretrained=True).to(device).eval()
        bb.crops_dir = args.crops  # type: ignore[assignment]
        s, y, gaps = _scores(bb, device)
        if len(ptype) != len(y):
            log.warning("порядок пар не совпал (%d vs %d) — пропуск", len(ptype), len(y))
            return
        pos = y == 1
        pos25 = pos & (gaps >= 25)
        r: dict[str, float] = {}
        for tag, negmask in [
            ("negatives: ВСЕ", y == 0),
            ("negatives: easy (случайные)", (y == 0) & (ptype == "negative_cross_group")),
            ("negatives: age-controlled", (y == 0) & (ptype == "negative_age_controlled")),
        ]:
            for ptag, pmask in [("overall", pos), ("25+", pos25)]:
                m = pmask | negmask
                r[f"{tag} | {ptag}"] = float(roc_auc(s[m], y[m]))
                r[f"{tag} | {ptag} (n_neg)"] = float(negmask.sum())
        out[f"{name}:frozen"] = r
        for k, v in r.items():
            if "n_neg" not in k:
                log.info("%-14s %-42s AUC=%.4f", name, k, v)
        del bb
        torch.cuda.empty_cache()

    dst = data_path("metrics_dir", "hardneg_probe.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
