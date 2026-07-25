"""Трудный cross-age бенчмарк: позитивы 25+ против ПОХОЖИХ импостеров, а не случайных.

    uv run python scripts/hardneg_benchmark.py

ЗАЧЕМ. Замороженный AdaFace IR-101 решает наш тест на our.25+ = 0.9675, тогда как весь наш
пайплайн даёт 0.838 — отсюда вывод рецензента TNNLS о практической значимости. Диагностика
(scripts/hardneg_probe.py) показала причину: в тесте нет трудных негативов вовсе (4412 случайных
`negative_cross_group` + 695 `negative_age_controlled`), вся сложность сидит в позитивах. Контроль
по возрасту стоит всего −0.019, т.е. атрибутивного ужесточения мало.

ЧТО ДЕЛАЕМ. Оставляем позитивы как есть (тот же человек через годы) и заменяем негативы на
ПОХОЖИХ импостеров: другой человек, тот же кажущийся пол, близкий кажущийся возраст (|Δ|<=5) и
максимальное косинусное сходство. Если современные модели на этом проваливаются — датасет
становится бенчмарком, который ломает SOTA, и запас, где наши данные могут помогать, возвращается.

ЦИРКУЛЯРНОСТЬ — главная ловушка. Майнить негативы тем же энкодером, которым потом меряешь, значит
искусственно занижать именно его. Поэтому майним НЕЗАВИСИМЫМ w600k_r50 (он используется в проекте
для кластеризации и НЕ входит в список оцениваемых). Оговорка, которую нельзя опускать: r100 с
майнером одного семейства (ArcFace), AdaFace — другого, поэтому r100 может быть наказан сильнее.

Негативы берутся только среди личностей ТЕСТОВОГО сплита — иначе утечка train-личностей в тест.

Пишет metrics/hardneg_benchmark.json.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import torch

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl
from age_gap.common.logging import get_logger
from age_gap.evaluation.metrics import roc_auc
from age_gap.models.backbones import make_backbone
from age_gap.models.embeddings import load_embeddings
from age_gap.training.finetune import _bb_prep, _crop_path

log = get_logger(__name__)


@torch.no_grad()
def _embed_faces(backbone, face_ids, crops_dir, device, batch=64):
    """Эмбеддинги замороженного бэкбона по ЛИЦАМ (а не парам) -> {face_id: vec}."""
    prep = _bb_prep(backbone)
    out: dict[str, np.ndarray] = {}
    buf_id, buf_im = [], []

    def flush():
        if not buf_id:
            return
        t = torch.from_numpy(np.stack(buf_im)).to(device)
        z = backbone(t).cpu().numpy()
        for k, fid in enumerate(buf_id):
            out[fid] = z[k]
        buf_id.clear()
        buf_im.clear()

    for fid in face_ids:
        p = _crop_path(fid, crops_dir)
        im = cv2.imread(str(p)) if p.exists() else None
        if im is None:
            continue
        buf_im.append(prep(im))
        buf_id.append(fid)
        if len(buf_id) >= batch:
            flush()
    flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+",
                    default=["adaface_ir101", "arcface_r100", "adaface_ir50"])
    ap.add_argument("--crops", default="faces")
    ap.add_argument("--max-age-diff", type=float, default=5.0)
    args = ap.parse_args()

    device = torch_device()
    pairs = [r for r in read_jsonl(data_path("data_dir", "processed", "pairs.jsonl"))
             if r.get("split") == "test"]
    pos = [r for r in pairs if r["label"] == 1]
    log.info("тестовых позитивов: %d (из них 25+: %d)",
             len(pos), sum(1 for r in pos if (r.get("age_gap") or -1) >= 25))

    # личность каждого лица + пул кандидатов-импостеров (только из ТЕСТОВЫХ личностей)
    grp: dict[str, str] = {}
    for r in pairs:
        grp[r["face_a"]] = r["identity_group_a"]
        grp[r["face_b"]] = r["identity_group_b"]
    ga = {r["face_id"]: (float(r["age_est"]), int(r["gender"]))
          for r in read_jsonl(data_path("data_dir", "interim", "face_genderage.jsonl"))}
    mine = load_embeddings()                       # независимый майнер (w600k_r50)
    pool = [f for f in grp if f in mine and f in ga]
    log.info("пул импостеров: %d лиц из %d личностей", len(pool), len(set(grp.values())))

    M = np.stack([mine[f] for f in pool]).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    p_grp = np.array([grp[f] for f in pool])
    p_age = np.array([ga[f][0] for f in pool])
    p_gen = np.array([ga[f][1] for f in pool])

    # --- майним по одному ПОХОЖЕМУ импостеру на каждый позитив ---
    negs: list[tuple[str, str]] = []
    skipped = 0
    for r in pos:
        a = r["face_a"]
        if a not in mine or a not in ga:
            skipped += 1
            continue
        va = mine[a].astype(np.float32)
        va = va / (np.linalg.norm(va) + 1e-9)
        age_a, gen_a = ga[a]
        ok = (p_grp != grp[a]) & (p_gen == gen_a) & (np.abs(p_age - age_a) <= args.max_age_diff)
        if not ok.any():
            skipped += 1
            continue
        sims = M @ va
        sims[~ok] = -2.0
        negs.append((a, pool[int(np.argmax(sims))]))
    log.info("трудных негативов намайнено: %d (пропущено %d)", len(negs), skipped)

    need = sorted({r["face_a"] for r in pos} | {r["face_b"] for r in pos}
                  | {b for _, b in negs})
    gaps = np.array([(r.get("age_gap") or -1) for r in pos])

    out: dict[str, dict[str, float]] = {}
    for name in args.backbones:
        bb = make_backbone(name, pretrained=True).to(device).eval()
        bb.crops_dir = args.crops  # type: ignore[assignment]
        emb = _embed_faces(bb, need, args.crops, device)

        def cos(x, y, emb=emb):
            if x not in emb or y not in emb:
                return None
            return float(np.dot(emb[x], emb[y]))

        sp = [(cos(r["face_a"], r["face_b"]), g) for r, g in zip(pos, gaps, strict=True)]
        sp = [(s, g) for s, g in sp if s is not None]
        sn = [c for a, b in negs if (c := cos(a, b)) is not None]

        pos_s = np.array([s for s, _ in sp])
        pos_g = np.array([g for _, g in sp])
        neg_s = np.array(sn)
        r: dict[str, float] = {"n_pos": float(len(pos_s)), "n_neg": float(len(neg_s))}
        for tag, mask in [("overall", np.ones(len(pos_s), bool)), ("25+", pos_g >= 25)]:
            s = np.concatenate([pos_s[mask], neg_s])
            y = np.concatenate([np.ones(mask.sum()), np.zeros(len(neg_s))])
            r[f"HARD {tag}"] = float(roc_auc(s, y))
            r[f"HARD {tag} n_pos"] = float(mask.sum())
        out[f"{name}:frozen"] = r
        log.info("%-14s HARD overall=%.4f  HARD 25+=%.4f  (pos=%d/%d, neg=%d)", name,
                 r["HARD overall"], r["HARD 25+"], int(r["HARD 25+ n_pos"]), len(pos_s), len(neg_s))
        del bb, emb
        torch.cuda.empty_cache()

    out["_ref_easy_negatives"] = {
        "adaface_ir101 our.25+": 0.9675, "arcface_r100 our.25+": 0.9420,
        "adaface_ir50 our.25+": 0.9308, "наш дообученный FaceNet our.25+": 0.838,
        "майнер": "w600k_r50 (независим от оцениваемых; одно семейство с r100 — оговорка)",
    }
    dst = data_path("metrics_dir", "hardneg_benchmark.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)
    print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
