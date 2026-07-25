"""Кривая сложности негативов: артефакт ли обвал на трудных импостерах?

    uv run python scripts/hardneg_curve.py

Первый замер (scripts/hardneg_benchmark.py) дал резкое: замороженный AdaFace IR-101 падает на
25+ с 0.9675 (случайные негативы) до 0.4877 (самый похожий импостер) — ниже случайного угадывания.
Прежде чем этому верить, надо снять два возражения.

(1) СОСТЯЗАТЕЛЬНЫЙ ОТБОР. Брать top-1 самого похожего импостера из ~6000 кандидатов — это не
    «честная сложность», а атака: мы адверсарно выбираем негатив и сравниваем со СЛУЧАЙНЫМ
    позитивом. Поэтому здесь строится КРИВАЯ по рангу (1, 10, 50, 200, случайный): если обвал
    градуированный и монотонный по рангу — это настоящая ось сложности, а не одна выбросная точка.

(2) МАЛАЯ ВЫБОРКА. Позитивов с разрывом 25+ всего 154 — без доверительных интервалов такие
    числа читать нельзя. Бутстрап по позитивам И негативам, 1000 итераций.

Дополнительный контроль на смещение майнера: майнер (w600k_r50) одного семейства с ArcFace r100,
поэтому r100 может быть наказан сильнее AdaFace. Если порядок моделей на кривой сохраняется тот
же, что на лёгких негативах, — смещение майнера не определяет результат.

Пишет metrics/hardneg_curve.json.
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
from age_gap.training.finetune import _bb_prep, _crop_path, load_finetuned

log = get_logger(__name__)
RANKS = [1, 10, 50, 200]


@torch.no_grad()
def _embed_faces(backbone, face_ids, crops_dir, device, batch=64):
    prep = _bb_prep(backbone)
    out: dict[str, np.ndarray] = {}
    buf_id, buf_im = [], []

    def flush():
        if not buf_id:
            return
        z = backbone(torch.from_numpy(np.stack(buf_im)).to(device)).cpu().numpy()
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


def _auc_ci(pos_s, neg_s, n_boot=1000, seed=0):
    """AUC + перцентильный бутстрап (ресэмплим позитивы и негативы независимо)."""
    s = np.concatenate([pos_s, neg_s])
    y = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
    point = float(roc_auc(s, y))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        p = pos_s[rng.integers(0, len(pos_s), len(pos_s))]
        n = neg_s[rng.integers(0, len(neg_s), len(neg_s))]
        vals.append(roc_auc(np.concatenate([p, n]),
                            np.concatenate([np.ones(len(p)), np.zeros(len(n))])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


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
    grp: dict[str, str] = {}
    for r in pairs:
        grp[r["face_a"]] = r["identity_group_a"]
        grp[r["face_b"]] = r["identity_group_b"]
    ga = {r["face_id"]: (float(r["age_est"]), int(r["gender"]))
          for r in read_jsonl(data_path("data_dir", "interim", "face_genderage.jsonl"))}
    mine = load_embeddings()
    pool = [f for f in grp if f in mine and f in ga]
    M = np.stack([mine[f] for f in pool]).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    p_grp = np.array([grp[f] for f in pool])
    p_age = np.array([ga[f][0] for f in pool])
    p_gen = np.array([ga[f][1] for f in pool])
    log.info("позитивов %d (25+: %d), пул импостеров %d",
             len(pos), sum(1 for r in pos if (r.get("age_gap") or -1) >= 25), len(pool))

    # для каждого якоря — ранжированный список подходящих импостеров
    anchors, gaps = [], []
    neg_by_rank: dict[str, list[str]] = {f"rank{r}": [] for r in RANKS}
    neg_by_rank["random"] = []
    rng = np.random.default_rng(0)
    for r in pos:
        a = r["face_a"]
        if a not in mine or a not in ga:
            continue
        va = mine[a].astype(np.float32)
        va /= np.linalg.norm(va) + 1e-9
        age_a, gen_a = ga[a]
        ok = np.nonzero((p_grp != grp[a]) & (p_gen == gen_a)
                        & (np.abs(p_age - age_a) <= args.max_age_diff))[0]
        if len(ok) < max(RANKS) + 1:
            continue
        order = ok[np.argsort(-(M[ok] @ va))]
        anchors.append(a)
        gaps.append(r.get("age_gap") or -1)
        for k in RANKS:
            neg_by_rank[f"rank{k}"].append(pool[int(order[k - 1])])
        neg_by_rank["random"].append(pool[int(rng.choice(ok))])
    gaps = np.asarray(gaps)
    log.info("якорей с достаточным пулом: %d (25+: %d)", len(anchors), int((gaps >= 25).sum()))

    need = sorted(set(anchors) | {r["face_b"] for r in pos}
                  | {f for v in neg_by_rank.values() for f in v})
    pos_b = {r["face_a"]: r["face_b"] for r in pos}

    out: dict[str, dict] = {}
    for spec in args.backbones:
        # "name" -> замороженный; "name:tuned" -> наш дообученный на парах bb_<name>_pairs.pt
        name, _, kind = spec.partition(":")
        if kind == "tuned":
            ckpt = data_path("models_dir", f"bb_{name}_pairs.pt")
            if not ckpt.exists():
                log.warning("нет чекпойнта %s — пропуск", ckpt)
                continue
            bb = load_finetuned(ckpt, device)
        else:
            bb = make_backbone(name, pretrained=True).to(device).eval()
        bb.crops_dir = args.crops  # type: ignore[assignment]
        emb = _embed_faces(bb, need, args.crops, device)
        res: dict[str, dict] = {}
        for tag in ["random", *[f"rank{k}" for k in reversed(RANKS)]]:
            ps, ns, keep = [], [], []
            for i, a in enumerate(anchors):
                b, nf = pos_b[a], neg_by_rank[tag][i]
                if a in emb and b in emb and nf in emb:
                    ps.append(float(np.dot(emb[a], emb[b])))
                    ns.append(float(np.dot(emb[a], emb[nf])))
                    keep.append(gaps[i])
            ps, ns, keep = np.array(ps), np.array(ns), np.array(keep)
            m25 = keep >= 25
            for lab, mask in [("overall", np.ones(len(ps), bool)), ("25+", m25)]:
                pt, lo, hi = _auc_ci(ps[mask], ns)
                res[f"{tag} | {lab}"] = {"auc": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                         "n_pos": int(mask.sum()), "n_neg": int(len(ns))}
            log.info("%-22s %-8s overall=%.4f  25+=%.4f [%.3f,%.3f]", spec, tag,
                     res[f"{tag} | overall"]["auc"], res[f"{tag} | 25+"]["auc"],
                     *res[f"{tag} | 25+"]["ci95"])
        out[spec] = res
        del bb, emb
        torch.cuda.empty_cache()

    out["_note"] = {
        "майнер": "w600k_r50 — независим от оцениваемых; одного семейства с ArcFace r100",
        "смысл": "rank1 = самый похожий импостер (состязательный отбор), random = как в статье",
        "контроль": "если порядок моделей сохраняется как на лёгких негативах, "
                    "смещение майнера не определяет результат",
    }
    dst = data_path("metrics_dir", "hardneg_curve.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
