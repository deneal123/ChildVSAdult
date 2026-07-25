"""Парное сравнение на трудном бенчмарке: наш дообученный слабый бэкбон против замороженной SOTA.

    uv run python scripts/hardneg_paired.py

Кривая сложности дала, что на rank10 FaceNet+наши пары (0.7190) и замороженный AdaFace IR-101
(0.7045) неразличимы, а на rank1 мы впереди (0.5628 против 0.4743). Но там сравнивались
НЕЗАВИСИМЫЕ доверительные интервалы, а это неверный тест: обе модели считаются на ОДНИХ И ТЕХ ЖЕ
парах, значит их ошибки скоррелированы. Правильно — парный бутстрап по ЯКОРЯМ: ресэмплим якоря,
пересчитываем обе AUC на одной и той же выборке и смотрим распределение РАЗНОСТИ.

Это единственное место, где утверждение «догоняем/обходим SOTA» может быть подтверждено или снято.

Пишет metrics/hardneg_paired.json.
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


def _load(spec, device, crops):
    name, _, kind = spec.partition(":")
    bb = (load_finetuned(data_path("models_dir", f"bb_{name}_pairs.pt"), device)
          if kind == "tuned" else make_backbone(name, pretrained=True).to(device).eval())
    bb.crops_dir = crops
    return bb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="facenet:tuned", help="наша модель")
    ap.add_argument("--b", default="adaface_ir101", help="замороженная SOTA")
    ap.add_argument("--crops", default="faces")
    ap.add_argument("--max-age-diff", type=float, default=5.0)
    ap.add_argument("--n-boot", type=int, default=2000)
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

    anchors, gaps, negs = [], [], {f"rank{k}": [] for k in RANKS}
    negs["random"] = []
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
            negs[f"rank{k}"].append(pool[int(order[k - 1])])
        negs["random"].append(pool[int(rng.choice(ok))])
    gaps = np.asarray(gaps)
    pos_b = {r["face_a"]: r["face_b"] for r in pos}
    need = sorted(set(anchors) | set(pos_b.values()) | {f for v in negs.values() for f in v})

    E = {}
    for spec in (args.a, args.b):
        bb = _load(spec, device, args.crops)
        E[spec] = _embed_faces(bb, need, args.crops, device)
        del bb
        torch.cuda.empty_cache()

    out: dict[str, dict] = {}
    for tag in ["random", *[f"rank{k}" for k in reversed(RANKS)]]:
        # общие якоря, где обе модели дали эмбеддинги -> строго парное сравнение
        idx = [i for i, a in enumerate(anchors)
               if all(a in E[s] and pos_b[a] in E[s] and negs[tag][i] in E[s] for s in E)]
        idx25 = [i for i in idx if gaps[i] >= 25]
        sc = {s: (np.array([np.dot(E[s][anchors[i]], E[s][pos_b[anchors[i]]]) for i in idx25]),
                  np.array([np.dot(E[s][anchors[i]], E[s][negs[tag][i]]) for i in idx]))
              for s in E}

        def auc(ps, ns):
            return roc_auc(np.concatenate([ps, ns]),
                           np.concatenate([np.ones(len(ps)), np.zeros(len(ns))]))

        pa, pb_ = auc(*sc[args.a]), auc(*sc[args.b])
        r = np.random.default_rng(0)
        d = []
        for _ in range(args.n_boot):
            ip = r.integers(0, len(idx25), len(idx25))       # ОДНИ И ТЕ ЖЕ якоря обеим моделям
            inn = r.integers(0, len(idx), len(idx))
            d.append(auc(sc[args.a][0][ip], sc[args.a][1][inn])
                     - auc(sc[args.b][0][ip], sc[args.b][1][inn]))
        d = np.asarray(d)
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[tag] = {args.a: round(float(pa), 4), args.b: round(float(pb_), 4),
                    "delta": round(float(pa - pb_), 4),
                    "delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
                    "P(delta>0)": round(float((d > 0).mean()), 4),
                    "n_pos": len(idx25), "n_neg": len(idx)}
        log.info("%-8s %s=%.4f  %s=%.4f  Δ=%+.4f [%+.4f,%+.4f]  P(Δ>0)=%.3f",
                 tag, args.a, pa, args.b, pb_, pa - pb_, lo, hi, (d > 0).mean())

    dst = data_path("metrics_dir", "hardneg_paired.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
