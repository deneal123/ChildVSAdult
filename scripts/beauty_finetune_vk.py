"""W4-lite: адаптация beauty-модели к ТВОЕМУ вкусу по парным меткам (learning-to-rank).

    uv run python scripts/beauty_finetune_vk.py --pairs reports/rating/pairs.jsonl

При малом числе пар дообучать backbone нельзя (переобучится), поэтому учим ЛЁГКИЙ линейный
re-ranker поверх ЗАМОРОЖЕННЫХ DINOv2-эмбеддингов: PCA -> логистический RankSVM на разностях пар
(w·(emb_win − emb_lose) > 0). 5-fold по парам; сравниваем held-out pairwise-accuracy с SCUT-baseline
(знак sa−sb). Пишет metrics/beauty_rerank.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from age_gap import beauty
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="reports/rating/pairs.jsonl")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pca", type=int, default=40)
    ap.add_argument("--C", type=float, default=0.2, help="регуляризация LogReg (меньше = сильнее)")
    args = ap.parse_args()

    import torch
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    rows = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    log.info("Решительных пар: %d", len(rows))

    # эмбеддинги DINOv2 для всех участвующих лиц
    device = beauty.pick_device()
    wpath = resolve_path(args.weights) if args.weights else resolve_path("data_beauty", "weights", "beauty_dinov2.pt")
    ckpt = torch.load(wpath, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "clip")
    model = beauty.BeautyRegressor(backbone, unfreeze_top=int(ckpt["unfreeze_vision"])).to(device)
    model.load_state_dict(ckpt["state_dict"])

    ids = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    hd = data_path("data_dir", "interim", "faces_hires")
    paths = [hd / f"{i}.jpg" for i in ids]
    emb = beauty.embed_paths(model, [p if p.exists() else None for p in paths], device, backbone)
    idx = {i: k for k, i in enumerate(ids)}
    ok = ~np.isnan(emb).any(1)
    rows = [r for r in rows if ok[idx[r["a"]]] and ok[idx[r["b"]]]]
    log.info("Пар с эмбеддингами обоих лиц: %d", len(rows))

    A = np.array([idx[r["a"]] for r in rows])
    B = np.array([idx[r["b"]] for r in rows])
    y = np.array([1 if r["winner"] == r["a"] else 0 for r in rows])
    base = np.array([1 if r["sa"] > r["sb"] else 0 for r in rows])  # SCUT-baseline на этой же паре

    rr_acc, bl_acc, n_te = [], [], []
    for tr, te in KFold(n_splits=args.folds, shuffle=True, random_state=0).split(rows):
        k = min(args.pca, len(tr) - 1, emb.shape[1])
        sc = StandardScaler().fit(emb)
        pca = PCA(n_components=k, random_state=0).fit(sc.transform(emb))
        z = pca.transform(sc.transform(emb))
        # обучающие разности в обе стороны (баланс классов)
        dtr = np.vstack([z[A[tr]] - z[B[tr]], z[B[tr]] - z[A[tr]]])
        ytr = np.concatenate([y[tr], 1 - y[tr]])
        clf = LogisticRegression(C=args.C, max_iter=2000).fit(dtr, ytr)
        w = clf.coef_[0]
        s = z @ w  # скор re-ranker'а на все лица
        pred = (s[A[te]] > s[B[te]]).astype(int)
        rr_acc.append(float((pred == y[te]).mean()))
        bl_acc.append(float((base[te] == y[te]).mean()))
        n_te.append(len(te))

    out = {
        "n_pairs": len(rows), "n_faces": int(ok.sum()), "folds": args.folds, "pca": args.pca, "C": args.C,
        "reranker_heldout_acc": round(float(np.average(rr_acc, weights=n_te)), 4),
        "scut_baseline_heldout_acc": round(float(np.average(bl_acc, weights=n_te)), 4),
        "reranker_per_fold": [round(a, 3) for a in rr_acc],
        "note": "pair-level CV (лица могут пересекаться train/test) — предварительная проба адаптации; "
                "для чистой генерализации нужен face-disjoint eval и больше пар",
    }
    dst = data_path("metrics_dir", "beauty_rerank.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("reranker_heldout_acc", "scut_baseline_heldout_acc", "n_pairs")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
