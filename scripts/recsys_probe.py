"""Проба recsys-архитектуры: ЗАМОРОЖЕННЫЙ эмбеддинг красоты + лёгкая пер-юзерная голова.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_probe.py

Идея (как в приложении знакомств): тяжёлую модель гоняем ОДИН раз офлайн -> кешируем вектор на
каждое лицо. На юзера — дешёвая линейная голова (ridge), которая учится на его метках/свайпах и
доучивается онлайн. Проверяем:
  * догоняет ли frozen+linear полноценный fine-tune (0.378 OOF / 0.721 на парах);
  * КРИВАЯ ОБУЧЕНИЯ: сколько взаимодействий нужно, чтобы персонализация включилась;
  * cold-start: что даёт популяционный prior (SCUT-модель) без единого свайпа.

Пишет metrics/recsys_probe.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from age_gap import beauty
from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)


def _model(w, device):
    ck = torch.load(resolve_path(w), map_location=device, weights_only=False)
    bb = ck.get("backbone", "dinov2")
    m = beauty.BeautyRegressor(bb, unfreeze_top=int(ck["unfreeze_vision"])).to(device)
    m.load_state_dict(ck["state_dict"])
    return m, bb, float(ck.get("mu", 0.0)), float(ck.get("sd", 1.0))


def _fit_head(E, y, tr, alpha=50.0):
    """Наивная пер-юзерная голова: ridge прямо на замороженном эмбеддинге (вариант A)."""
    sc = StandardScaler().fit(E[tr])
    rg = Ridge(alpha=alpha).fit(sc.transform(E[tr]), y[tr])
    return lambda X: rg.predict(sc.transform(X))


def _fit_residual_head(E, y, prior, tr, n_pca=40, alpha=200.0):
    """ПРАВИЛЬНАЯ схема: персонализация как ОСТАТОК поверх популяционного приора.

        score = prior(face) + lam(n) * personal_residual(emb)

    lam растёт с числом меток (усадка к приору): при n->0 схема вырождается в generic-модель,
    поэтому персонализация НЕ МОЖЕТ стать хуже cold-start. PCA сжимает 768-d, чтобы голова
    не цеплялась за стилевые направления домена.
    """
    from sklearn.decomposition import PCA

    sc = StandardScaler().fit(E[tr])
    k = min(n_pca, len(tr) - 1, E.shape[1])
    pca = PCA(n_components=k, random_state=0).fit(sc.transform(E[tr]))
    z = pca.transform(sc.transform(E[tr]))
    # приор в шкале метки (линейная калибровка prior -> y на train)
    a, b = np.polyfit(prior[tr], y[tr], 1)
    resid = y[tr] - (a * prior[tr] + b)
    rg = Ridge(alpha=alpha).fit(z, resid)
    n = len(tr)
    lam = n / (n + 300.0)  # усадка к приору: мало меток -> доверяем generic-модели

    def predict(X, pr):
        return (a * pr + b) + lam * rg.predict(pca.transform(sc.transform(X)))
    return predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="reports/rating/ratings.jsonl")
    ap.add_argument("--encoder", default="data_beauty/weights/beauty_dinov2.pt",
                    help="ЗАМОРОЖЕННЫЙ энкодер (SCUT-beauty) — источник эмбеддинга красоты")
    ap.add_argument("--pairs", default="reports/rating/pairs.jsonl")
    args = ap.parse_args()

    device = beauty.pick_device()
    model, bb, mu, sd = _model(args.encoder, device)

    rows = [r for r in read_jsonl(resolve_path(args.ratings)) if r.get("score")]
    hd = data_path("data_dir", "interim", "faces_hires")
    rows = [r for r in rows if (hd / f"{r['face_id']}.jpg").exists()]
    paths = [hd / f"{r['face_id']}.jpg" for r in rows]
    y = np.array([float(r["score"]) for r in rows], dtype=np.float64)
    log.info("Меток: %d", len(y))

    # ЗАМОРОЖЕННЫЕ эмбеддинги (тяжёлая модель — один раз, офлайн, кешируется)
    E = beauty.embed_paths(model, paths, device, bb, batch=64).astype(np.float64)
    log.info("Эмбеддинги: %s", E.shape)

    # cold-start: предсказание самой SCUT-модели (популяционный prior, 0 свайпов)
    zs = beauty.score_paths(model, paths, device, mu, sd, bb, batch=64)
    cold = float(spearmanr(y, zs).statistic)
    log.info("COLD-START (SCUT prior, 0 меток): spearman=%.4f", cold)

    # frozen + linear head, 5-fold OOF
    oof = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(E):
        oof[te] = _fit_head(E, y, tr)(E[te])
    full = float(spearmanr(y, oof).statistic)
    log.info("FROZEN+LINEAR (все %d меток, OOF): spearman=%.4f", len(y), full)

    # ВАРИАНТ B: остаток поверх приора + усадка (правильная recsys-схема)
    oof_b = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(E):
        oof_b[te] = _fit_residual_head(E, y, zs, tr)(E[te], zs[te])
    full_b = float(spearmanr(y, oof_b).statistic)
    log.info("RESIDUAL+PRIOR (все %d меток, OOF): spearman=%.4f", len(y), full_b)

    # КРИВАЯ ОБУЧЕНИЯ: сколько взаимодействий нужно (обе схемы)
    rng = np.random.default_rng(0)
    curve = []
    for n in [25, 50, 100, 200, 400, 800, 1200]:
        if n >= len(y):
            break
        sa, sb = [], []
        for _ in range(5):  # усредняем по случайным подвыборкам
            perm = rng.permutation(len(y))
            tr, te = perm[:n], perm[n:]
            sa.append(float(spearmanr(y[te], _fit_head(E, y, tr)(E[te])).statistic))
            sb.append(float(spearmanr(y[te], _fit_residual_head(E, y, zs, tr)(E[te], zs[te])).statistic))
        curve.append({"n_labels": n, "naive": round(float(np.mean(sa)), 4),
                      "residual_prior": round(float(np.mean(sb)), 4)})
        log.info("  n=%4d -> naive=%.4f | residual+prior=%.4f",
                 n, curve[-1]["naive"], curve[-1]["residual_prior"])

    # независимый тест: 215 пар then/now (головы обучены на ВСЕХ метках natural)
    prs = [r for r in read_jsonl(resolve_path(args.pairs)) if r.get("winner")]
    ids = sorted({r["a"] for r in prs} | {r["b"] for r in prs})
    tn = resolve_path("data", "interim", "faces_hires")
    pth = [tn / f"{i}.jpg" for i in ids]
    live = [p if p.exists() else None for p in pth]
    Ep = beauty.embed_paths(model, live, device, bb, batch=64)
    pr_p = beauty.score_paths(model, live, device, mu, sd, bb, batch=64)  # приор на этих лицах
    ok = ~np.isnan(Ep).any(1) & np.isfinite(pr_p)

    def pair_acc(scores):
        smap = dict(zip(ids, scores, strict=True))
        used = [r for r in prs if np.isfinite(smap[r["a"]]) and np.isfinite(smap[r["b"]])
                and smap[r["a"]] != smap[r["b"]]]
        return float(np.mean([1.0 if (smap[r["a"]] > smap[r["b"]]) == (r["winner"] == r["a"]) else 0.0
                              for r in used]))

    allx = np.arange(len(y))
    s_a = np.full(len(ids), np.nan)
    s_a[ok] = _fit_head(E, y, allx)(Ep[ok].astype(np.float64))
    s_b = np.full(len(ids), np.nan)
    s_b[ok] = _fit_residual_head(E, y, zs, allx)(Ep[ok].astype(np.float64), pr_p[ok])
    acc_a, acc_b, acc_prior = pair_acc(s_a), pair_acc(s_b), pair_acc(pr_p)
    log.info("ПАРЫ then/now: prior=%.4f | naive=%.4f | residual+prior=%.4f", acc_prior, acc_a, acc_b)

    out = {
        "architecture": "frozen beauty embedding (heavy model offline) + light per-user head",
        "n_labels": int(len(y)), "embedding_dim": int(E.shape[1]),
        "cold_start_scut_prior_spearman": round(cold, 4),
        "naive_head": {"oof_spearman": round(full, 4), "pairs_accuracy": round(acc_a, 4)},
        "residual_prior_head": {"oof_spearman": round(full_b, 4), "pairs_accuracy": round(acc_b, 4)},
        "prior_only_pairs_accuracy": round(acc_prior, 4),
        "learning_curve": curve,
        "reference_finetuned": {"oof_spearman": 0.3777, "pairs_accuracy": 0.7209},
    }
    dst = data_path("metrics_dir", "recsys_probe.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
