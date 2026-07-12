"""КОМПОЗИТНЫЙ БЕЙЗЛАЙН рексиса знакомств + новые гипотезы улучшения.

    ENV_FOR_DYNACONF=natural uv run python scripts/recsys_baseline.py

Все куски были провалидированы по отдельности, но ВМЕСТЕ не мерились. Здесь собираем композит
и получаем ОДНО число, которое дальше улучшаем.

КОМПОЗИТ (baseline):
  score(A,B) = prior(B) + w_A·z_B              — популяционный приор + пер-юзерный residual
  показ      = Thompson (не жадный: жадный выжигает каталог)
  ранжирование матчей = calibrated reciprocal: score_A(B) * rank_B(A)
  голова     = байесовская, whitened PCA-40, λ=300

ГИПОТЕЗЫ:
  H1 онбординг: k чистых Likert-оценок ДО свайпов (чистая метка стоит дороже свайпа)
  H2 размерность головы: 10/20/40/80 PCA — bias-variance при малом N

Метрики (то, что важно продукту):
  quality@100 — качественных матчей на 100 показов (обе стороны в топ-25% друг у друга)
  right@100   — доля «свайпнул бы вправо» среди показанных
  taste       — Spearman головы с истинным вкусом на отложенных лицах

Пишет metrics/recsys_baseline.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from age_gap.common.io import data_path, resolve_path
from age_gap.common.logging import get_logger
from age_gap.recsys import BayesianLinearHead

log = get_logger(__name__)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pct(v):
    return np.argsort(np.argsort(v)) / (len(v) - 1)


def run(Z, prior, y, dim, n_swipes, onboard_k, rng, taste_corr=0.80, lam=300.0,
        n_show=100, alpha=1.0, eng=None, pop_dim=40):
    """Один прогон композита. Возвращает (taste, right@100, quality@100).

    ВАЖНО: ``pop_dim`` (размерность вкусов ПОПУЛЯЦИИ) зафиксирована — иначе, меняя ``dim``,
    мы бы меняли сам мир, а не только ёмкость головы, и сравнение было бы нечестным.
    """
    n = len(y)
    # --- популяция кандидатов: вкусы + переборчивость (для реципрокности) ---
    sd_p = float(np.std(prior))
    w_norm = sd_p * np.sqrt(1.0 / taste_corr ** 2 - 1.0)
    W = rng.normal(size=(n, pop_dim))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    W *= w_norm
    S = prior[None, :] + W @ Z[:, :pop_dim].T
    # --- фокусный юзер A: истинный вкус = его оценки ---
    a_idx = int(np.argmin(np.abs(prior - np.quantile(prior, 0.5))))
    zA = Z[a_idx, :pop_dim]
    pct_B_of_A = np.array([(S[i] < (prior[a_idx] + W[i] @ zA)).mean() for i in range(n)])  # ранг A у B

    # свайп юзера A (возможно контаминированный)
    zy = (y - y.mean()) / (y.std() + 1e-9)
    latent = zy if alpha == 1.0 else alpha * zy + (1 - alpha) * ((eng - eng.mean()) / (eng.std() + 1e-9))
    p_swipe = _sig(2.0 * (latent - np.quantile(latent, 0.77)))
    swipe = (rng.random(n) < p_swipe).astype(float)

    a, b = np.polyfit(prior, swipe, 1)
    pr = a * prior + b
    head = BayesianLinearHead(dim, prior_precision=lam)

    hold = rng.permutation(n)[:400]
    pool = np.setdiff1d(np.arange(n), np.append(hold, a_idx))

    # --- H1: онбординг на ЧИСТЫХ оценках (шкала 1..5, не свайп) ---
    if onboard_k:
        ob = rng.choice(pool, size=onboard_k, replace=False)
        ay, by = np.polyfit(prior[pool], y[pool], 1)
        for j in ob:
            head.update(Z[j, :dim], y[j] - (ay * prior[j] + by))
        pool = np.setdiff1d(pool, ob)

    # --- сессия: Thompson-показ, онлайн-обучение на свайпах ---
    remaining = list(pool)
    shown = []
    for _ in range(n_swipes):
        if not remaining:
            break
        idx = np.array(remaining)
        s = pr[idx] + head.thompson(Z[idx, :dim], rng)
        j = int(idx[int(np.argmax(s))])
        shown.append(j)
        head.update(Z[j, :dim], swipe[j] - pr[j])
        remaining.remove(j)

    from scipy.stats import spearmanr
    taste = float(spearmanr(y[hold], pr[hold] + head.predict(Z[hold, :dim])[0]).statistic)

    # --- выдача после обучения: калиброванная реципрокность ---
    # ВАЖНО: оцениваем на СВЕЖИХ лицах (holdout), а не на остатке каталога — иначе меряем
    # истощение (Thompson выел верхушку), а не качество головы.
    cand = hold
    s_A = pr[cand] + head.predict(Z[cand, :dim])[0]
    key = _pct(s_A) * pct_B_of_A[cand]                 # calibrated reciprocal
    sel = cand[np.argsort(-key)[:n_show]]
    right = float((y[sel] >= 3).mean())
    pct_A_of_B = _pct(y)
    quality = float(np.mean((pct_A_of_B[sel] > 0.75) & (pct_B_of_A[sel] > 0.75)))
    return taste, right, quality


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    z = np.load(resolve_path("data_beauty", "cache", "recsys_features.npz"), allow_pickle=True)
    E, prior, y = z["E"], z["prior"], z["y"]
    sc = StandardScaler().fit(E)
    Z = PCA(80, random_state=0, whiten=True).fit_transform(sc.transform(E))

    results = {}

    def bench(name, **kw):
        out = np.array([run(Z, prior, y, rng=np.random.default_rng(900 + s), **kw)
                        for s in range(args.seeds)])
        m = out.mean(0)
        results[name] = {"taste": round(float(m[0]), 4), "right@100": round(float(m[1]), 4),
                         "quality@100": round(float(m[2]), 4)}
        log.info("%-34s taste=%.3f  right@100=%.3f  quality@100=%.3f", name, *m)

    # ---- BASELINE (композит: dim=40, без онбординга) ----
    for ns in [0, 100, 400]:
        bench(f"baseline/dim40/swipes{ns}", dim=40, n_swipes=ns, onboard_k=0)

    # ---- H2: размерность головы × число свайпов (растёт ли оптимум с N?) ----
    for d in [10, 20, 30, 40]:
        for ns in [50, 100, 200, 400]:
            bench(f"H2/dim{d}/swipes{ns}", dim=d, n_swipes=ns, onboard_k=0)

    # ---- H4: усадка λ при лучшем dim ----
    for lam in [100.0, 300.0, 800.0]:
        bench(f"H4/dim20/lam{lam:g}/swipes100", dim=20, n_swipes=100, onboard_k=0, lam=lam)

    # ---- H1+H2: компаундится ли онбординг с лучшим dim? ----
    for k in [0, 20, 50]:
        for ns in [0, 100]:
            bench(f"BEST/dim20/onboard{k}/swipes{ns}", dim=20, n_swipes=ns, onboard_k=k)

    dst = data_path("metrics_dir", "recsys_baseline.json")
    dst.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
