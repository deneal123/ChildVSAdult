"""Референс-реализация пер-юзерной головы для приложения знакомств.

Архитектура (проверена в scripts/recsys_probe.py — бьёт полный fine-tune):

    score(user, face) = prior(face)  +  w_user · PCA(emb(face))

  * ``prior``  — популяционная beauty-модель, считается ОДИН раз офлайн;
  * ``emb``    — замороженный вектор лица, кешируется рядом с фото;
  * ``w_user`` — байесовская линейная голова на ОСТАТКЕ (метка − приор), обновляется ОНЛАЙН
                 на каждом свайпе за O(d²) (d≈40 -> микросекунды).

Байес даёт три вещи:
  1. **Cold-start безопасен**: при 0 свайпов апостериор сидит в нуле -> score = prior.
  2. **Усадка автоматическая**: не нужен ad-hoc λ(n) — её роль играет prior_precision.
  3. **Неопределённость** -> active learning (Thompson / uncertainty sampling).

⚠️ КРИТИЧНО: вход головы обязан быть WHITENED (PCA(whiten=True) поверх StandardScaler).
Без этого компоненты имеют разные дисперсии, усадка действует неравномерно (по «сильным» осям
её фактически нет), и холодный старт ПРОВАЛИВАЕТСЯ ниже приора на первых десятках свайпов —
измерено: pairs-accuracy падала 0.688 -> 0.535. С whitening усадка = n/(n+λ) по всем осям и
провал исчезает (0.688 -> 0.687 на 25 свайпах). Рабочая точка: dim=40, prior_precision≈300.
"""

from __future__ import annotations

import numpy as np


class BayesianLinearHead:
    """Онлайн байесовская линейная регрессия на остатке. Сопряжённая, без градиентов."""

    def __init__(self, dim: int, prior_precision: float = 1.0, noise_var: float = 1.0):
        self.dim = dim
        self.noise = float(noise_var)
        self.lam = np.eye(dim) * float(prior_precision)  # precision (Λ)
        self.b = np.zeros(dim)                            # Λ·μ
        self.n = 0

    @property
    def mean(self) -> np.ndarray:
        return np.linalg.solve(self.lam, self.b)

    @property
    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.lam)

    def update(self, x: np.ndarray, residual: float, weight: float = 1.0) -> None:
        """Один свайп/оценка: rank-1 апдейт апостериора.

        ``weight`` — вес наблюдения. Для IPS передавайте 1/propensity (с клиппингом):
        так снимается смещение экспозиции, когда показ был не случайным.
        """
        w = float(weight)
        self.lam += w * np.outer(x, x) / self.noise
        self.b += w * x * float(residual) / self.noise
        self.n += 1

    def update_batch(self, X: np.ndarray, r: np.ndarray) -> None:
        self.lam += X.T @ X / self.noise
        self.b += X.T @ r / self.noise
        self.n += len(r)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (мат.ожидание остатка, предиктивная дисперсия)."""
        cov = self.cov
        mu = X @ self.mean
        var = np.einsum("ij,jk,ik->i", X, cov, X) + self.noise
        return mu, var

    def thompson(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Сэмпл из апостериора -> скор (exploration/exploitation «из коробки»)."""
        w = rng.multivariate_normal(self.mean, self.cov, method="eigh")
        return X @ w


# ---------------------------------------------------------------- политики показа


def pick_random(cand: np.ndarray, rng: np.random.Generator, **_) -> int:
    return int(rng.choice(cand))


def pick_uncertainty(cand: np.ndarray, rng: np.random.Generator, head=None, Z=None, **_) -> int:
    """Active learning: показываем то, в чём голова МЕНЕЕ всего уверена -> быстрее учится."""
    _, var = head.predict(Z[cand])
    return int(cand[int(np.argmax(var))])


def pick_thompson(cand: np.ndarray, rng: np.random.Generator, head=None, Z=None, prior=None, **_) -> int:
    """Баланс: показываем привлекательных, но с исследованием."""
    s = prior[cand] + head.thompson(Z[cand], rng)
    return int(cand[int(np.argmax(s))])


def pick_exploit(cand: np.ndarray, rng: np.random.Generator, head=None, Z=None, prior=None, **_) -> int:
    """ЖАДНАЯ политика — только лучшие по текущему скору. Демонстрирует смещение экспозиции:
    голова видит лишь верхушку -> учится на усечённом диапазоне и деградирует."""
    mu, _ = head.predict(Z[cand])
    return int(cand[int(np.argmax(prior[cand] + mu))])


POLICIES = {"random": pick_random, "uncertainty": pick_uncertainty,
            "thompson": pick_thompson, "exploit": pick_exploit}
