"""Преобразование «старения» для synthetic-ageing baseline (docs/deep-research.md).

Интерфейс ``AgingTransform`` — pluggable. Доступны две реализации:

* ``HeuristicAging`` — ПРОКСИ, не настоящее старение лица: имитирует доменный сдвиг
  (качество/эра снимка: гамма, насыщенность, размытие, шум). Нужен лишь чтобы прогнать
  harness end-to-end, без зависимостей и весов.
* ``FRANAging`` — РЕАЛЬНАЯ генеративная aging-модель: Face Re-Aging Network (U-Net, реплика
  Disney «Production-Ready Face Re-Aging», репо timroelofs123/face_reaging, MIT). Делает
  настоящее биологическое старение с контролем целевого возраста. Веса (~124 МБ) скачиваются
  с HuggingFace при первом обращении. Это убирает оговорку «прокси» в synthetic-baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch


@runtime_checkable
class AgingTransform(Protocol):
    def __call__(self, img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


class HeuristicAging:
    """ПРОКСИ старения: доменная пертурбация (НЕ реальное биологическое старение)."""

    def __init__(self, strength: float = 1.0) -> None:
        self.strength = float(strength)

    def __call__(self, img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        s = self.strength
        out = img_bgr.astype(np.float32)

        # Гамма/яркость (старые снимки — иной тон).
        gamma = 1.0 + rng.uniform(-0.25, 0.25) * s
        out = (255.0 * np.power(np.clip(out / 255.0, 0, 1), gamma)).astype(np.float32)

        # Снижение насыщенности (выцветание).
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(
            np.float32
        )
        hsv[..., 1] *= 1.0 - 0.3 * s * rng.uniform(0.5, 1.0)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(
            np.float32
        )

        # Лёгкое размытие + шум (потеря качества).
        if rng.uniform() < 0.7:
            k = 3
            out = cv2.GaussianBlur(out, (k, k), sigmaX=0.6 * s).astype(np.float32)
        out += rng.normal(0, 3.0 * s, out.shape).astype(np.float32)

        return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------------
# FRAN: реальная aging-модель (U-Net). Архитектура — точная реплика из репозитория
# timroelofs123/face_reaging (MIT), повторяющего Disney «Production-Ready Face Re-Aging».
# Вход 5 каналов: RGB + source_age/100 + target_age/100; выход — delta, добавляемая к RGB.
# --------------------------------------------------------------------------------------------

FRAN_REPO_ID = "timroelofs123/face_re-aging"
FRAN_WEIGHTS_FILE = "best_unet_model.pth"


def _build_fran_unet() -> torch.nn.Module:
    """Собрать U-Net FRAN (требует torch + antialiased_cnns)."""
    import antialiased_cnns
    import torch
    import torch.nn as nn

    class DownLayer(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.layer = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=1),
                antialiased_cnns.BlurPool(in_ch, stride=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer(x)

    class UpLayer(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.blur_upsample = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0),
                antialiased_cnns.BlurPool(out_ch, stride=1),
            )
            self.layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            x = self.blur_upsample(x)
            return self.layer(torch.cat([x, skip], dim=1))

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.init_conv = nn.Sequential(
                nn.Conv2d(5, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
            )
            self.down1 = DownLayer(64, 128)
            self.down2 = DownLayer(128, 256)
            self.down3 = DownLayer(256, 512)
            self.down4 = DownLayer(512, 1024)
            self.up1 = UpLayer(1024, 512)
            self.up2 = UpLayer(512, 256)
            self.up3 = UpLayer(256, 128)
            self.up4 = UpLayer(128, 64)
            self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x0 = self.init_conv(x)
            x1 = self.down1(x0)
            x2 = self.down2(x1)
            x3 = self.down3(x2)
            x4 = self.down4(x3)
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.up4(x, x0)
            return self.final_conv(x)

    return UNet()


def fran_weights_path(ckpt: Path | str | None = None) -> Path:
    """Локальный путь к весам FRAN; при отсутствии — скачать с HuggingFace (~124 МБ)."""
    from age_gap.common.io import data_path, resolve_path

    if ckpt is not None:
        return resolve_path(str(ckpt))
    local = Path(resolve_path(str(data_path("models_dir", "fran_unet.pth"))))
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=FRAN_REPO_ID, filename=FRAN_WEIGHTS_FILE)
    local.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy(downloaded, local)
    return local


class FRANAging:
    """РЕАЛЬНОЕ старение лица (FRAN U-Net) с контролем целевого возраста.

    На каждом вызове выбирает источник/цель так, чтобы получить КРОСС-ВОЗРАСТНУЮ пару с
    большим разрывом (именно такие пары важны для cross-age бенчмарков): по умолчанию старит
    «вверх» (ребёнок/молодой → пожилой); с вероятностью ``p_young`` омолаживает «вниз».

    Наши кропы уже выровнены (norm_crop), поэтому скользящее окно/детекция из оригинального
    репо не нужны: вход ресайзим до ``input_size``, прогоняем одним окном, delta добавляем к
    исходнику, ресайзим обратно к размеру кропа.
    """

    def __init__(
        self,
        ckpt: Path | str | None = None,
        device: str | None = None,
        input_size: int = 512,
        source_age_range: tuple[int, int] = (8, 22),
        target_age_range: tuple[int, int] = (55, 78),
        p_young: float = 0.25,
        weights_only: bool = True,
        real_gaps: list[int] | None = None,
    ) -> None:
        import torch

        from age_gap.common.device import torch_device

        self.device = device or torch_device()
        self.input_size = int(input_size)
        self.source_age_range = source_age_range
        self.target_age_range = target_age_range
        self.p_young = float(p_young)
        # Parity mode: when given the real training pairs' age-gaps, synthetic pairs are aged
        # to the SAME gap distribution (not a fixed wide 8-22 -> 55-78), so the real-vs-synthetic
        # comparison is not confounded by a gap mismatch -- the reviewer's parity request.
        self.real_gaps = list(real_gaps) if real_gaps else None

        net = _build_fran_unet()
        state = torch.load(
            str(fran_weights_path(ckpt)), map_location="cpu", weights_only=weights_only
        )
        net.load_state_dict(state)
        self.net = net.to(self.device).eval()

    def _ages(self, rng: np.random.Generator) -> tuple[float, float]:
        if self.real_gaps is not None:  # parity: gap drawn from the real-pair distribution
            slo, shi = self.source_age_range
            gap = float(self.real_gaps[int(rng.integers(len(self.real_gaps)))])
            src = float(rng.integers(slo, shi + 1))
            tgt = min(src + gap, 90.0)
            if rng.uniform() < self.p_young:  # омоложение: меняем направление
                src, tgt = tgt, src
            return src, tgt
        slo, shi = self.source_age_range
        tlo, thi = self.target_age_range
        if rng.uniform() < self.p_young:  # омоложение: меняем роли диапазонов
            src = float(rng.integers(tlo, thi + 1))
            tgt = float(rng.integers(slo, shi + 1))
        else:  # старение
            src = float(rng.integers(slo, shi + 1))
            tgt = float(rng.integers(tlo, thi + 1))
        return src, tgt

    def __call__(self, img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import torch

        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rs = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rs).permute(2, 0, 1)
        src, tgt = self._ages(rng)
        src_ch = torch.full_like(t[:1], src / 100.0)
        tgt_ch = torch.full_like(t[:1], tgt / 100.0)
        inp = torch.cat([t, src_ch, tgt_ch], dim=0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            delta = self.net(inp).squeeze(0)
        out = (t.to(self.device) + delta).clamp(0, 1)
        out_np = (out.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        out_np = cv2.resize(out_np, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


def make_aging(kind: str = "proxy", **kwargs: object) -> AgingTransform:
    """Фабрика aging-преобразования: ``proxy`` (HeuristicAging) | ``fran`` (FRANAging)."""
    if kind == "proxy":
        return HeuristicAging(strength=float(kwargs.get("strength", 1.0)))  # type: ignore[arg-type]
    if kind == "fran":
        return FRANAging()
    raise ValueError(f"неизвестный aging kind: {kind!r} (ожидалось proxy|fran)")
