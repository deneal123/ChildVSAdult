"""Выбор устройства вычислений (CPU/GPU) для ONNX Runtime и torch.

insightface (детекция, ArcFace) работает через ONNX Runtime, а не torch, поэтому для GPU
нужен пакет ``onnxruntime-gpu`` и доступный ``CUDAExecutionProvider``. torch (adapter)
использует свой CUDA отдельно. Режим берётся из settings ([default.compute].device):
"auto" (по умолчанию) / "cpu" / "cuda".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from age_gap.common.logging import get_logger
from age_gap.settings import settings

log = get_logger(__name__)

_CUDA_RUNTIME_READY = False


def _configured_device() -> str:
    compute = getattr(settings, "compute", None)
    return str(getattr(compute, "device", "auto")).lower() if compute else "auto"


def _prepare_cuda_runtime() -> None:
    """Сделать CUDA/cuDNN-DLL из nvidia-*-cu12 wheels видимыми для onnxruntime (Windows).

    onnxruntime-gpu ищет cublas/cuDNN рядом; pip-wheels кладут их в
    site-packages/nvidia/*/bin. Добавляем эти каталоги в пути поиска DLL и подгружаем
    зависимые библиотеки через onnxruntime.preload_dlls.
    """
    global _CUDA_RUNTIME_READY
    if _CUDA_RUNTIME_READY:
        return

    for sp in map(Path, sys.path):
        nvidia = sp / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            if bin_dir.is_dir():
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(bin_dir))  # type: ignore[attr-defined]
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("preload_dlls не удался: %s", exc)

    _CUDA_RUNTIME_READY = True


# Опции CUDA-провайдера: EXHAUSTIVE-поиск алгоритмов свёртки (дефолт ORT) вызывал
# CUDNN_STATUS_INTERNAL_ERROR на части кадров — переходим на DEFAULT (стабильно и быстро).
# max_workspace оставляем включённым (1): без него свёртки уходят в медленный fallback.
_CUDA_PROVIDER_OPTIONS = {
    "cudnn_conv_algo_search": "DEFAULT",
    "cudnn_conv_use_max_workspace": "1",
}


def onnx_providers(prefer: str | None = None) -> list:
    """ONNX-провайдеры: CUDA (со стабильными опциями cuDNN) первым, если доступен."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    mode = (prefer or _configured_device()).lower()
    if mode != "cpu" and "CUDAExecutionProvider" in available:
        _prepare_cuda_runtime()  # сделать CUDA/cuDNN-DLL видимыми
        return [("CUDAExecutionProvider", _CUDA_PROVIDER_OPTIONS), "CPUExecutionProvider"]
    if mode == "cuda" and "CUDAExecutionProvider" not in available:
        log.warning(
            "device=cuda запрошен, но CUDAExecutionProvider недоступен "
            "(нужен onnxruntime-gpu + совместимый CUDA). Откат на CPU."
        )
    return ["CPUExecutionProvider"]


def _provider_names(providers: list) -> list[str]:
    return [p[0] if isinstance(p, tuple) else p for p in providers]


def onnx_ctx_id(prefer: str | None = None) -> int:
    """ctx_id для insightface: 0 (GPU) если CUDA-провайдер доступен, иначе -1 (CPU)."""
    return 0 if "CUDAExecutionProvider" in _provider_names(onnx_providers(prefer)) else -1


def torch_device(prefer: str | None = None) -> str:
    """Устройство для torch: 'cuda' если доступно и не запрещено, иначе 'cpu'."""
    import torch

    mode = (prefer or _configured_device()).lower()
    if mode != "cpu" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
