"""Реестр обучаемых face-бэкбонов для мультибэкбонового честного эксперимента.

Все бэкбоны имеют единый интерфейс: ``forward`` (L2-нормированный эмбеддинг),
``preprocess(img_bgr, bgr=True)`` (свой размер/нормировка/выравнивание), ``trainable_scopes``
(head/tail для дообучения) и ``input_size``. Это позволяет одним протоколом
(frozen → дообучение на наших парах → внешние бенчмарки) сравнивать backbone разной силы.
"""

from __future__ import annotations

from torch import nn

# Каноничные имена backbone -> (роль, loss, обуч. данные) для таблиц/документации.
# Покрывают разные loss (softmax/ArcFace/AdaFace), архитектуры и силу — для §1d/§1f.
BACKBONES: dict[str, str] = {
    "facenet": "InceptionResnetV1 / softmax / CASIA-WebFace (слабый)",
    "arcface_r50_casia": "iResNet50 / ArcFace / CASIA-FaceV5 (слабый, др. арх.)",
    "adaface_ir50": "IR-50 / AdaFace / WebFace4M (сильный, депт-контроль к r100)",
    "arcface_r100": "iResNet100 / ArcFace / large-scale (сильный)",
    "adaface_ir101": "IR-101 / AdaFace / WebFace12M (сильный, др. loss)",
}


def make_backbone(name: str = "facenet", pretrained: bool = True) -> nn.Module:
    """Создать backbone по имени из ``BACKBONES``."""
    if name == "facenet":
        from age_gap.models.facenet import FaceNetBackbone

        return FaceNetBackbone(pretrained="casia-webface" if pretrained else None)
    if name.startswith("arcface"):
        from age_gap.models.iresnet import ArcFaceBackbone

        return ArcFaceBackbone(name, pretrained=pretrained)
    if name.startswith("adaface"):
        from age_gap.models.adaface_ir import AdaFaceBackbone

        return AdaFaceBackbone(name, pretrained=pretrained)
    raise ValueError(f"неизвестный backbone: {name!r} (доступно: {list(BACKBONES)})")
