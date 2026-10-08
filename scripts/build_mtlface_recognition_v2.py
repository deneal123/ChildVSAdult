"""Strict local common-backbone assembly; no joint/FAS parity claim or download."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from age_gap.common.manifest import file_record
from scripts.mtlface_components_v2 import MapAgeHead, SpatialAgeIdentitySplit
from scripts.mtlface_recognition_v2 import SpatialRecognitionAdapter
from scripts.mtlface_training_v2 import CosFaceHead


def clean_backbone_state(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("tensor state dictionary required")
    state, skipped = {}, []
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("string state keys required")
        normalized = key
        while normalized.startswith(("module.", "arcface.", "backbone.")):
            normalized = normalized.split(".", 1)[1]
        if normalized.startswith(("head.", "logits.", "fc_head.")):
            skipped.append(key)
            continue
        if normalized in state:
            raise ValueError("state key collision after prefix normalization")
        if not isinstance(value, torch.Tensor):
            raise ValueError("backbone tensor values required")
        state[normalized] = value
    return state, sorted(skipped)


def strict_load(module, payload):
    state, skipped = clean_backbone_state(payload)
    expected = module.state_dict()
    if set(state) != set(expected):
        raise ValueError("exact backbone state keys required; partial loading refused")
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError("exact backbone state shape/dtype required")
        if not torch.isfinite(value).all():
            raise ValueError("finite backbone state required")
    module.load_state_dict(state, strict=True)
    return dict(
        backbone_tensors=len(state),
        skipped_classifier_tensors=len(skipped),
        missing_keys=0,
        unexpected_backbone_keys=0,
    )


def build_model(weights, classes, *, seed, scope="head"):
    if scope not in ("head", "tail", "full"):
        raise ValueError("head/tail/full scope required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("nonnegative integer seed required")
    from age_gap.models.iresnet import ArcFaceBackbone

    weights = Path(weights)
    before = file_record(weights)
    torch.manual_seed(seed)
    backbone = ArcFaceBackbone("arcface_r50_casia", pretrained=False)
    initialization = strict_load(
        backbone.net, torch.load(weights, map_location="cpu", weights_only=True)
    )
    if before != file_record(weights):
        raise RuntimeError("initial weights changed during loading")
    if scope != "full":
        for name, parameter in backbone.net.named_parameters():
            parameter.requires_grad_(
                any(name.startswith(prefix + ".") for prefix in backbone.trainable_scopes[scope])
            )
    # Full scope preserves the backbone constructor's intrinsic frozen affine
    # flags; head/tail follow its declared scope map. This is a common adaptation.
    channels = backbone.net.bn2.num_features
    spatial_size = math.isqrt(backbone.net.fc.in_features // channels)
    if channels * spatial_size**2 != backbone.net.fc.in_features:
        raise ValueError("square spatial output contract required")
    dimension = backbone.net.fc.out_features
    model = SpatialRecognitionAdapter(
        backbone,
        SpatialAgeIdentitySplit(channels),
        MapAgeHead(channels, spatial_size),
        MapAgeHead(channels, spatial_size),
    )
    identity_head = CosFaceHead(dimension, classes)
    model.set_batchnorm_policy()
    return (
        model,
        identity_head,
        dict(
            weights=before,
            initialization=initialization,
            seed=seed,
            backbone="arcface_r50_casia",
            scope=scope,
            channels=channels,
            spatial_size=spatial_size,
            embedding_dimension=dimension,
            cosface_scale=identity_head.scale,
            cosface_margin=identity_head.margin,
            batchnorm_policy="frozen_all",
            full_joint_fas=False,
        ),
    )
