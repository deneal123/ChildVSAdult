"""Create a checksum-pinned inventory of every pretrained/LLM pipeline dependency."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from age_gap.common.io import data_path, resolve_path
from age_gap.common.manifest import file_record, write_experiment_manifest
from age_gap.datasets.llm_age_extractor import SYSTEM_PROMPT
from age_gap.settings import settings


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def main() -> None:
    user = Path.home()
    weights = {
        "insightface_retinaface": {
            "path": user / ".insightface/models/buffalo_l/det_10g.onnx",
            "source": "InsightFace buffalo_l model pack",
            "role": "detection and five-point landmarks",
            "declared_training_data": "not declared in the distributed artifact",
        },
        "insightface_arcface_r50": {
            "path": user / ".insightface/models/buffalo_l/w600k_r50.onnx",
            "source": "InsightFace buffalo_l model pack",
            "role": "clustering and near-duplicate detection only",
            "declared_training_data": "WebFace600K (model filename/InsightFace model-zoo designation)",
        },
        "insightface_genderage": {
            "path": user / ".insightface/models/buffalo_l/genderage.onnx",
            "source": "InsightFace buffalo_l model pack",
            "role": "apparent gender/age metadata",
            "declared_training_data": "not declared in the distributed artifact",
        },
        "facenet_casia": {
            "path": user / ".cache/torch/checkpoints/20180408-102900-casia-webface.pt",
            "source": "facenet-pytorch InceptionResnetV1 pretrained=casia-webface",
            "role": "weak trainable backbone",
            "declared_training_data": "CASIA-WebFace",
        },
        "arcface_r50_casia": {
            "path": Path(str(data_path("models_dir", "arcface_r50_casia.pth"))),
            "source": "JustinLeee/FaceMind_ArcFace_iResNet50_CASIA_FaceV5",
            "role": "weak architecture control",
            "declared_training_data": "CASIA-FaceV5",
        },
        "arcface_r100": {
            "path": Path(str(data_path("models_dir", "arcface_r100.pth"))),
            "source": "marcelohaps/arcface-torch backbone.pth",
            "role": "strong backbone",
            "declared_training_data": "not declared in the downloaded checkpoint artifact",
        },
        "adaface_ir50": {
            "path": Path(str(data_path("models_dir", "adaface_ir50.pt"))),
            "source": "minchul/cvlface_adaface_ir50_webface4m",
            "role": "strong depth control",
            "declared_training_data": "WebFace4M",
        },
        "adaface_ir101": {
            "path": Path(str(data_path("models_dir", "adaface_ir101.pt"))),
            "source": "minchul/cvlface_adaface_ir101_webface12m",
            "role": "strong mechanism-study backbone",
            "declared_training_data": "WebFace12M",
        },
    }
    missing = [name for name, item in weights.items() if not item["path"].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model weights: {missing}")
    model_rows = {}
    input_paths: list[Path] = []
    for name, item in weights.items():
        path = item.pop("path")
        input_paths.append(path)
        model_rows[name] = {**item, "artifact": file_record(path)}

    cache = Path(str(data_path("data_dir", "interim", "llm_age_cache.jsonl")))
    prompt_file = Path(resolve_path("src", "age_gap", "datasets", "llm_age_extractor.py"))
    settings_file = Path(resolve_path("src", "age_gap", "settings", "settings.toml"))
    input_paths.extend([prompt_file, settings_file])
    llm = {
        "provider": "GigaChat",
        "configured_model": str(settings.gigachat.model_name),
        "system_prompt_sha256": sha256_file_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "prompt_source": file_record(prompt_file),
        "settings_source": file_record(settings_file),
        "cached_outputs": file_record(cache) if cache.exists() else None,
        "decoding_parameters": {
            "temperature": getattr(settings.gigachat.llm_params, "temperature", None),
            "top_p": getattr(settings.gigachat.llm_params, "top_p", None),
            "max_gen_tokens": getattr(settings.gigachat.llm_params, "max_gen_tokens", None),
        },
    }
    if cache.exists():
        input_paths.append(cache)

    payload = {
        "packages": {
            name: _version(name)
            for name in [
                "insightface",
                "onnxruntime-gpu",
                "facenet-pytorch",
                "torch",
                "opencv-python-headless",
                "gigachat",
            ]
        },
        "models": model_rows,
        "llm": llm,
        "pretraining_overlap_audit": {
            "status": "not directly testable from distributed checkpoints",
            "reason": (
                "The downloaded checkpoints do not include complete pretraining identity/image "
                "manifests, and CACD-VS distributes anonymized pair filenames. Dataset names are "
                "reported above; exact identity- or image-level intersection cannot be verified "
                "without upstream manifests. This is retained as a dependency limitation."
            ),
        },
    }
    output = Path(str(data_path("metrics_dir", "model_inventory.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="model-and-llm-dependency-inventory",
        parameters={"includes_weight_checksums": True, "includes_secrets": False},
        metrics={"models": len(model_rows), "cached_llm_outputs": cache.exists()},
        inputs=input_paths,
        outputs=[output],
    )
    print(f"OK: {output} ({len(model_rows)} model artifacts)")


def sha256_file_bytes(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


if __name__ == "__main__":
    main()
