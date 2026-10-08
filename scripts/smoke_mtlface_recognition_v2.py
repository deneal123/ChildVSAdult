"""One real-crop CPU recognition step; not a scientific result or full FAS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.build_mtlface_recognition_v2 import build_model
from scripts.mtlface_epoch_v2 import load_bound_dataset
from scripts.mtlface_training_v2 import recognition_step, recognition_weights_snapshot


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def verified(path, experiment):
    native = json.loads(Path(path).read_text(encoding="utf-8"))
    if native.get("experiment") != experiment:
        raise ValueError("wrong prerequisite type")
    for record in native["inputs"] + native["outputs"]:
        if file_record(resolve(record["path"])) != record:
            raise ValueError("prerequisite hash mismatch")
    return native


def smoke_step(model, head, dataset):
    # Selection is metadata-only and declared before outcome inspection.
    explicit = next((i for i, row in enumerate(dataset.records) if row.age is not None), None)
    missing = next((i for i, row in enumerate(dataset.records) if row.age is None), None)
    if explicit is None or missing is None:
        raise ValueError("both explicit and missing ages required for this smoke")
    indices = [explicit, missing]
    items = [dataset[index] for index in indices]
    batch = (
        torch.stack([r[0] for r in items]),
        torch.tensor([r[1] for r in items]),
        torch.tensor([r[2] for r in items]),
    )
    before = recognition_weights_snapshot(model, head)
    optimizer = torch.optim.SGD(
        [p for module in (model, head) for p in module.parameters() if p.requires_grad], lr=1e-5
    )
    model.eval()
    with torch.no_grad():
        embeddings = model(batch[0])
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("nonfinite real-crop embedding")
    result = recognition_step(model, head, optimizer, batch)
    for name, value in model.named_buffers():
        if not torch.equal(value, before["model"][name]):
            raise ValueError("frozen model buffer changed")
    updated = {}
    for name, module in (
        ("backbone", model.backbone),
        ("separation", model.separation),
        ("age_head", model.age_head),
        ("age_adversary", model.age_adversary),
    ):
        updated[name] = sum(
            not torch.equal(value, before["model"][name + "." + key])
            for key, value in module.named_parameters()
        )
        for key, value in module.named_parameters():
            if not value.requires_grad and not torch.equal(
                value, before["model"][name + "." + key]
            ):
                raise ValueError("frozen parameter changed")
    updated["identity_head"] = sum(
        not torch.equal(value, before["identity_head"][key])
        for key, value in head.named_parameters()
    )
    if not all(updated.values()):
        raise ValueError("joint update did not reach all required modules")
    model.eval()
    with torch.no_grad():
        after_embeddings = model(batch[0])
    if not torch.isfinite(after_embeddings).all():
        raise FloatingPointError("nonfinite post-update embedding")
    return dict(
        batch_size=2,
        supervised_age_rows=1,
        missing_age_rows=1,
        step=result,
        updated_parameter_tensors=updated,
        frozen_buffers_unchanged=True,
        frozen_parameters_unchanged=True,
        embeddings_finite=True,
        embedding_shape=list(embeddings.shape),
    ), indices


def main():
    parser = argparse.ArgumentParser()
    for name in ("faces", "initialization", "weights", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    initialization = verified(args.initialization, "mtlface-common-recognition-initialization")
    if initialization["metrics"].get("initialization_complete") is not True:
        raise ValueError("completed initialization prerequisite required")
    classes = initialization["metrics"]["identity_classes"]
    model, head, metadata = build_model(args.weights, classes, seed=42, scope="head")
    if metadata != initialization["metrics"]["initialization"]:
        raise ValueError("initialization contract mismatch")
    dataset, faces = load_bound_dataset(args.faces, model.preprocess)
    if dataset.n_classes != classes:
        raise ValueError("face-list classifier dimension mismatch")
    sources = list(
        dict.fromkeys(
            resolve(r["path"]).resolve()
            for native in (initialization, faces)
            for r in native["inputs"] + native["outputs"]
        )
    )
    sources += [
        args.faces.resolve(),
        args.initialization.resolve(),
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts/mtlface_epoch_v2.py",
        PROJECT_ROOT / "scripts/prepare_sota_face_list.py",
    ]
    sources = list(dict.fromkeys(sources))
    before = [file_record(p) for p in sources]
    metrics, indices = smoke_step(model, head, dataset)
    metrics |= dict(
        real_crop_smoke_complete=True,
        training_complete=False,
        scientific_evaluation_complete=False,
        full_joint_fas=False,
        publication_ready=False,
    )
    if before != [file_record(p) for p in sources]:
        raise RuntimeError("smoke inputs changed")
    args.out.mkdir(parents=True)
    private = args.out / "private"
    private.mkdir()
    selection = private / "selection.json"
    selection.write_text(json.dumps(dict(dataset_indices=indices)) + "\n", encoding="utf-8")
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        args.out / "summary.manifest.json",
        experiment="mtlface-common-recognition-real-crop-smoke",
        parameters=dict(
            seed=42,
            scope="head",
            device="cpu",
            threads=1,
            optimizer="SGD",
            learning_rate=1e-5,
            steps=1,
            selection="first explicit and first missing age",
            batchnorm_policy="frozen_all",
            model_discarded=True,
        ),
        metrics=metrics,
        inputs=sources,
        outputs=[summary, selection],
    )
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
