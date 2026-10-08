"""One real recognition+FAS smoke, not trained-system scientific evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.build_mtlface_recognition_v2 import build_model
from scripts.mtlface_epoch_v2 import load_bound_dataset
from scripts.mtlface_fas_networks_v2 import AgingModule, PatchDiscriminator, provenance
from scripts.mtlface_fas_training_v2 import fas_step
from scripts.mtlface_joint_adapter_v2 import CommonJointAdapter
from scripts.mtlface_training_v2 import recognition_step, recognition_weights_snapshot
from scripts.run_mtlface_campaign_v2 import collect_sources, load_prerequisites


def run_joint_smoke(model, head, dataset, generator, discriminator):
    explicit = next((i for i, row in enumerate(dataset.records) if row.age is not None), None)
    missing = next((i for i, row in enumerate(dataset.records) if row.age is None), None)
    if explicit is None or missing is None:
        raise ValueError("explicit and missing age smoke rows required")
    indices = [explicit, missing]
    items = [dataset[index] for index in indices]
    images = torch.stack([item[0] for item in items])
    identities = torch.tensor([item[1] for item in items], dtype=torch.int64)
    ages = torch.tensor([item[2] for item in items], dtype=torch.int64)
    target_group = sum(
        dataset.records[explicit].age > boundary for boundary in (10, 20, 30, 40, 50, 60)
    )
    groups = torch.full((2,), target_group, dtype=torch.int64)
    targets = images[:1].repeat(2, 1, 1, 1)
    model.set_batchnorm_policy()
    before_fr = recognition_weights_snapshot(model, head)
    optimizer = torch.optim.SGD(
        [p for m in (model, head) for p in m.parameters() if p.requires_grad], lr=1e-5
    )
    print("joint smoke: recognition step", flush=True)
    fr = recognition_step(model, head, optimizer, (images, identities, ages))
    after_fr = recognition_weights_snapshot(model, head)
    updated_fr = sum(
        not torch.equal(value, before_fr["model"][name]) for name, value in model.named_parameters()
    )
    updated_head = sum(
        not torch.equal(value, before_fr["identity_head"][name])
        for name, value in head.named_parameters()
    )
    for name, value in model.named_buffers():
        if not torch.equal(value, before_fr["model"][name]):
            raise ValueError("recognition frozen BN/buffer changed")
    if not updated_fr or not updated_head:
        raise ValueError("recognition model/head update required")
    del before_fr
    before_g = generator.conv3.weight.detach().clone()
    before_d = discriminator.conv1.weight.detach().clone()
    # Smoke policy follows pinned-code Adam beta1, not the paper's beta1.
    g_opt = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.99))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.99))
    print("joint smoke: FAS D/G step", flush=True)
    fas = fas_step(
        model,
        generator,
        discriminator,
        g_opt,
        d_opt,
        images,
        targets,
        groups,
        generator_bn_policy="adapt",
    )
    for name, value in model.state_dict().items():
        if not torch.equal(value, after_fr["model"][name]):
            raise ValueError("recognizer state changed during FAS")
    for name, value in head.state_dict().items():
        if not torch.equal(value, after_fr["identity_head"][name]):
            raise ValueError("identity head changed during FAS")
    if torch.equal(generator.conv3.weight, before_g) or torch.equal(
        discriminator.conv1.weight, before_d
    ):
        raise ValueError("actual G/D parameter update required")
    model.eval()
    with torch.no_grad():
        embeddings = model(images)
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError("nonfinite post-joint embedding")
    return dict(
        recognition=fr,
        fas=fas,
        updated_recognition_parameter_tensors=updated_fr,
        updated_classifier_parameter_tensors=updated_head,
        recognition_buffers_frozen=True,
        recognizer_state_unchanged_during_fas=True,
        identity_head_unchanged_during_fas=True,
        embeddings_finite=True,
        embedding_shape=list(embeddings.shape),
        target_group=target_group,
        joint_steps_complete=True,
        training_complete=False,
        scientific_evaluation_complete=False,
        full_method_parity=False,
        publication_ready=False,
    ), indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("faces", "initialization", "smoke", "weights", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    init, faces, smoke, contract = load_prerequisites(
        initialization_path=args.initialization,
        faces_path=args.faces,
        smoke_path=args.smoke,
        weights=args.weights,
        scope="head",
        device="cpu",
    )
    sources = collect_sources(
        initialization_path=args.initialization,
        faces_path=args.faces,
        smoke_path=args.smoke,
        weights=args.weights,
        initialization=init,
        faces=faces,
        smoke=smoke,
        extra_sources=[
            Path(__file__),
            *[
                PROJECT_ROOT / ("scripts/" + name + ".py")
                for name in (
                    "mtlface_fas_networks_v2",
                    "mtlface_fas_training_v2",
                    "mtlface_joint_adapter_v2",
                )
            ],
        ],
    )
    before = [file_record(path) for path in sources]
    base, head, metadata = build_model(
        args.weights, init["metrics"]["identity_classes"], seed=42, scope="head"
    )
    if metadata != contract:
        raise ValueError("initialization metadata mismatch")
    model = CommonJointAdapter(base.backbone, base.separation, base.age_head, base.age_adversary)
    del base
    dataset, native = load_bound_dataset(args.faces, model.preprocess)
    if native != faces or dataset.n_classes != head.classes:
        raise ValueError("dataset contract mismatch")
    generator, discriminator = AgingModule(), PatchDiscriminator()
    metrics, indices = run_joint_smoke(model, head, dataset, generator, discriminator)
    metrics["real_crop_joint_smoke_complete"] = True
    if before != [file_record(path) for path in sources]:
        raise RuntimeError("joint smoke sources changed")
    args.out.mkdir(parents=True)
    private = args.out / "private"
    private.mkdir()
    selection = private / "selection.json"
    selection.write_text(json.dumps(dict(dataset_indices=indices)) + "\n", encoding="utf-8")
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest = args.out / "summary.manifest.json"
    write_experiment_manifest(
        manifest,
        experiment="mtlface-common-real-crop-joint-smoke",
        parameters=dict(
            seed=42,
            scope="head",
            device="cpu",
            threads=1,
            selection="first explicit and first missing age; first explicit target repeated twice",
            selection_not_uniform_training_stream=True,
            recognition_optimizer="SGD",
            recognition_lr=1e-5,
            fas_optimizer="Adam",
            fas_lr=1e-4,
            fas_betas=[0.5, 0.99],
            generator_bn_policy="adapt",
            recognition_bn_policy="frozen_all",
            batch_size=2,
            recognition_steps=1,
            fas_steps=1,
            model_discarded=True,
            architecture_reference=provenance(),
        ),
        metrics=metrics,
        inputs=sources,
        outputs=[summary, selection],
    )
    if before != [file_record(path) for path in sources]:
        manifest.unlink(missing_ok=True)
        raise RuntimeError("sources changed during joint manifest write")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
