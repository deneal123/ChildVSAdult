"""Serial, fixed-last joint MTLFace adaptation; not published-system parity.

Native CLI binds recognition initialization, real joint smoke and target metadata
preflight. No downloads, checkpoint selection by outcomes or implicit resume.
Preflight never constructs neural models. Synthetic injections are labelled.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.build_mtlface_recognition_v2 import build_model
from scripts.mtlface_epoch_v2 import load_bound_dataset
from scripts.mtlface_fas_networks_v2 import AgingModule, PatchDiscriminator, provenance
from scripts.mtlface_joint_adapter_v2 import CommonJointAdapter
from scripts.mtlface_joint_epoch_v3 import JointEpochFailure, run_joint_epoch
from scripts.mtlface_target_stream_v3 import TargetAgeStream
from scripts.run_mtlface_campaign_v2 import (
    _check_seeded_contract,
    _require_finite,
    collect_sources,
    load_prerequisites,
    resolve,
    verified,
)


@dataclass(frozen=True)
class CampaignConfig:
    epochs: int
    batch_size: int
    fr_lr: float
    g_lr: float
    d_lr: float
    generator_bn_policy: str
    seeds: tuple[int, ...] = (42, 1, 2)
    scope: str = "head"
    betas: tuple[float, float] = (0.5, 0.99)

    def validate(self):
        for value in (self.epochs, self.batch_size):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("positive integer epoch/batch budget required")
        for value in (self.fr_lr, self.g_lr, self.d_lr):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("finite positive learning rates required")
        if self.scope not in ("head", "tail", "full") or self.generator_bn_policy not in ("adapt", "frozen"):
            raise ValueError("explicit scope and generator BN policy required")
        if (
            not self.seeds or len(set(self.seeds)) != len(self.seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**63 - 10000 for s in self.seeds)
        ):
            raise ValueError("unique nonnegative bounded seeds required")
        if len(self.betas) != 2 or any(
            isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b < 1
            for b in self.betas
        ):
            raise ValueError("explicit Adam betas in [0,1) required")


def flags():
    return dict(full_method_parity=False, common_budget_matched=False,
                scientific_evaluation_complete=False, publication_ready=False)


def default_models(weights, classes, *, seed, scope):
    base, head, metadata = build_model(weights, classes, seed=seed, scope=scope)
    model = CommonJointAdapter(base.backbone, base.separation, base.age_head, base.age_adversary)
    return model, head, metadata, AgingModule(), PatchDiscriminator()


def default_dataset(model, faces):
    return load_bound_dataset(faces, model.preprocess)


def load_inputs(paths, config):
    init, faces, smoke, contract = load_prerequisites(
        initialization_path=paths["initialization"], faces_path=paths["faces"],
        smoke_path=paths["smoke"], weights=paths["weights"], scope=config.scope, device="cpu",
    )
    joint = verified(paths["joint_smoke"], "mtlface-common-real-crop-joint-smoke")
    targets = verified(paths["targets"], "mtlface-target-stream-metadata-preflight")
    if joint["metrics"].get("real_crop_joint_smoke_complete") is not True:
        raise ValueError("completed real joint smoke required")
    if targets["metrics"].get("metadata_sampling_preflight_complete") is not True:
        raise ValueError("completed native target preflight required")
    for native in (joint, targets):
        if native["metrics"].get("training_complete") is not False:
            raise ValueError("prerequisite must not claim completed training")
        if native["metrics"].get("scientific_evaluation_complete") is not False:
            raise ValueError("prerequisite scientific scope mismatch")
        if file_record(paths["faces"]) not in native["inputs"]:
            raise ValueError("joint/target prerequisites must bind the exact face list")
    for name in ("initialization", "weights"):
        if file_record(paths[name]) not in joint["inputs"]:
            raise ValueError("joint smoke initial weights/initialization mismatch")
    if joint["parameters"].get("scope") != config.scope or joint["parameters"].get("device") != "cpu":
        raise ValueError("joint smoke scope/device mismatch")
    if targets["parameters"].get("policy") != "uniform-group-then-uniform-row-with-replacement":
        raise ValueError("target policy mismatch")
    counts = faces["metrics"]["counts"]
    if targets["metrics"].get("retained_images") != counts["retained_images"] or targets["metrics"].get("retained_recorded_identities") != counts["retained_people"]:
        raise ValueError("target preflight count mismatch")
    return init, faces, smoke, contract, joint, targets


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fingerprints(modules):
    return {
        role: {
            name: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, p in module.named_parameters()
        }
        for role, module in modules.items()
    }


def _weights_snapshot(modules):
    result = {}
    for role, module in modules.items():
        result[role] = {}
        for name, value in module.state_dict().items():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError("nonfinite final checkpoint")
            result[role][name] = value.detach().cpu().clone()
    return result


def _check_epoch(result, samples, batch_size, rows):
    _require_finite(result, "joint epoch")
    ledger = result["ledger"]
    batches = math.ceil(samples / batch_size)
    if (
        ledger.get("epoch_complete") is not True
        or ledger.get("completed_source_samples") != samples
        or ledger.get("source_unique_rows") != samples
        or ledger.get("source_decoded_views") != samples
        or ledger.get("target_sampled_draws") != samples
        or ledger.get("target_decoded_views") != samples
        or ledger.get("completed_batches") != batches
        or ledger.get("optimizer_steps") != dict(fr=batches, generator=batches, discriminator=batches)
        or ledger.get("optimizer_step_attempts") != ledger.get("optimizer_steps")
        or ledger.get("entry_forward_images") != dict(encoder_fr=samples, encoder_fas=2 * samples,
                                                       generator_fas=samples, discriminator_fas=3 * samples)
        or rows != batches
    ):
        raise ValueError("full joint epoch and persisted batch coverage required")
    means = result.get("sample_weighted_means", {})
    if not {"recognition", "fas"} <= means.keys():
        raise ValueError("recognition and FAS epoch summaries required")
    for stage, names in (("recognition", ("total", "gradient_norm")),
                         ("fas", ("discriminator_loss", "generator_loss", "discriminator_gradient_norm", "generator_gradient_norm"))):
        for name in names:
            value = means[stage].get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("numeric complete joint epoch metrics required")


def run_campaign(*, paths, out, config, preflight_only=False,
                 model_factory=None, dataset_factory=None, epoch_runner=None):
    config.validate()
    if not isinstance(preflight_only, bool):
        raise ValueError("boolean preflight flag required")
    if out.exists():
        raise FileExistsError("fresh output required; implicit resume refused")
    torch.set_num_threads(1)
    init, faces, smoke, contract, joint, targets = load_inputs(paths, config)
    injected = any(fn is not None for fn in (model_factory, dataset_factory, epoch_runner))
    model_factory = model_factory or default_models
    dataset_factory = dataset_factory or default_dataset
    epoch_runner = epoch_runner or run_joint_epoch
    extras = [Path(__file__).resolve(), paths["joint_smoke"], paths["targets"]]
    extras += [resolve(r["path"]) for m in (joint, targets) for r in m["inputs"] + m["outputs"]]
    extras += [Path(inspect.getsourcefile(fn)) for fn in (model_factory, dataset_factory, epoch_runner)]
    extras += [PROJECT_ROOT / ("scripts/" + name + ".py") for name in (
        "mtlface_joint_epoch_v3", "mtlface_joint_adapter_v2", "mtlface_fas_training_v2",
        "mtlface_fas_networks_v2", "mtlface_target_stream_v3",
    )]
    sources = collect_sources(
        initialization_path=paths["initialization"], faces_path=paths["faces"],
        smoke_path=paths["smoke"], weights=paths["weights"],
        initialization=init, faces=faces, smoke=smoke, extra_sources=extras,
    )
    before = [file_record(p) for p in sources]

    def stable():
        if before != [file_record(p) for p in sources]:
            raise RuntimeError("campaign input/code changed; completion not credited")

    protocol = dict(
        **asdict(config), device="cpu", threads=1, recognition_bn_policy="frozen_all",
        recognition_optimizer="plain SGD", fas_optimizer="Adam",
        checkpoint_selection="last_epoch", weights_only_not_resumable=True,
        target_seed_rule="training seed + 10000; distinct continuing CPU generators",
        target_preflight_is_policy_check_not_same_seed_draws=True,
        workflow="per source batch: target decode, FR, D, G",
        implementation_mode="injected-test" if injected else "native-joint-adaptation",
        architecture_reference=provenance(), preflight_only=preflight_only,
        gan_weight=75.0, identity_weight=0.002, age_weight=10.0,
    )
    out.mkdir(parents=True)
    summary_path, manifest = out / "summary.json", out / "summary.manifest.json"

    def bind(path, experiment, metrics, outputs, parameters=protocol):
        stable()
        write_experiment_manifest(path, experiment=experiment, parameters=parameters,
                                  metrics=metrics, inputs=sources, outputs=outputs)
        try:
            stable()
        except Exception:
            path.unlink(missing_ok=True)
            raise

    if preflight_only:
        metrics = dict(preflight_complete=True, training_complete=False, **flags(),
                       retained_images=faces["metrics"]["counts"]["retained_images"],
                       identity_classes=init["metrics"]["identity_classes"])
        _dump(summary_path, metrics)
        bind(manifest, "mtlface-common-joint-campaign-preflight", metrics, [summary_path])
        return metrics
    private = out / "private"
    private.mkdir()
    cells, outputs, cell_manifests = [], [], []
    for seed in config.seeds:
        cell_dir = private / f"seed_{seed}"
        cell_dir.mkdir()
        history, batch_paths = [], []
        torch.manual_seed(seed)
        model, head, metadata, generator, discriminator = model_factory(
            paths["weights"], init["metrics"]["identity_classes"], seed=seed, scope=config.scope,
        )
        _check_seeded_contract(metadata, contract, seed=seed, weights=paths["weights"])
        modules = dict(model=model, identity_head=head, generator=generator, discriminator=discriminator)
        for module in modules.values():
            module.to("cpu")
        initial = _fingerprints(modules)
        dataset, native = dataset_factory(model, paths["faces"])
        if native != faces or len(dataset) != faces["metrics"]["counts"]["retained_images"] or dataset.n_classes != init["metrics"]["identity_classes"]:
            raise ValueError("full dataset/native/classifier contract mismatch")
        bindings = {resolve(r["path"]).resolve(): r for r in faces["inputs"]}
        stream = TargetAgeStream(dataset, torch.Generator().manual_seed(seed + 10000), bindings)
        source_rng = torch.Generator().manual_seed(seed)
        fr_opt = torch.optim.SGD([p for m in (model, head) for p in m.parameters() if p.requires_grad], lr=config.fr_lr)
        g_opt = torch.optim.Adam(generator.parameters(), lr=config.g_lr, betas=config.betas)
        d_opt = torch.optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=config.betas)
        for epoch in range(1, config.epochs + 1):
            batch_path = cell_dir / f"epoch_{epoch}_batches.jsonl"
            batch_paths.append(batch_path)
            rows = 0
            with batch_path.open("x", encoding="utf-8") as log:
                def progress(ledger, seed=seed, epoch=epoch):
                    nonlocal rows
                    _require_finite(ledger, "batch ledger")
                    compact = {k: v for k, v in ledger.items() if k not in ("target_stream_before", "target_stream_after")}
                    compact["target_rng_current_sha256"] = ledger["target_stream_after"]["rng_current_sha256"]
                    log.write(json.dumps(dict(seed=seed, epoch=epoch, ledger=compact)) + "\n")
                    log.flush()
                    rows += 1
                    print(f"seed={seed} epoch={epoch}/{config.epochs} batch={rows}", flush=True)
                try:
                    result = epoch_runner(
                        model, head, generator, discriminator, fr_opt, g_opt, d_opt, stream,
                        batch_size=config.batch_size, source_rng=source_rng, device="cpu",
                        generator_bn_policy=config.generator_bn_policy, progress=progress,
                    )
                except JointEpochFailure as error:
                    _dump(cell_dir / f"epoch_{epoch}_failure.json", dict(
                        seed=seed, epoch=epoch, error=str(error), ledger=error.ledger,
                        training_complete=False, **flags(),
                    ))
                    raise
            _check_epoch(result, len(dataset), config.batch_size, rows)
            history.append(dict(epoch=epoch, **result))
        final = _fingerprints(modules)
        changes = {role: sum(value != initial[role][name] for name, value in final[role].items()) for role in modules}
        if any(count < 1 for count in changes.values()):
            raise ValueError("actual model/head/G/D parameter changes required")
        checkpoint, history_path = cell_dir / "last.pt", cell_dir / "history.json"
        torch.save(_weights_snapshot(modules), checkpoint)
        _dump(history_path, history)
        metrics = dict(seed=seed, training_complete=True, epochs_executed=config.epochs,
                       epochs_requested=config.epochs, selected_epoch=config.epochs,
                       changed_parameter_tensors=changes, **flags())
        cell_manifest = cell_dir / "cell.manifest.json"
        bind(cell_manifest, "mtlface-common-joint-campaign-cell", metrics,
             [checkpoint, history_path, *batch_paths], protocol | dict(seed=seed, target_seed=seed + 10000))
        verified(cell_manifest, "mtlface-common-joint-campaign-cell")
        cell_manifests.append(cell_manifest)
        cells.append(dict(seed=seed, manifest=file_record(cell_manifest), checkpoint=file_record(checkpoint)))
        outputs.extend([cell_manifest, checkpoint, history_path, *batch_paths])
        del modules, model, head, generator, discriminator, fr_opt, g_opt, d_opt, dataset, stream
    for path in cell_manifests:
        verified(path, "mtlface-common-joint-campaign-cell")
    metrics = dict(campaign_complete=True, training_complete=True, completed_seeds=list(config.seeds),
                   cells=cells, **flags())
    _dump(summary_path, metrics)
    bind(manifest, "mtlface-common-joint-campaign", metrics, [summary_path, *outputs])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("faces", "initialization", "smoke", "joint-smoke", "targets", "weights", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("epochs", "batch-size"):
        parser.add_argument("--" + name, type=int, required=True)
    for name in ("fr-lr", "g-lr", "d-lr"):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--generator-bn-policy", choices=("adapt", "frozen"), required=True)
    parser.add_argument("--scope", choices=("head", "tail", "full"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    parser.add_argument("--betas", type=float, nargs=2, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_interop_threads(1)
    config = CampaignConfig(args.epochs, args.batch_size, args.fr_lr, args.g_lr, args.d_lr,
                            args.generator_bn_policy, tuple(args.seeds), args.scope, tuple(args.betas))
    paths = {name: getattr(args, name) for name in ("faces", "initialization", "smoke", "joint_smoke", "targets", "weights")}
    print(json.dumps(run_campaign(paths=paths, out=args.out, config=config, preflight_only=args.preflight_only)))


if __name__ == "__main__":
    main()
