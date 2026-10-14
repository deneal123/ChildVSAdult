"""Full CUDA ROC-v2 comparison for prepared permutation1/2, three fixed checkpoints each.

Version2 preserves the immutable permutation42 evaluator and metric definitions.
Pi E380 proposal reviewed and corrected by root; native execution remains separate
from synthetic verification. No full SOTA, causal-source or human-gold claim.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_controls import SEEDS, paired_infer
from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs
from scripts.run_oriented_campaign import PROTOCOL, paths_from, training_contract, verified

PROPOSAL_VERSION = "oriented-noise-evaluator-v2-root-e385"

#: Permutation (intervention) seeds of the version-2 preflights.  Distinct from training seeds.
PERMUTATION_SEEDS = (1, 2)
#: Full training-seed set that every version-2 campaign must have completed.
TRAINING_SEEDS = SEEDS
#: ROC-v2 endpoint-age negative-matching seed; part of the metric protocol, not an intervention.
ROC_V2_PROTOCOL_SEED = 42

NOISE_PREFLIGHT_EXPERIMENT = "oriented-uniform-noise-training-preflight-v2"
NOISE_CAMPAIGN_EXPERIMENT = "oriented-uniform-cuda-noise-training-v2"
NOISE_CELL_EXPERIMENT = "oriented-native-cuda-noise-cell-v2"
COMPONENT_EXPERIMENT = "pair-contrastive-backbone-finetune"
CLEAN_EVAL_EXPERIMENT = "oriented-cuda-controls-full-fgnet-roc-v2"
OUTPUT_EXPERIMENT = "oriented-cuda-noise-permutation-full-fgnet-roc-v2"

CLEAN_EVAL_PATH = PROJECT_ROOT / "metrics/oriented_cuda_fgnet_20261004/summary.manifest.json"
FGNET_CACHE = PROJECT_ROOT / "data/external/fgnet_crops.npz"

#: Minimal clean/noise contract published by the version-2 preflight (strictly typed).
CLEAN_NOISE_CONTRACT = {
    "positive_pairs": 600,
    "negative_pairs": 600,
    "all_train_images": 1096,
    "recorded_train_people": 500,
    "fixed_negatives_and_heldout_unchanged": True,
    "endpoint_image_multiplicities_unchanged": True,
}

#: Canonical sources whose exact bytes must be bound into the evaluation manifest.
_SOURCE_RELATIVE = (
    "scripts/evaluate_oriented_noise_cuda_v1.py",  # original v1 evaluator (immutable)
    "scripts/evaluate_oriented_noise_cuda_v2.py",  # this proposal, when promoted
    "scripts/run_oriented_noise_cuda_v1.py",
    "scripts/run_oriented_noise_cuda_v2.py",
    "scripts/prepare_oriented_noise_training.py",
    "scripts/prepare_oriented_noise_training_v2.py",
    "tests/test_oriented_noise_cuda_evaluation_v1.py",
    "scripts/evaluate_oriented_controls.py",
    "scripts/evaluate_oriented_cuda_v1.py",
    "scripts/benchmark_metrics_v2.py",
    "scripts/verification_metrics_v2.py",
    "scripts/reevaluate_fgnet_metrics_v2.py",
    "scripts/run_oriented_campaign.py",
    "scripts/run_oriented_cuda_campaign_v1.py",
)


def _same_schema(actual, expected):
    """Strict typed equality (``True`` never equals ``1``); mirrors run v2's guard."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _same_schema(actual[k], v) for k, v in expected.items()
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _same_schema(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


def canonical_permutation_seed(value):
    """Accept only the canonical version-2 permutation seeds 1/2 (never training seed 42)."""
    if type(value) is not int or value not in PERMUTATION_SEEDS:
        raise ValueError(
            "canonical permutation seed 1/2 required; permutation seed is not a training seed"
        )
    return value


def pairs_for(root, permutation_seed):
    """Prepared noise arm for one permutation seed under the shared ``--pairs`` root."""
    canonical_permutation_seed(permutation_seed)
    return Path(root) / f"seed{permutation_seed}" / "private" / "partial_noise_arm.jsonl"


def source_paths():
    """Canonical v1/v2 producer, evaluator and test sources bound into the manifest inputs."""
    return [Path(__file__), *(PROJECT_ROOT / rel for rel in _SOURCE_RELATIVE)]


def require_training_seeds(metrics):
    """Require exactly the full training-seed set 42/1/2, strictly typed."""
    if not _same_schema(metrics.get("training_seeds"), list(TRAINING_SEEDS)):
        raise ValueError("training seeds 42/1/2 required, distinct from permutation seed")


def require_preflight(native, *, permutation_seed, pairs):
    """Bind an explicit permutation seed and its pair arm to a version-2 preflight."""
    canonical_permutation_seed(permutation_seed)
    if not _same_schema(
        native.get("parameters"), PROTOCOL | dict(permutation_seed=permutation_seed)
    ):
        raise ValueError("noise v2 preflight protocol mismatch")
    metrics = native["metrics"]
    if metrics.get("preflight_complete") is not True or not _same_schema(
        metrics.get("permutation_seed"), permutation_seed
    ):
        raise ValueError("completed selected-permutation v2 preflight required")
    require_training_seeds(metrics)
    if file_record(pairs) not in native["inputs"]:
        raise ValueError("noise pair file not bound by v2 preflight")
    contract = metrics.get("clean_noise_contract", {})
    if any(
        not _same_schema(contract.get(key), expected)
        for key, expected in CLEAN_NOISE_CONTRACT.items()
    ):
        raise ValueError("clean/noise budget or exposure mismatch")


def bound_preflight(native):
    """Require exactly one bound v2 preflight; never scan arbitrary private JSON."""
    found_paths = {}
    for record in native["inputs"]:
        candidate = Path(record["path"])
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if not candidate.name.endswith(".manifest.json"):
            continue
        found = json.loads(candidate.read_text(encoding="utf-8"))
        if type(found) is dict and found.get("experiment") == NOISE_PREFLIGHT_EXPERIMENT:
            found_paths[candidate.resolve(strict=True)] = candidate
    if len(found_paths) != 1:
        raise ValueError("noise v2 campaign must bind exactly one version-2 preflight")
    return next(iter(found_paths.values()))


def readiness(binding, *, permutation_seed, pairs):
    """Validate a version-2 campaign, its bound preflight and every seed's cell chain."""
    canonical_permutation_seed(permutation_seed)
    native = verified(binding, NOISE_CAMPAIGN_EXPERIMENT)
    if not _same_schema(
        native["parameters"], PROTOCOL | dict(device="cuda", permutation_seed=permutation_seed)
    ):
        raise ValueError("noise v2 CUDA campaign protocol mismatch")
    preflight = bound_preflight(native)
    require_preflight(
        verified(preflight, NOISE_PREFLIGHT_EXPERIMENT),
        permutation_seed=permutation_seed,
        pairs=pairs,
    )
    metrics = native["metrics"]
    if metrics.get("training_complete") is not True or not _same_schema(metrics.get("completed_cells"), 3):
        raise ValueError("three completed noise training seeds required")
    selected = {}
    for seed in TRAINING_SEEDS:
        checkpoint = binding.parent / "private" / f"noise_s{seed}.pt"
        component_path = checkpoint.with_suffix(".manifest.json")
        cell_path = binding.parent / f"noise_s{seed}.manifest.json"
        if any(
            file_record(p) not in native["outputs"]
            for p in (checkpoint, component_path, cell_path)
        ):
            raise ValueError("noise cell/component/checkpoint not bound by v2 campaign")
        cell = verified(cell_path, NOISE_CELL_EXPERIMENT)
        parameters = cell["parameters"]
        if cell["metrics"].get("training_complete") is not True:
            raise ValueError("completed noise cell required")
        expected = PROTOCOL | dict(device="cuda", seed=seed, permutation_seed=permutation_seed)
        if any(
            not _same_schema(parameters.get(key), value) for key, value in expected.items()
        ):
            raise ValueError("noise v2 cell protocol mismatch")
        if any(file_record(p) not in cell["outputs"] for p in (checkpoint, component_path)):
            raise ValueError("noise cell linkage mismatch")
        component = verified(component_path, COMPONENT_EXPERIMENT)
        training_contract(component, pairs, seed)
        if file_record(checkpoint) not in component["outputs"]:
            raise ValueError("component does not bind noise checkpoint")
        selected[seed] = checkpoint
    if set(selected) != set(TRAINING_SEEDS):
        raise ValueError("missing noise training seed")
    return native, selected


def build_inputs(*, preflights, pairs, campaigns, clean_path, natives, clean):
    """Deduplicated provenance: v1/v2 sources, both arms' preflights, campaigns and clean eval."""
    return list(
        dict.fromkeys(
            [
                *source_paths(),
                *preflights,
                *pairs,
                *campaigns,
                clean_path,
                FGNET_CACHE,
                *(p for native in natives for p in paths_from(native)),
                *paths_from(clean),
            ]
        )
    )


def aligned_clean(data, left, right, labels, gaps, subjects):
    """Unchanged v1 clean-score realignment guard (endpoint/pair order and finiteness)."""
    expected = dict(
        left=left,
        right=right,
        labels=labels,
        source_gap=gaps,
        subject_a=subjects[left],
        subject_b=subjects[right],
    )
    if any(not np.array_equal(data[key], value) for key, value in expected.items()):
        raise ValueError("saved clean pair order/metadata differs")
    scores = np.stack([data[f"cross_s{seed}"] for seed in SEEDS])
    frozen = data["frozen"]
    if (
        scores.shape != (3, len(labels))
        or frozen.shape != labels.shape
        or not np.isfinite(scores).all()
        or not np.isfinite(frozen).all()
    ):
        raise ValueError("finite aligned clean/frozen scores required")
    return scores, frozen


def arm_results(clean_scores, noise, labels, gaps, subjects, left, right, frozen):
    """Unchanged v1 ROC-v2 inference: overall and large-gap shared-subject paired contrast."""
    result = {}
    for name, mask in (
        ("overall", np.ones(len(labels), bool)),
        ("large_gap_25plus", gaps >= 25),
    ):
        result[name] = paired_infer(
            clean_scores[:, mask],
            noise[:, mask],
            labels[mask],
            subjects[left][mask],
            subjects[right][mask],
            frozen=frozen[mask],
        )
        result[name]["scope"] = (
            "raw noisy CROSS minus clean CROSS; smaller EER is better; test-derived ROC "
            "not deployment calibration; intervals not multiplicity corrected"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutation-seed1", type=Path, required=True)
    parser.add_argument("--permutation-seed2", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh noise evaluation output required")
    campaigns = {1: args.permutation_seed1, 2: args.permutation_seed2}
    pairs, natives, checkpoints, preflights = {}, {}, {}, {}
    for permutation_seed in PERMUTATION_SEEDS:
        pairs[permutation_seed] = pairs_for(args.pairs, permutation_seed)
        binding = campaigns[permutation_seed] / "training-bound.manifest.json"
        native, selected = readiness(
            binding, permutation_seed=permutation_seed, pairs=pairs[permutation_seed]
        )
        natives[permutation_seed] = native
        checkpoints[permutation_seed] = selected
        preflights[permutation_seed] = bound_preflight(native)
    clean_path = CLEAN_EVAL_PATH
    clean = verified(clean_path, CLEAN_EVAL_EXPERIMENT)
    if clean["metrics"].get("execution_complete") is not True:
        raise ValueError("completed clean evaluation required")
    if not args.execute:
        print(
            "both permutation arms, three training seeds each, version-2 preflight/campaign "
            "and clean scores ready; no inference",
            flush=True,
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    from age_gap.training.finetune import load_finetuned
    from scripts.reevaluate_fgnet_metrics_v2 import protocol

    torch.set_num_threads(1)
    cache = FGNET_CACHE
    inputs = build_inputs(
        preflights=list(preflights.values()),
        pairs=list(pairs.values()),
        campaigns=[d / "training-bound.manifest.json" for d in campaigns.values()],
        clean_path=clean_path,
        natives=natives.values(),
        clean=clean,
    )
    before = [file_record(path) for path in inputs]
    with np.load(cache, allow_pickle=False) as data:
        crops, subjects, ages = data["crops"], data["subjects"], data["ages"]
    if crops.shape != (len(subjects), 112, 112, 3) or crops.dtype != np.uint8:
        raise ValueError("full aligned BGR crops required")
    left, right, labels, gaps, coverage = protocol(subjects, ages)
    with np.load(clean_path.parent / "private/scores.npz", allow_pickle=False) as data:
        clean_scores, frozen = aligned_clean(data, left, right, labels, gaps, subjects)
    result = {
        "execution_complete": True,
        "publication_ready": False,
        "coverage": coverage,
        "permutation_seeds": list(PERMUTATION_SEEDS),
        "training_seeds": list(TRAINING_SEEDS),
        "evaluator_version": PROPOSAL_VERSION,
        "low_role": "clean CROSS",
        "cross_role": "noisy CROSS",
        "direction": "noise_minus_clean",
        "limitation": (
            "fixed permutations/checkpoints; not empirical identity-error rate, co-occurrence "
            "or causal-source evidence"
        ),
    }
    outputs = []
    for permutation_seed in PERMUTATION_SEEDS:
        embeddings, noise = {}, []
        for seed in TRAINING_SEEDS:
            model = load_finetuned(checkpoints[permutation_seed][seed], "cuda").eval()
            vectors = []
            with torch.no_grad():
                for start in range(0, len(crops), 32):
                    batch = np.stack(
                        [model.preprocess(image, True) for image in crops[start : start + 32]]
                    )
                    vectors.append(model(torch.from_numpy(batch).to("cuda")).cpu().numpy())
            vectors = np.concatenate(vectors)
            if (
                vectors.shape != (len(crops), 512)
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
            ):
                raise ValueError("finite full unit embeddings required")
            embeddings[seed] = vectors
            noise.append(np.sum(vectors[left] * vectors[right], axis=1))
            del model
            print(
                f"CUDA noise permutation{permutation_seed} full pool embedded: seed{seed}",
                flush=True,
            )
        noise = np.stack(noise)
        result[f"permutation_seed{permutation_seed}"] = arm_results(
            clean_scores, noise, labels, gaps, subjects, left, right, frozen
        )
        private = args.out / "private"
        private.mkdir(parents=True, exist_ok=True)
        for seed, vectors in embeddings.items():
            path = private / f"noise_perm{permutation_seed}_s{seed}.npz"
            np.savez_compressed(path, indices=np.arange(len(crops)), embeddings=vectors)
            outputs.append(path)
        scores = private / f"scores_perm{permutation_seed}.npz"
        np.savez_compressed(
            scores,
            clean=clean_scores,
            noise=noise,
            frozen=frozen,
            labels=labels,
            subject_a=subjects[left],
            subject_b=subjects[right],
            left=left,
            right=right,
            source_gap=gaps,
        )
        outputs.append(scores)
    if before != [file_record(path) for path in inputs]:
        raise RuntimeError("noise evaluation input changed")
    summary = args.out / "summary.json"
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment=OUTPUT_EXPERIMENT,
        parameters=dict(
            device="cuda",
            permutation_seeds=list(PERMUTATION_SEEDS),
            training_seeds=list(TRAINING_SEEDS),
            bootstrap_seed=0,
            n_boot=2000,
            protocol_seed=ROC_V2_PROTOCOL_SEED,
            tolerance=2,
            evaluator_version=PROPOSAL_VERSION,
        ),
        metrics=result,
        inputs=inputs,
        outputs=[summary, *outputs],
    )
    validate_written_inputs(target, before)


if __name__ == "__main__":
    main()
