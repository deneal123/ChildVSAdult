"""Focused synthetic tests for the proposed version-2 noise evaluator.

Task ``noise-evaluator-v2-e380``, check date 2026-10-09.

No GPU, CUDA, native inference, training, downloading, large hashing or real-data
processing is performed here.  Every prerequisite is a tiny synthetic object or a
``monkeypatch`` stub under ``tmp_path``.  The tests assert that the proposal:

* requires the explicit ``--permutation-seed1``/``--permutation-seed2`` campaign arms and
  the shared ``--pairs`` root, and binds them to the v2 preflight/campaign;
* requires all three training seeds ``42/1/2`` and keeps them distinct from the permutation
  seeds (whose numeric set nonetheless overlaps);
* version-checks the v2 protocol instead of silently accepting a hard-coded 42;
* validates the v2 cell/component/checkpoint ancestry and the campaign->preflight binding;
* binds the original v1 evaluator source;
* preserves the unchanged clean-score realignment and ROC-v2 inference body.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_oriented_noise_cuda_v2 as evaluator


# --------------------------------------------------------------------------- #
# seed concepts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [1, 2])
def test_accepts_authorised_permutation_seeds(value):
    assert evaluator.canonical_permutation_seed(value) == value


@pytest.mark.parametrize("value", [True, False, "1", 1.0, 0, 3, 42, None])
def test_rejects_bool_nontraining_and_42_as_permutation_seed(value):
    with pytest.raises(ValueError, match="permutation seed"):
        evaluator.canonical_permutation_seed(value)


def test_permutation_and_training_seed_sets_are_distinct_but_overlap():
    assert evaluator.PERMUTATION_SEEDS == (1, 2)
    assert evaluator.TRAINING_SEEDS == (42, 1, 2)
    assert 42 not in evaluator.PERMUTATION_SEEDS
    # Concepts differ even though the numeric sets share {1, 2}.
    assert set(evaluator.PERMUTATION_SEEDS) & set(evaluator.TRAINING_SEEDS) == {1, 2}


@pytest.mark.parametrize("seeds", [[42, 1], [1, 2], [42, 1, 2, 3], [42, True, 2]])
def test_training_seed_set_must_be_exactly_42_1_2(seeds):
    with pytest.raises(ValueError, match="training seeds"):
        evaluator.require_training_seeds({"training_seeds": seeds})


def test_training_seed_set_all_three_accepted():
    evaluator.require_training_seeds({"training_seeds": [42, 1, 2]})


def test_roc_v2_protocol_seed_is_not_an_intervention_seed():
    assert evaluator.ROC_V2_PROTOCOL_SEED == 42
    assert evaluator.ROC_V2_PROTOCOL_SEED not in evaluator.PERMUTATION_SEEDS


def test_pairs_for_derives_per_seed_arm():
    root = Path("prepared/noise")
    assert evaluator.pairs_for(root, 1) == root / "seed1/private/partial_noise_arm.jsonl"
    assert evaluator.pairs_for(root, 2) == root / "seed2/private/partial_noise_arm.jsonl"
    with pytest.raises(ValueError, match="permutation seed"):
        evaluator.pairs_for(root, 42)


# --------------------------------------------------------------------------- #
# preflight binding: explicit permutation seed and shared --pairs arm
# --------------------------------------------------------------------------- #
def synthetic_preflight(permutation_seed=1, pairs="pairs"):
    return dict(
        parameters=evaluator.PROTOCOL | dict(permutation_seed=permutation_seed),
        inputs=[dict(path=pairs)],
        metrics=dict(
            preflight_complete=True,
            permutation_seed=permutation_seed,
            training_seeds=[42, 1, 2],
            clean_noise_contract=dict(evaluator.CLEAN_NOISE_CONTRACT),
        ),
    )


def _bind(monkeypatch, pairs="pairs"):
    monkeypatch.setattr(evaluator, "file_record", lambda path: dict(path=pairs))


@pytest.mark.parametrize("seed", [1, 2])
def test_matching_preflight_accepted(monkeypatch, seed):
    _bind(monkeypatch)
    evaluator.require_preflight(synthetic_preflight(seed), permutation_seed=seed, pairs="unused")


@pytest.mark.parametrize("selected,declared", [(1, 2), (2, 1), (1, 42)])
def test_permutation_mismatch_rejected(monkeypatch, selected, declared):
    _bind(monkeypatch)
    with pytest.raises(ValueError, match="mismatch"):
        evaluator.require_preflight(
            synthetic_preflight(declared), permutation_seed=selected, pairs="unused"
        )


def test_incomplete_preflight_rejected(monkeypatch):
    _bind(monkeypatch)
    changed = synthetic_preflight(1)
    changed["metrics"]["preflight_complete"] = False
    with pytest.raises(ValueError, match="completed"):
        evaluator.require_preflight(changed, permutation_seed=1, pairs="unused")


def test_preflight_training_seed_change_rejected(monkeypatch):
    _bind(monkeypatch)
    changed = synthetic_preflight(1)
    changed["metrics"]["training_seeds"] = [1]
    with pytest.raises(ValueError, match="training seeds"):
        evaluator.require_preflight(changed, permutation_seed=1, pairs="unused")


def test_preflight_missing_protocol_key_rejected(monkeypatch):
    """A hard-coded-42 style protocol cannot masquerade as the versioned v2 protocol."""
    _bind(monkeypatch)
    changed = synthetic_preflight(1)
    del changed["parameters"]["sampling"]
    with pytest.raises(ValueError, match="mismatch"):
        evaluator.require_preflight(changed, permutation_seed=1, pairs="unused")


@pytest.mark.parametrize(
    "key,value",
    [
        ("all_train_images", 1121),
        ("fixed_negatives_and_heldout_unchanged", False),
        ("endpoint_image_multiplicities_unchanged", False),
        ("positive_pairs", 599),
        ("negative_pairs", True),
    ],
)
def test_incompatible_contract_rejected(monkeypatch, key, value):
    _bind(monkeypatch)
    changed = synthetic_preflight(1)
    changed["metrics"]["clean_noise_contract"][key] = value
    with pytest.raises(ValueError, match="exposure"):
        evaluator.require_preflight(changed, permutation_seed=1, pairs="unused")


def test_unbound_pairs_rejected(monkeypatch):
    monkeypatch.setattr(evaluator, "file_record", lambda path: dict(path="different"))
    with pytest.raises(ValueError, match="not bound"):
        evaluator.require_preflight(synthetic_preflight(1), permutation_seed=1, pairs="unused")


# --------------------------------------------------------------------------- #
# campaign/cell/component ancestry
# --------------------------------------------------------------------------- #
def readiness_fixture(monkeypatch, permutation_seed=1, tmp_path=None):
    """Synthetic linkage records; real hashing is covered by native execution."""
    binding = Path("campaign/training-bound.manifest.json")
    preflight = Path("preflight/summary.manifest.json")
    native = dict(
        parameters=evaluator.PROTOCOL | dict(device="cuda", permutation_seed=permutation_seed),
        metrics=dict(training_complete=True, completed_cells=3),
        inputs=[dict(path=str(preflight))],
        outputs=[],
    )
    cells, components, calls = {}, {}, []
    for seed in evaluator.TRAINING_SEEDS:
        ckpt = binding.parent / "private" / f"noise_s{seed}.pt"
        component = ckpt.with_suffix(".manifest.json")
        cell = binding.parent / f"noise_s{seed}.manifest.json"
        records = [dict(path=str(p)) for p in (ckpt, component, cell)]
        native["outputs"].extend(records)
        cells[str(cell)] = dict(
            metrics=dict(training_complete=True),
            parameters=evaluator.PROTOCOL
            | dict(seed=seed, device="cuda", permutation_seed=permutation_seed),
            outputs=records[:2],
        )
        components[str(component)] = dict(outputs=records[:1])

    def verified(path, experiment):
        if experiment == evaluator.NOISE_CAMPAIGN_EXPERIMENT:
            return native
        if experiment == evaluator.NOISE_PREFLIGHT_EXPERIMENT:
            return synthetic_preflight(permutation_seed)
        if experiment == evaluator.NOISE_CELL_EXPERIMENT:
            return cells[str(path)]
        assert experiment == evaluator.COMPONENT_EXPERIMENT
        return components[str(path)]

    monkeypatch.setattr(evaluator, "verified", verified)
    monkeypatch.setattr(evaluator, "file_record", lambda p: dict(path=str(p)))
    monkeypatch.setattr(evaluator, "training_contract", lambda *args: calls.append(args))
    monkeypatch.setattr(evaluator, "bound_preflight", lambda native: preflight)
    return binding, native, cells, components, calls


@pytest.mark.parametrize("permutation_seed", [1, 2])
def test_readiness_links_all_three_seeds_and_components(monkeypatch, permutation_seed):
    binding, _, _, _, calls = readiness_fixture(monkeypatch, permutation_seed)
    _, selected = evaluator.readiness(
        binding, permutation_seed=permutation_seed, pairs="pairs"
    )
    assert set(selected) == {42, 1, 2}
    assert [call[2] for call in calls] == list(evaluator.TRAINING_SEEDS)
    assert all(call[1] == "pairs" for call in calls)


@pytest.mark.parametrize(
    "change",
    [
        "incomplete",
        "missing_cell",
        "cpu",
        "wrong_permutation",
        "wrong_seed",
        "unbound_component",
        "unbound_checkpoint",
        "missing_protocol_key",
    ],
)
def test_readiness_rejects_incompatible_records(monkeypatch, change):
    binding, native, cells, components, _ = readiness_fixture(monkeypatch, 1)
    cell = cells[str(binding.parent / "noise_s42.manifest.json")]
    component = components[str(binding.parent / "private/noise_s42.manifest.json")]
    if change == "incomplete":
        native["metrics"]["completed_cells"] = 2
    elif change == "missing_cell":
        native["outputs"].pop(2)
    elif change == "cpu":
        native["parameters"]["device"] = "cpu"
    elif change == "wrong_permutation":
        cell["parameters"]["permutation_seed"] = 2
    elif change == "wrong_seed":
        cell["parameters"]["seed"] = 1
    elif change == "unbound_component":
        cell["outputs"] = cell["outputs"][:1]
    elif change == "unbound_checkpoint":
        component["outputs"] = []
    else:
        del cell["parameters"]["sampling"]
    with pytest.raises(ValueError):
        evaluator.readiness(binding, permutation_seed=1, pairs="pairs")


def test_readiness_requires_matching_campaign_permutation(monkeypatch):
    binding, _, _, _, _ = readiness_fixture(monkeypatch, 1)
    with pytest.raises(ValueError, match="campaign protocol"):
        evaluator.readiness(binding, permutation_seed=2, pairs="pairs")


def test_readiness_requires_campaign_bound_preflight(monkeypatch):
    binding, _, _, _, _ = readiness_fixture(monkeypatch, 1)
    monkeypatch.setattr(
        evaluator,
        "bound_preflight",
        lambda native: (_ for _ in ()).throw(ValueError("noise v2 campaign does not bind")),
    )
    with pytest.raises(ValueError, match="does not bind"):
        evaluator.readiness(binding, permutation_seed=1, pairs="pairs")


# --------------------------------------------------------------------------- #
# provenance / source ancestry
# --------------------------------------------------------------------------- #
def test_source_paths_bind_original_v1_evaluator():
    paths = {p.as_posix() for p in evaluator.source_paths()}
    assert any(p.endswith("scripts/evaluate_oriented_noise_cuda_v1.py") for p in paths)
    assert any(p.endswith("scripts/evaluate_oriented_noise_cuda_v2.py") for p in paths)
    assert any(p.endswith("scripts/run_oriented_noise_cuda_v2.py") for p in paths)
    assert any(p.endswith("tests/test_oriented_noise_cuda_evaluation_v1.py") for p in paths)


def test_build_inputs_dedupes_and_binds(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluator, "source_paths", lambda: [tmp_path / "src.py"])
    monkeypatch.setattr(evaluator, "paths_from", lambda native: [tmp_path / "inner.py"])
    clean = tmp_path / "clean.json"
    inputs = evaluator.build_inputs(
        preflights=[tmp_path / "preflight.json"],
        pairs=[tmp_path / "pairs.jsonl"],
        campaigns=[tmp_path / "binding.json"],
        clean_path=clean,
        natives=[object(), object()],
        clean=object(),
    )
    resolved = [Path(i) for i in inputs]
    assert len(resolved) == len(set(resolved))
    assert tmp_path / "preflight.json" in resolved
    assert tmp_path / "pairs.jsonl" in resolved
    assert clean in resolved
    assert evaluator.FGNET_CACHE in resolved
    assert resolved.count(tmp_path / "inner.py") == 1


def test_proposal_has_no_hardcoded_permutation_seed():
    """Static guard: no `permutation_seed=42` / `permutation_seed == 42` literal in source."""
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    assert "permutation_seed=42" not in source
    assert "permutation_seed == 42" not in source
    assert "permutation_seed != 42" not in source


# --------------------------------------------------------------------------- #
# CLI contract
# --------------------------------------------------------------------------- #
def cli_args(tmp_path, **overrides):
    base = dict(
        permutation_seed1=str(tmp_path / "c1"),
        permutation_seed2=str(tmp_path / "c2"),
        pairs=str(tmp_path / "pairs"),
        out=str(tmp_path / "out"),
    )
    base.update(overrides)
    argv = ["evaluator"]
    for key, value in base.items():
        argv += [f"--{key.replace('_', '-')}", value]
    return argv


def test_main_requires_both_explicit_permutation_seed_arms(monkeypatch, tmp_path):
    argv = cli_args(tmp_path)
    argv = [a for a in argv if a not in ("--permutation-seed2", str(tmp_path / "c2"))]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        evaluator.main()


def test_main_requires_pairs_root(monkeypatch, tmp_path):
    argv = cli_args(tmp_path)
    argv = [a for a in argv if a not in ("--pairs", str(tmp_path / "pairs"))]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        evaluator.main()


def test_main_fresh_output_guard(monkeypatch, tmp_path):
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(sys, "argv", cli_args(tmp_path))
    with pytest.raises(FileExistsError):
        evaluator.main()


# --------------------------------------------------------------------------- #
# unchanged clean-score realignment and inference body (carried from v1)
# --------------------------------------------------------------------------- #
def fixture():
    left, right = np.array([0, 1]), np.array([0, 2])
    labels, gaps, subjects = np.array([1, 0]), np.array([25, 25]), np.array(["a", "b", "c"])
    data = dict(
        left=left.copy(),
        right=right.copy(),
        labels=labels.copy(),
        source_gap=gaps.copy(),
        subject_a=subjects[left],
        subject_b=subjects[right],
        frozen=np.array([0.9, 0.1]),
    )
    for seed in (42, 1, 2):
        data[f"cross_s{seed}"] = np.array([0.8, 0.2])
    return data, left, right, labels, gaps, subjects


def test_same_order_clean_scores():
    values = fixture()
    scores, frozen = evaluator.aligned_clean(*values)
    assert scores.shape == (3, 2)
    assert frozen.tolist() == [0.9, 0.1]


@pytest.mark.parametrize(
    "field", ["left", "right", "labels", "source_gap", "subject_a", "subject_b"]
)
def test_misalignment_rejected(field):
    values = fixture()
    data = values[0]
    if data[field].dtype.kind in "US":
        data[field][0] = "wrong"
    else:
        data[field][0] += 1
    with pytest.raises(ValueError, match="metadata"):
        evaluator.aligned_clean(*values)


def test_nonfinite_rejected():
    values = fixture()
    values[0]["cross_s42"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluator.aligned_clean(*values)


def test_arm_results_uses_paired_infer_and_both_strata(monkeypatch):
    calls = []
    monkeypatch.setattr(
        evaluator,
        "paired_infer",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"auc": 0.5},
    )
    data, left, right, labels, gaps, subjects = fixture()
    clean_scores, frozen = evaluator.aligned_clean(data, left, right, labels, gaps, subjects)
    noise = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    result = evaluator.arm_results(
        clean_scores, noise, labels, gaps, subjects, left, right, frozen
    )
    assert set(result) == {"overall", "large_gap_25plus"}
    assert len(calls) == 2
    assert all(call[1]["frozen"].shape == (2,) for call in calls)
    assert all("noisy CROSS minus clean CROSS" in row["scope"] for row in result.values())


@pytest.mark.parametrize("count", [3., True, "3"])
def test_completed_cell_count_is_type_exact(monkeypatch, count):
    binding, native, _, _, _ = readiness_fixture(monkeypatch)
    native["metrics"]["completed_cells"] = count
    with pytest.raises(ValueError, match="three completed"):
        evaluator.readiness(binding, permutation_seed=1, pairs="pairs")


def test_preflight_discovery_does_not_read_arbitrary_private_json(tmp_path):
    import json
    private = tmp_path / "private.json"
    private.write_text("not-json; must never be opened", encoding="utf-8")
    manifest = tmp_path / "summary.manifest.json"
    manifest.write_text(json.dumps(dict(experiment=evaluator.NOISE_PREFLIGHT_EXPERIMENT)), encoding="utf-8")
    native = dict(inputs=[dict(path=str(private)), dict(path=str(manifest))])
    assert evaluator.bound_preflight(native) == manifest


def test_preflight_discovery_refuses_ambiguous_originals(tmp_path):
    import json
    inputs = []
    for name in ("one", "two"):
        manifest = tmp_path / f"{name}.manifest.json"
        manifest.write_text(json.dumps(dict(experiment=evaluator.NOISE_PREFLIGHT_EXPERIMENT)), encoding="utf-8")
        inputs.append(dict(path=str(manifest)))
    with pytest.raises(ValueError, match="exactly one"):
        evaluator.bound_preflight(dict(inputs=inputs))


def test_preflight_discovery_skips_non_object_manifest_without_attribute_error(tmp_path):
    candidate = tmp_path / "list.manifest.json"
    candidate.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        evaluator.bound_preflight(dict(inputs=[dict(path=str(candidate))]))


def test_clean_alignment_guard_keeps_immutable_v1_math():
    import ast

    from scripts import evaluate_oriented_noise_cuda_v1 as original
    def body(path):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "aligned_clean")
        return [ast.dump(n) for n in node.body if not
                (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    assert body(evaluator.__file__) == body(original.__file__)
