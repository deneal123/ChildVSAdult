"""Generated dictionary tests only; no native metrics, weights or images."""

import copy

import pytest

from scripts import fresh_common_presentation_contract_v1 as contract


def fixture():
    def model(delta=0):
        return dict(
            negative_mean=0.1,
            positive_mean=0.5,
            negative_variance=0.1,
            positive_variance=0.1,
            standardized_mean_difference=1 + delta,
            ci95=[0.5, 2],
            delta_vs_frozen=delta,
            delta_ci95=[0, 0] if delta == 0 else [-0.1, 0.4],
        )

    def family(keys):
        return dict(
            separability=dict(
                status="ok",
                version="fixed-pair-subject-separability-v1",
                positive_weight="owner multiplicity once",
                negative_weight="endpoint multiplicity product",
                n_pairs=5308,
                n_persons=82,
                requested_draws=2000,
                valid_draws=2000,
                seed=0,
                models={k: model(0 if k == "frozen" else 0.2) for k in keys},
            ),
            diagnostics={
                k: dict(
                    representation=dict(linear_cka=0.9, mean_cosine_drift=0.1, n_images=650),
                    paired_age_error=dict(
                        delta_mae_person=-0.1,
                        ci95=[-0.5, 0.2],
                        n_persons=82,
                        resamples=2000,
                        seed=0,
                    ),
                )
                for k in keys
                if k != "frozen"
            },
        )

    cell = "random_head_lr1e-06_s42"
    return dict(
        experiment=contract.EXPERIMENT,
        parameters=copy.deepcopy(contract.PARAMETERS),
        metrics=dict(
            execution_complete=True,
            publication_ready=False,
            fresh_weak_lineage=True,
            legacy_weak_history_recovered=False,
            fixed_oof_bootstrap=True,
            neural_inference_replayed=False,
            mechanism_complete=False,
            n_images=650,
            n_pairs=5308,
            strong_cells=[cell],
            weak_keys=list(contract.WEAK_KEYS),
            weak_upstream_bindings=["native1", "native2", "native42"],
            results=dict(strong=family(("frozen", cell)), weak=family(contract.WEAK_KEYS)),
        ),
    )


def test_literal_fresh_family_no_pooling():
    value = fixture()
    payload = contract.collect(value)
    assert payload["weak_keys"] == ["frozen", "tuned_s1", "tuned_s2", "tuned_s42"]
    assert payload["observed_strong_cells"] == 1
    assert payload["mechanism_complete"] is False
    assert "training-seed population" in payload["limitation"]
    payload["results"]["weak"]["diagnostics"].clear()
    assert value["metrics"]["results"]["weak"]["diagnostics"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("fresh_weak_lineage", False),
        ("legacy_weak_history_recovered", True),
        ("mechanism_complete", True),
        ("execution_complete", 1),
        ("publication_ready", True),
        ("weak_keys", ["frozen", "tuned_seed42", "tuned_seed1", "tuned_seed2"]),
        ("strong_cells", ["random_head_lr1e-06_s42", "random_head_lr1e-06_s1"]),
        ("weak_upstream_bindings", ["same", "same", "other"]),
        ("n_images", 650.0),
    ],
)
def test_refuses_false_or_legacy_contract(key, value):
    native = fixture()
    native["metrics"][key] = value
    with pytest.raises(ValueError):
        contract.collect(native)


@pytest.mark.parametrize(
    "key,value",
    [
        ("age_alpha", True),
        ("bootstrap_seed", True),
        ("protocol", "completed strong bound image-index pairs"),
        ("extra", 1),
    ],
)
def test_refuses_parameter_type_or_schema_drift(key, value):
    native = fixture()
    native["parameters"][key] = value
    with pytest.raises(ValueError):
        contract.collect(native)
