import copy

import pytest

from scripts.render_fixed_checkpoint_means_v1 import GROUPS, METRICS, number, table


def fixture():
    cell = dict(fixed_seed_list=[42, 1, 2], publication_ready=False,
        training_identity_independence="unverified",
        bootstrap=dict(shared_weights_across_models=True),
        frozen={name: dict(point=.9) for name in METRICS},
        aggregate={name: dict(mean=.8, sd=.01, mean_checkpoint_delta=-.1,
                             delta_ci95=[-.2, -.05]) for name in METRICS})
    return dict(publication_ready=False, full_matrix_complete=False,
                groups={key: {s: copy.deepcopy(cell) for s in ("overall", "large_gap_25plus")} for key in GROUPS})


def test_exact_twenty_native_rows_and_explicit_sd_ci():
    tex = table(fixture(), "overall")
    assert tex.count("0.8000") == 20
    assert tex.count("[-0.2000, -0.0500]") == 20
    assert tex.count("Head,") == tex.count("Tail,") == 10
    assert "Mean$\\pm$SD" in tex and "mean-$\\Delta$" in tex
    assert "Discrete minimax EER" in tex
    assert "TAR@FAR=0.1\\%" in tex


@pytest.mark.parametrize("bad", [True, 1, float("nan"), float("inf"), "0.1"])
def test_non_native_float_refused(bad):
    with pytest.raises(ValueError):
        number(bad)


@pytest.mark.parametrize("kind", ["scope", "seed", "model", "ci", "sd", "metric", "identity"])
def test_unsupported_or_changed_contract_refused(kind):
    summary = fixture()
    cell = summary["groups"][GROUPS[0]]["overall"]
    if kind == "scope":
        summary["full_matrix_complete"] = True
    elif kind == "seed":
        cell["fixed_seed_list"] = [42, True, 2]
    elif kind == "model":
        summary["groups"].pop(GROUPS[-1])
    elif kind == "ci":
        cell["aggregate"][METRICS[0]]["delta_ci95"] = [.2, -.2]
    elif kind == "sd":
        cell["aggregate"][METRICS[0]]["sd"] = -.01
    elif kind == "metric":
        cell["aggregate"].pop(METRICS[-1])
    else:
        cell["training_identity_independence"] = "verified"
    with pytest.raises(ValueError):
        table(summary, "overall")
