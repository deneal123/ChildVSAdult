from copy import deepcopy

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts.evaluate_clean_noise import compare_common_inputs, load_binding, paired_effect, run


def fixture():
    rng = np.random.default_rng(17)
    clean, noise = rng.normal(size=(3, 8)), rng.normal(size=(3, 8))
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    a = np.array(["p0", "p1", "p2", "p3", "p0", "p1", "p2", "p3"])
    b = np.array(["p0", "p1", "p2", "p3", "q0", "q1", "q2", "q3"])
    return clean, noise, y, a, b


def test_common_subject_draws_match_independent_reference():
    clean, noise, y, a, b = fixture()
    result = paired_effect(clean, noise, y, a, b, n_boot=80, seed=9)
    subjects = np.unique(np.concatenate((a, b)))
    rng = np.random.default_rng(9)
    draws = []
    for _ in range(80):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        counts = {s: int(np.sum(sampled == s)) for s in subjects}
        weights = np.array([counts[x] if label == 1 else counts[x] * counts[z]
                            for label, x, z in zip(y, a, b, strict=True)])
        if not all(np.any(weights[y == label]) for label in (0, 1)):
            continue
        draws.append([roc_auc_score(y, nr, sample_weight=weights) - roc_auc_score(y, cr, sample_weight=weights)
                      for cr, nr in zip(clean, noise, strict=True)])
    draws = np.asarray(draws)
    assert result["bootstrap"]["n_valid"] == len(draws)
    assert result["mean_checkpoint_ci95"] == pytest.approx(np.percentile(draws.mean(axis=1), [2.5, 97.5]))
    for index, seed in enumerate((42, 1, 2)):
        assert result["per_seed"][str(seed)]["ci95"] == pytest.approx(np.percentile(draws[:, index], [2.5, 97.5]))
    assert result == paired_effect(clean, noise, y, a, b, n_boot=80, seed=9)
    assert result["publication_ready"] is False
    assert result["causal_source_evidence"] is False


def test_mean_checkpoint_effect_is_not_score_ensemble_auc():
    y, a, b = np.array([1, 0]), np.array(["p", "p"]), np.array(["p", "q"])
    clean = np.array([[.8, .2]] * 3)
    noise = np.array([[.9, 0], [0, .8], [.1, .2]])
    result = paired_effect(clean, noise, y, a, b, n_boot=40)
    assert result["mean_checkpoint_delta_auc"] == pytest.approx(-2/3)
    ensemble_effect = roc_auc_score(y, noise.mean(axis=0)) - roc_auc_score(y, clean.mean(axis=0))
    assert ensemble_effect == -.5
    assert result["mean_checkpoint_delta_auc"] != ensemble_effect
    assert result["leave_one_subject_out"]["mean_delta_min"] is None
    assert result["leave_one_subject_out"]["unavailable"] == 2


def test_identical_scores_have_exact_zero_conditional_difference():
    clean, _, y, a, b = fixture()
    result = paired_effect(clean, clean.copy(), y, a, b, n_boot=20)
    assert result["mean_checkpoint_delta_auc"] == 0
    assert result["mean_checkpoint_ci95"] == [0, 0]
    assert "not training-seed population" in result["bootstrap"]["conditioning"]


def test_direction_and_leave_one_subject_extrema():
    clean, noise, y, a, b = fixture()
    result = paired_effect(clean, noise, y, a, b, n_boot=30)
    reverse = paired_effect(noise, clean, y, a, b, n_boot=30)
    assert result["mean_checkpoint_delta_auc"] == pytest.approx(-reverse["mean_checkpoint_delta_auc"])
    assert result["mean_checkpoint_ci95"] == pytest.approx([-v for v in reversed(reverse["mean_checkpoint_ci95"])])
    loo = []
    for s in np.unique(np.concatenate((a, b))):
        mask = (a != s) & (b != s)
        loo.append(np.mean([roc_auc_score(y[mask], n[mask]) - roc_auc_score(y[mask], c[mask])
                            for c, n in zip(clean, noise, strict=True)]))
    assert result["leave_one_subject_out"]["mean_delta_min"] == pytest.approx(min(loo))
    assert result["leave_one_subject_out"]["mean_delta_max"] == pytest.approx(max(loo))


@pytest.mark.parametrize("attack", ["shape", "seeds", "nan", "class", "floatlabel", "missing",
                                   "noisylabel", "samename", "bootstrap", "seed"])
def test_invalid_or_training_metadata_refused(attack):
    clean, noise, y, a, b = fixture()
    kwargs = dict(n_boot=5)
    if attack == "shape":
        noise = noise[:, :-1]
    elif attack == "seeds":
        clean, noise = clean[:2], noise[:2]
    elif attack == "nan":
        noise[0, 0] = np.nan
    elif attack == "class":
        y[:] = 1
    elif attack == "floatlabel":
        y = y.astype(float)
    elif attack == "missing":
        a = a.astype("U10")
        a[0] = "unknown"
    elif attack == "noisylabel":
        b[0] = "q0"
    elif attack == "samename":
        b[4] = a[4]
    elif attack == "bootstrap":
        kwargs["n_boot"] = 0
    elif attack == "seed":
        kwargs["seed"] = -1
    with pytest.raises(ValueError):
        paired_effect(clean, noise, y, a, b, **kwargs)


def bindings():
    required = ("src/age_gap/training/finetune.py", "src/age_gap/models/facenet.py",
                "scripts/train_matched_agegap_arms.py", "data/external/fgnet_crops.npz",
                "metrics/model_inventory.json", "data/interim/restricted_matched_arms_20261003/private/cross_arm.jsonl",
                "data/interim/faces/private.jpg", "cache/20180408-102900-casia-webface.pt")
    clean = {"inputs": [{"path": p, "bytes": 10, "sha256": "a" * 64} for p in required]}
    return clean, deepcopy(clean)


def test_common_source_crop_weight_contract():
    clean, noise = bindings()
    compare_common_inputs(clean, noise)
    noise["inputs"].append({"path": "scripts/train_partial_noise.py", "bytes": 5, "sha256": "b" * 64})
    compare_common_inputs(clean, noise)


@pytest.mark.parametrize("attack", ["mismatch", "missing", "crop", "weight", "duplicate", "bothweightsmissing"])
def test_mismatched_scientific_bindings_refused(attack):
    clean, noise = bindings()
    if attack == "mismatch":
        noise["inputs"][0]["sha256"] = "b" * 64
    elif attack == "missing":
        noise["inputs"].pop(0)
    elif attack == "crop":
        noise["inputs"].pop(-2)
    elif attack == "weight":
        noise["inputs"].pop(-1)
    elif attack == "duplicate":
        noise["inputs"].append(noise["inputs"][0] | {"sha256": "b" * 64})
    elif attack == "bothweightsmissing":
        noise["inputs"].pop(-1)
        clean["inputs"].pop(-1)
    with pytest.raises(ValueError):
        compare_common_inputs(clean, noise)


def test_pending_bound_campaign_never_creates_results(tmp_path):
    with pytest.raises(FileNotFoundError, match="completed"):
        run(tmp_path / "clean", tmp_path / "noise", tmp_path / "cm", tmp_path / "nm", tmp_path / "out")
    assert not (tmp_path / "out").exists()
    with pytest.raises(FileNotFoundError):
        load_binding(tmp_path / "missing", "expected")


def test_extra_clean_source_or_configuration_must_be_shared():
    clean, noise = bindings()
    for suffix in (".py", ".toml"):
        record = {"path": "additional_source" + suffix, "bytes": 5, "sha256": "a" * 64}
        clean["inputs"].append(record)
        with pytest.raises(ValueError, match="source/config"):
            compare_common_inputs(clean, noise)
        noise["inputs"].append(record.copy())
        compare_common_inputs(clean, noise)


def test_pending_cli_is_explicit_without_fake_ci(tmp_path, monkeypatch, capsys):
    import scripts.evaluate_clean_noise as module

    args = ["evaluate_clean_noise", "--clean", str(tmp_path / "clean"), "--noise", str(tmp_path / "noise"),
            "--out", str(tmp_path / "out")]
    monkeypatch.setattr("sys.argv", args)
    module.main()
    output = capsys.readouterr().out
    assert "pending_completed_bound_campaigns" in output
    assert "ci95" not in output and not (tmp_path / "out").exists()
    monkeypatch.setattr("sys.argv", args + ["--execute"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
