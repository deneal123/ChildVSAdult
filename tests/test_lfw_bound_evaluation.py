"""Statistical-contract tests; synthetic scores/crops, no downloads or face data."""
import hashlib
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts import evaluate_lfw_bound as ev


def fixture_vectors():
    labels = np.tile([1, 1, 0, 0], 3)
    folds = np.repeat(np.arange(3), 4)
    a = np.tile(["p", "q", "p", "r"], 3)
    b = np.tile(["p", "q", "q", "q"], 3)
    scores = np.array([.8, .4, .5, .1, .7, .3, .6, -.1, .9, .2, .4, 0])
    return scores, labels, folds, a, b


@pytest.mark.parametrize("weighted", [False, True])
def test_histogram_accuracy_equals_fixed_grid_brute_force(weighted):
    s, y, f, _, _ = fixture_vectors()
    w = np.arange(1, len(s) + 1) if weighted else np.ones(len(s))
    result = ev.official_fold_accuracy(s, y, f, w)
    for row in result["folds"]:
        train, test = f != row["fold"], f == row["fold"]
        correct = (s[train, None] >= ev.THRESHOLDS) == y[train, None]
        best = np.argmax((correct * w[train, None]).sum(axis=0))
        assert row["threshold"] == ev.THRESHOLDS[best]
        expected = np.average((s[test] >= ev.THRESHOLDS[best]) == y[test], weights=w[test])
        assert row["accuracy"] == pytest.approx(expected)
    assert result["accuracy"] == pytest.approx(np.mean([r["accuracy"] for r in result["folds"]]))


def test_heldout_scores_cannot_change_own_calibration_threshold():
    s, y, f, _, _ = fixture_vectors()
    before = ev.official_fold_accuracy(s, y, f)
    s[f == 0] = [-1, -.9, .99, 1]
    after = ev.official_fold_accuracy(s, y, f)
    assert before["folds"][0]["threshold"] == after["folds"][0]["threshold"]
    assert before["folds"][0]["accuracy"] != after["folds"][0]["accuracy"]


def test_auc_is_exact_and_low_far_ties_are_not_split():
    s = np.array([.90001, .90002, .90003, .1])
    y = np.array([0, 1, 0, 1])
    w = np.array([1, 3, 4, 2])
    assert ev.roc_points(s, y, w)["roc_auc"] == roc_auc_score(y, s, sample_weight=w)
    # A positive tied with a negative cannot be accepted at 1% FAR on two negatives.
    assert ev.roc_points([.8, .8, .7, .2], [1, 0, 1, 0])["tar@far=0.01"] == 0


def test_person_multiplicity_not_squared_for_positive_pairs():
    assert np.array_equal(ev.subject_weights([1, 0], np.array([3, 2]),
                                             np.array([0, 0]), np.array([0, 1])), [3, 6])


def test_shared_draws_identical_models_have_exact_zero_gain_and_are_deterministic():
    s, y, f, a, b = fixture_vectors()
    models = {"frozen": s, "tuned_seed1": s.copy(), "tuned_seed2": s.copy()}
    result = ev.paired_statistics(models, y, f, a, b, n_boot=40, seed=17)
    assert result == ev.paired_statistics(models, y, f, a, b, n_boot=40, seed=17)
    for metric in ev.METRICS:
        assert result["models"]["tuned_seed1"]["paired_gain"][metric]["subject_ci95"] == [0, 0]
        assert result["seed_aggregate"][metric]["fixed_checkpoint_mean_gain_subject_ci95"] == [0, 0]
    assert result["bootstrap"]["n_valid_accuracy"] < 40
    assert result["bootstrap"]["folds_are_subject_disjoint"] is False
    assert result["bootstrap"]["includes_training_seed_population_uncertainty"] is False


def test_accuracy_calibration_is_repeated_for_each_eligible_draw(monkeypatch):
    original = ev.official_fold_accuracy
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[3] if len(args) > 3 else None)
        return original(*args, **kwargs)

    monkeypatch.setattr(ev, "official_fold_accuracy", observed)
    s, y, f, a, b = fixture_vectors()
    result = ev.paired_statistics({"frozen": s, "tuned_seed42": s}, y, f, a, b, n_boot=20)
    assert len(calls) == 2 * (1 + result["bootstrap"]["n_valid_accuracy"])
    assert all(value is not None for value in calls[2:])


def test_fold_empty_draws_are_excluded_only_from_accuracy():
    y, f = np.array([1, 0, 1, 0]), np.array([0, 0, 1, 1])
    a, b = ["a", "a", "c", "c"], ["a", "b", "c", "d"]
    result = ev.paired_statistics({"frozen": np.array([.8, .2, .7, .1]),
                                   "tuned": np.array([.8, .2, .7, .1])}, y, f, a, b,
                                  n_boot=100, seed=23)
    assert result["bootstrap"]["n_valid_roc"] > result["bootstrap"]["n_valid_accuracy"] > 0
    assert result["bootstrap"]["folds_are_subject_disjoint"] is True


def test_fixed_checkpoint_mean_auc_ci_matches_independent_shared_draw_calculation():
    s, y, f, a, b = fixture_vectors()
    models = {"frozen": s, "tuned_seed1": np.roll(s, 1), "tuned_seed2": -s}
    result = ev.paired_statistics(models, y, f, a, b, n_boot=40, seed=19)
    people = sorted(set(a) | set(b))
    rng, differences = np.random.default_rng(19), []
    for _ in range(40):
        counts = np.bincount(rng.integers(len(people), size=len(people)), minlength=len(people))
        by_person = dict(zip(people, counts, strict=True))
        weights = np.array([by_person[left] if label == 1 else by_person[left] * by_person[right]
                            for label, left, right in zip(y, a, b, strict=True)])
        if min(weights[y == 0].sum(), weights[y == 1].sum()) == 0:
            continue
        values = {key: roc_auc_score(y, score, sample_weight=weights) for key, score in models.items()}
        differences.append((values["tuned_seed1"] + values["tuned_seed2"]) / 2 - values["frozen"])
    expected = np.percentile(differences, [2.5, 97.5])
    assert result["seed_aggregate"]["roc_auc"]["fixed_checkpoint_mean_gain_subject_ci95"] == pytest.approx(expected)


def test_seed_mean_is_mean_of_metrics_not_score_ensemble():
    s, y, f, a, b = fixture_vectors()
    models = {"frozen": s, "tuned_seed1": s, "tuned_seed2": -s}
    result = ev.paired_statistics(models, y, f, a, b, n_boot=30)
    for metric in ev.METRICS:
        per_seed = [result["models"][key]["metrics"][metric] for key in models if key != "frozen"]
        assert result["seed_aggregate"][metric]["mean"] == np.mean(per_seed)
    assert result["seed_aggregate"]["eer"]["mean"] != ev.roc_points(np.zeros(len(s)), y)["eer"]


@pytest.mark.parametrize("bad", [None, "", 123])
def test_missing_person_metadata_is_rejected(bad):
    s, y, f, a, b = fixture_vectors()
    a, b = a.astype(object), b.astype(object)
    a[0] = b[0] = bad
    with pytest.raises(ValueError, match="metadata"):
        ev.paired_statistics({"frozen": s, "tuned": s}, y, f, a, b)


def test_person_class_conflict_rejected():
    s, y, f, a, b = fixture_vectors()
    b[0] = "z"
    with pytest.raises(ValueError, match="contradicts"):
        ev.paired_statistics({"frozen": s, "tuned": s}, y, f, a, b)


@pytest.mark.parametrize("case", ["nan", "range", "labels", "folds", "float_folds", "weights"])
def test_invalid_vectors_rejected(case):
    s, y, f, _, _ = fixture_vectors()
    w = np.ones(len(s))
    if case == "nan":
        s[0] = np.nan
    elif case == "range":
        s[0] = 1.5
    elif case == "labels":
        y = y.astype(float)
        y[0] = .5
    elif case == "folds":
        f += 1
    elif case == "float_folds":
        f = f.astype(float)
    else:
        w[0] = -1
    with pytest.raises(ValueError):
        ev.official_fold_accuracy(s, y, f, w)


def test_empty_weighted_class_fails_accuracy_not_silent_nan():
    s, y, f, _, _ = fixture_vectors()
    w = np.ones(len(s))
    w[(f == 0) & (y == 0)] = 0
    with pytest.raises(ValueError, match="retain both"):
        ev.official_fold_accuracy(s, y, f, w)


def test_unverified_cache_refused_before_replay(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "verified_result", lambda *args: {"row_binding_verified": True, "full_transform_replay": False})
    monkeypatch.setattr(ev, "verify_binding", lambda *args: pytest.fail("unverified cache must not reach replay"))
    with pytest.raises(ValueError, match="completed full"):
        ev.load_bound(tmp_path / "summary.json", tmp_path, tmp_path / "pairs.txt", tmp_path)


@pytest.mark.parametrize("change", ["input", "output", "missing_manifest"])
def test_real_manifest_integrity_is_enforced_before_replay(monkeypatch, tmp_path, change):
    target, source = tmp_path / "summary.json", tmp_path / "source.txt"
    target.write_text(json.dumps({"row_binding_verified": True, "full_transform_replay": True}), encoding="utf-8")
    source.write_text("original input", encoding="utf-8")
    manifest = target.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"inputs": [ev.file_record(source)],
                                    "outputs": [ev.file_record(target)]}), encoding="utf-8")
    if change == "input":
        source.write_text("changed input", encoding="utf-8")
    elif change == "output":
        target.write_text("{}", encoding="utf-8")
    else:
        manifest.unlink()
    monkeypatch.setattr(ev, "verify_binding", lambda *args: pytest.fail("tampered manifest reached replay"))
    with pytest.raises((ValueError, FileNotFoundError)):
        ev.load_bound(target, tmp_path, tmp_path / "pairs.txt", tmp_path)


def test_fallback_sensitivity_is_explicit_descriptive_subset():
    rows = [{"image_a": "a", "image_b": "b"}, {"image_a": "c", "image_b": "b"},
            {"image_a": "b", "image_b": "c"}]
    sources = {"a": {"detected": False}, "b": {"detected": True}, "c": {"detected": True}}
    result = ev.detection_sensitivity({"frozen": [.9, .8, .1]}, [1, 1, 0], rows, sources)
    assert result["endpoint_fallback_count"] == 1
    assert result["pairs_with_any_fallback"] == 1
    assert result["both_detected_roc"]["frozen"]["roc_auc"] == 1
    assert "accuracy" not in result["both_detected_roc"]["frozen"]


def test_grid_hash_is_replayable():
    expected = hashlib.sha256(ev.THRESHOLDS.astype("<f8").tobytes()).hexdigest()
    assert expected == ev.THRESHOLD_SHA256
    assert ev.THRESHOLDS.flags.writeable is False
