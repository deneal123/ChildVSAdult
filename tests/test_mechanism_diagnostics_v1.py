import numpy as np
import pytest

from scripts.mechanism_diagnostics_v1 import age_probe, drift, paired_age_error


def fixture():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(60, 4))
    return x, 30 + 5 * x[:, 0], np.repeat(np.arange(20), 3)


def test_person_disjoint_and_fit_only_scaling():
    x, y, p = fixture()
    result = age_probe(x, y, p)
    assert result["status"] == "ok"
    for ledger in result["fold_ledger"]:
        assert not set(ledger["fit_persons"]) & set(ledger["test_persons"])
        train = np.isin(p.astype(str), ledger["fit_persons"])
        np.testing.assert_allclose(ledger["fit_mean"], x[train].mean(axis=0))
    for person in set(result["persons"]):
        assert len({f for g, f in zip(result["persons"], result["folds"], strict=True)
                    if g == person}) == 1


def test_signal_beats_constant():
    x, y, p = fixture()
    result = age_probe(x, y, p)
    assert result["mae_person"] < .3
    assert result["mae_image"] < result["mean_baseline_mae_image"]


def test_missing_metadata_disclosed():
    x, y, p = fixture()
    y[0] = np.nan
    people = p.astype(object)
    people[1] = None
    result = age_probe(x, y, people)
    assert result["n_missing_age"] == result["n_missing_person"] == 1
    assert result["n_scored"] == 58
    assert 0 not in result["row_indices"] and 1 not in result["row_indices"]


def test_no_person_fallback():
    x, y, _ = fixture()
    assert age_probe(x, y, [None] * 60)["status"].startswith("unavailable")


def test_paired_identity_and_identical_errors():
    result = age_probe(*fixture())
    paired = paired_age_error(result, result, resamples=20)
    assert paired["ci95"] == [0, 0]
    changed = {**result, "folds": [9] * len(result["folds"])}
    with pytest.raises(ValueError):
        paired_age_error(result, changed)


def test_cka_invariance_and_drift():
    x, _, _ = fixture()
    assert drift(x, x)["linear_cka"] == pytest.approx(1)
    assert drift(x, -x)["linear_cka"] == pytest.approx(1)
    assert drift(x, x)["mean_cosine_drift"] == pytest.approx(0, abs=1e-14)


def test_nonfinite_and_shape_rejected():
    x, y, p = fixture()
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        age_probe(x, y, p)
    with pytest.raises(ValueError):
        drift(np.ones((2, 3)), np.ones((3, 3)))


def test_embedding_cache_rejects_permuted_indices(tmp_path):
    from scripts.run_mechanism_diagnostics_v1 import aligned_vectors

    vectors = np.zeros((3, 512))
    vectors[:, 0] = 1
    path = tmp_path / "cache.npz"
    np.savez(path, indices=[1, 0, 2], embeddings=vectors)
    with pytest.raises(ValueError, match="index order"):
        aligned_vectors(path, 3)


def test_embedding_cache_rejects_nonunit_vectors(tmp_path):
    from scripts.run_mechanism_diagnostics_v1 import aligned_vectors

    path = tmp_path / "cache.npz"
    np.savez(path, indices=np.arange(3), embeddings=np.ones((3, 512)))
    with pytest.raises(ValueError, match="unit embeddings"):
        aligned_vectors(path, 3)
