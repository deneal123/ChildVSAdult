import numpy as np
import pytest

from scripts.benchmark_metrics_v2 import KEYS, infer
from scripts.fgnet_retrieval_study import embedding_cache_key
from scripts.reevaluate_fgnet_metrics_v2 import full_embeddings, protocol


def test_all_checkpoints_identical_gives_zero_mean_delta_not_ensemble():
    scores = {key: [.9, .8, .2, .3] for key in KEYS}
    result = infer(scores, [1, 1, 0, 0], [1, 2, 1, 2], [1, 2, 2, 3], n_boot=20)
    for metric in result["three_checkpoint_aggregate"].values():
        assert metric["mean_checkpoint_delta"] == 0
        assert metric["delta_ci95"] == [0, 0]
    assert result["bootstrap"]["positive_weight"] == "one shared person multiplicity"


def test_missing_checkpoint_and_invalid_labels_rejected():
    with pytest.raises(ValueError):
        infer({"frozen": [.9, .8]}, [1, 0], [1, 2], [1, 3], n_boot=1)
    with pytest.raises(ValueError):
        infer({key: [.9, .8] for key in KEYS}, [True, False], [1, 2], [1, 3], n_boot=1)


def test_reconstructed_protocol_balances_source_gap_strata():
    a, b, labels, gaps, coverage = protocol(np.array([1, 1, 2, 2]), np.array([1, 30, 1, 30], dtype=np.uint8))
    assert labels.tolist() == [1, 1, 0, 0]
    assert gaps.tolist() == [29] * 4
    assert coverage["retained_positives"] == 2
    assert len(a) == len(b) == 4


def test_reconstruction_matches_original_global_loader(tmp_path):
    from age_gap.evaluation.fgnet import load_pairs

    subjects = np.array([3, 1, 3, 2, 1, 2])
    ages = np.array([1, 1, 30, 1, 30, 30])
    source = tmp_path / "fake-crops.npz"
    # Each synthetic crop is its original index; no real images are read.
    np.savez(source, crops=np.arange(len(ages)), subjects=subjects, ages=ages)
    old_a, old_b, old_labels, old_gaps = load_pairs(source, protocol="endpoint_age_matched")
    a, b, labels, gaps, _ = protocol(subjects, ages)
    for reconstructed, original in zip((a, b, labels, gaps), (old_a, old_b, old_labels, old_gaps), strict=True):
        assert np.array_equal(reconstructed, original)


def cache(path, indices, vectors):
    key = embedding_cache_key(role="frozen_casia_webface", weights_sha256="weights", source_sha256="source", indices=np.array(indices))
    np.savez(path, indices=np.array(indices), embeddings=np.array(vectors, dtype=np.float32), key=key)


def test_full_cache_reordered_without_subset_substitution(tmp_path):
    path = tmp_path / "full.npz"
    cache(path, [1, 0], [[0., 1.], [1., 0.]])
    assert np.array_equal(full_embeddings(path, n_crops=2, role="frozen_casia_webface", weights_sha="weights", source_sha="source"), np.eye(2))


@pytest.mark.parametrize("indices,vectors", [([0], [[1., 0.]]), ([0, 0], [[1., 0.], [0., 1.]]),
                                          ([0, 1], [[2., 0.], [0., 1.]])])
def test_incomplete_aliased_or_nonunit_cache_rejected(tmp_path, indices, vectors):
    path = tmp_path / "bad.npz"
    cache(path, indices, vectors)
    with pytest.raises(ValueError):
        full_embeddings(path, n_crops=2, role="frozen_casia_webface", weights_sha="weights", source_sha="source")


def test_model_key_mismatch_rejected(tmp_path):
    path = tmp_path / "bad-model.npz"
    cache(path, [0, 1], [[1., 0.], [0., 1.]])
    with pytest.raises(ValueError):
        full_embeddings(path, n_crops=2, role="frozen_casia_webface", weights_sha="different", source_sha="source")
