import numpy as np
import pytest

from scripts.calfw_pixel_binding import build_index, check_pair_labels, match_endpoints, pixel_key


def image(value=0, shape=(3, 4, 3)):
    return np.full(shape, value, dtype=np.uint8)


def test_exact_pixel_identity_join_ignores_endpoint_order():
    index = build_index([('Person_A_0001.jpg', image()), ('Person_B_0001.jpg', image(1))])
    mapping, summary = match_endpoints(index, [image(1), image(), image(1)])
    assert mapping == ['Person_B', 'Person_A', 'Person_B']
    assert summary['exact_pixel_person_coverage_complete']
    assert not summary['source_provenance_verified_by_this_helper']
    assert not summary['subject_ci_available']
    assert 'Person_A' not in str(summary)


def test_cross_person_pixel_alias_is_ambiguous_not_first_match():
    index = build_index([('Person_A_0001.jpg', image()), ('Person_B_0001.jpg', image())])
    mapping, summary = match_endpoints(index, [image()])
    assert mapping == [None]
    assert summary['ambiguous_person_endpoints'] == 1
    assert not summary['exact_pixel_person_coverage_complete']


def test_same_person_image_alias_can_resolve_person_not_image():
    index = build_index([('Person_A_0001.jpg', image()), ('Person_A_0002.jpg', image())])
    mapping, summary = match_endpoints(index, [image()])
    assert mapping == ['Person_A']
    assert summary['same_person_image_alias_endpoints'] == 1


def test_unknown_endpoint_stays_unknown():
    index = build_index([('Person_A_0001.jpg', image())])
    mapping, summary = match_endpoints(index, [image(2)])
    assert mapping == [None] and summary['unmatched_endpoints'] == 1


def test_conflicting_name_pixels_rejected():
    with pytest.raises(ValueError, match='conflicting'):
        build_index([('Person_A_0001.jpg', image()), ('Person_A_0001.jpg', image(1))])


def test_hash_covers_shape_and_noncontiguous_pixels():
    assert pixel_key(image(shape=(3, 4, 3))) != pixel_key(image(shape=(4, 3, 3)))
    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    assert pixel_key(pixels[:, ::-1]) == pixel_key(pixels[:, ::-1].copy())


@pytest.mark.parametrize('pixels', [np.zeros((3, 4, 3)), np.zeros((3, 4), dtype=np.uint8),
                                   np.zeros((0, 4, 3), dtype=np.uint8)])
def test_invalid_pixels_rejected(pixels):
    with pytest.raises(ValueError):
        pixel_key(pixels)


def test_pair_check_missing_does_not_validate_and_conflicts_counted():
    out = check_pair_labels(['A', 'A', 'A', 'B', 'A', None, 'A', 'B'], np.array([1, 0, 1, 1]))
    assert out['resolved_pairs_checked'] == 3
    assert out['unresolved_pairs'] == 1 and out['identity_label_conflicts'] == 1
    assert not out['all_pair_labels_verified']


@pytest.mark.parametrize('labels', [np.array([1.]), np.array([2]), np.array([[1]])])
def test_invalid_labels_rejected(labels):
    with pytest.raises(ValueError):
        check_pair_labels(['A', 'A'], labels)


def test_all_resolved_pairs_consistent():
    out = check_pair_labels(['A', 'A', 'A', 'B'], np.array([True, False]))
    assert out['all_pair_labels_verified']


@pytest.mark.parametrize('bad', ['', 42, float('nan')])
def test_missing_other_endpoint_does_not_hide_invalid_person_token(bad):
    with pytest.raises(ValueError, match='person tokens'):
        check_pair_labels([bad, None], np.array([1]))


def test_empty_sources_and_empty_endpoints_rejected():
    with pytest.raises(ValueError):
        build_index([])
    with pytest.raises(ValueError):
        match_endpoints(build_index([('A_0001.jpg', image())]), [])
