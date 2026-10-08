import pytest

from scripts.probe_facenet_head_cache_v2 import balanced_indices, class_counts


def test_balanced_source_order_and_unique_rows():
    labels = [1] * 40 + [0] * 40
    selected = balanced_indices(labels)
    assert selected[:4] == [0, 40, 1, 41]
    assert len(set(selected)) == 64
    counts = class_counts([labels[i] for i in selected])
    assert [(row["positive"], row["negative"]) for row in counts] == [
        (32, 32),
        (24, 24),
        (3, 3),
        (1, 0),
        (32, 32),
    ]
    assert labels == [1] * 40 + [0] * 40


@pytest.mark.parametrize("labels", [[1, 1], [0, 0], [1, 2], [True, 0], [1.0, 0], ["1", 0]])
def test_invalid_or_insufficient_labels(labels):
    with pytest.raises(ValueError):
        balanced_indices(labels, per_class=1)


@pytest.mark.parametrize("count", [True, 0, -1, 1.5])
def test_invalid_per_class(count):
    with pytest.raises(ValueError):
        balanced_indices([1, 0], per_class=count)


@pytest.mark.parametrize("sizes", [(0,), (3,), (True,), ()])
def test_invalid_schedule(sizes):
    with pytest.raises(ValueError):
        class_counts([1, 0], sizes)
