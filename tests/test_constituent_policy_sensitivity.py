import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.audit_constituent_policy_sensitivity import sensitivity
from tests.test_constituent_integrity import fixture


def test_all_cached_single_is_distinct_from_known_noisy_exclusion():
    result = sensitivity(*fixture())
    assert result["pair_counts_after_any_noisy_group_exclusion"]["test_positive"] == 1
    assert result["pair_counts_after_all_cached_single_filter"]["test_positive"] == 0
    assert result["retained_groups_unresolved_after_known_noisy_exclusion"] == 1


def test_explicit_positive_group_denominators():
    post, mapping, pre, final, cache, pairs = fixture()
    pre[1]["identity_review"] = final[1]["identity_review"] = None
    result = sensitivity(post, mapping, pre, final, cache[:2], pairs)
    assert result["positive_groups"] == 2
    assert result["positive_group_representative_categories"] == {"missing": 1, "single": 1}
    assert result["positive_group_single_fraction_all"] == 0.5
    assert result["positive_group_single_fraction_cached"] == 1.0


@pytest.mark.parametrize("label", [0, 1])
def test_group_relationship_checked_per_class_not_universal_diagonal_guard(label):
    values = fixture()
    pairs = values[-1]
    if label == 1:
        pairs[0]["identity_group_b"] = "g3"
    else:
        pairs[1]["identity_group_b"] = "g3"
    with pytest.raises(ValueError):
        sensitivity(*values)


def test_empty_positive_group_denominator_yields_none_not_accuracy():
    values = fixture()
    result = sensitivity(*values[:-1], [])
    assert result["positive_group_single_fraction_all"] is None
    assert result["positive_group_single_fraction_cached"] is None


def test_sensitivity_does_not_mutate_records():
    values = fixture()
    original = deepcopy(values)
    sensitivity(*values)
    assert values == original


def test_real_sensitivity_is_bound_and_does_not_rebalance_or_retrain():
    root = Path(__file__).resolve().parents[1]
    directory = root / "metrics/constituent_policy_sensitivity_20261003"
    manifest = json.loads((directory / "summary.manifest.json").read_text(encoding="utf-8"))
    for record in manifest["inputs"] + manifest["outputs"]:
        path = (root / record["path"]).resolve()
        assert path.is_relative_to(root)
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == record["sha256"]
        assert path.stat().st_size == record["bytes"]
    result = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert manifest["metrics"] == result
    assert result["positive_groups"] == 13142
    assert result["positive_group_representative_categories"] == {"missing": 192, "single": 12839, "unknown": 111}
    assert result["retained_groups_unresolved_after_known_noisy_exclusion"] == 669
    assert result["pair_counts_after_any_noisy_group_exclusion"]["train_positive"] != result["pair_counts_after_any_noisy_group_exclusion"]["train_negative"]
    assert "no true-identity purity or retrained effect" in result["scope"]
    main = (root / "latex/papers/journal-1-tbiom/en/main.tex").read_text(encoding="utf-8")
    assert "12{,}839/13{,}142 positive groups" in main
    assert "97.7\\% overall; 99.1\\% among cached representative decisions" in main
    supplement = (root / "latex/papers/journal-1-tbiom/en/supplement.tex").read_text(encoding="utf-8")
    assert "not a balanced training arm or a measured performance effect" in supplement
    for count in (17, 54, 669, 711, 204, 20741, 12839, 13142):
        assert f"{count:,}".replace(",", "{,}") in supplement
    assert "unknown 111 and missing 192" in supplement
