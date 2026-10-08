import numpy as np
import pytest

from scripts.audit_oriented_nuisance import endpoint_image_tv, verified


def test_uniform_exposure_preservation_not_weighted():
    paired = [
        ({"face_a": "a", "face_b": "b"}, {"face_a": "a", "face_b": "d"}),
        ({"face_a": "c", "face_b": "d"}, {"face_a": "c", "face_b": "b"}),
    ]
    assert endpoint_image_tv(paired, [1.0, 1.0], "b") == 0.0
    assert endpoint_image_tv(paired, [3.0, 1.0], "b") == 0.5
    assert endpoint_image_tv(paired, [3.0, 1.0], "a") == 0.0


@pytest.mark.parametrize("weights", [[0.0, 0.0], [-1.0, 1.0], [np.nan, 1.0], [1.0]])
def test_invalid_mass(weights):
    with pytest.raises(ValueError):
        endpoint_image_tv([({}, {}), ({}, {})], weights, "b")


def test_wrong_prerequisite_fails_before_source_access(tmp_path):
    p = tmp_path / "summary.manifest.json"
    p.write_text('{"experiment":"other","metrics":{"execution_complete":true}}', encoding="utf-8")
    with pytest.raises(ValueError, match="correctly typed"):
        verified(p, tmp_path, "required")
