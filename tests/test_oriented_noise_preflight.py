from copy import deepcopy

import pytest

from age_gap.common.manifest import file_record
from scripts.prepare_oriented_noise_training import (
    require_crop_coverage,
    require_preflight,
    require_preparation,
)
from scripts.run_oriented_campaign import PROTOCOL


def preflight():
    return dict(
        parameters=deepcopy(PROTOCOL),
        metrics=dict(
            protocol=deepcopy(PROTOCOL), preflight_complete=True,
            crop_audit=dict(all_decodable=True, missing_rows_dropped=0),
        ),
    )


def test_frozen_preflight_contract():
    require_preflight(preflight())


@pytest.mark.parametrize("defect", ["protocol", "metrics_protocol", "pending", "decode", "dropped"])
def test_refuses_incompatible_preflight(defect):
    native = preflight()
    if defect == "protocol":
        native["parameters"]["batchnorm_policy"] = "adapt_all"
    elif defect == "metrics_protocol":
        native["metrics"]["protocol"]["batch_size"] = 32
    elif defect == "pending":
        native["metrics"]["preflight_complete"] = False
    elif defect == "decode":
        native["metrics"]["crop_audit"]["all_decodable"] = False
    else:
        native["metrics"]["crop_audit"]["missing_rows_dropped"] = 1
    with pytest.raises(ValueError):
        require_preflight(native)


def test_declared_permutation_and_clean_linkage(tmp_path):
    clean, noisy = tmp_path / "clean.jsonl", tmp_path / "noise.jsonl"
    clean.write_text("{}", encoding="utf-8")
    noisy.write_text("{}", encoding="utf-8")
    native = dict(parameters={"seed": 42}, inputs=[file_record(clean)], outputs=[file_record(noisy)])
    require_preparation(native, clean, noisy)
    for field in ("inputs", "outputs"):
        bad = deepcopy(native)
        bad[field] = []
        with pytest.raises(ValueError):
            require_preparation(bad, clean, noisy)
    native["parameters"]["seed"] = 1
    with pytest.raises(ValueError):
        require_preparation(native, clean, noisy)


def test_crop_coverage_requires_actual_unchanged_bytes(tmp_path):
    path = tmp_path / "face.jpg"
    path.write_bytes(b"fake decoded-file fixture, not an image")
    native = dict(inputs=[file_record(path)])
    assert require_crop_coverage(native, {"face"}, lambda _: path) == 1
    with pytest.raises(ValueError):
        require_crop_coverage(dict(inputs=[]), {"face"}, lambda _: path)
    path.write_bytes(b"changed bytes")
    with pytest.raises(ValueError):
        require_crop_coverage(native, {"face"}, lambda _: path)
