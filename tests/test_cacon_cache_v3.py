import copy
import json

import cv2
import numpy as np
import pytest

from age_gap.common.manifest import file_record
from scripts.cacon_cache_v3 import load_bound_three_views, validate_rows


@pytest.fixture
def fixture(tmp_path):
    faces_path = tmp_path / "faces" / "summary.manifest.json"
    faces_path.parent.mkdir()
    faces_path.write_text("{}", encoding="utf-8")
    weights, code = tmp_path / "generator.pt", tmp_path / "generator.py"
    weights.write_bytes(b"test-only-weights-not-real-generator")
    code.write_text("# test-only\n", encoding="utf-8")
    bins = [[16, 20], [21, 25], [26, 30]]
    groups = np.random.Generator(np.random.PCG64(42)).integers(0, 3, size=3).tolist()
    face_rows, rows = [], []
    for i, group in enumerate(groups):
        source, generated = tmp_path / f"source{i}.png", tmp_path / f"generated{i}.png"
        assert cv2.imwrite(str(source), np.full((16, 16, 3), i + 50, dtype=np.uint8))
        assert cv2.imwrite(str(generated), np.full((16, 16, 3), i + 100, dtype=np.uint8))
        face_rows.append(dict(face_id=f"face{i}", crop_path=str(source), age=18 + 5 * i))
        rows.append(dict(face_id=f"face{i}", source=file_record(source), generated=file_record(generated),
                         target_group=group, target_age_bin=bins[group], draw_index=i,
                         target_reference=None, target_reference_face_id=None))
    native = dict(
        experiment="cacon-three-view-generated-cache", command=["synthetic-test-fixture"],
        inputs=[file_record(faces_path), file_record(weights), file_record(code), *[r["source"] for r in rows]],
        outputs=[r["generated"] for r in rows],
        parameters=dict(seed=42, sampling="continuing-PCG64-uniform-group-with-replacement", target_age_bins=bins,
                        producer=dict(family="synthetic-test-only", version="fixture-v1", status="adapted",
                                      conditioning="label-only-adaptation",
                                      reference_url="https://arxiv.org/html/2312.11195v2#S2.SS2",
                                      weights=[file_record(weights)], implementation=[file_record(code)])),
        metrics=dict(cache_complete=True, training_complete=False, full_method_parity=False, generated_images=3),
    )
    return rows, face_rows, native, faces_path


def test_full_declared_cache_rows_do_not_claim_generator_fidelity(fixture):
    rows, faces, native, path = fixture
    records = validate_rows(rows, faces, native, file_record(path))
    assert [r.face_id for r in records] == [r["face_id"] for r in faces]


@pytest.mark.parametrize("kind", ["parity", "training", "incomplete", "no_command", "no_face_binding", "weights", "code", "version", "status", "sampling", "seed", "bins", "coverage", "order", "draw_float", "target_group", "age_bin", "source", "output", "duplicate"])
def test_invalid_lineage_refused(fixture, kind):
    rows, faces, native, path = copy.deepcopy(fixture)
    if kind == "parity":
        native["metrics"]["full_method_parity"] = True
    elif kind == "training":
        native["metrics"]["training_complete"] = True
    elif kind == "incomplete":
        native["metrics"]["cache_complete"] = False
    elif kind == "no_command":
        native["command"] = []
    elif kind == "no_face_binding":
        native["inputs"] = native["inputs"][1:]
    elif kind == "weights":
        native["parameters"]["producer"]["weights"] = []
    elif kind == "code":
        native["parameters"]["producer"]["implementation"] = []
    elif kind == "version":
        native["parameters"]["producer"]["version"] = "latest"
    elif kind == "status":
        native["parameters"]["producer"]["status"] = "verified-author-parity"
    elif kind == "sampling":
        native["parameters"]["sampling"] = "unknown"
    elif kind == "seed":
        native["parameters"]["seed"] = True
    elif kind == "bins":
        native["parameters"]["target_age_bins"][0] = [10, 20]
    elif kind == "coverage":
        rows.pop()
    elif kind == "order":
        rows.reverse()
    elif kind == "draw_float":
        rows[0]["draw_index"] = 0.0
    elif kind == "target_group":
        rows[0]["target_group"] = 6
    elif kind == "age_bin":
        rows[0]["target_age_bin"] = [0, 4]
    elif kind == "source":
        rows[0]["source"] = rows[1]["source"]
    elif kind == "output":
        native["outputs"] = []
    elif kind == "duplicate":
        rows[1]["face_id"] = rows[0]["face_id"]
    with pytest.raises(ValueError):
        validate_rows(rows, faces, native, file_record(path))


def test_loading_bound_cache_keeps_fidelity_false(fixture, tmp_path, monkeypatch):
    import scripts.cacon_cache_v3 as module

    rows, face_rows, cache, faces_path = fixture
    private = faces_path.parent / "private"
    private.mkdir()
    face_rows_path = private / "faces.jsonl"
    face_rows_path.write_text("".join(json.dumps(row) + "\n" for row in face_rows), encoding="utf-8")
    faces = dict(inputs=[row["source"] for row in rows], outputs=[file_record(face_rows_path)])
    monkeypatch.setattr(module, "load_bound_dataset", lambda *a: (None, faces))
    index = tmp_path / "cache_rows.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cache["parameters"]["index"] = file_record(index)
    cache["outputs"].append(file_record(index))
    path = tmp_path / "cache.manifest.json"
    path.write_text(json.dumps(cache), encoding="utf-8")
    dataset, audit = load_bound_three_views(
        path, faces_path, lambda image: image.transpose(2, 0, 1).astype(np.float32), np.random.default_rng(99),
    )
    assert len(dataset) == 3 and audit["cache_lineage_verified"]
    assert dataset.generator_provenance_verified is False
    assert not audit["generator_provenance_verified"] and not audit["full_method_parity"]
    first = dataset[0]
    assert len(first) == 3 and first[2].shape == (3, 16, 16)
    assert first[2].mean().item() == 100
    generated = module.resolve(rows[0]["generated"]["path"])
    generated.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        dataset[0]


def test_target_image_conditioning_binds_age_reference(fixture):
    rows, faces, native, path = fixture
    producer = native["parameters"]["producer"]
    producer["conditioning"] = "target-image-and-label"
    producer["status"] = "reference-reproduction-unverified"
    sources = [row["source"] for row in rows]
    for row in rows:
        group = row["target_group"]
        row["target_reference"] = sources[group]
        row["target_reference_face_id"] = faces[group]["face_id"]
    assert len(validate_rows(rows, faces, native, file_record(path))) == 3
    faces[0]["age"] = None
    with pytest.raises(ValueError, match="source age"):
        validate_rows(rows, faces, native, file_record(path))


def test_label_only_cannot_claim_reference_reproduction(fixture):
    rows, faces, native, path = fixture
    native["parameters"]["producer"]["status"] = "reference-reproduction-unverified"
    with pytest.raises(ValueError, match="label-only"):
        validate_rows(rows, faces, native, file_record(path))


def test_target_label_can_record_unverified_reference_reproduction(fixture):
    rows, faces, native, path = fixture
    native["parameters"]["producer"].update(
        conditioning="target-label", status="reference-reproduction-unverified",
    )
    assert len(validate_rows(rows, faces, native, file_record(path))) == 3
    assert native["metrics"]["full_method_parity"] is False


@pytest.mark.parametrize("field", ["target_reference", "target_reference_face_id"])
def test_target_label_refuses_inference_reference(fixture, field):
    rows, faces, native, path = fixture
    native["parameters"]["producer"]["conditioning"] = "target-label"
    rows[0][field] = rows[0]["source"] if field == "target_reference" else faces[0]["face_id"]
    with pytest.raises(ValueError, match="conceal target-image"):
        validate_rows(rows, faces, native, file_record(path))
