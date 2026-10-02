from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from scripts.build_lfw_bound_cache import (
    construct,
    decode_bgr,
    parse_protocol,
    sha256_file,
    source_path,
    verify_binding,
)

LANDMARKS = [[87, 100], [164, 100], [125, 137], [96, 173], [155, 173]]


@pytest.fixture
def protocol(tmp_path):
    path = tmp_path / "pairs.txt"
    path.write_text("2 1\nAlpha 1 2\nAlpha 1 Beta 1\nBeta 1 2\nAlpha 2 Beta 2\n", encoding="utf-8")
    raw = tmp_path / "raw"
    for index, (person, instance) in enumerate((("Alpha", 1), ("Alpha", 2), ("Beta", 1), ("Beta", 2))):
        directory = raw / person
        directory.mkdir(parents=True, exist_ok=True)
        pixels = np.random.default_rng(index).integers(0, 256, (250, 250, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(directory / f"{person}_{instance:04d}.jpg")
    return path, raw


@pytest.fixture
def built(tmp_path, protocol):
    path, raw = protocol
    pairs, _, _ = parse_protocol(path)
    cache, rows, sources, summary = construct(tmp_path / "private", pairs, path, raw,
                                             lambda image: LANDMARKS)
    return cache, rows, sources, path, raw, summary


def rewrite_cache(cache: Path, **changes):
    with np.load(cache, allow_pickle=False) as z:
        values = {key: z[key] for key in z.files}
    values.update(changes)
    np.savez_compressed(cache, **values)


def test_official_contiguous_folds_and_named_images(protocol):
    path, _ = protocol
    pairs, folds, per_class = parse_protocol(path)
    assert (folds, per_class) == (2, 1)
    assert [p.fold for p in pairs] == [0, 0, 1, 1]
    assert [p.label for p in pairs] == [1, 0, 1, 0]
    assert pairs[0].image_a == "Alpha/Alpha_0001.jpg"


@pytest.mark.parametrize("content", [
    "1\nAlpha 1 2\n", "1 1\nAlpha 1 2\n", "0 1\n", "1 1\nAlpha 1 Beta 2\nAlpha 1 2\n",
    "1 1\n../Alpha 1 2\nAlpha 1 Beta 1\n",
    "1 1\nAlpha 1 1\nAlpha 1 Beta 1\n",
    "1 1\nAlpha 1 2\nAlpha 1 Alpha 2\n",
    "1 1\nAlpha 0 2\nAlpha 1 Beta 1\n",
    "1 1\nAlpha 1 2\nAlpha 1 Beta 1\n\n",
])
def test_refuses_malformed_protocol(tmp_path, content):
    path = tmp_path / "invalid.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_protocol(path)


def test_refuses_missing_or_escaped_source(protocol):
    _, raw = protocol
    with pytest.raises(ValueError):
        source_path(raw, "Alpha/missing.jpg")
    with pytest.raises(ValueError):
        source_path(raw, "../../pairs.txt")


def test_streaming_decoder_matches_local_sklearn_full_canvas(protocol):
    from sklearn.datasets._lfw import _load_imgs

    _, raw = protocol
    source = raw / "Alpha/Alpha_0001.jpg"
    official = _load_imgs([str(source)], slice_=None, color=True, resize=1.0)[0]
    expected = cv2.cvtColor((official * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    assert np.array_equal(decode_bgr(source), expected)


def test_constructed_cache_has_full_replayable_binding_and_no_legacy_claim(built):
    cache, rows, _, path, raw, summary = built
    verdict = verify_binding(cache, path, raw)
    assert verdict["all_endpoints_checked"] == 8
    assert verdict["unique_sources_checked"] == 4
    assert verdict["full_transform_replay"] is True
    assert verdict["training_identity_independence"] == "unverified"
    assert summary["legacy_cache_replaced"] is False
    first = json.loads(rows.read_text(encoding="utf-8").splitlines()[0])
    assert first["subject_a"] == first["subject_b"]


def test_rejects_modulo_fold_substitution(built):
    cache, _, _, path, raw, _ = built
    rewrite_cache(cache, fold_ids=np.arange(4) % 2)
    with pytest.raises(ValueError, match="contiguous folds"):
        verify_binding(cache, path, raw)


def test_equal_label_blocks_do_not_prove_pixel_row_binding(built):
    cache, _, _, path, raw, _ = built
    with np.load(cache, allow_pickle=False) as z:
        a = z["a"]
    a[[0, 2]] = a[[2, 0]]
    rewrite_cache(cache, a=a)
    with pytest.raises(ValueError, match="crop pixel mismatch"):
        verify_binding(cache, path, raw)


def test_rejects_person_mapping_even_if_metadata_checksum_is_rebound(built):
    cache, rows, _, path, raw, _ = built
    payload = [json.loads(line) for line in rows.read_text(encoding="utf-8").splitlines()]
    payload[0]["subject_a"] = "wrong-person"
    rows.write_text("\n".join(json.dumps(row) for row in payload) + "\n", encoding="utf-8")
    rewrite_cache(cache, rows_sha256=np.array(sha256_file(rows)))
    with pytest.raises(ValueError, match="person/image row mapping"):
        verify_binding(cache, path, raw)


def test_rejects_raw_image_changes(built):
    cache, _, _, path, raw, _ = built
    source = raw / "Alpha/Alpha_0001.jpg"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="raw source checksum"):
        verify_binding(cache, path, raw)


def test_rejects_bad_detection_rate(built):
    cache, _, _, path, raw, _ = built
    rewrite_cache(cache, miss_rate=np.float64(0.1))
    with pytest.raises(ValueError, match="misses exceed"):
        verify_binding(cache, path, raw)


def test_rejects_nonempty_output_and_preserves_previous_cache(built):
    cache, _, _, path, raw, _ = built
    original = sha256_file(cache)
    pairs, _, _ = parse_protocol(path)
    with pytest.raises(FileExistsError):
        construct(cache.parent, pairs, path, raw, lambda image: LANDMARKS)
    assert sha256_file(cache) == original


def test_rejects_changed_recorded_transform_during_full_replay(built):
    cache, _, sources, path, raw, _ = built
    records = json.loads(sources.read_text(encoding="utf-8"))
    records["Alpha/Alpha_0001.jpg"]["landmarks"][0][0] += 10
    sources.write_text(json.dumps(records), encoding="utf-8")
    rewrite_cache(cache, sources_sha256=np.array(sha256_file(sources)))
    with pytest.raises(ValueError, match="transform replay mismatch"):
        verify_binding(cache, path, raw)
