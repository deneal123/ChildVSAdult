"""Тесты обезличивания релиза (DATA_GOVERNANCE §4) — чистая логика, без файлов."""

from __future__ import annotations

from age_gap.common.schemas import IdentityGroup, Pair
from age_gap.datasets.anonymize import age_bucket, anon_group, anon_pair, hash_id

_SALT = b"test-salt-32-bytes-long-padding!!"
_RAW = "-77072632_457318692_f0"


def test_hash_id_deterministic_and_opaque():
    h1 = hash_id(_RAW, _SALT)
    h2 = hash_id(_RAW, _SALT)
    assert h1 == h2  # детерминирован
    assert h1 != hash_id(_RAW, b"other-salt-32-bytes-long-pad!!!!")  # зависит от соли
    assert h1 != hash_id("-77072632_457318693_f0", _SALT)  # разные входы → разные
    # необратимость на уровне строки: ни одной сырой подстроки (hex16, без '-'/'_')
    assert all(m not in h1 for m in ("-77072632", "_f", "-"))
    assert len(h1) == 16 and all(c in "0123456789abcdef" for c in h1)


def test_age_bucket():
    assert age_bucket(None) == "unknown"
    assert age_bucket(5) == "0-17"
    assert age_bucket(17) == "0-17"
    assert age_bucket(18) == "18-29"
    assert age_bucket(40) == "30-44"
    assert age_bucket(70) == "45+"


def test_anon_pair_excludes_minor_and_hides_raw():
    p = Pair(
        pair_id=f"pos_{_RAW}__x",
        face_a=_RAW,
        face_b="-1_2_f1",
        label=1,
        pair_type="positive_same_post",
        age_gap=20,
    )
    out = anon_pair(p, _SALT, exclude_faces=set())
    assert out is not None
    # ни одного сырого идентификатора в выводе
    blob = str(out)
    assert all(m not in blob for m in ("-77072632", "_f0", "_f1"))
    assert out["label"] == 1 and out["age_gap"] == 20
    # пара, задевающая несовершеннолетнее лицо, исключается
    assert anon_pair(p, _SALT, exclude_faces={_RAW}) is None


def test_anon_group_drops_minor_faces():
    g = IdentityGroup(
        identity_group_id="vk_-77072632_1",
        source_post_id="vk_-77072632_1",
        faces=["-77072632_a_f0", "-77072632_b_f0"],
        apparent_gender="F",
        apparent_age=10,
    )
    # одно лицо несовершеннолетнее → остаётся одно, группа не None
    out = anon_group(g, _SALT, minor_faces={"-77072632_a_f0"})
    assert out is not None and len(out["faces"]) == 1
    assert out["apparent_age_bucket"] == "0-17"
    assert all("-77072632" not in f for f in out["faces"])
    # все лица несовершеннолетние → None (группа исключается целиком)
    assert anon_group(g, _SALT, minor_faces={"-77072632_a_f0", "-77072632_b_f0"}) is None
