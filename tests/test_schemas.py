"""Round-trip сериализации схем (to_dict/from_dict)."""

from __future__ import annotations

import json

from age_gap.common.schemas import (
    AgeLabel,
    Comment,
    FaceCrop,
    IdentityGroup,
    Pair,
    Photo,
    Provenance,
    RawPost,
)


def _roundtrip(obj):
    cls = type(obj)
    return cls.from_dict(json.loads(json.dumps(obj.to_dict())))


def test_rawpost_roundtrip():
    post = RawPost(
        post_id="vk_-1_2",
        caption="мне 25",
        photos=[Photo(photo_id="p1", url="http://x/1.jpg", order=0)],
        comments=[Comment(comment_id="c1", text="похож", likes_count=3)],
        provenance=Provenance(origin_url="http://x", collection_date="2026-06-15"),
    )
    rt = _roundtrip(post)
    assert rt.post_id == "vk_-1_2"
    assert rt.photos[0].photo_id == "p1"
    assert rt.comments[0].text == "похож"
    assert rt.provenance.collection_date == "2026-06-15"


def test_facecrop_roundtrip():
    fc = FaceCrop(face_id="f1", photo_id="p1", bbox=[1, 2, 3, 4], is_usable=True, det_score=0.9)
    rt = _roundtrip(fc)
    assert rt.face_id == "f1" and rt.is_usable and rt.bbox == [1, 2, 3, 4]


def test_identity_group_roundtrip():
    g = IdentityGroup(
        identity_group_id="g1",
        source_post_id="g1",
        faces=["f1", "f2"],
        age_labels=[AgeLabel(face_id="f1", age=7, photo_reference="first")],
    )
    rt = _roundtrip(g)
    assert rt.faces == ["f1", "f2"]
    assert rt.age_labels[0].age == 7


def test_pair_roundtrip():
    p = Pair(
        pair_id="pos_f1__f2",
        face_a="f1",
        face_b="f2",
        label=1,
        pair_type="positive_same_post",
        age_a=7,
        age_b=31,
        age_gap=24,
        split="train",
    )
    rt = _roundtrip(p)
    assert rt.age_gap == 24 and rt.split == "train" and rt.label == 1
