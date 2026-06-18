"""Тесты сопоставления возраста лицам (позиционный маппинг, TODO §8 правило 2)."""

from __future__ import annotations

from age_gap.common.schemas import AgeLabel, FaceCrop
from age_gap.datasets.age_anchors import RegexAgeExtractor
from age_gap.datasets.identity_groups import _map_ages


def _face(photo_id: str) -> FaceCrop:
    return FaceCrop(face_id=f"{photo_id}_f0", photo_id=photo_id, is_usable=True)


def _seq(*photo_ids: str) -> dict[str, int]:
    """seq_by_photo для полного списка фото поста."""
    return {pid: i for i, pid in enumerate(photo_ids)}


def test_singles_preserve_caption_order():
    ages = [lbl.age for lbl in RegexAgeExtractor().extract("здесь 16 лет, а здесь уже 26 лет")]
    assert ages == [16, 26]


def test_positional_mapping_when_counts_match():
    faces = [_face("p0"), _face("p1")]
    seq = _seq("p0", "p1")
    labels = _map_ages("здесь 16 лет, а здесь 26 лет", faces, seq, 2, RegexAgeExtractor())
    by_face = {lbl.face_id: lbl.age for lbl in labels}
    assert by_face == {"p0_f0": 16, "p1_f0": 26}
    assert all(lbl.face_id is not None for lbl in labels)


def test_no_positional_mapping_when_counts_differ():
    faces = [_face("p0"), _face("p1"), _face("p2")]  # 3 лица, 2 возраста
    seq = _seq("p0", "p1", "p2")
    labels = _map_ages("16 и 26 лет", faces, seq, 3, RegexAgeExtractor())
    assert all(lbl.face_id is None for lbl in labels)


def test_explicit_left_right_still_maps():
    faces = [_face("p0"), _face("p1")]
    seq = _seq("p0", "p1")
    labels = _map_ages("слева 5 лет, справа 30", faces, seq, 2, RegexAgeExtractor())
    by_face = {lbl.face_id: lbl.age for lbl in labels}
    assert by_face.get("p0_f0") == 5
    assert by_face.get("p1_f0") == 30


class _FakeExtractor:
    """Экстрактор, отдающий заранее заданные метки (имитирует LLM с position_N)."""

    def __init__(self, labels):
        self._labels = labels
        self.last_n_photos = None

    def extract(self, caption, n_photos=None):  # noqa: ANN001
        self.last_n_photos = n_photos
        return list(self._labels)


def test_position_n_reference_maps_to_face():
    faces = [_face("p0"), _face("p1"), _face("p2")]
    seq = _seq("p0", "p1", "p2")
    ext = _FakeExtractor(
        [
            AgeLabel(age=8, photo_reference="position_0", source="llm", mapping_confidence=0.6),
            AgeLabel(age=35, photo_reference="position_2", source="llm", mapping_confidence=0.6),
        ]
    )
    labels = _map_ages("сложная подпись", faces, seq, 3, ext)
    by_face = {lbl.face_id: lbl.age for lbl in labels}
    assert by_face.get("p0_f0") == 8
    assert by_face.get("p2_f0") == 35


def test_position_out_of_range_not_mapped():
    faces = [_face("p0"), _face("p1")]
    seq = _seq("p0", "p1")
    ext = _FakeExtractor([AgeLabel(age=8, photo_reference="position_5", source="llm")])
    labels = _map_ages("x", faces, seq, 2, ext)
    assert labels[0].face_id is None


def test_position_aligns_to_photo_seq_when_middle_face_rejected():
    # В посте 3 фото (p0,p1,p2), но лицо p1 отбраковано -> usable: p0, p2.
    # position_2 (возраст для 3-го фото) должен привязаться к лицу p2, НЕ к p1/смещению.
    faces = [_face("p0"), _face("p2")]
    seq = _seq("p0", "p1", "p2")  # полный список фото, total=3
    ext = _FakeExtractor(
        [
            AgeLabel(age=10, photo_reference="position_0", source="llm"),
            AgeLabel(age=40, photo_reference="position_2", source="llm"),
        ]
    )
    labels = _map_ages("в детстве и сейчас", faces, seq, 3, ext)
    by_face = {lbl.face_id: lbl.age for lbl in labels}
    assert by_face.get("p0_f0") == 10
    assert by_face.get("p2_f0") == 40
    # position_1 отсутствует среди меток; p2 получил верный возраст (а не возраст position_1).


def test_total_photos_passed_to_extractor():
    faces = [_face("p0"), _face("p2")]
    seq = _seq("p0", "p1", "p2")
    ext = _FakeExtractor([])
    _map_ages("x", faces, seq, 3, ext)
    assert ext.last_n_photos == 3  # передаётся ОБЩЕЕ число фото, не число usable-лиц (2)
