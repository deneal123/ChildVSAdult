"""Draft governance promises must not contradict the English manuscript.

Text-level regression only: no legal review, release audit or human age validation.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_governance_does_not_offer_private_biometric_derivatives():
    text = (ROOT / "docs/DATA_GOVERNANCE.md").read_text(encoding="utf-8")
    assert "не предлагаются для доступа" in text
    assert "построчные протоколы" in text
    assert "черновики, а не" in text
    assert "псевдонимы, не анонимизация" in text
    assert "обратная связь с VK-профилем невозможна" not in text
    assert "код, конфиги, веса обученной модели" not in text


def test_governance_discloses_retrospective_minors_and_pending_review():
    text = (ROOT / "docs/DATA_GOVERNANCE.md").read_text(encoding="utf-8")
    assert "уже использовались в агрегированном" in text
    assert "нельзя утверждать, что они исключены из исследовательского корпуса" in text
    assert "последующее решение нельзя называть предварительным" in text
    assert "не являются завершённым или одобренным DPIA" in text
    assert "Пропущенный возраст нельзя считать доказательством совершеннолетия" in text
    assert "8 201 minor-лицо исключено" not in text
    assert "Опора — научно-исследовательский режим" not in text


def test_english_master_has_matching_release_and_review_limits():
    text = (ROOT / "latex/papers/journal-1-tbiom/en/main.tex").read_text(encoding="utf-8")
    assert "images depicting minors were included in aggregate research analyses" in text
    assert "no trained weights, embeddings, crops, captions" in text
    assert "salted hashes are pseudonyms rather than anonymization" in text
    assert "formal retrospective determination" in text
