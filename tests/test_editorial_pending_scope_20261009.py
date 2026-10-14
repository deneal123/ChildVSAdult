"""Draft factual guards only; no legal, ethical or release certification."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "latex/papers/journal-1-tbiom/submission"


def test_request_discloses_actual_caption_processing_and_api_review():
    text = (SUBMISSION / "ethics_request_hse.md").read_text(encoding="utf-8")
    assert "исключены из обработки" not in text
    assert "Подписи постов использованы для извлечения возраста" in text
    assert "передачу текста поставщику API" in text
    assert "VK и независимый Reddit-источник" in text


def test_request_does_not_offer_public_or_current_private_derivatives():
    text = (SUBMISSION / "ethics_request_hse.md").read_text(encoding="utf-8")
    assert "деперсонализированный протокол пар" not in text
    assert "Controlled access и DUA — черновики" in text
    assert "не предлагаются сейчас по запросу" in text
    assert "псевдонимизация, не подтверждённая анонимизация" in text


def test_request_distinguishes_research_minors_and_released_aggregates():
    text = (SUBMISSION / "ethics_request_hse.md").read_text(encoding="utf-8")
    assert "Агрегаты могут отражать эту категорию" in text
    assert "сами изображения и построчные" in text
    assert "не попадают ни в один публикуемый набор или иной артефакт" not in text


def test_portal_does_not_claim_future_determination_as_current_fact():
    text = (SUBMISSION / "portal_answers.md").read_text(encoding="utf-8")
    assert "was subsequently obtained" not in text
    assert "No formal institutional determination has yet been obtained" in text
    assert "replace this pending-status sentence" in text


def test_cover_scientific_and_institutional_gates_remain_pending():
    text = (SUBMISSION / "cover_letter.tex").read_text(encoding="utf-8")
    assert "Working draft, not submission-ready" in text
    assert "No formal institutional determination has yet been obtained" in text
    assert "full-method reproduction remains pending" in text
    assert "not a completed matrix" in text
    assert "larger three-seed bidirectional" not in text


def test_cover_does_not_claim_package_or_reproducibility_certification():
    text = (SUBMISSION / "cover_letter.tex").read_text(encoding="utf-8")
    assert "Final package disclosure, licenses and artifact availability must be verified" in text
    assert "External data alone do not make private-corpus-trained checkpoints available" in text
    assert "can be reproduced independently" not in text
    assert "not under review elsewhere" not in text
    assert "Before submission, all authors must confirm" in text
