
from __future__ import annotations

import types

import httpx
import pytest
from bs4 import BeautifulSoup

import src.risk_scoring as risk_scoring
from src.datamodels import LLMRiskResult, WarningLetterMetadata


def _metadata() -> WarningLetterMetadata:
    return WarningLetterMetadata(
        url="https://example.com/letter",
        posted_date="01/20/2025",
        issue_date_str="01/10/2025",
        company_name="Acme Pharma",
        issuing_office="CDER",
        subject="Sterility/Data Integrity",
    )


def _llm_result() -> LLMRiskResult:
    return LLMRiskResult(
        patient_safety_risk_score=0.9,
        systemic_quality_breakdown_score=0.8,
        data_integrity_risk_score=0.7,
        remediation_complexity_score=0.6,
        regulatory_escalation_risk_score=0.5,
        key_risk_drivers=["Sterility"],
        cited_findings=["Aseptic failures"],
        affected_quality_systems=["Sterility Assurance"],
        recommended_follow_up_actions=["Audit CAPA"],
        reasoning_summary="High concern.",
        confidence="HIGH",
        requires_human_review=True,
    )


def test_clean_block_filters_noise_and_normalizes_text() -> None:
    assert risk_scoring._clean_block(" FDA Ref 123 ") is None
    assert risk_scoring._clean_block("January 1, 2025") is None
    cleaned = risk_scoring._clean_block("  Important finding (b)(4)   here  ")
    assert cleaned == "Important finding here"


def test_dedupe_preserve_order() -> None:
    assert risk_scoring._dedupe_preserve_order(["a", "b", "a", "c"]) == ["a", "b", "c"]


def test_finalize_text_raises_for_empty_or_too_short() -> None:
    with pytest.raises(risk_scoring.LetterBodyExtractionError):
        risk_scoring._finalize_text([])

    with pytest.raises(risk_scoring.LetterBodyExtractionError):
        risk_scoring._finalize_text(["too short"])


def test_extract_blocks_with_prefix_and_stop_at_end() -> None:
    html = '''
    <article id="main-content">
        <p>Header text</p>
        <p>During an inspection of your firm, FDA observed major violations.</p>
        <p>Your sterility controls were inadequate.</p>
        <p>Sincerely,</p>
    </article>
    '''
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("article#main-content")
    assert container is not None

    blocks = risk_scoring._extract_blocks_with_prefix(container)

    assert blocks == [
        "During an inspection of your firm, FDA observed major violations.",
        "Your sterility controls were inadequate.",
    ]


def test_fetch_letter_body_extracts_main_text() -> None:
    html = '''
    <article id="main-content">
        <p><strong>Warning Letter</strong></p>
        <p>During an inspection of your firm, FDA observed major violations that created patient risk.</p>
        <p>Your sterility controls were inadequate and documentation was incomplete.</p>
        <p>Sincerely,</p>
    </article>
    '''

    class FakeResponse:
        text = html
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def get(self, url: str) -> FakeResponse:
            return FakeResponse()

    body = risk_scoring.fetch_letter_body(FakeClient(), "https://example.com")
    assert "During an inspection of your firm" in body
    assert "documentation was incomplete" in body
    assert "Sincerely" not in body


def test_calculate_final_risk_level_and_combine_scores() -> None:
    score = risk_scoring.combine_scores(_llm_result())
    assert score == 0.75
    assert risk_scoring.calculate_final_risk_level(0.9) == "CRITICAL"
    assert risk_scoring.calculate_final_risk_level(0.75) == "HIGH"
    assert risk_scoring.calculate_final_risk_level(0.6) == "MODERATE"
    assert risk_scoring.calculate_final_risk_level(0.4) == "ELEVATED"


def test_score_letter_uses_fetch_llm_and_subject_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(risk_scoring, "fetch_letter_body", lambda client, url: "x" * 100)
    monkeypatch.setattr(risk_scoring, "build_messages", lambda metadata, body_text: [{"role": "user", "content": body_text}])
    monkeypatch.setattr(risk_scoring, "call_llm_json", lambda messages: _llm_result())

    scored = risk_scoring.score_letter(client=types.SimpleNamespace(), metadata=_metadata())

    assert scored.metadata.company_name == "Acme Pharma"
    assert scored.subject_tokens == ["Sterility", "Data Integrity"]
    assert scored.final_risk_score == 0.75
    assert scored.final_risk_level == "HIGH"
