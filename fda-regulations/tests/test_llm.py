
from __future__ import annotations

import types

import pytest

import src.llm as llm
from src.datamodels import WarningLetterMetadata


def _metadata() -> WarningLetterMetadata:
    return WarningLetterMetadata(
        url="https://example.com/letter",
        posted_date="01/20/2025",
        issue_date_str="01/10/2025",
        company_name="Acme Pharma",
        issuing_office="CDER",
        subject="Sterility/Data Integrity",
    )


def test_build_messages_contains_metadata_and_body() -> None:
    messages = llm.build_messages(_metadata(), "Body text here.")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "strict JSON only" in messages[0]["content"]
    assert "Acme Pharma" in messages[1]["content"]
    assert "Body text here." in messages[1]["content"]


def test_extract_json_text_returns_outermost_json_object() -> None:
    text = 'prefix ```json {"a": 1, "b": 2} ``` suffix'
    assert llm.extract_json_text(text) == '{"a": 1, "b": 2}'


def test_extract_json_text_raises_when_missing_json() -> None:
    with pytest.raises(ValueError):
        llm.extract_json_text("no json here")


def test_validate_llm_result_normalizes_values() -> None:
    payload = {
        "patient_safety_risk_score": 1.2,
        "systemic_quality_breakdown_score": 0.81,
        "data_integrity_risk_score": 0.55,
        "remediation_complexity_score": 0.4,
        "regulatory_escalation_risk_score": 0.1,
        "key_risk_drivers": [" Sterility ", "", 123],
        "cited_findings": [" Finding A "],
        "affected_quality_systems": [" CAPA ", None],
        "recommended_follow_up_actions": [" Audit ", ""],
        "reasoning_summary": "Summary.",
        "confidence": " High ",
        "requires_human_review": True,
    }

    result = llm.validate_llm_result(payload)

    assert result.patient_safety_risk_score == 1.0
    assert result.key_risk_drivers == ["Sterility"]
    assert result.affected_quality_systems == ["CAPA"]
    assert result.confidence == "high"
    assert result.requires_human_review is True


def test_validate_llm_result_rejects_bad_payload() -> None:
    with pytest.raises(ValueError):
        llm.validate_llm_result({"reasoning_summary": "x", "requires_human_review": "yes"})


def test_call_llm_json_retries_and_returns_valid_result(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            types.SimpleNamespace(text="not json"),
            types.SimpleNamespace(
                text='''
                ```json
                {
                  "patient_safety_risk_score": 0.9,
                  "systemic_quality_breakdown_score": 0.8,
                  "data_integrity_risk_score": 0.7,
                  "remediation_complexity_score": 0.6,
                  "regulatory_escalation_risk_score": 0.5,
                  "key_risk_drivers": ["Sterility"],
                  "cited_findings": ["Aseptic failures"],
                  "affected_quality_systems": ["Sterility Assurance"],
                  "recommended_follow_up_actions": ["Audit CAPA"],
                  "reasoning_summary": "High concern.",
                  "confidence": "medium",
                  "requires_human_review": true
                }
                ```
                '''
            ),
        ]
    )

    class FakeModels:
        def generate_content(self, model: str, contents: str) -> object:
            return next(responses)

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(llm, "get_client", lambda: FakeClient())
    monkeypatch.setattr(llm.time, "sleep", lambda *_args, **_kwargs: None)

    result = llm.call_llm_json(
        messages=[{"role": "user", "content": "score this"}],
        max_retries=2,
        retry_delay_seconds=0.0,
    )

    assert result.patient_safety_risk_score == 0.9
    assert result.confidence == "medium"


def test_call_llm_json_raises_after_all_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModels:
        def generate_content(self, model: str, contents: str) -> object:
            return types.SimpleNamespace(text="still not json")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(llm, "get_client", lambda: FakeClient())
    monkeypatch.setattr(llm.time, "sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError):
        llm.call_llm_json(
            messages=[{"role": "user", "content": "score this"}],
            max_retries=2,
            retry_delay_seconds=0.0,
        )
