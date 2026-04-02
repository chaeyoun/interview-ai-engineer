
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils import (
    load_metadata,
    load_scored_letters,
    normalize_token,
    parse_date,
    require_columns,
    split_subject_tokens,
    validate_bool,
    validate_confidence,
    validate_risk_level,
    validate_score_range,
    validate_str,
    write_json_file,
)


def test_parse_date_returns_date_for_valid_mmddyyyy() -> None:
    parsed = parse_date("01/15/2025")
    assert parsed is not None
    assert parsed.year == 2025
    assert parsed.month == 1
    assert parsed.day == 15


def test_parse_date_returns_none_for_invalid_value() -> None:
    assert parse_date("2025-01-15") is None


def test_split_subject_tokens_and_normalize_token() -> None:
    tokens = split_subject_tokens("Sterility / Data Integrity /  Labeling ")
    assert tokens == ["Sterility", "Data Integrity", "Labeling"]
    assert normalize_token("  Data Integrity  ") == "data integrity"


def test_require_columns_raises_for_missing_columns() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError):
        require_columns(df, ["a", "c"], "demo")


def test_validation_helpers() -> None:
    assert validate_score_range(0.5, "score") == 0.5
    assert validate_risk_level(" high ") == "HIGH"
    assert validate_confidence(" medium ") == "MEDIUM"
    assert validate_str("abc", "field") == "abc"
    assert validate_bool(True, "flag") is True

    with pytest.raises(ValueError):
        validate_score_range(1.1, "score")
    with pytest.raises(ValueError):
        validate_risk_level("bad")
    with pytest.raises(ValueError):
        validate_confidence("bad")
    with pytest.raises(ValueError):
        validate_str(123, "field")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_bool("yes", "flag")  # type: ignore[arg-type]


def test_write_json_file_creates_output(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "result.json"
    payload = {"letters": 3, "status": "ok"}

    write_json_file(payload, output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_load_metadata_skips_invalid_rows(tmp_path: Path) -> None:
    payload = [
        {
            "url": "https://example.com/1",
            "posted_date": "01/20/2025",
            "issue_date_str": "01/10/2025",
            "company_name": "Acme Pharma",
            "issuing_office": "CDER",
            "subject": "Sterility",
        },
        {"url": "missing-fields"},
        "not-a-dict",
    ]
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    results = load_metadata(path)

    assert len(results) == 1
    assert results[0].company_name == "Acme Pharma"


def test_load_scored_letters_skips_invalid_rows_and_keeps_valid_one(tmp_path: Path) -> None:
    payload = [
        {
            "metadata": {
                "url": "https://example.com/1",
                "posted_date": "01/20/2025",
                "issue_date_str": "01/10/2025",
                "company_name": "Acme Pharma",
                "issuing_office": "CDER",
                "subject": "Sterility/Data Integrity",
            },
            "subject_tokens": ["Sterility", "Data Integrity"],
            "final_risk_score": 0.84,
            "final_risk_level": "HIGH",
            "llm_analysis": {
                "patient_safety_risk_score": 0.9,
                "systemic_quality_breakdown_score": 0.8,
                "data_integrity_risk_score": 0.7,
                "remediation_complexity_score": 0.6,
                "regulatory_escalation_risk_score": 0.5,
                "key_risk_drivers": ["Sterility"],
                "cited_findings": ["Aseptic processing failures"],
                "affected_quality_systems": ["Sterility Assurance"],
                "recommended_follow_up_actions": ["Audit CAPA"],
                "reasoning_summary": "High patient impact.",
                "confidence": "HIGH",
                "requires_human_review": True,
            },
        },
        {
            "metadata": {
                "url": "https://example.com/2",
                "posted_date": "01/20/2025",
                "issue_date_str": "01/10/2025",
                "company_name": "Broken Pharma",
                "issuing_office": "CDER",
                "subject": "Sterility",
            },
            "subject_tokens": ["Sterility"],
            "final_risk_score": 1.2,
            "final_risk_level": "HIGH",
            "llm_analysis": {
                "patient_safety_risk_score": 0.9,
                "systemic_quality_breakdown_score": 0.8,
                "data_integrity_risk_score": 0.7,
                "remediation_complexity_score": 0.6,
                "regulatory_escalation_risk_score": 0.5,
                "key_risk_drivers": ["Sterility"],
                "cited_findings": ["Aseptic processing failures"],
                "affected_quality_systems": ["Sterility Assurance"],
                "recommended_follow_up_actions": ["Audit CAPA"],
                "reasoning_summary": "Bad score.",
                "confidence": "HIGH",
                "requires_human_review": True,
            },
        },
    ]
    path = tmp_path / "scored.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    results = load_scored_letters(path)

    assert len(results) == 1
    assert results[0].metadata.company_name == "Acme Pharma"
    assert results[0].final_risk_level == "HIGH"
