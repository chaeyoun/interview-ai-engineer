
from __future__ import annotations

import pandas as pd

from src.datamodels import FlattenedRow, LLMRiskResult, ScoredLetter, WarningLetterMetadata
from src.risk_analysis import (
    build_base_dataframe,
    build_key_insights,
    build_office_risk_summary,
    build_pair_dataframe,
    build_quality_system_dataframe,
    build_repeat_company_summary,
    flatten_scored_letters,
)


def _scored_letter(
    company_name: str,
    url: str,
    final_risk_score: float,
    final_risk_level: str,
    subject_tokens: list[str],
    affected_quality_systems: list[str],
) -> ScoredLetter:
    metadata = WarningLetterMetadata(
        url=url,
        posted_date="01/20/2025",
        issue_date_str="01/10/2025",
        company_name=company_name,
        issuing_office="CDER",
        subject="/".join(subject_tokens),
    )
    llm = LLMRiskResult(
        patient_safety_risk_score=0.9,
        systemic_quality_breakdown_score=0.8,
        data_integrity_risk_score=0.7,
        remediation_complexity_score=0.6,
        regulatory_escalation_risk_score=0.5,
        key_risk_drivers=["Sterility"],
        cited_findings=["Aseptic failures"],
        affected_quality_systems=affected_quality_systems,
        recommended_follow_up_actions=["Audit CAPA"],
        reasoning_summary="High concern.",
        confidence="HIGH",
        requires_human_review=True,
    )
    return ScoredLetter(
        metadata=metadata,
        subject_tokens=subject_tokens,
        final_risk_score=final_risk_score,
        final_risk_level=final_risk_level,
        llm_analysis=llm,
    )


def test_flatten_scored_letters_normalizes_subject_tokens() -> None:
    scored = [
        _scored_letter(
            company_name="Acme Pharma",
            url="https://example.com/1",
            final_risk_score=0.8,
            final_risk_level="HIGH",
            subject_tokens=[" Sterility ", "Data Integrity"],
            affected_quality_systems=["CAPA", " Sterility Assurance "],
        )
    ]

    flattened = flatten_scored_letters(scored)

    assert len(flattened) == 1
    assert flattened[0].subject_tokens == ["data integrity", "sterility"]
    assert flattened[0].affected_quality_systems == ["CAPA", "Sterility Assurance"]


def test_build_base_dataframe_adds_derived_columns() -> None:
    rows = [
        FlattenedRow(
            url="https://example.com/1",
            company_name="Acme Pharma",
            issue_date="01/10/2025",
            posted_date="01/20/2025",
            issuing_office="CDER",
            subject="Sterility",
            subject_tokens=["sterility"],
            final_risk_score=0.8,
            final_risk_level="HIGH",
            patient_safety_risk_score=0.9,
            systemic_quality_breakdown_score=0.8,
            data_integrity_risk_score=0.7,
            remediation_complexity_score=0.6,
            regulatory_escalation_risk_score=0.5,
            affected_quality_systems=["CAPA"],
        )
    ]

    df = build_base_dataframe(rows)

    assert "issue_date_dt" in df.columns
    assert "posted_date_dt" in df.columns
    assert "issue_month" in df.columns
    assert "posting_lag_days" in df.columns
    assert "is_high_or_critical" in df.columns
    assert bool(df.iloc[0]["is_high_or_critical"]) is True
    assert int(df.iloc[0]["posting_lag_days"]) == 10


def test_build_pair_dataframe_and_quality_system_dataframe() -> None:
    df = pd.DataFrame(
        [
            {
                "url": "https://example.com/1",
                "company_name": "Acme Pharma",
                "subject_tokens": ["sterility", "data integrity", "sterility"],
                "final_risk_score": 0.8,
                "final_risk_level": "HIGH",
                "is_high_or_critical": True,
                "affected_quality_systems": ["CAPA", "Sterility Assurance"],
            }
        ]
    )

    pair_df = build_pair_dataframe(df)
    qs_df = build_quality_system_dataframe(df)

    assert len(pair_df) == 1
    assert pair_df.iloc[0]["subject_token_pair"] == "data integrity + sterility"
    assert len(qs_df) == 2


def test_build_office_risk_summary_and_repeat_company_summary() -> None:
    df = pd.DataFrame(
        [
            {
                "issuing_office": "CDER",
                "url": "https://example.com/1",
                "company_name": "Acme Pharma",
                "final_risk_score": 0.8,
                "is_high_or_critical": True,
                "patient_safety_risk_score": 0.9,
                "systemic_quality_breakdown_score": 0.8,
                "data_integrity_risk_score": 0.7,
                "remediation_complexity_score": 0.6,
                "regulatory_escalation_risk_score": 0.5,
                "subject_tokens": ["sterility"],
            },
            {
                "issuing_office": "CDER",
                "url": "https://example.com/2",
                "company_name": "Acme Pharma",
                "final_risk_score": 0.7,
                "is_high_or_critical": True,
                "patient_safety_risk_score": 0.8,
                "systemic_quality_breakdown_score": 0.7,
                "data_integrity_risk_score": 0.6,
                "remediation_complexity_score": 0.5,
                "regulatory_escalation_risk_score": 0.4,
                "subject_tokens": ["data integrity"],
            },
            {
                "issuing_office": "CDER",
                "url": "https://example.com/3",
                "company_name": "Other Pharma",
                "final_risk_score": 0.5,
                "is_high_or_critical": False,
                "patient_safety_risk_score": 0.5,
                "systemic_quality_breakdown_score": 0.5,
                "data_integrity_risk_score": 0.5,
                "remediation_complexity_score": 0.5,
                "regulatory_escalation_risk_score": 0.5,
                "subject_tokens": ["labeling"],
            },
        ]
    )

    office_summary = build_office_risk_summary(df)
    repeat_summary = build_repeat_company_summary(df)

    assert office_summary[0]["issuing_office"] == "CDER"
    assert office_summary[0]["letter_count"] == 3
    assert repeat_summary["repeat_company_count"] == 1
    assert repeat_summary["one_time_company_count"] == 1
    assert repeat_summary["top_repeat_companies"][0]["company_name"] == "Acme Pharma"


def test_build_key_insights_generates_readable_strings() -> None:
    summary = {
        "dataset_overview": {
            "total_usable_letters": 10,
            "total_scored_letters": 12,
        },
        "office_risk_summary": [
            {"issuing_office": "CDER", "avg_final_risk_score": 0.8, "letter_count": 4}
        ],
        "subject_token_risk_summary": [
            {"subject_token": "sterility", "avg_final_risk_score": 0.82, "letter_count": 3}
        ],
        "subject_token_pair_risk_summary": [
            {"subject_token_pair": "sterility + data integrity", "avg_final_risk_score": 0.85}
        ],
        "repeat_company_summary": {
            "repeat_vs_non_repeat_risk": {
                "repeat_company_avg_final_risk_score": 0.77,
                "non_repeat_company_avg_final_risk_score": 0.55,
            }
        },
        "quality_system_risk_summary": [
            {"quality_system": "CAPA", "avg_final_risk_score": 0.81, "letter_count": 2}
        ],
    }

    insights = build_key_insights(summary)

    assert len(insights) >= 5
    assert any("Analyzed 10 usable scored letters" in item for item in insights)
    assert any("highest-average-risk issuing office was CDER" in item for item in insights)
    assert any("Repeat-letter companies had an average final risk score of 0.77" in item for item in insights)
