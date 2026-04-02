import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import TypedDict

import pandas as pd

from src.datamodels import WarningLetterMetadata, ScoredLetter, LLMRiskResult
from src.plotting import plot_bar, plot_line, plot_barh

logger = logging.getLogger(__name__)

ALLOWED_RISK_LEVELS: set[str] = {"ELEVATED", "MODERATE", "HIGH", "CRITICAL"}
ALLOWED_CONFIDENCE_LEVELS: set[str] = {"LOW", "MEDIUM", "HIGH"}


class RepeatCompanySummary(TypedDict, total=False):
    top_repeat_companies: list[dict[str, object]]


class RiskSummary(TypedDict, total=False):
    office_risk_summary: list[dict[str, object]]
    subject_token_risk_summary: list[dict[str, object]]
    subject_token_pair_risk_summary: list[dict[str, object]]
    monthly_risk_trends: list[dict[str, object]]
    quality_system_risk_summary: list[dict[str, object]]
    repeat_company_summary: RepeatCompanySummary
    baseline_final_risk_score: float


def parse_date(value: str) -> date | None:
    """Parse an FDA-style MM/DD/YYYY date string."""
    try:
        return datetime.strptime(value, "%m/%d/%Y").date()
    except ValueError:
        logger.warning("Failed to parse issue date: %s", value)
        return None


def split_subject_tokens(subject: str) -> list[str]:
    """Split a slash-delimited subject field into normalized tokens."""
    return [part.strip() for part in subject.split("/") if part.strip()]


def normalize_token(token: str) -> str:
    return token.strip().lower()


def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    context: str,
) -> None:
    """Raise an error if required DataFrame columns are missing."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} requires columns {missing}, but they were not found.")


def validate_score_range(value: float, field_name: str) -> float:
    """Validate that a score is within the expected 0.0-1.0 range."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")
    return value


def validate_risk_level(value: str) -> str:
    """Validate that the final risk level is one of the allowed labels."""
    normalized = value.strip().upper()
    if normalized not in ALLOWED_RISK_LEVELS:
        raise ValueError(
            f"final_risk_level must be one of {sorted(ALLOWED_RISK_LEVELS)}, got {value!r}"
        )
    return normalized


def validate_confidence(value: object) -> str:
    """Validate that confidence is one of the allowed labels."""
    text = validate_str(value, "confidence").strip().upper()
    if text not in ALLOWED_CONFIDENCE_LEVELS:
        raise ValueError(
            f"confidence must be one of {sorted(ALLOWED_CONFIDENCE_LEVELS)}, got {text!r}"
        )
    return text


def validate_str(value: object, field_name: str) -> str:
    """Validate that a value is a string."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
    return value


def validate_bool(value: object, field_name: str) -> bool:
    """Validate that a value is a boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean, got {type(value).__name__}")
    return value


def write_json_file(payload: list[dict[str, object]] | dict[str, object], output_path: Path) -> None:
    """Write a JSON-serializable payload to disk, creating parent directories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to write JSON to {output_path}") from exc

    logger.info("Saved JSON to %s", output_path)


def write_df_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
    except OSError as exc:
        raise RuntimeError(f"Failed to write CSV to {path}") from exc

    logger.info("Saved CSV to %s", path)


def load_metadata(path: Path) -> list[WarningLetterMetadata]:
    """Load warning letter metadata records from a JSON file.

    Invalid records are skipped and counted in logs rather than causing the
    entire load to fail.
    """
    records_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records_raw, list):
        raise ValueError("Metadata JSON must be a list of objects.")

    required_keys = {
        "url",
        "posted_date",
        "issue_date_str",
        "company_name",
        "issuing_office",
        "subject",
    }

    results: list[WarningLetterMetadata] = []
    skipped_non_dict = 0
    skipped_missing_keys = 0

    for idx, item in enumerate(records_raw):
        if not isinstance(item, dict):
            skipped_non_dict += 1
            continue

        if not required_keys.issubset(item.keys()):
            skipped_missing_keys += 1
            continue

        results.append(
            WarningLetterMetadata(
                url=str(item["url"]),
                posted_date=str(item["posted_date"]),
                issue_date_str=str(item["issue_date_str"]),
                company_name=str(item["company_name"]),
                issuing_office=str(item["issuing_office"]),
                subject=str(item["subject"]),
            )
        )

    logger.info(
        (
            "Loaded %s metadata records from %s "
            "(skipped_non_dict=%s, skipped_missing_keys=%s)"
        ),
        len(results),
        path,
        skipped_non_dict,
        skipped_missing_keys,
    )
    return results


def load_scored_letters(path: Path) -> list[ScoredLetter]:
    """Load scored warning letter records from a JSON file.

    Invalid records are skipped and counted in logs rather than causing the
    entire load to fail.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Scored JSON must be a list of objects.")

    results: list[ScoredLetter] = []
    skipped_non_dict = 0
    skipped_missing_nested = 0
    skipped_invalid_payload = 0

    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            skipped_non_dict += 1
            continue

        metadata_raw = item.get("metadata")
        llm_raw = item.get("llm_analysis")

        if not isinstance(metadata_raw, dict) or not isinstance(llm_raw, dict):
            skipped_missing_nested += 1
            continue

        try:
            metadata = WarningLetterMetadata(
                url=str(metadata_raw["url"]),
                posted_date=str(metadata_raw["posted_date"]),
                issue_date_str=str(metadata_raw["issue_date_str"]),
                company_name=str(metadata_raw["company_name"]),
                issuing_office=str(metadata_raw["issuing_office"]),
                subject=str(metadata_raw["subject"]),
            )

            patient_safety_risk_score = validate_score_range(
                float(llm_raw["patient_safety_risk_score"]),
                "patient_safety_risk_score",
            )
            systemic_quality_breakdown_score = validate_score_range(
                float(llm_raw["systemic_quality_breakdown_score"]),
                "systemic_quality_breakdown_score",
            )
            data_integrity_risk_score = validate_score_range(
                float(llm_raw["data_integrity_risk_score"]),
                "data_integrity_risk_score",
            )
            remediation_complexity_score = validate_score_range(
                float(llm_raw["remediation_complexity_score"]),
                "remediation_complexity_score",
            )
            regulatory_escalation_risk_score = validate_score_range(
                float(llm_raw["regulatory_escalation_risk_score"]),
                "regulatory_escalation_risk_score",
            )
            final_risk_score = validate_score_range(
                float(item["final_risk_score"]),
                "final_risk_score",
            )
            final_risk_level = validate_risk_level(str(item["final_risk_level"]))

            llm_analysis = LLMRiskResult(
                patient_safety_risk_score=patient_safety_risk_score,
                systemic_quality_breakdown_score=systemic_quality_breakdown_score,
                data_integrity_risk_score=data_integrity_risk_score,
                remediation_complexity_score=remediation_complexity_score,
                regulatory_escalation_risk_score=regulatory_escalation_risk_score,
                key_risk_drivers=[str(x) for x in llm_raw.get("key_risk_drivers", [])],
                cited_findings=[str(x) for x in llm_raw.get("cited_findings", [])],
                affected_quality_systems=[str(x) for x in llm_raw.get("affected_quality_systems", [])],
                recommended_follow_up_actions=[str(x) for x in llm_raw.get("recommended_follow_up_actions", [])],
                reasoning_summary=str(llm_raw["reasoning_summary"]),
                confidence=validate_confidence(llm_raw["confidence"]),
                requires_human_review=validate_bool(llm_raw["requires_human_review"], "requires_human_review")
            )

            scored_letter = ScoredLetter(
                metadata=metadata,
                subject_tokens=[str(x) for x in item.get("subject_tokens", [])],
                final_risk_score=final_risk_score,
                final_risk_level=final_risk_level,
                llm_analysis=llm_analysis,
            )
        except (KeyError, TypeError, ValueError) as exc:
            skipped_invalid_payload += 1
            logger.warning(
                "Skipping invalid scored record at index=%s in %s: %s",
                idx,
                path,
                exc,
            )
            continue

        results.append(scored_letter)

    logger.info(
        (
            "Loaded %s scored records from %s "
            "(skipped_non_dict=%s, skipped_missing_nested=%s, skipped_invalid_payload=%s)"
        ),
        len(results),
        path,
        skipped_non_dict,
        skipped_missing_nested,
        skipped_invalid_payload,
    )
    return results


def write_summary_csvs(
    summary: RiskSummary,
    output_dir: Path,
    target_year: int,
) -> None:
    """Write all summary tables to CSV files."""
    csv_dir = output_dir / f"risk_intelligence_csv_{target_year}"
    csv_dir.mkdir(parents=True, exist_ok=True)

    office_rows = summary.get("office_risk_summary", [])
    token_rows = summary.get("subject_token_risk_summary", [])
    pair_rows = summary.get("subject_token_pair_risk_summary", [])
    monthly_rows = summary.get("monthly_risk_trends", [])
    quality_rows = summary.get("quality_system_risk_summary", [])

    repeat_summary = summary.get("repeat_company_summary", {})
    repeat_rows = []
    if isinstance(repeat_summary, dict):
        repeat_rows = repeat_summary.get("top_repeat_companies", [])

    write_df_csv(pd.DataFrame(office_rows), csv_dir / "office_risk_summary.csv")
    write_df_csv(pd.DataFrame(token_rows), csv_dir / "subject_token_risk_summary.csv")
    write_df_csv(pd.DataFrame(pair_rows), csv_dir / "subject_token_pair_risk_summary.csv")
    write_df_csv(pd.DataFrame(monthly_rows), csv_dir / "monthly_risk_trends.csv")
    write_df_csv(pd.DataFrame(repeat_rows), csv_dir / "top_repeat_companies.csv")
    write_df_csv(pd.DataFrame(quality_rows), csv_dir / "quality_system_risk_summary.csv")


def write_png_charts(
    summary: RiskSummary,
    output_dir: Path,
    target_year: int,
) -> None:
    """Render PNG charts from summary tables."""
    chart_dir = output_dir / f"risk_intelligence_charts_{target_year}"
    chart_dir.mkdir(parents=True, exist_ok=True)

    office_df = pd.DataFrame(summary.get("office_risk_summary", []))
    token_df = pd.DataFrame(summary.get("subject_token_risk_summary", []))
    monthly_df = pd.DataFrame(summary.get("monthly_risk_trends", []))
    quality_df = pd.DataFrame(summary.get("quality_system_risk_summary", []))

    plot_bar(
        office_df,
        "issuing_office",
        "avg_final_risk_score",
        "Average Final Risk Score by Issuing Office",
        "Issuing Office",
        "Average Final Risk Score",
        chart_dir / "office_avg_final_risk_score.png",
        top_n=12,
    )

    plot_barh(
        token_df,
        "subject_token",
        "risk_lift_vs_baseline",
        "Risk Lift vs Baseline by Subject Token",
        "Risk Lift vs Baseline",
        "Subject Token",
        chart_dir / "subject_token_risk_lift_vs_baseline.png",
        top_n=12,
        max_label_chars=40,
        wrap_width=28,
        figsize=(12, 8),
    )

    plot_barh(
        token_df,
        "subject_token",
        "high_or_critical_rate",
        "High/Critical Rate by Subject Token",
        "High/Critical Rate",
        "Subject Token",
        chart_dir / "subject_token_high_or_critical_rate.png",
        top_n=12,
        max_label_chars=40,
        wrap_width=28,
        figsize=(12, 8),
    )

    plot_line(
        monthly_df,
        "issue_month",
        "avg_final_risk_score",
        "Monthly Average Final Risk Score",
        "Issue Month",
        "Average Final Risk Score",
        chart_dir / "monthly_avg_final_risk_score.png",
    )

    plot_line(
        monthly_df,
        "issue_month",
        "high_or_critical_rate",
        "Monthly High/Critical Rate",
        "Issue Month",
        "High/Critical Rate",
        chart_dir / "monthly_high_or_critical_rate.png",
    )

    plot_bar(
        quality_df,
        "quality_system",
        "risk_lift_vs_baseline",
        "Risk Lift vs Baseline by Affected Quality System",
        "Affected Quality System",
        "Risk Lift vs Baseline",
        chart_dir / "quality_system_risk_lift_vs_baseline.png",
        top_n=10,
        max_label_chars=30,
        wrap_width=16,
        figsize=(14, 8),
    )