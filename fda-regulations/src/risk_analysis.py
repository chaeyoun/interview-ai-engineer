import argparse
import logging
from itertools import combinations
from pathlib import Path

import pandas as pd
from dataclasses import asdict

from src.datamodels import ScoredLetter, FlattenedRow
from src.utils import load_scored_letters, write_json_file, write_summary_csvs, write_png_charts, normalize_token, require_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build severity-aware risk intelligence summary from scored letters."
    )
    parser.add_argument(
        "--scored-path",
        type=Path,
        required=True,
        help="Path to scored letters JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write summary JSON, CSV files, and PNG charts.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        required=True,
        help="Target year used for naming analysis outputs.",
    )
    return parser.parse_args()


def flatten_scored_letters(scored_records: list[ScoredLetter]) -> list[FlattenedRow]:
    """Flatten scored letter objects into row-oriented records for analysis."""
    flattened: list[FlattenedRow] = []

    for scored in scored_records:
        normalized_tokens = sorted(
            {
                normalize_token(token)
                for token in scored.subject_tokens
                if token.strip()
            }
        )

        flattened.append(
            FlattenedRow(
                url=scored.metadata.url,
                company_name=scored.metadata.company_name,
                issue_date=scored.metadata.issue_date_str,
                posted_date=scored.metadata.posted_date,
                issuing_office=scored.metadata.issuing_office,
                subject=scored.metadata.subject,
                subject_tokens=normalized_tokens,
                final_risk_score=scored.final_risk_score,
                final_risk_level=scored.final_risk_level,
                patient_safety_risk_score=scored.llm_analysis.patient_safety_risk_score,
                systemic_quality_breakdown_score=scored.llm_analysis.systemic_quality_breakdown_score,
                data_integrity_risk_score=scored.llm_analysis.data_integrity_risk_score,
                remediation_complexity_score=scored.llm_analysis.remediation_complexity_score,
                regulatory_escalation_risk_score=scored.llm_analysis.regulatory_escalation_risk_score,
                affected_quality_systems=[
                    item.strip()
                    for item in scored.llm_analysis.affected_quality_systems
                    if item.strip()
                ],
            )
        )

    return flattened


def build_base_dataframe(flattened: list[FlattenedRow]) -> pd.DataFrame:
    """
    Build the base pandas DataFrame from flattened records.

    Adds derived columns:
    - issue_date_dt / posted_date_dt (datetime)
    - is_high_or_critical (boolean flag)

    Args:
        flattened: Flattened scored-letter rows ready for DataFrame construction.

    Returns:
        pd.DataFrame: Enriched DataFrame for downstream analysis.
    """
    df = pd.DataFrame([asdict(row) for row in flattened])
    if df.empty:
        return df

    df["issue_date_dt"] = pd.to_datetime(
        df["issue_date"],
        format="%m/%d/%Y",
        errors="coerce",
    )
    df["posted_date_dt"] = pd.to_datetime(
        df["posted_date"],
        format="%m/%d/%Y",
        errors="coerce",
    )

    df["issue_month"] = df["issue_date_dt"].dt.strftime("%Y-%m").where(
        df["issue_date_dt"].notna(),
        None,
    )
    df["posting_lag_days"] = (
        df["posted_date_dt"] - df["issue_date_dt"]
    ).dt.days

    df["is_high_or_critical"] = df["final_risk_level"].isin(["HIGH", "CRITICAL"])

    return df

def build_token_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand subject tokens into a token-level DataFrame.

    Each row becomes one (letter, token) pair using explode(),
    enabling token-level aggregation.

    Args:
        df (pd.DataFrame): Base DataFrame.

    Returns:
        pd.DataFrame: Token-level DataFrame.
    """
    if df.empty: 
        return pd.DataFrame()

    require_columns(
        df,
        ["subject_tokens"],
        "build_token_dataframe",
    )

    token_df = df.explode("subject_tokens").copy()
    token_df = token_df[token_df["subject_tokens"].notna() & (token_df["subject_tokens"] != "")]
    return token_df


def build_pair_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate pairwise combinations of subject tokens per letter.

    For each row, computes all unique token pairs (combinations of 2),
    used for co-occurrence risk analysis.
    """
    if df.empty:
        return pd.DataFrame()

    require_columns(
        df,
        [
            "subject_tokens",
            "url",
            "company_name",
            "final_risk_score",
            "final_risk_level",
            "is_high_or_critical",
        ],
        "build_pair_dataframe",
    )
    pair_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        tokens = sorted(set(row["subject_tokens"]))
        for left, right in combinations(tokens, 2):
            pair_rows.append(
                {
                    "subject_token_pair": f"{left} + {right}",
                    "url": row["url"],
                    "company_name": row["company_name"],
                    "final_risk_score": row["final_risk_score"],
                    "final_risk_level": row["final_risk_level"],
                    "is_high_or_critical": row["is_high_or_critical"],
                }
            )

    return pd.DataFrame(pair_rows)


def build_quality_system_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Expand affected quality systems into one row per (letter, quality system).

    Args:
        df: Base risk-analysis DataFrame.

    Returns:
        A DataFrame suitable for quality-system-level aggregation.
    """
    if df.empty:
        return pd.DataFrame()

    require_columns(
        df,
        ["affected_quality_systems"],
        "build_quality_system_dataframe",
    )

    qs_df = df.explode("affected_quality_systems").copy()
    qs_df = qs_df[
        qs_df["affected_quality_systems"].notna()
        & (qs_df["affected_quality_systems"] != "")
    ]
    return qs_df


def build_dataset_overview(df: pd.DataFrame, scored_count: int) -> dict[str, object]:
    if not df.empty:
        require_columns(df, ["issue_date_dt"], "build_dataset_overview")

    issue_dates = df["issue_date_dt"].dropna() if not df.empty else pd.Series(dtype="datetime64[ns]")

    return {
        "total_scored_letters": scored_count,
        "total_usable_letters": int(len(df)),
        "total_skipped_letters": int(max(scored_count - len(df), 0)),
        "issue_date_range": {
            "min": issue_dates.min().date().isoformat() if not issue_dates.empty else None,
            "max": issue_dates.max().date().isoformat() if not issue_dates.empty else None,
        },
    }


def build_risk_distribution(df: pd.DataFrame) -> dict[str, object]:
    """
    Compute overall risk distribution statistics.

    Includes:
    - Counts per risk level
    - Summary stats (mean, median, min, max) for final risk score
    - Summary stats for each LLM sub-score

    Args:
        df (pd.DataFrame): Base DataFrame.

    Returns:
        dict: Risk distribution summary.
    """
    if df.empty:
        return {
            "final_risk_level_counts": [],
            "final_risk_score_summary": {},
            "llm_subscore_summary": {},
        }

    require_columns(
        df,
        [
            "final_risk_level",
            "final_risk_score",
            "patient_safety_risk_score",
            "systemic_quality_breakdown_score",
            "data_integrity_risk_score",
            "remediation_complexity_score",
            "regulatory_escalation_risk_score",
        ],
        "build_risk_distribution",
    )

    level_counts = (
        df["final_risk_level"]
        .value_counts(dropna=False)
        .rename_axis("name")
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    def summarize(series: pd.Series) -> dict[str, float]:
        series = series.dropna()
        if series.empty:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(float(series.mean()), 3),
            "median": round(float(series.median()), 3),
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
        }

    return {
        "final_risk_level_counts": level_counts,
        "final_risk_score_summary": summarize(df["final_risk_score"]),
        "llm_subscore_summary": {
            "patient_safety_risk_score": summarize(df["patient_safety_risk_score"]),
            "systemic_quality_breakdown_score": summarize(df["systemic_quality_breakdown_score"]),
            "data_integrity_risk_score": summarize(df["data_integrity_risk_score"]),
            "remediation_complexity_score": summarize(df["remediation_complexity_score"]),
            "regulatory_escalation_risk_score": summarize(df["regulatory_escalation_risk_score"]),
        },
    }


def build_office_risk_summary(df: pd.DataFrame) -> list[dict[str, object]]:
    """
    Aggregate risk metrics by issuing office.

    Computes:
    - Letter count
    - Average final risk score
    - High/critical rate
    - Average LLM sub-scores

    Sorted by highest risk.

    Args:
        df (pd.DataFrame): Base DataFrame.

    Returns:
        list[dict]: Office-level summary records.
    """
    if df.empty: 
        return []

    require_columns(
        df,
        [
            "issuing_office",
            "url",
            "final_risk_score",
            "is_high_or_critical",
            "patient_safety_risk_score",
            "systemic_quality_breakdown_score",
            "data_integrity_risk_score",
            "remediation_complexity_score",
            "regulatory_escalation_risk_score",
        ],
        "build_office_risk_summary",
    )
    summary = (
        df.groupby("issuing_office", as_index=False)
        .agg(
            letter_count=("url", "count"),
            avg_final_risk_score=("final_risk_score", "mean"),
            high_or_critical_rate=("is_high_or_critical", "mean"),
            avg_patient_safety_risk_score=("patient_safety_risk_score", "mean"),
            avg_systemic_quality_breakdown_score=("systemic_quality_breakdown_score", "mean"),
            avg_data_integrity_risk_score=("data_integrity_risk_score", "mean"),
            avg_remediation_complexity_score=("remediation_complexity_score", "mean"),
            avg_regulatory_escalation_risk_score=("regulatory_escalation_risk_score", "mean"),
        )
        .sort_values(
            by=["avg_final_risk_score", "letter_count", "issuing_office"],
            ascending=[False, False, True],
        )
    )

    numeric_cols = [col for col in summary.columns if col != "issuing_office" and col != "letter_count"]
    summary[numeric_cols] = summary[numeric_cols].round(3)
    return summary.to_dict(orient="records")


def build_subject_token_risk_summary(
    token_df: pd.DataFrame,
    baseline_final_risk_score: float,
    min_letters: int = 5,
) -> list[dict[str, object]]:
    """
    Aggregate risk metrics by individual subject token.

    Adds:
    - risk_lift_vs_baseline: average final risk score minus overall dataset baseline
    - minimum sample-size filter to reduce noisy one-off tokens
    """
    if token_df.empty:
        return []

    require_columns(
        token_df,
        [
            "subject_tokens",
            "url",
            "final_risk_score",
            "is_high_or_critical",
            "patient_safety_risk_score",
            "data_integrity_risk_score",
            "regulatory_escalation_risk_score",
        ],
        "build_subject_token_risk_summary",
    )

    summary = (
        token_df.groupby("subject_tokens", as_index=False)
        .agg(
            letter_count=("url", "count"),
            avg_final_risk_score=("final_risk_score", "mean"),
            high_or_critical_rate=("is_high_or_critical", "mean"),
            avg_patient_safety_risk_score=("patient_safety_risk_score", "mean"),
            avg_data_integrity_risk_score=("data_integrity_risk_score", "mean"),
            avg_regulatory_escalation_risk_score=("regulatory_escalation_risk_score", "mean"),
        )
        .rename(columns={"subject_tokens": "subject_token"})
    )

    summary = summary[summary["letter_count"] >= min_letters].copy()
    if summary.empty:
        return []

    summary["risk_lift_vs_baseline"] = (
        summary["avg_final_risk_score"] - baseline_final_risk_score
    )

    summary = summary.sort_values(
        by=["risk_lift_vs_baseline", "letter_count", "subject_token"],
        ascending=[False, False, True],
    )

    numeric_cols = [
        col for col in summary.columns
        if col not in {"subject_token", "letter_count"}
    ]
    summary[numeric_cols] = summary[numeric_cols].round(3)
    return summary.to_dict(orient="records")


def build_subject_token_pair_risk_summary(pair_df: pd.DataFrame) -> list[dict[str, object]]:
    """
    Aggregate risk metrics for token pairs.

    Measures how combinations of subject tokens correlate with risk.

    Args:
        pair_df (pd.DataFrame): Pair-level DataFrame.

    Returns:
        list[dict]: Token pair summary.
    """
    if pair_df.empty: 
        return []

    require_columns(
        pair_df,
        [
            "subject_token_pair",
            "url",
            "final_risk_score",
            "is_high_or_critical",
        ],
        "build_subject_token_pair_risk_summary",
    )
    summary = (
        pair_df.groupby("subject_token_pair", as_index=False)
        .agg(
            letter_count=("url", "count"),
            avg_final_risk_score=("final_risk_score", "mean"),
            high_or_critical_rate=("is_high_or_critical", "mean"),
        )
        .sort_values(
            by=["avg_final_risk_score", "letter_count", "subject_token_pair"],
            ascending=[False, False, True],
        )
    )

    summary[["avg_final_risk_score", "high_or_critical_rate"]] = summary[
        ["avg_final_risk_score", "high_or_critical_rate"]
    ].round(3)
    return summary.to_dict(orient="records")


def build_monthly_risk_trends(df: pd.DataFrame) -> list[dict[str, object]]:
    """
    Compute monthly trends of risk metrics.

    Aggregates by issue_month:
    - Letter count
    - Average risk score
    - High/critical rate

    Args:
        df (pd.DataFrame): Base DataFrame.

    Returns:
        list[dict]: Monthly trend records.
    """
    if df.empty: 
        return []

    require_columns(
        df,
        [
            "issue_month",
            "url",
            "final_risk_score",
            "is_high_or_critical",
        ],
        "build_monthly_risk_trends",
    )

    summary = (
        df.dropna(subset=["issue_month"])
        .groupby("issue_month", as_index=False)
        .agg(
            letter_count=("url", "count"),
            avg_final_risk_score=("final_risk_score", "mean"),
            high_or_critical_rate=("is_high_or_critical", "mean"),
        )
        .sort_values("issue_month")
    )

    summary[["avg_final_risk_score", "high_or_critical_rate"]] = summary[
        ["avg_final_risk_score", "high_or_critical_rate"]
    ].round(3)
    return summary.to_dict(orient="records")


def build_repeat_company_summary(df: pd.DataFrame) -> dict[str, object]:
    """
    Analyze repeat vs one-time companies.

    Includes:
    - Count of repeat vs one-time companies
    - Top repeat offenders
    - Risk comparison between repeat and non-repeat companies

    Args:
        df (pd.DataFrame): Base DataFrame.

    Returns:
        dict: Repeat company analysis.
    """
    if df.empty:
        return {
            "repeat_company_count": 0,
            "one_time_company_count": 0,
            "top_repeat_companies": [],
            "repeat_vs_non_repeat_risk": {
                "repeat_company_avg_final_risk_score": 0.0,
                "non_repeat_company_avg_final_risk_score": 0.0,
            },
        }
    require_columns(
        df,
        [
            "company_name",
            "url",
            "final_risk_score",
            "subject_tokens",
        ],
        "build_repeat_company_summary",
    )

    company_counts = df.groupby("company_name").size().rename("letter_count").reset_index()
    repeat_companies = set(company_counts.loc[company_counts["letter_count"] > 1, "company_name"])
    one_time_companies = set(company_counts.loc[company_counts["letter_count"] == 1, "company_name"])

    repeat_df = df[df["company_name"].isin(repeat_companies)].copy()
    non_repeat_df = df[df["company_name"].isin(one_time_companies)].copy()

    if repeat_df.empty:
        top_repeat_companies = []
    else:
        top_repeat = (
            repeat_df.groupby("company_name", as_index=False)
            .agg(
                letter_count=("url", "count"),
                avg_final_risk_score=("final_risk_score", "mean"),
            )
        )

        subject_map = (
            repeat_df.groupby("company_name")["subject_tokens"]
            .apply(lambda x: "; ".join(sorted({token for tokens in x for token in tokens})))
            .rename("subjects")
            .reset_index()
        )

        top_repeat = top_repeat.merge(subject_map, on="company_name", how="left")
        top_repeat = top_repeat.sort_values(
            by=["letter_count", "avg_final_risk_score", "company_name"],
            ascending=[False, False, True],
        )
        top_repeat["avg_final_risk_score"] = top_repeat["avg_final_risk_score"].round(3)
        top_repeat_companies = top_repeat.head(15).to_dict(orient="records")

    return {
        "repeat_company_count": int(len(repeat_companies)),
        "one_time_company_count": int(len(one_time_companies)),
        "top_repeat_companies": top_repeat_companies,
        "repeat_vs_non_repeat_risk": {
            "repeat_company_avg_final_risk_score": round(float(repeat_df["final_risk_score"].mean()), 3)
            if not repeat_df.empty else 0.0,
            "non_repeat_company_avg_final_risk_score": round(float(non_repeat_df["final_risk_score"].mean()), 3)
            if not non_repeat_df.empty else 0.0,
        },
    }


def build_posting_lag_summary(df: pd.DataFrame) -> dict[str, object]:
    lag_df = df.dropna(subset=["posting_lag_days"]).copy()
    if lag_df.empty:
        return {"overall": {}, "by_office": []}

    require_columns(
        lag_df,
        [
            "posting_lag_days",
            "issuing_office",
            "url",
        ],
        "build_posting_lag_summary",
    )

    by_office = (
        lag_df.groupby("issuing_office", as_index=False)
        .agg(
            letter_count=("url", "count"),
            mean_posting_lag_days=("posting_lag_days", "mean"),
            median_posting_lag_days=("posting_lag_days", "median"),
        )
        .sort_values(
            by=["mean_posting_lag_days", "letter_count", "issuing_office"],
            ascending=[False, False, True],
        )
    )

    by_office[["mean_posting_lag_days", "median_posting_lag_days"]] = by_office[
        ["mean_posting_lag_days", "median_posting_lag_days"]
    ].round(3)

    return {
        "overall": {
            "mean_posting_lag_days": round(float(lag_df["posting_lag_days"].mean()), 3),
            "median_posting_lag_days": round(float(lag_df["posting_lag_days"].median()), 3),
            "max_posting_lag_days": int(lag_df["posting_lag_days"].max()),
        },
        "by_office": by_office.to_dict(orient="records"),
    }


def build_quality_system_risk_summary(
    qs_df: pd.DataFrame,
    baseline_final_risk_score: float,
    min_letters: int = 5,
) -> list[dict[str, object]]:
    """
    Aggregate risk metrics by affected quality system.

    Adds:
    - risk_lift_vs_baseline
    - minimum sample-size filter
    """
    if qs_df.empty:
        return []

    require_columns(
        qs_df,
        [
            "affected_quality_systems",
            "url",
            "final_risk_score",
            "is_high_or_critical",
            "systemic_quality_breakdown_score",
            "regulatory_escalation_risk_score",
        ],
        "build_quality_system_risk_summary",
    )

    summary = (
        qs_df.groupby("affected_quality_systems", as_index=False)
        .agg(
            letter_count=("url", "count"),
            avg_final_risk_score=("final_risk_score", "mean"),
            high_or_critical_rate=("is_high_or_critical", "mean"),
            avg_systemic_quality_breakdown_score=("systemic_quality_breakdown_score", "mean"),
            avg_regulatory_escalation_risk_score=("regulatory_escalation_risk_score", "mean"),
        )
        .rename(columns={"affected_quality_systems": "quality_system"})
    )

    summary = summary[summary["letter_count"] >= min_letters].copy()
    if summary.empty:
        return []

    summary["risk_lift_vs_baseline"] = (
        summary["avg_final_risk_score"] - baseline_final_risk_score
    )

    summary = summary.sort_values(
        by=["risk_lift_vs_baseline", "letter_count", "quality_system"],
        ascending=[False, False, True],
    )

    numeric_cols = [
        col for col in summary.columns
        if col not in {"quality_system", "letter_count"}
    ]
    summary[numeric_cols] = summary[numeric_cols].round(3)
    return summary.to_dict(orient="records")


def build_key_insights(summary: dict[str, object]) -> list[str]:
    """Generate human-readable key insights from summary results."""
    insights: list[str] = []

    overview = summary.get("dataset_overview", {})
    if isinstance(overview, dict):
        total_usable = overview.get("total_usable_letters")
        total_scored = overview.get("total_scored_letters")
        if total_usable is not None and total_scored is not None:
            insights.append(
                f"Analyzed {total_usable} usable scored letters out of {total_scored} total scored records."
            )

    office_rows = summary.get("office_risk_summary", [])
    if isinstance(office_rows, list) and office_rows:
        top_office = office_rows[0]
        if isinstance(top_office, dict):
            insights.append(
                f"The highest-average-risk issuing office was {top_office.get('issuing_office')} "
                f"(avg final risk {top_office.get('avg_final_risk_score')}, {top_office.get('letter_count')} letters)."
            )

    token_rows = summary.get("subject_token_risk_summary", [])
    if isinstance(token_rows, list) and token_rows:
        top_token = token_rows[0]
        if isinstance(top_token, dict):
            insights.append(
                f"The highest-average-risk subject token was {top_token.get('subject_token')} "
                f"(avg final risk {top_token.get('avg_final_risk_score')}, {top_token.get('letter_count')} letters)."
            )

    pair_rows = summary.get("subject_token_pair_risk_summary", [])
    if isinstance(pair_rows, list) and pair_rows:
        top_pair = pair_rows[0]
        if isinstance(top_pair, dict):
            insights.append(
                f"The highest-average-risk token pair was {top_pair.get('subject_token_pair')} "
                f"(avg final risk {top_pair.get('avg_final_risk_score')})."
            )

    repeat_summary = summary.get("repeat_company_summary", {})
    if isinstance(repeat_summary, dict):
        repeat_vs_non_repeat = repeat_summary.get("repeat_vs_non_repeat_risk", {})
        if isinstance(repeat_vs_non_repeat, dict):
            repeat_avg = repeat_vs_non_repeat.get("repeat_company_avg_final_risk_score")
            non_repeat_avg = repeat_vs_non_repeat.get("non_repeat_company_avg_final_risk_score")
            if repeat_avg is not None and non_repeat_avg is not None:
                insights.append(
                    f"Repeat-letter companies had an average final risk score of {repeat_avg}, "
                    f"compared with {non_repeat_avg} for one-time companies."
                )

    quality_rows = summary.get("quality_system_risk_summary", [])
    if isinstance(quality_rows, list) and quality_rows:
        top_system = quality_rows[0]
        if isinstance(top_system, dict):
            insights.append(
                f"The highest-average-risk affected quality system was {top_system.get('quality_system')} "
                f"(avg final risk {top_system.get('avg_final_risk_score')}, {top_system.get('letter_count')} letters)."
            )

    return insights


def run_analysis(
    target_year: int,
    output_dir: Path,
    scored_path: Path | None = None,
) -> Path:
    """Run risk analysis and return the summary JSON path."""
    resolved_scored_path = (
        scored_path
        if scored_path is not None
        else output_dir / f"risk_scores_{target_year}.json"
    )

    scored_records = load_scored_letters(resolved_scored_path)
    if not scored_records:
        raise RuntimeError(f"No scored records found in {resolved_scored_path}")

    flattened = flatten_scored_letters(scored_records)
    if not flattened:
        raise RuntimeError("No usable scored records found after flattening.")

    df = build_base_dataframe(flattened)
    token_df = build_token_dataframe(df)
    pair_df = build_pair_dataframe(df)
    quality_sys_df = build_quality_system_dataframe(df)

    baseline_final_risk_score = float(df["final_risk_score"].mean())
    summary: dict[str, object] = {
        "dataset_overview": build_dataset_overview(df=df, scored_count=len(scored_records)),
        "risk_distribution": build_risk_distribution(df),
        "office_risk_summary": build_office_risk_summary(df),
        "subject_token_risk_summary": build_subject_token_risk_summary(
            token_df,
            baseline_final_risk_score=baseline_final_risk_score,
            min_letters=5,
        ),
        "subject_token_pair_risk_summary": build_subject_token_pair_risk_summary(pair_df),
        "monthly_risk_trends": build_monthly_risk_trends(df),
        "repeat_company_summary": build_repeat_company_summary(df),
        "posting_lag_summary": build_posting_lag_summary(df),
        "quality_system_risk_summary": build_quality_system_risk_summary(
            quality_sys_df,
            baseline_final_risk_score=baseline_final_risk_score,
            min_letters=5,
        ),
        "baseline_final_risk_score": round(baseline_final_risk_score, 3),
    }
    summary["key_insights"] = build_key_insights(summary)

    summary_path = output_dir / f"summary_{target_year}.json"
    write_json_file(summary, summary_path)
    write_summary_csvs(summary, output_dir, target_year)
    write_png_charts(summary, output_dir, target_year)

    logger.info("Saved JSON summary to %s", summary_path)
    logger.info("Saved CSV summaries to %s", output_dir / f"risk_intelligence_csv_{target_year}")
    logger.info("Saved PNG charts to %s", output_dir / f"risk_intelligence_charts_{target_year}")

    return summary_path


def main() -> None:
    """
    Pipeline for building risk intelligence outputs.

    Steps:
    1. Load scored records
    2. Flatten and preprocess data
    3. Build DataFrames (base, token, pair)
    4. Compute all summary metrics
    5. Generate insights
    6. Save JSON, CSV, and visualization outputs

    Entry point for CLI execution.
    """
    args = parse_args()
    run_analysis(
        target_year=args.target_year,
        output_dir=args.output_dir,
        scored_path=args.scored_path,
    )

    
if __name__ == "__main__":
    main()