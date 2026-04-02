import json
import logging
import os
import time
from typing import Final

from dotenv import load_dotenv
from google import genai

from src.datamodels import LLMRiskResult, WarningLetterMetadata

logger = logging.getLogger(__name__)

MODEL: Final[str] = "gemini-3.1-flash-lite-preview"

USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/146.0.0.0 Safari/537.36"
)

_client: genai.Client | None = None


def get_google_api_key() -> str:
    """Load and return the Google API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return api_key


def get_client() -> genai.Client:
    """Return a lazily initialized Gemini client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_google_api_key())
    return _client


def build_messages(
    metadata: WarningLetterMetadata,
    body_text: str,
) -> list[dict[str, str]]:
    """Build the system and user messages for LLM risk assessment.

    Args:
        metadata: Structured metadata for one FDA warning letter.
        body_text: Cleaned warning letter body text to analyze.

    Returns:
        A list of role/content message dictionaries.
    """
    system_prompt = (
        "You are an FDA regulatory risk analyst for a pharmaceutical quality intelligence system. "
        "Assess one FDA warning letter and return strict JSON only.\n\n"
        "Evaluate this letter relative to other FDA warning letters, not relative to ordinary compliance documents. "
        "Do not assign uniformly high scores simply because the input is a warning letter. "
        "Reserve the highest scores for letters that indicate unusually high patient safety risk, "
        "broad systemic quality breakdown, serious data integrity concerns, unusually high remediation burden, "
        "or elevated risk of regulatory escalation.\n\n"
        "Return an object with exactly these keys:\n"
        "- patient_safety_risk_score: float between 0.0 and 1.0\n"
        "- systemic_quality_breakdown_score: float between 0.0 and 1.0\n"
        "- data_integrity_risk_score: float between 0.0 and 1.0\n"
        "- remediation_complexity_score: float between 0.0 and 1.0\n"
        "- regulatory_escalation_risk_score: float between 0.0 and 1.0\n"
        "- key_risk_drivers: array of short strings\n"
        "- cited_findings: array of short strings grounded in the provided text\n"
        "- affected_quality_systems: array of short strings\n"
        "- recommended_follow_up_actions: array of short strings\n"
        "- reasoning_summary: short paragraph\n"
        "- confidence: one of low, medium, high\n"
        "- requires_human_review: boolean\n\n"
        "Base the assessment only on the warning letter text provided. "
        "Prefer concrete findings over generic summaries. "
        "Return strict JSON only."
    )

    user_prompt = (
        f"Company: {metadata.company_name}\n"
        f"Issuing office: {metadata.issuing_office}\n"
        f"Issue date: {metadata.issue_date_str}\n"
        f"Subject: {metadata.subject}\n"
        "Warning letter body:\n"
        f"{body_text}\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llm_json(
    messages: list[dict[str, str]],
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
) -> LLMRiskResult:
    """Call the LLM, extract JSON from the response, and validate it.

    Args:
        messages: Prompt messages returned by build_messages().
        max_retries: Number of attempts before failing.
        retry_delay_seconds: Base backoff delay between attempts.

    Returns:
        A validated LLMRiskResult instance.

    Raises:
        RuntimeError: If all attempts fail.
    """
    prompt = "\n\n".join(message["content"] for message in messages)
    last_error: Exception | None = None
    client = get_client()

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            text = (response.text or "").strip()
            if not text:
                raise ValueError("Empty response from LLM.")

            json_text = extract_json_text(text)
            parsed = json.loads(json_text)
            return validate_llm_result(parsed)

        except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
            last_error = exc
            logger.warning(
                "LLM call failed on attempt %s/%s: %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(retry_delay_seconds * attempt)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts.") from last_error


def _normalize_score(value: object, field_name: str) -> float:
    """Convert a numeric score to a normalized 0.0-1.0 float."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    return round(max(0.0, min(float(value), 1.0)), 3)


def _normalize_string_list(value: object, field_name: str) -> list[str]:
    """Normalize a list of strings by stripping whitespace and dropping empty values."""
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")

    cleaned: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized = item.strip()
            if normalized:
                cleaned.append(normalized)

    return cleaned


def _normalize_confidence(value: object) -> str:
    """Normalize and validate the confidence label."""
    if not isinstance(value, str):
        raise ValueError("confidence must be a string.")

    normalized = value.strip().lower()
    if normalized not in {"low", "medium", "high"}:
        raise ValueError("confidence must be low, medium, or high.")

    return normalized


def extract_json_text(text: str) -> str:
    """Extract the outermost JSON object substring from raw model output.

    Args:
        text: Raw text returned by the LLM.

    Returns:
        A substring expected to contain a JSON object.

    Raises:
        ValueError: If no JSON object boundaries can be found.
    """
    stripped = text.strip()
    decoder = json.JSONDecoder()

    for i, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(stripped[i:])
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict):
            return stripped[i:i + end]

    raise ValueError(f"No valid JSON object found in LLM output:\n{text}")


def validate_llm_result(payload: object) -> LLMRiskResult:
    """Validate and normalize a parsed LLM response payload."""
    if not isinstance(payload, dict):
        raise ValueError("LLM output must be a JSON object.")

    patient_raw = payload.get("patient_safety_risk_score")
    systemic_raw = payload.get("systemic_quality_breakdown_score")
    data_integrity_raw = payload.get("data_integrity_risk_score")
    remediation_raw = payload.get("remediation_complexity_score")
    escalation_raw = payload.get("regulatory_escalation_risk_score")

    key_risk_drivers_raw = payload.get("key_risk_drivers")
    cited_findings_raw = payload.get("cited_findings")
    affected_quality_systems_raw = payload.get("affected_quality_systems")
    recommended_actions_raw = payload.get("recommended_follow_up_actions")

    reasoning_summary_raw = payload.get("reasoning_summary")
    confidence_raw = payload.get("confidence")
    requires_human_review_raw = payload.get("requires_human_review")

    if not isinstance(reasoning_summary_raw, str):
        raise ValueError("reasoning_summary must be a string.")
    if not isinstance(requires_human_review_raw, bool):
        raise ValueError("requires_human_review must be a boolean.")

    return LLMRiskResult(
        patient_safety_risk_score=_normalize_score(
            patient_raw,
            "patient_safety_risk_score",
        ),
        systemic_quality_breakdown_score=_normalize_score(
            systemic_raw,
            "systemic_quality_breakdown_score",
        ),
        data_integrity_risk_score=_normalize_score(
            data_integrity_raw,
            "data_integrity_risk_score",
        ),
        remediation_complexity_score=_normalize_score(
            remediation_raw,
            "remediation_complexity_score",
        ),
        regulatory_escalation_risk_score=_normalize_score(
            escalation_raw,
            "regulatory_escalation_risk_score",
        ),
        key_risk_drivers=_normalize_string_list(
            key_risk_drivers_raw,
            "key_risk_drivers",
        ),
        cited_findings=_normalize_string_list(
            cited_findings_raw,
            "cited_findings",
        ),
        affected_quality_systems=_normalize_string_list(
            affected_quality_systems_raw,
            "affected_quality_systems",
        ),
        recommended_follow_up_actions=_normalize_string_list(
            recommended_actions_raw,
            "recommended_follow_up_actions",
        ),
        reasoning_summary=reasoning_summary_raw.strip(),
        confidence=_normalize_confidence(confidence_raw),
        requires_human_review=requires_human_review_raw,
    )