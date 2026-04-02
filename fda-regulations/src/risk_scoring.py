import argparse
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Final

import httpx
from bs4 import BeautifulSoup, Tag

from src.datamodels import WarningLetterMetadata, ScoredLetter, LLMRiskResult
from src.llm import USER_AGENT, build_messages, call_llm_json
from src.utils import load_metadata, split_subject_tokens, write_json_file


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


START_PREFIXES: Final[tuple[str, ...]] = (
    "Dear ",
    "This warning letter",
    "The U.S. Food and Drug Administration",
    "During an inspection",
    "FDA inspected",
    "The United States Food and Drug Administration",
    "This is to advise you that the United States"
)

END_PREFIXES: Final[tuple[str, ...]] = (
    "Please notify this office in writing within",
    "Your firm’s response should be sent",
    "Finally, you should know that this letter is not intended",
    "Sincerely,",
    "CC:",
)

REMOVE_PHRASES: Final[tuple[str, ...]] = (
    "Please provide a response on how you plan to address this deficiency.",
    "Please provide a response on how you plan to address the above deficiencies.",
    "Please continue to provide updates on the progress of your corrections and/or corrective actions",
)

BODY_CONTAINER_SELECTORS: Final[tuple[str, ...]] = (
    "article#main-content div[role='main']",
    "article#main-content",
    "main div[role='main']",
    "main",
)

MIN_BODY_LENGTH: Final[int] = 50
DEFAULT_BODY_CHAR_LIMIT: Final[int] = 20000


class LetterBodyExtractionError(RuntimeError):
    """Raised when warning letter body extraction fails."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the scoring pipeline."""
    parser = argparse.ArgumentParser(
        description="Score FDA warning letters from metadata JSON."
    )
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--max-letters", type=int, default=None)
    parser.add_argument("--target-year", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def _find_body_container(soup: BeautifulSoup) -> Tag:
    """Find best candidate container for warning letter body."""
    for selector in BODY_CONTAINER_SELECTORS:
        container = soup.select_one(selector)
        if container is not None:
            return container
    raise LetterBodyExtractionError("No body container found.")


def _iter_content_nodes(container: Tag) -> list[Tag]:
    """Return paragraph/list nodes in order."""
    return [n for n in container.find_all(["p", "ul", "ol"]) if isinstance(n, Tag)]


def _find_warning_anchor(container: Tag) -> Tag | None:
    """Find 'Warning Letter' anchor."""
    for node in _iter_content_nodes(container):
        strong = node.find("strong")
        if strong and strong.get_text(strip=True).lower() == "warning letter":
            return node
    return None


def _is_start_block(text: str) -> bool:
    """Return whether a cleaned block looks like the start of the letter body."""
    normalized = text.lower().strip()
    return any(normalized.startswith(p.lower()) for p in START_PREFIXES)


def _is_end_block(text: str) -> bool:
    """Return whether a cleaned block looks like the end of the letter body."""
    normalized = text.lower().strip()
    return any(normalized.startswith(p.lower()) for p in END_PREFIXES)


def _clean_block(text: str) -> str | None:
    """Clean single text block."""
    text = text.strip()
    if not text:
        return None

    if text.startswith("FDA Ref"):
        return None

    if re.fullmatch(r"[A-Za-z]+ \d{1,2}, \d{4}", text):
        return None

    text = re.sub(r"\(b\)\(4\)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    if any(p in text for p in REMOVE_PHRASES):
        return None

    if len(text) < 5:
        return None

    return text


def _dedupe_preserve_order(blocks: list[str]) -> list[str]:
    """Remove duplicate text blocks while preserving original order."""
    seen: set[str] = set()
    result: list[str] = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            result.append(b)
    return result


def _extract_blocks_after_anchor(container: Tag, anchor: Tag) -> list[str]:
    """Extract cleaned text blocks that appear after the 'Warning Letter' anchor."""
    blocks: list[str] = []
    started: bool = False

    for node in _iter_content_nodes(container):
        if node is anchor:
            started = True
            continue

        if not started:
            continue

        raw = node.get_text(separator="\n", strip=True)
        cleaned = _clean_block(raw)
        if cleaned is None:
            continue

        if _is_end_block(cleaned):
            break

        blocks.append(cleaned)

    return _dedupe_preserve_order(blocks)


def _finalize_text(blocks: list[str]) -> str:
    """Join cleaned blocks into final body text and validate minimum length."""
    if not blocks:
        raise LetterBodyExtractionError("No blocks extracted.")

    text = "\n\n".join(blocks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    if len(text) < MIN_BODY_LENGTH:
        raise LetterBodyExtractionError("Extracted text too short.")

    return text


def _extract_blocks_with_prefix(container: Tag) -> list[str]:
    """Extract cleaned text blocks starting from detected content prefixes."""
    blocks: list[str] = []
    started = False

    for node in _iter_content_nodes(container):
        raw = node.get_text(separator="\n", strip=True)
        cleaned = _clean_block(raw)
        if cleaned is None:
            continue

        if not started:
            if _is_start_block(cleaned):
                started = True
            else:
                continue

        if _is_end_block(cleaned):
            break

        blocks.append(cleaned)

    return _dedupe_preserve_order(blocks)


def _extract_all_clean_blocks(container: Tag) -> list[str]:
    """Extract all cleaned text blocks without requiring a start prefix."""
    blocks: list[str] = []

    for node in _iter_content_nodes(container):
        raw = node.get_text(separator="\n", strip=True)
        cleaned = _clean_block(raw)
        if cleaned is None:
            continue

        if _is_end_block(cleaned):
            break

        blocks.append(cleaned)

    return _dedupe_preserve_order(blocks)


def fetch_letter_body(client: httpx.Client, url: str) -> str:
    """Fetch and extract the main body text from an FDA warning letter page."""
    response = client.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    container = _find_body_container(soup)

    anchor = _find_warning_anchor(container)

    if anchor is not None:
        blocks = _extract_blocks_after_anchor(container, anchor)
        if blocks:
            return _finalize_text(blocks)

    blocks = _extract_blocks_with_prefix(container)
    if blocks:
        return _finalize_text(blocks)

    blocks = _extract_all_clean_blocks(container)
    return _finalize_text(blocks)

def calculate_final_risk_level(score: float) -> str:
    """Map a normalized final risk score to a categorical risk label."""
    if score >= 0.85:
        return "CRITICAL"
    if score >= 0.70:
        return "HIGH"
    if score >= 0.55:
        return "MODERATE"
    return "ELEVATED"


def combine_scores(llm_result: LLMRiskResult) -> float:
    """Combine LLM sub-scores into one normalized final risk score."""
    llm_composite = (
        0.30 * llm_result.patient_safety_risk_score
        + 0.25 * llm_result.systemic_quality_breakdown_score
        + 0.20 * llm_result.data_integrity_risk_score
        + 0.15 * llm_result.remediation_complexity_score
        + 0.10 * llm_result.regulatory_escalation_risk_score
    )
    return round(max(0.0, min(llm_composite, 1.0)), 3)


def score_letter(
    client: httpx.Client,
    metadata: WarningLetterMetadata,
    body_char_limit: int | None = DEFAULT_BODY_CHAR_LIMIT,
) -> ScoredLetter:
    """Fetch one letter body, run LLM scoring, and return a scored record."""

    body = fetch_letter_body(client, metadata.url)

    if body_char_limit:
        body = body[:body_char_limit]

    llm_result = call_llm_json(
        build_messages(metadata=metadata, body_text=body)
    )

    final_score = combine_scores(llm_result)
    level = calculate_final_risk_level(final_score)

    return ScoredLetter(
        metadata=metadata,
        subject_tokens=split_subject_tokens(metadata.subject),
        final_risk_score=final_score,
        final_risk_level=level,
        llm_analysis=llm_result,
    )


def run_scoring(
    target_year: int,
    output_dir: Path,
    metadata_path: Path | None = None,
    max_letters: int | None = None,
) -> tuple[Path, Path]:
    """Run the scoring pipeline and return output paths for scored and failed records."""

    scored_path = output_dir / f"risk_scores_{target_year}.json"
    failed_path = output_dir / f"failed_letters_{target_year}.json"

    metadata_path = metadata_path or output_dir / f"fda_warning_letters_{target_year}_metadata.json"
    records = load_metadata(metadata_path)

    if max_letters is not None:
        records = records[:max_letters]

    scored: list[ScoredLetter] = []
    failed: list[dict[str, str]] = []

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        for i, meta in enumerate(records, 1):
            try:
                scored.append(score_letter(client, meta))
                logger.info("Scored %s/%s: %s", i, len(records), meta.company_name)
            except (
                httpx.HTTPError,
                LetterBodyExtractionError,
                RuntimeError,
                ValueError,
                TypeError,
            ) as exc:
                logger.warning("Failed to score %s: %s", meta.url, exc)
                failed.append({"url": meta.url, "error": str(exc)})

    write_json_file([asdict(s) for s in scored], scored_path)
    write_json_file(failed, failed_path)

    return scored_path, failed_path


def main() -> None:
    args = parse_args()
    run_scoring(
        target_year=args.target_year,
        output_dir=args.output_dir,
        metadata_path=args.metadata_path,
        max_letters=args.max_letters,
    )


if __name__ == "__main__":
    main()