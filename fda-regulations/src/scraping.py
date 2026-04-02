import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Final
import argparse
import asyncio

import httpx
from bs4 import BeautifulSoup

from src.datamodels import WarningLetterMetadata
from src.utils import parse_date

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PageFetchError(Exception):
    """Raised when a page cannot be fetched or parsed after retries."""


class FDAScraper:
    """Scrape FDA warning letter metadata from the website.
    """

    AJAX_URL: Final[str] = "https://www.fda.gov/datatables/views/ajax"
    BASE_SITE_URL: Final[str] = "https://www.fda.gov"
    PAGE_SIZE: Final[int] = 10 # FDA website has 10 items per page
    MAX_COLUMNS: Final[int] = 8 # max columns in table of FDA warning letters website
    MAX_ATTEMPTS: Final[int] = 2
    BATCH_SIZE: Final[int] = 10

    def __init__(self) -> None:
        """Initialize a reusable HTTP client with browser-like request headers."""
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/146.0.0.0 Safari/537.36"
                ),
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Referer": (
                    "https://www.fda.gov/"
                    "inspections-compliance-enforcement-and-criminal-investigations/"
                    "compliance-actions-and-activities/warning-letters"
                ),
            },
            timeout=30.0,
            follow_redirects=True,
        )


    async def close(self) -> None:
        """Release the underlying HTTP client resources."""
        await self._client.aclose()


    def _build_ajax_params(self, page: int) -> dict[str, str | int]:
        """Build query parameters for the FDA DataTables AJAX endpoint.

        These parameters were derived from the browser's network requests.
        The endpoint expects a DataTables-compatible request shape, including
        pagination fields such as ``draw``, ``start``, ``length``, and per-column
        metadata under ``columns[...]``.

        Note:
            Some fields may look redundant, but undocumented AJAX endpoints can
            be sensitive to request shape. Keep only the parameters confirmed to
            be required by endpoint testing.
        """
        start = page * self.PAGE_SIZE

        params: dict[str, str | int] = {
            "search_api_fulltext": "",
            "search_api_fulltext_issuing_office": "",
            "field_letter_issue_datetime": "All",
            "field_change_date_closeout_letter": "",
            "field_change_date_response_letter": "",
            "field_change_date_2": "All",
            "field_letter_issue_datetime_2": "",
            "draw": page + 1,
            "start": start,
            "length": self.PAGE_SIZE,
            "search[value]": "",
            "search[regex]": "false",
            "_drupal_ajax": "1",
            "_wrapper_format": "drupal_ajax",
            "pager_element": "0",
            "view_args": "",
            "view_base_path": (
                "inspections-compliance-enforcement-and-criminal-investigations/"
                "compliance-actions-and-activities/warning-letters/datatables-data"
            ),
            "view_display_id": "warning_letter_solr_block",
            "view_name": "warning_letter_solr_index",
            "view_path": (
                "/inspections-compliance-enforcement-and-criminal-investigations/"
                "compliance-actions-and-activities/warning-letters"
            ),
        }

        for column_index in range(self.MAX_COLUMNS):
            params[f"columns[{column_index}][data]"] = str(column_index)
            params[f"columns[{column_index}][name]"] = ""
            params[f"columns[{column_index}][searchable]"] = "true"
            params[f"columns[{column_index}][orderable]"] = (
                "false" if column_index == 7 else "true"
            )
            params[f"columns[{column_index}][search][value]"] = ""
            params[f"columns[{column_index}][search][regex]"] = "false"

        return params


    def _strip_html(self, value: str) -> str:
        """Extract normalized plain text from an HTML fragment."""
        return BeautifulSoup(value, "html.parser").get_text(strip=True)


    def _parse_metadata_row(self, row: list[object]) -> WarningLetterMetadata | None:
        """Convert one raw datatable row into structured warning letter metadata.

        Each row contains HTML-encoded cell values. This method extracts the
        warning letter URL from the company column, strips HTML from display
        fields, and returns a typed metadata object.
        """
        if len(row) < 5:
            return None

        posted_date_raw = row[0]
        issue_date_raw = row[1]
        company_cell_raw = row[2]
        issuing_office_raw = row[3]
        subject_raw = row[4]

        if not all(isinstance(cell, str) for cell in row[:5]):
            return None

        posted_date = self._strip_html(posted_date_raw)
        issue_date_str = self._strip_html(issue_date_raw)

        company_cell = BeautifulSoup(company_cell_raw, "html.parser")
        link_tag = company_cell.find("a", href=True)
        if link_tag is None:
            return None

        href = link_tag["href"]
        url = href if href.startswith("http") else f"{self.BASE_SITE_URL}{href}"

        company_name = company_cell.get_text(strip=True)
        issuing_office = self._strip_html(issuing_office_raw)
        subject = self._strip_html(subject_raw)

        return WarningLetterMetadata(
            url=url,
            posted_date=posted_date,
            issue_date_str=issue_date_str,
            company_name=company_name,
            issuing_office=issuing_office,
            subject=subject,
        )

    
    def _extract_rows_from_payload(self, payload: object, page: int) -> list[object]:
        """Normalize and extract row data from an FDA AJAX response.
        This function handles these variations and extracts the underlying row list.

        Raises:
            PageFetchError: If no valid row structure can be extracted.
        """
        if isinstance(payload, dict):
            rows = payload.get("data")
            if isinstance(rows, list):
                return rows
            raise PageFetchError(
                f"AJAX payload dict missing valid 'data' list on page {page}"
            )

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue

                candidate = item.get("data")
                if isinstance(candidate, dict):
                    nested_rows = candidate.get("data")
                    if isinstance(nested_rows, list):
                        return nested_rows

                if isinstance(candidate, list):
                    return candidate

            raise PageFetchError(
                f"AJAX payload list missing parsable row data on page {page}"
            )

        raise PageFetchError(
            f"Unexpected payload type {type(payload).__name__} on page {page}"
        )


    async def get_letters_from_page(self, page: int) -> list[WarningLetterMetadata]:
        """Fetch and parse one page of warning letter metadata.

        Sends a request to the FDA AJAX endpoint using the page-specific
        DataTables parameters, validates the response shape, and converts
        each valid row into a metadata object.
        """
        logger.info("Fetching AJAX page %s", page)

        last_error: Exception | None = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                response = await self._client.get(
                    self.AJAX_URL,
                    params=self._build_ajax_params(page),
                )
                response.raise_for_status()
                payload = response.json()
                rows = self._extract_rows_from_payload(payload, page)

                results: list[WarningLetterMetadata] = []
                for row in rows:
                    if not isinstance(row, list):
                        continue

                    metadata = self._parse_metadata_row(row)
                    if metadata is not None:
                        results.append(metadata)

                logger.info("Parsed %s rows from AJAX page %s", len(results), page)
                return results

            except (
                httpx.HTTPStatusError,
                httpx.RequestError,
                json.JSONDecodeError,
                PageFetchError,
            ) as error:
                last_error = error
                if attempt == self.MAX_ATTEMPTS:
                    break

                backoff_seconds = 2 ** (attempt - 1)
                logger.warning(
                    "Attempt %s/%s failed for page %s: %s. Retrying in %ss...",
                    attempt,
                    self.MAX_ATTEMPTS,
                    page,
                    error,
                    backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)

        raise PageFetchError(
            f"Failed to fetch page {page} after {self.MAX_ATTEMPTS} attempts: {last_error}"
        )

    
    async def _fetch_page_batch(
        self,
        start_page: int,
        end_page: int,
    ) -> list[tuple[int, list[WarningLetterMetadata] | Exception]]:
        """Fetch a batch of pages concurrently.

        This method launches concurrent requests for a range of pages using
        asyncio.gather, allowing controlled parallelism at the batch level.

        Each result is returned alongside its page index to preserve ordering.

        Args:
            start_page: Inclusive start page index.
            end_page: Exclusive end page index.

        Returns:
            A list of tuples (page, result), where result is either:
            - list[WarningLetterMetadata] on success
            - Exception if the page fetch failed
        """
        tasks = [
            self.get_letters_from_page(page)
            for page in range(start_page, end_page)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            (page, result)
            for page, result in zip(range(start_page, end_page), results)
        ]


    async def get_letters_for_year(
        self,
        target_year: int,
        max_pages: int,
    ) -> list[WarningLetterMetadata]:
        """Collect warning letter metadata for a specific year.

        Pages are fetched in fixed-size batches to balance concurrency and stability.
        Each batch is processed sequentially, while pages inside a batch are fetched
        concurrently.
        """
        collected: list[WarningLetterMetadata] = []
        seen_urls: set[str] = set()
        consecutive_empty_pages = 0

        for batch_start in range(0, max_pages, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, max_pages)
            logger.info("Fetching page batch %s-%s", batch_start, batch_end - 1)

            batch_results = await self._fetch_page_batch(batch_start, batch_end)

            for page, page_result in batch_results:
                if isinstance(page_result, Exception):
                    logger.error("Skipping page %s due to fetch error: %s", page, page_result)
                    continue

                if not page_result:
                    consecutive_empty_pages += 1
                    logger.info("Empty page %s detected.", page)

                    if consecutive_empty_pages >= 3:
                        logger.info(
                            "Stopping after %s consecutive empty pages.",
                            consecutive_empty_pages,
                        )
                        return collected
                    continue

                consecutive_empty_pages = 0
                kept_count = 0

                for item in page_result:
                    parsed_date = parse_date(item.issue_date_str)
                    if parsed_date is None:
                        logger.debug(
                            "Could not parse issue_date_str=%r for url=%s",
                            item.issue_date_str,
                            item.url,
                        )
                        continue

                    if parsed_date.year != target_year:
                        continue

                    if item.url in seen_urls:
                        continue

                    seen_urls.add(item.url)
                    collected.append(item)
                    kept_count += 1

                logger.info(
                    "Page %s: parsed=%s, kept_for_%s=%s, total=%s",
                    page,
                    len(page_result),
                    target_year,
                    kept_count,
                    len(collected),
                )

        return collected



def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments for the scraping pipeline."""
    parser = argparse.ArgumentParser(
        description="Scrape FDA warning letter metadata for a target year."
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=2025,
        help="Issue year to collect.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=330,
        help="Maximum number of AJAX pages to scan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write metadata JSON.",
    )
    return parser.parse_args()


def save_metadata(records: list[WarningLetterMetadata], output_path: Path) -> None:
    """Save metadata records to a JSON file.

    Dataclass instances are converted to dictionaries before serialization
    because the standard JSON encoder does not handle custom Python objects.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)


async def run_scraping(
    target_year: int,
    output_dir: Path,
    max_pages: int,
) -> Path:
    """Run metadata scraping and return the metadata output path."""
    output_path = output_dir / f"fda_warning_letters_{target_year}_metadata.json"

    scraper = FDAScraper()
    try:
        metadata_list = await scraper.get_letters_for_year(
            target_year=target_year,
            max_pages=max_pages,
        )
    finally:
        await scraper.close()

    if not metadata_list:
        raise RuntimeError(f"No warning letters found for year {target_year}.")

    save_metadata(metadata_list, output_path)
    logger.info("Saved %s metadata records to %s", len(metadata_list), output_path)
    return output_path


def main() -> None:
    """Run the scraping pipeline."""
    args = parse_args()
    asyncio.run(run_scraping(
        target_year=args.target_year,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
    ))


if __name__ == "__main__":
    main()