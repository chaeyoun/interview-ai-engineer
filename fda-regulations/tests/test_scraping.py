
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.scraping import FDAScraper, PageFetchError, save_metadata


def test_build_ajax_params_contains_expected_pagination_fields() -> None:
    scraper = FDAScraper()
    params = scraper._build_ajax_params(page=3)

    assert params["draw"] == 4
    assert params["start"] == 30
    assert params["length"] == scraper.PAGE_SIZE
    assert params["columns[7][orderable]"] == "false"


def test_parse_metadata_row_builds_absolute_url() -> None:
    scraper = FDAScraper()
    row = [
        "<span>01/20/2025</span>",
        "<span>01/10/2025</span>",
        '<a href="/inspections/example-letter">Acme Pharma</a>',
        "<span>CDER</span>",
        "<span>Sterility</span>",
    ]

    metadata = scraper._parse_metadata_row(row)

    assert metadata is not None
    assert metadata.url == "https://www.fda.gov/inspections/example-letter"
    assert metadata.company_name == "Acme Pharma"
    assert metadata.subject == "Sterility"


def test_extract_rows_from_payload_handles_dict_and_nested_list() -> None:
    scraper = FDAScraper()

    rows = [["a"], ["b"]]
    assert scraper._extract_rows_from_payload({"data": rows}, page=1) == rows
    assert scraper._extract_rows_from_payload([{"data": {"data": rows}}], page=1) == rows

    with pytest.raises(PageFetchError):
        scraper._extract_rows_from_payload("bad", page=1)


@pytest.mark.anyio
async def test_get_letters_from_page_parses_successful_response(monkeypatch: pytest.MonkeyPatch) -> None:
    scraper = FDAScraper()

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return {
                "data": [
                    [
                        "01/20/2025",
                        "01/10/2025",
                        '<a href="/inspections/example-letter">Acme Pharma</a>',
                        "CDER",
                        "Sterility",
                    ]
                ]
            }

    async def fake_get(url: str, params: dict[str, str | int]) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr(scraper._client, "get", fake_get)

    results = await scraper.get_letters_from_page(0)

    assert len(results) == 1
    assert results[0].company_name == "Acme Pharma"


@pytest.mark.anyio
async def test_get_letters_for_year_filters_duplicates_and_stops_after_empty_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    scraper = FDAScraper()

    from src.datamodels import WarningLetterMetadata

    page0 = [
        WarningLetterMetadata(
            url="https://example.com/1",
            posted_date="01/20/2025",
            issue_date_str="01/10/2025",
            company_name="A",
            issuing_office="CDER",
            subject="Sterility",
        ),
        WarningLetterMetadata(
            url="https://example.com/1",
            posted_date="01/20/2025",
            issue_date_str="01/10/2025",
            company_name="A duplicate",
            issuing_office="CDER",
            subject="Sterility",
        ),
    ]

    async def fake_fetch_page_batch(start_page: int, end_page: int):
        mapping = {
            0: [(0, page0), (1, [])],
            10: [(10, []), (11, [])],
        }
        return mapping.get(start_page, [])

    monkeypatch.setattr(scraper, "_fetch_page_batch", fake_fetch_page_batch)
    results = await scraper.get_letters_for_year(target_year=2025, max_pages=20)

    assert len(results) == 1
    assert results[0].url == "https://example.com/1"


def test_save_metadata_writes_json(tmp_path: Path) -> None:
    from src.datamodels import WarningLetterMetadata

    output_path = tmp_path / "metadata.json"
    records = [
        WarningLetterMetadata(
            url="https://example.com/1",
            posted_date="01/20/2025",
            issue_date_str="01/10/2025",
            company_name="Acme Pharma",
            issuing_office="CDER",
            subject="Sterility",
        )
    ]

    save_metadata(records, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["company_name"] == "Acme Pharma"
