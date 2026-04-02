import argparse
import logging
from pathlib import Path
import asyncio

from src.scraping import run_scraping
from src.risk_scoring import run_scoring
from src.risk_analysis import run_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full FDA warning letter pipeline."
    )
    parser.add_argument(
        "--target-year",
        type=int,
        required=True,
        help="Target year for the end-to-end pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for all pipeline outputs.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum number of FDA AJAX pages to scrape.",
    )
    parser.add_argument(
        "--max-letters",
        type=int,
        default=None,
        help="Optional cap on the number of letters to score.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    logger.info("Starting full pipeline for target year %s", args.target_year)

    metadata_path = await run_scraping(
        target_year=args.target_year,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
    )

    scored_path, failed_path = run_scoring(
        target_year=args.target_year,
        output_dir=args.output_dir,
        metadata_path=metadata_path,
        max_letters=args.max_letters,
    )

    summary_path = run_analysis(
        target_year=args.target_year,
        output_dir=args.output_dir,
        scored_path=scored_path,
    )

    logger.info("Pipeline complete.")
    logger.info("Metadata saved to %s", metadata_path)
    logger.info("Scored results saved to %s", scored_path)
    logger.info("Failed records saved to %s", failed_path)
    logger.info("Summary saved to %s", summary_path)


def main() -> None:
    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 
    asyncio.run(async_main())


if __name__ == "__main__":
    main()