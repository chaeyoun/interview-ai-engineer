# FDA Warning Letter Risk Intelligence

This project is a production-oriented proof of concept that transforms FDA Warning Letter data into structured risk intelligence.

The system is designed to:
- collect regulatory signals from FDA warning letters
- assess risk using LLM-based structured analysis
- generate aggregated insights for downstream decision-making

The pipeline is intentionally simple, modular, and runnable either end-to-end or by individual stage.

---

## Overview

The system is divided into three main components:

1. **Web Scraping**
   - Collects FDA Warning Letter metadata via the FDA AJAX endpoint
   - Outputs structured metadata JSON

2. **LLM-based Risk Scoring**
   - Fetches each warning letter body
   - Performs structured risk assessment using an LLM
   - Combines LLM sub-scores into a final risk score

3. **Risk Analysis**
   - Aggregates scored results
   - Produces summary JSON, CSV tables, and PNG charts

---

## Project Structure

```bash
.
├── fda-regulations/
│   ├── main.py
│   ├── .env
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── README.md
│   ├── REPORT.md
│   ├── src/
│   │   ├── scraping.py
│   │   ├── risk_scoring.py
│   │   ├── risk_analysis.py
│   │   ├── llm.py
│   │   ├── datamodels.py
│   │   ├── plotting.py
│   │   └── utils.py
│   └── tests/
│       ├── test_utils.py
│       ├── test_risk_scoring.py
│       ├── test_risk_analysis.py
│       ├── test_scraping.py
│       └── test_llm.py
├── Dockerfile
└── README.md
```

### Module Responsibilities

- `datamodels.py`: Defines structured data models used across the pipeline
- `scraping.py`: Collects FDA warning letter metadata
- `risk_scoring.py`: Extracts letter body and performs LLM-based risk scoring
- `risk_analysis.py`: Aggregates scored results into structured insights
- `plotting.py`: Generates charts from analysis outputs
- `utils.py`: Provides shared utilities for I/O, validation, and data handling
- `llm.py`: Handles LLM interaction and response normalization

---

## Setup

### Docker

Build the image:

```bash
docker build -t fda .
```

### Environment Variables

This project requires a Google API key for LLM scoring.
Create a `.env` file in the project root/fda-regulations:

```bash
GOOGLE_API_KEY=your_api_key_here
```

---

## Run the Pipeline

### Option 1. Run the full pipeline at once

From the project folder (`fda-regulations/`):

**Run with Docker:**
```bash
docker run --rm -it --env-file .env -v "$(pwd)/output:/app/output" fda --target-year {year}
```
**Run locally:**
```bash
uv run python main.py --target-year {year}
```

Example:

```bash
uv run python main.py --target-year 2025
```

This runs all three stages in order:
1. scraping
2. risk scoring
3. risk analysis

Generated outputs:

```bash
output/
├── fda_warning_letters_{year}_metadata.json
├── risk_scores_{year}.json
├── failed_letters_{year}.json
├── summary_{year}.json
├── risk_intelligence_csv_{year}/
└── risk_intelligence_charts_{year}/
```

---

### Option 2. Run each stage separately

From the project folder:

### 1. Web Scraping

```bash
uv run python -m src.scraping --target-year {year}
```

This stage:
- calls the FDA Warning Letter AJAX endpoint
- converts each row into structured metadata
- saves metadata as JSON

Output:

```bash
/output/fda_warning_letters_{year}_metadata.json
```

### 2. Risk Scoring

```bash
uv run python -m src.risk_scoring \
  --target-year {year} \
  --metadata-path output/fda_warning_letters_{year}_metadata.json
```

This stage:
- loads metadata
- fetches warning letter body text
- runs LLM-based structured risk analysis
- combines LLM sub-scores into a final risk score
- saves scored records and failed records

Outputs:

```bash
/output/risk_scores_{year}.json
/output/failed_letters_{year}.json
```

### 3. Risk Analysis

```bash
uv run python -m src.risk_analysis \
  --target-year {year} \
  --scored-path output/risk_scores_{year}.json
```

This stage:
- builds structured summaries
- generates CSV tables
- produces PNG charts

Outputs:

```bash
/output/summary_{year}.json
/output/risk_intelligence_csv_{year}/
/output/risk_intelligence_charts_{year}/
```

---

## Outputs

The full pipeline produces:
- metadata JSON
- scored JSON
- failed-record JSON
- summary JSON
- CSV tables
- PNG charts

---

## Notes

- Each stage can be run independently.
- The full pipeline can be run from `main.py`.
- LLM scoring requires a valid `GOOGLE_API_KEY`.
- Risk analysis requires completed scoring output.
- Some warning letters may fail during scoring if body text cannot be extracted.
- Detailed methodology, assumptions, and design decisions are documented in `REPORT.md`.
- The LLM model can be switched in `llm.py` if needed.
