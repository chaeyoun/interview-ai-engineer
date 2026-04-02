# REPORT

## Problem Framing

The goal of this project is to transform unstructured FDA warning letter narratives into structured, actionable risk signals for pharmaceutical manufacturing.

While structured APIs provide easy access to metadata, they lack the detailed context needed to understand regulatory violations. In contrast, warning letters contain rich narrative descriptions of:
- violation types
- affected quality systems
- severity signals

This project prioritizes information density and system design over ease of access by working directly with unstructured regulatory text.


## Data Acquisition

- **Source**: FDA Warning Letters (public website)
- **Access Method**: Reverse-engineered AJAX endpoint
- **Reasoning**:
  - The FDA page uses a DataTables frontend
  - Data is dynamically fetched via AJAX
  - Static HTML scraping is incomplete and unreliable

By replicating the AJAX requests, the system retrieves structured data efficiently without browser automation.


## Data Filtering and Cleaning

### Inclusion Criteria
- Valid issue date
- Matching target year
- Unique URLs
- Successfully extracted body text
- Sufficient body length for analysis

### Exclusion Criteria
- Missing or invalid dates
- Records outside target year
- Duplicate entries
- Letters where:
  - body container is missing
  - extracted text is empty or too short
  - LLM output is invalid or fails validation

### Text Cleaning

Warning letters contain structural noise. The system removes:
- headers and titles
- greetings
- closing signatures
- repeated instructional content

This ensures only meaningful content is passed to the LLM.


## Risk Scoring Design

### Body Extraction

FDA warning letters follow semi-structured formats but include significant noise.  
The system extracts only the core narrative using:

1. Anchor-based extraction (preferred)
2. Prefix-based fallback
3. Full cleaned block fallback

This multi-stage approach improves robustness across varying document formats.


### LLM Structured Extraction

The LLM is used as a **structured feature extractor**, not a free-form text generator.

It produces:
- key_risk_drivers
- cited_findings
- affected_quality_systems
- recommended_follow_up_actions
- multiple sub-risk scores:
  - patient safety risk
  - systemic quality breakdown
  - data integrity risk
  - remediation complexity
  - regulatory escalation risk

This design ensures:
- consistent schema
- machine-readable outputs
- reliable downstream aggregation

By constraining the output format, the system reduces variability and improves interpretability.


### Final Risk Score

The final risk score is computed deterministically as a weighted combination of LLM sub-scores.

This approach:
- avoids reliance on opaque LLM reasoning
- ensures reproducibility
- allows clear interpretation of contributing factors

The LLM is responsible for extracting structured evidence, while scoring remains deterministic.


## Analysis Design

The scoring stage produces nested objects (e.g., metadata + LLM outputs), which are not directly suitable for aggregation.

To enable analysis:

1. Flatten scored records into tabular format
2. Convert into a DataFrame
3. Perform aggregation using pandas

This enables:
- group-level analysis (office, company, subject tokens)
- severity-aware metrics (average risk, high/critical rate)
- system-level insights (quality system failures)

This design separates:
- object-oriented scoring logic
- table-based analytical processing

and allows the system to scale as new features are added.


## Results

This analysis reveals that regulatory risk is not uniformly distributed, but instead concentrated in specific domains, time periods, and systemic failure patterns.

All visualizations referenced in this section are generated from the analysis pipeline and saved under:

`fda-regulations/output/risk_intelligence_charts_2025/`

---

### Overall Risk Distribution

The dataset contains 689 FDA warning letters spanning 2025.

- Mean final risk score: **0.507**
- Distribution:
  - ELEVATED: 367
  - HIGH: 149
  - MODERATE: 88
  - CRITICAL: 85

This indicates that while most warning letters fall into moderate-risk categories, a substantial portion (~34%) represent high or critical regulatory risk.

---

### Temporal Risk Patterns

Risk is not uniformly distributed over time.

- Early year (Jan–Feb): highest risk levels (~0.65 average)
- Mid-year (Mar–May): significant drop (~0.46 average)
- Late year (Dec): rising trend again
- Lowest point observed around August

High/Critical rates follow a similar pattern.

**Insight:**  
Regulatory severity shows temporal clustering, suggesting that high-risk enforcement actions are concentrated in specific periods rather than evenly distributed.

---

### Risk by Issuing Office

Significant variation exists across regulatory domains:

**High-risk domains:**
- Office of Manufacturing Quality (~0.85 average)
- Device and biologics-related offices

**Lower-risk domains:**
- Center for Tobacco Products (~0.21 average)

**Insight:**  
The meaning of a warning letter depends heavily on domain context.  
Manufacturing-related enforcement corresponds to significantly higher risk than administrative or labeling violations.

---

### Quality System Drivers of Risk

Risk lift analysis shows that failures in the following systems are most strongly associated with high severity:

- Document Control
- Quality Unit
- Packaging & Labeling Systems
- Supplier / Purchasing Controls

**Insight:**  
High-risk cases are driven by systemic quality breakdowns rather than isolated violations.

---

### Subject Token Risk Signals

Certain subject tokens strongly correlate with high risk:

**High-risk tokens:**
- CGMP
- QSR
- Medical Devices
- Finished Pharmaceuticals
- Active Pharmaceutical Ingredient (API)

These tokens show large risk lift vs baseline:
- API: +0.346
- Finished pharmaceuticals: +0.327
- CGMP/QSR: +0.302

**Insight:**  
Manufacturing-related violations are the strongest predictors of severe regulatory risk.

---

### Low-Risk Regulatory Categories

Certain categories consistently show lower risk:

- Foreign Supplier Verification Program (FSVP)
- Tobacco-related violations
- Labeling and administrative issues

These cases typically involve compliance gaps rather than direct product safety risks.

**Insight:**  
Not all warning letters indicate product risk; many represent regulatory or administrative non-compliance.


## Key Insights

- Regulatory risk is primarily driven by **systemic quality failures**, not isolated defects  
- **Manufacturing-related violations** (CGMP, QSR, API, finished products) are the strongest predictors of high severity  
- Risk varies significantly by **regulatory domain** (device/drug vs tobacco)  
- High-risk enforcement actions show **temporal clustering**, not uniform distribution  


## Key Design Decisions

### AJAX-Based Data Retrieval

Instead of using Selenium or static HTML parsing, the system directly replicates AJAX requests used by the FDA frontend.

Benefits:
- lower latency
- reduced overhead
- more reliable structured data access

Trade-off:
- depends on undocumented endpoints
- may break if frontend changes


### Single-Stage LLM Pipeline

Two approaches were considered:
1. Multi-stage LLM pipeline (summarization → extraction → scoring)
2. Single structured extraction call + deterministic scoring

The second approach was chosen because:
- lower latency
- lower cost
- reduced error propagation
- simpler pipeline

Summarization can be added later as a fallback for extremely long documents.


### pandas-Based Analysis Layer

Initial approaches using plain Python were less maintainable for complex aggregation.

pandas was chosen because:
- natural fit for tabular data
- concise groupby/aggregation operations
- built-in support for:
  - explode (token-level analysis)
  - aggregation metrics
  - sorting and ranking
- direct integration with visualization tools

Alternative tools (e.g., Polars) were considered, but pandas provided sufficient performance and better familiarity for this prototype.


### Data Model Design

Metadata is embedded directly within each scored record rather than duplicated across structures.

This:
- avoids redundancy
- simplifies schema evolution
- eliminates the need for joins
- integrates naturally with flattening for analysis


## Limitations

- LLM outputs may vary in phrasing and require normalization
- Extraction quality depends on prompt design and model behavior
- Some fields (e.g., cited findings) may require clustering for consistency
- Dataset is limited by filtering and extraction success
- The system is designed for analytical insight, not as a replacement for expert review


## Next Steps

- Introduce normalization and clustering for cited findings
- Improve robustness of LLM extraction (retry, stricter schema validation)
- Optimize aggregation logic for scalability
- Support incremental updates and scheduled execution
- Integrate outputs into dashboards or alerting systems