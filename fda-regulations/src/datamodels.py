from dataclasses import dataclass

@dataclass(slots=True)
class WarningLetterMetadata:
    url: str
    posted_date: str
    issue_date_str: str
    company_name: str
    issuing_office: str
    subject: str


@dataclass(slots=True)
class LLMRiskResult:
    patient_safety_risk_score: float
    systemic_quality_breakdown_score: float
    data_integrity_risk_score: float
    remediation_complexity_score: float
    regulatory_escalation_risk_score: float

    key_risk_drivers: list[str]
    cited_findings: list[str]
    affected_quality_systems: list[str]
    recommended_follow_up_actions: list[str]

    reasoning_summary: str
    confidence: str
    requires_human_review: bool


@dataclass(slots=True)
class ScoredLetter:
    metadata: WarningLetterMetadata
    subject_tokens: list[str]
    final_risk_score: float
    final_risk_level: str
    llm_analysis: LLMRiskResult


@dataclass(slots=True)
class FlattenedRow:
    url: str
    company_name: str
    issue_date: str
    posted_date: str
    issuing_office: str
    subject: str
    subject_tokens: list[str]
    final_risk_score: float
    final_risk_level: str
    patient_safety_risk_score: float
    systemic_quality_breakdown_score: float
    data_integrity_risk_score: float
    remediation_complexity_score: float
    regulatory_escalation_risk_score: float
    affected_quality_systems: list[str]