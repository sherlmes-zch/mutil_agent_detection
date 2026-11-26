"""Versioned, serializable contracts for the audit workflow.

The models deliberately contain no LangChain or LangGraph imports.  They can
therefore be used as structured-output schemas, persisted graph state, or
ordinary JSON documents without coupling the protocol to an orchestration
implementation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


PROTOCOL_VERSION = "0.2"


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class ProtocolModel(BaseModel):
    """Base model with strict, JSON-friendly defaults."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )


class ClaimKind(str, Enum):
    RISK = "risk"
    PROTECTION = "protection"
    FACT = "fact"


class ClaimStatus(str, Enum):
    CANDIDATE = "candidate"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    WITHDRAWN = "withdrawn"


class EvidenceStatus(str, Enum):
    CANDIDATE = "candidate"
    SUPPORTS = "supports"
    NEGATES = "negates"
    INCONCLUSIVE = "inconclusive"


class QueryKind(str, Enum):
    SOURCE_SEARCH = "source_search"
    CPG = "cpg"
    ALLOCATION = "allocation_relation"
    NUMERIC_RANGE = "numeric_range"
    PARAMETER_MAPPING = "parameter_mapping"
    SIDE_EFFECT = "side_effect"
    PATH_REACHABILITY = "path_reachability"
    SYMBOLIC_EXECUTION = "symbolic_execution"
    SMT = "smt"


class Backend(str, Enum):
    SOURCE = "source"
    JOERN = "joern"
    SMT = "smt"
    KLEE = "klee"
    OTHER = "other"


class QueryDecision(str, Enum):
    PROPOSED = "proposed"
    SELECTED = "selected"
    EXECUTED = "executed"
    SKIPPED = "skipped"
    FAILED = "failed"


class AuditPhase(str, Enum):
    INITIALIZED = "initialized"
    ANALYZING = "analyzing"
    DIAGNOSING = "diagnosing"
    PLANNING = "planning"
    INVESTIGATING = "investigating"
    VERIFYING = "verifying"
    UPDATING = "updating"
    COMPLETED = "completed"
    STOPPED = "stopped"


class Prediction(str, Enum):
    VULNERABLE = "vulnerable"
    BENIGN = "benign"
    UNKNOWN = "unknown"


class SourceLocation(ProtocolModel):
    """Stable source anchor; temporary CPG node ids are intentionally absent."""

    file: str
    function: str | None = None
    line: int | None = Field(default=None, ge=1)
    column: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    code_hash: str | None = None

    @field_validator("file")
    @classmethod
    def file_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source file cannot be empty")
        return value


class SourceSnapshot(ProtocolModel):
    version: str
    root: str | None = None
    commit: str | None = None
    tool_config_hash: str | None = None
    dependencies_hash: str | None = None


class ProgramScope(ProtocolModel):
    """The semantic range to which a claim or evidence is allowed to apply."""

    snapshot: SourceSnapshot
    entrypoint: str
    call_context: list[str] = Field(default_factory=list)
    locations: list[SourceLocation] = Field(default_factory=list)
    target_operation: str | None = None
    target_object: str | None = None
    path_condition: str | None = None
    state_variables: list[str] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)


class Budget(ProtocolModel):
    token_limit: int | None = Field(default=None, ge=0)
    tool_call_limit: int | None = Field(default=None, ge=0)
    time_limit_seconds: float | None = Field(default=None, ge=0)
    path_limit: int | None = Field(default=None, ge=0)


class AnalysisInput(ProtocolModel):
    """Immutable request entering the graph for one independently audited sample."""

    protocol_version: str = PROTOCOL_VERSION
    sample_id: str
    function_name: str
    source: str
    scope: ProgramScope
    budget: Budget = Field(default_factory=Budget)
    dataset_split: Literal["train", "validation", "test", "external"] | None = None
    expected_label: int | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(ProtocolModel):
    id: str = Field(default_factory=lambda: _id("claim"))
    kind: ClaimKind
    text: str
    status: ClaimStatus = ClaimStatus.CANDIDATE
    scope: ProgramScope
    premise_ids: list[str] = Field(default_factory=list)
    source_anchors: list[SourceLocation] = Field(default_factory=list)
    created_by: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    withdrawn_reason: str | None = None
    # Role-specific, model-authored details retained for later diagnosis and queries.
    analysis: dict[str, Any] = Field(default_factory=dict)


class DisputeFact(ProtocolModel):
    id: str
    statement: str
    evidence_refs: list[str] = Field(default_factory=list)


class DisputeClaimRelation(ProtocolModel):
    risk_candidate_id: str
    protection_ids: list[str] = Field(default_factory=list)
    relation: Literal[
        "supports", "contradicts", "limits", "does_not_address", "unresolved", "unrelated"
    ]
    object_relation: Literal["same", "overlapping", "different", "unknown"] = "unknown"
    property_relation: Literal["matches", "mismatches", "not_applicable", "unknown"] = "unknown"
    path_coverage: Literal["complete", "partial", "none", "unknown"] = "unknown"
    explanation: str
    evidence_refs: list[str] = Field(default_factory=list)


class DisputePremise(ProtocolModel):
    id: str
    description: str
    relation: Literal[
        "supports", "contradicts", "limits", "does_not_address", "unresolved", "unrelated"
    ] = "unresolved"
    status: Literal["supported", "unknown", "refuted"] = "unknown"
    claim_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    decisive: bool = False


class InvestigationTarget(ProtocolModel):
    premise_id: str
    question: str
    discriminating_outcomes: list[str] = Field(min_length=2, max_length=4)
    suggested_query_kinds: list[QueryKind] = Field(default_factory=list)
    evidence_requirement: str


class Dispute(ProtocolModel):
    id: str = Field(default_factory=lambda: _id("dispute"))
    scope: ProgramScope
    risk_claim_id: str
    protection_claim_id: str
    diagnosis_status: Literal["complete", "partial", "insufficient_evidence"] = "insufficient_evidence"
    alignment_status: Literal["aligned", "partially_aligned", "misaligned", "unresolved"] = "unresolved"
    summary: str = ""
    facts: list[DisputeFact] = Field(default_factory=list)
    claim_relations: list[DisputeClaimRelation] = Field(default_factory=list)
    premises: list[DisputePremise] = Field(default_factory=list)
    investigation_targets: list[InvestigationTarget] = Field(default_factory=list)
    diagnosis_error: str | None = None
    # Compatibility projections; unresolved and decisive entries are premise IDs.
    shared_facts: list[str] = Field(default_factory=list)
    competing_premises: list[str] = Field(default_factory=list)
    unresolved_premises: list[str] = Field(default_factory=list)
    decisive_premises: list[str] = Field(default_factory=list)
    aligned: bool = False
    alignment_notes: str | None = None
    created_by: str = "disagreement_diagnoser"


class DiagnosisRiskCandidate(ProtocolModel):
    candidate_id: str
    model_entity_id: str | None = None
    candidate_class: Literal[
        "direct_violation", "conditional_interprocedural", "speculative"
    ] = "speculative"
    priority: int | None = None
    summary: str
    security_property: str | None = None
    risk_operation: str | None = None
    target_object: str | None = None
    address_source: str | None = None
    access_type: str | None = None
    alignment_requirement: str | None = None
    conversion_source_type: str | None = None
    conversion_target_type: str | None = None
    representation_requirement: str | None = None
    risk_path: list[dict[str, Any]] = Field(default_factory=list)
    required_conditions: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    source_ref_ids: list[str] = Field(default_factory=list)


class DiagnosisProtection(ProtocolModel):
    protection_id: str
    model_entity_id: str | None = None
    kind: str | None = None
    description: str
    protected_object: str | None = None
    path_condition: str | None = None
    coverage: Literal["complete", "partial", "unknown"] = "unknown"
    blocks_conditions: list[str] = Field(default_factory=list)
    bypass_paths: list[str] = Field(default_factory=list)
    source_ref_ids: list[str] = Field(default_factory=list)


class DiagnosisSourceRef(ProtocolModel):
    ref_id: str
    side: Literal["risk", "protection", "joern"]
    excerpt: str | None = None
    location: SourceLocation | None = None
    basis: str | None = None


class DiagnosisInput(ProtocolModel):
    """Bounded, label-free comparison context for the disagreement diagnostician."""

    sample_id: str
    scope: ProgramScope
    risk_claim_id: str
    risk_claim_status: ClaimStatus
    risk_claim_text: str
    risk_candidates: list[DiagnosisRiskCandidate] = Field(default_factory=list)
    risk_source_ref_ids: list[str] = Field(default_factory=list)
    protection_claim_id: str
    protection_claim_status: ClaimStatus
    protection_claim_text: str
    protections: list[DiagnosisProtection] = Field(default_factory=list)
    protection_invariants: list[dict[str, Any]] = Field(default_factory=list)
    protection_source_ref_ids: list[str] = Field(default_factory=list)
    source_refs: list[DiagnosisSourceRef] = Field(default_factory=list)
    relevant_skeleton: dict[str, Any] = Field(default_factory=dict)


class QueryOutcome(ProtocolModel):
    """A declared branch and the updates it permits before execution."""

    label: str
    meaning: str
    supports_premise_ids: list[str] = Field(default_factory=list)
    negates_premise_ids: list[str] = Field(default_factory=list)
    leaves_unknown_premise_ids: list[str] = Field(default_factory=list)


class QuerySpec(ProtocolModel):
    id: str = Field(default_factory=lambda: _id("query"))
    target_claim_ids: list[str] = Field(default_factory=list)
    target_premise_ids: list[str] = Field(default_factory=list)
    scope: ProgramScope
    kind: QueryKind
    backend: Backend
    question: str
    outcomes: list[QueryOutcome] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    evidence_requirement: str
    budget: Budget = Field(default_factory=Budget)
    impact: float | None = Field(default=None, ge=0, le=1)
    separation: float | None = Field(default=None, ge=0, le=1)
    answerability: float | None = Field(default=None, ge=0, le=1)
    estimated_cost: float | None = Field(default=None, ge=0)
    model_risk: float | None = Field(default=None, ge=0, le=1)
    decision: QueryDecision = QueryDecision.PROPOSED
    proposed_by: str = "query_planner"
    required_context: Literal["function", "repository", "build"] = "function"
    selection_score: float | None = None
    selection_reasons: list[str] = Field(default_factory=list)


class QueryPlanningInput(ProtocolModel):
    """Bounded, current-sample context for selecting a discriminating query."""

    sample_id: str
    risk_claim: Claim
    protection_claim: Claim
    dispute: Dispute
    scope: ProgramScope
    program_skeleton: dict[str, Any] = Field(default_factory=dict)
    budget: Budget = Field(default_factory=Budget)
    prior_queries: list[QuerySpec] = Field(default_factory=list)
    enabled_backends: list[Backend] = Field(default_factory=list)


class MemoryRecord(ProtocolModel):
    """A reusable investigation strategy, never a direct current-sample verdict."""

    id: str = Field(default_factory=lambda: _id("memory"))
    dispute_pattern: str
    description: str
    recommended_query_kinds: list[QueryKind] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    applicability_conditions: list[str] = Field(default_factory=list)
    common_failures: list[str] = Field(default_factory=list)
    follow_up_actions: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    source_split: Literal["training", "external", "curated"]
    frozen: bool = True


class Evidence(ProtocolModel):
    id: str = Field(default_factory=lambda: _id("evidence"))
    status: EvidenceStatus
    summary: str
    raw_result: str | None = None
    structured_facts: dict[str, Any] = Field(default_factory=dict)
    source_anchors: list[SourceLocation] = Field(default_factory=list)
    scope: ProgramScope
    query_id: str | None = None
    supports_claim_ids: list[str] = Field(default_factory=list)
    negates_claim_ids: list[str] = Field(default_factory=list)
    supports_premise_ids: list[str] = Field(default_factory=list)
    negates_premise_ids: list[str] = Field(default_factory=list)
    tool: Backend
    tool_version: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    analysis_boundary: str | None = None
    witness: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FactInvestigationInput(ProtocolModel):
    """Bounded input for interpreting one executed query result."""

    query: QuerySpec
    evidence: Evidence
    premises: list[DisputePremise] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    risk_candidates: list[DiagnosisRiskCandidate] = Field(default_factory=list)
    protections: list[DiagnosisProtection] = Field(default_factory=list)
    scope: ProgramScope


class FactStatement(ProtocolModel):
    id: str
    statement: str
    evidence_refs: list[str] = Field(default_factory=list)
    source_anchors: list[SourceLocation] = Field(default_factory=list)


class RiskCandidateAssessment(ProtocolModel):
    candidate_id: str
    status: Literal["supported", "refuted", "unknown"]
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)


class CandidateConditionAssessment(ProtocolModel):
    """Evidence status for one necessary condition of one risk candidate."""

    candidate_id: str
    premise_id: str
    status: Literal["supported", "refuted", "unknown"]
    reason: str
    evidence_refs: list[str] = Field(default_factory=list)


class ProtectionAssessment(ProtocolModel):
    protection_id: str
    coverage_status: Literal["effective", "ineffective", "unknown"]
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)


class CallerContract(ProtocolModel):
    caller_function: str
    call_site: str
    call_site_location: SourceLocation | None = None
    argument_expression: str
    allocation_expression: str | None = None
    read_length_expression: str | None = None
    required_size_expression: str | None = None
    short_read_guard: Literal["present", "absent", "unknown"] = "unknown"
    extent_relation: Literal["sufficient", "insufficient", "unknown"] = "unknown"
    evidence_refs: list[str] = Field(default_factory=list)


class PremiseUpdate(ProtocolModel):
    premise_id: str
    status: Literal["supported", "unknown", "refuted"]
    reason: str
    evidence_refs: list[str] = Field(default_factory=list)


class FactInvestigationResult(ProtocolModel):
    query_id: str
    investigation_status: Literal["complete", "partial", "inconclusive"]
    outcome_label: str | None = None
    facts: list[FactStatement] = Field(default_factory=list)
    risk_assessments: list[RiskCandidateAssessment] = Field(default_factory=list)
    candidate_condition_assessments: list[CandidateConditionAssessment] = Field(default_factory=list)
    protection_assessments: list[ProtectionAssessment] = Field(default_factory=list)
    caller_contracts: list[CallerContract] = Field(default_factory=list)
    premise_updates: list[PremiseUpdate] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    created_by: str = "fact_investigator"


class AuditEvent(ProtocolModel):
    id: str = Field(default_factory=lambda: _id("event"))
    phase: AuditPhase
    actor: str
    message: str
    related_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuditState(ProtocolModel):
    """Persistable LangGraph state for one sample; no cross-sample conclusions."""

    protocol_version: str = PROTOCOL_VERSION
    sample_id: str
    input: AnalysisInput
    phase: AuditPhase = AuditPhase.INITIALIZED
    claims: dict[str, Claim] = Field(default_factory=dict)
    disputes: dict[str, Dispute] = Field(default_factory=dict)
    queries: dict[str, QuerySpec] = Field(default_factory=dict)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    fact_investigations: dict[str, FactInvestigationResult] = Field(default_factory=dict)
    events: list[AuditEvent] = Field(default_factory=list)
    unresolved_premises: list[str] = Field(default_factory=list)
    remaining_budget: Budget = Field(default_factory=Budget)
    query_round: int = Field(default=0, ge=0)
    stop_reason: str | None = None


class FinalReport(ProtocolModel):
    protocol_version: str = PROTOCOL_VERSION
    sample_id: str
    prediction: Prediction
    decision_policy: Literal["strict", "balanced", "recall_first"] = "strict"
    decision_confidence: Literal["high", "medium", "low"] = "low"
    decision_evidence_grade: Literal["A", "B", "C", "none"] = "none"
    decision_basis: list[str] = Field(default_factory=list)
    decision_source_anchors: list[SourceLocation] = Field(default_factory=list)
    initial_evidence_summary: list[str] = Field(default_factory=list)
    vulnerability_type: str | None = None
    retained_claim_ids: list[str] = Field(default_factory=list)
    risk_claim_ids: list[str] = Field(default_factory=list)
    protection_claim_ids: list[str] = Field(default_factory=list)
    supported_premise_ids: list[str] = Field(default_factory=list)
    refuted_premise_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    risk_assessment: list[str] = Field(default_factory=list)
    protection_assessment: list[str] = Field(default_factory=list)
    disagreement_summary: str | None = None
    investigated_facts: list[FactStatement] = Field(default_factory=list)
    risk_assessments: list[RiskCandidateAssessment] = Field(default_factory=list)
    candidate_condition_assessments: list[CandidateConditionAssessment] = Field(default_factory=list)
    protection_assessments: list[ProtectionAssessment] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    unresolved_premises: list[str] = Field(default_factory=list)
    conclusion_scope: ProgramScope
    explanation: str
    limitations: list[str] = Field(default_factory=list)
    audit_event_ids: list[str] = Field(default_factory=list)


__all__ = [
    "PROTOCOL_VERSION",
    "AnalysisInput",
    "AuditEvent",
    "AuditPhase",
    "AuditState",
    "Backend",
    "Budget",
    "Claim",
    "ClaimKind",
    "ClaimStatus",
    "DiagnosisInput",
    "DiagnosisProtection",
    "DiagnosisRiskCandidate",
    "DiagnosisSourceRef",
    "Dispute",
    "DisputeClaimRelation",
    "DisputeFact",
    "DisputePremise",
    "Evidence",
    "FactInvestigationInput",
    "FactInvestigationResult",
    "FactStatement",
    "CallerContract",
    "EvidenceStatus",
    "FinalReport",
    "MemoryRecord",
    "InvestigationTarget",
    "Prediction",
    "ProgramScope",
    "PremiseUpdate",
    "RiskCandidateAssessment",
    "CandidateConditionAssessment",
    "ProtectionAssessment",
    "QueryDecision",
    "QueryPlanningInput",
    "QueryKind",
    "QueryOutcome",
    "QuerySpec",
    "SourceLocation",
    "SourceSnapshot",
]
