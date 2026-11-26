"""Structured model responses shared by the independent analyst agents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .protocols import Backend, QueryKind, SourceLocation


class PremiseDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable identifier, for example risk-premise-1")
    description: str
    status: Literal["supported", "unknown", "refuted"] = "unknown"
    evidence_refs: list[str] = Field(default_factory=list)


class ProtectionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable identifier, for example protection-1")
    kind: str
    description: str
    protected_object: str
    coverage: Literal["complete", "partial", "unknown"] = "unknown"
    covered_conditions: list[str] = Field(default_factory=list)
    bypass_paths: list[str] = Field(default_factory=list)
    source_anchors: list[SourceLocation] = Field(default_factory=list)


class ClaimDraft(BaseModel):
    """Strict JSON response accepted from either role-specific agent."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="One conditional, evidence-conscious claim")
    premise_ids: list[str] = Field(default_factory=list)
    source_anchors: list[SourceLocation] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    security_property: str | None = None
    risk_operation: str | None = None
    target_object: str | None = None
    risk_path: list[str] = Field(default_factory=list)
    required_conditions: list[PremiseDraft] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    protections: list[ProtectionDraft] = Field(default_factory=list)
    invariants: list[PremiseDraft] = Field(default_factory=list)


class AgentAnchorDraft(BaseModel):
    """A citation that can be checked against supplied source or Joern evidence."""

    model_config = ConfigDict(extra="forbid")

    # file/line are accepted for older responses, but the program derives them
    # from excerpt and basis before the claim can use the citation.
    file: str | None = None
    line: int | None = Field(default=None, ge=1)
    excerpt: str = Field(min_length=1, max_length=300)
    basis: Literal["function_source", "joern_skeleton"]


class PremiseAnalysisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    status: Literal["supported", "unknown", "refuted"] = "unknown"
    evidence_refs: list[AgentAnchorDraft] = Field(default_factory=list, max_length=5)


class RiskPathStepDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1)
    operation: str = Field(min_length=1, max_length=300)
    object: str = Field(min_length=1, max_length=160)
    effect: str = Field(min_length=1, max_length=300)
    source_anchors: list[AgentAnchorDraft] = Field(min_length=1, max_length=3)


class RiskCandidateDraft(BaseModel):
    """One independently testable and evidence-backed risk hypothesis."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1, max_length=80)
    candidate_class: Literal[
        "direct_violation", "conditional_interprocedural", "speculative"
    ] = "speculative"
    priority: int = Field(ge=1, le=3)
    summary: str = Field(min_length=1, max_length=800)
    security_property: str | None = Field(default=None, max_length=100)
    risk_operation: str = Field(min_length=1, max_length=300)
    target_object: str = Field(min_length=1, max_length=160)
    address_source: str | None = Field(default=None, max_length=300)
    access_type: str | None = Field(default=None, max_length=120)
    alignment_requirement: str | None = Field(default=None, max_length=240)
    conversion_source_type: str | None = Field(default=None, max_length=120)
    conversion_target_type: str | None = Field(default=None, max_length=120)
    representation_requirement: str | None = Field(default=None, max_length=300)
    risk_path: list[RiskPathStepDraft] = Field(min_length=1, max_length=6)
    required_conditions: list[PremiseAnalysisDraft] = Field(min_length=1, max_length=6)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=6)
    source_anchors: list[AgentAnchorDraft] = Field(min_length=1, max_length=8)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def path_orders_are_unique(self):
        orders = [step.order for step in self.risk_path]
        if len(orders) != len(set(orders)):
            raise ValueError("risk_path step orders must be unique")
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError("risk_path step orders must be consecutive from 1")
        return self


class RiskAnalysisDraft(BaseModel):
    """Ranked set of distinct, conditional risk-path hypotheses."""

    model_config = ConfigDict(extra="forbid")

    candidate_found: bool
    summary: str = Field(min_length=1, max_length=1200)
    candidates: list[RiskCandidateDraft] = Field(default_factory=list, max_length=3)
    # Legacy single-candidate fields remain accepted during schema migration.
    security_property: str | None = Field(default=None, max_length=100)
    risk_operation: str | None = Field(default=None, max_length=300)
    target_object: str | None = Field(default=None, max_length=160)
    address_source: str | None = Field(default=None, max_length=300)
    access_type: str | None = Field(default=None, max_length=120)
    alignment_requirement: str | None = Field(default=None, max_length=240)
    risk_path: list[RiskPathStepDraft] = Field(default_factory=list, max_length=8)
    required_conditions: list[PremiseAnalysisDraft] = Field(default_factory=list, max_length=8)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=8)
    source_anchors: list[AgentAnchorDraft] = Field(default_factory=list, max_length=8)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def path_orders_are_unique(self):
        if self.candidates:
            priorities = [candidate.priority for candidate in self.candidates]
            if priorities != list(range(1, len(priorities) + 1)):
                raise ValueError("candidate priorities must be consecutive, starting at 1")
            if not self.candidate_found:
                raise ValueError("candidate_found must be true when candidates are present")
            return self
        orders = [step.order for step in self.risk_path]
        if len(orders) != len(set(orders)):
            raise ValueError("risk_path step orders must be unique")
        if orders and sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError("risk_path step orders must be consecutive from 1")
        if not self.candidate_found and (self.risk_path or self.required_conditions):
            raise ValueError("a negative candidate result cannot include a risk path or conditions")
        if self.candidate_found and (
            not self.risk_operation or not self.target_object or not self.risk_path or not self.required_conditions
        ):
            raise ValueError("a legacy risk candidate requires an operation, target object, path, and trigger conditions")
        if not self.candidate_found and any((self.security_property, self.risk_operation, self.target_object)):
            raise ValueError("a negative candidate result cannot name a risk operation or object")
        return self


class ProtectionMechanismDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    kind: Literal[
        "bounds_check", "validation", "alignment", "type_safety", "ownership", "lifetime", "initialization",
        "error_handling", "caller_contract", "other",
    ]
    description: str = Field(min_length=1, max_length=500)
    protected_object: str = Field(min_length=1, max_length=160)
    path_condition: str = Field(min_length=1, max_length=300)
    coverage: Literal["complete", "partial", "unknown"] = "unknown"
    blocks_conditions: list[str] = Field(default_factory=list, max_length=8)
    bypass_paths: list[str] = Field(default_factory=list, max_length=8)
    source_anchors: list[AgentAnchorDraft] = Field(min_length=1, max_length=5)


class ProtectionAnalysisDraft(BaseModel):
    """Role-specific, conditional protection explanation."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=800)
    protections: list[ProtectionMechanismDraft] = Field(default_factory=list, max_length=8)
    invariants: list[PremiseAnalysisDraft] = Field(default_factory=list, max_length=8)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=8)
    source_anchors: list[AgentAnchorDraft] = Field(default_factory=list, max_length=8)
    confidence: float | None = Field(default=None, ge=0, le=1)


class DisputeFactDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    statement: str = Field(min_length=1, max_length=500)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)


class DisputeClaimRelationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_candidate_id: str = Field(min_length=1, max_length=80)
    protection_ids: list[str] = Field(default_factory=list, max_length=8)
    relation: Literal[
        "supports", "contradicts", "limits", "does_not_address", "unresolved", "unrelated"
    ]
    object_relation: Literal["same", "overlapping", "different", "unknown"] = "unknown"
    property_relation: Literal["matches", "mismatches", "not_applicable", "unknown"] = "unknown"
    path_coverage: Literal["complete", "partial", "none", "unknown"] = "unknown"
    explanation: str = Field(min_length=1, max_length=500)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)


class DisputePremiseDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    relation: Literal[
        "supports", "contradicts", "limits", "does_not_address", "unresolved", "unrelated"
    ] = "unresolved"
    status: Literal["supported", "unknown", "refuted"] = "unknown"
    claim_refs: list[str] = Field(default_factory=list, max_length=8)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)
    decisive: bool = False


class InvestigationTargetDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    premise_id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=500)
    discriminating_outcomes: list[str] = Field(min_length=2, max_length=4)
    suggested_query_kinds: list[QueryKind] = Field(default_factory=list, max_length=4)
    evidence_requirement: str = Field(min_length=1, max_length=500)


class DisputeDraft(BaseModel):
    """Structured alignment and premise-diagnosis result; contains no verdict."""

    model_config = ConfigDict(extra="forbid")

    diagnosis_status: Literal["complete", "partial", "insufficient_evidence"]
    alignment_status: Literal["aligned", "partially_aligned", "misaligned", "unresolved"]
    summary: str = Field(min_length=1, max_length=1200)
    facts: list[DisputeFactDraft] = Field(default_factory=list, max_length=12)
    claim_relations: list[DisputeClaimRelationDraft] = Field(default_factory=list, max_length=12)
    premises: list[DisputePremiseDraft] = Field(default_factory=list, max_length=16)
    investigation_targets: list[InvestigationTargetDraft] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_premise_references(self):
        premise_ids = [premise.id for premise in self.premises]
        if len(premise_ids) != len(set(premise_ids)):
            raise ValueError("dispute premise IDs must be unique")
        if any(target.premise_id not in premise_ids for target in self.investigation_targets):
            raise ValueError("every investigation target must refer to a declared premise")
        return self


class QueryOutcomeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=80)
    meaning: str = Field(min_length=1, max_length=300)
    supports_premise_ids: list[str] = Field(default_factory=list, max_length=8)
    negates_premise_ids: list[str] = Field(default_factory=list, max_length=8)
    leaves_unknown_premise_ids: list[str] = Field(default_factory=list, max_length=8)


class QueryCandidateDraft(BaseModel):
    """One testable query proposal with predeclared, premise-level outcomes."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1, max_length=80)
    target_claim_ids: list[str] = Field(min_length=1, max_length=4)
    target_premise_ids: list[str] = Field(min_length=1, max_length=8)
    kind: QueryKind
    backend: Backend
    question: str = Field(min_length=1, max_length=700)
    outcomes: list[QueryOutcomeDraft] = Field(min_length=2, max_length=4)
    assumptions: list[str] = Field(default_factory=list, max_length=8)
    evidence_requirement: str = Field(min_length=1, max_length=500)
    required_context: Literal["function", "repository", "build"] = "function"


class QueryPlanDraft(BaseModel):
    """Unranked candidate set; ranking and eligibility are program-controlled."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1000)
    candidates: list[QueryCandidateDraft] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def candidate_ids_are_unique(self):
        ids = [item.candidate_id for item in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("query candidate IDs must be unique")
        return self


class FactStatementDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    statement: str = Field(min_length=1, max_length=600)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)
    source_anchor_refs: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def bound_compatible_fields(cls, value):
        if isinstance(value, dict):
            source = dict(value)
            value = {
                "id": source.get("id") or source.get("fact_id") or "fact",
                "statement": source.get("statement") or source.get("fact") or "No statement supplied.",
                "evidence_refs": list(source.get("evidence_refs") or [])[:8],
                "source_anchor_refs": list(
                    source.get("source_anchor_refs") or source.get("source_anchors") or []
                )[:8],
            }
        return value


class RiskCandidateAssessmentDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1, max_length=80)
    status: Literal["supported", "refuted", "unknown"]
    rationale: str = Field(min_length=1, max_length=600)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def normalize_compatible_fields(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            alias = value.pop("assessment", None)
            value["status"] = value.get("status") or alias or "unknown"
            value["rationale"] = value.get("rationale") or value.pop("reason", None) or "No rationale supplied."
            value["evidence_refs"] = list(value.get("evidence_refs") or [])[:8]
        return value


class CandidateConditionAssessmentDraft(BaseModel):
    """Assessment of one necessary premise, never a sample-level verdict."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1, max_length=80)
    premise_id: str = Field(min_length=1, max_length=80)
    status: Literal["supported", "refuted", "unknown"]
    reason: str = Field(min_length=1, max_length=600)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def normalize_compatible_fields(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            value["status"] = value.get("status") or value.pop("assessment", None) or "unknown"
            value["reason"] = value.get("reason") or value.pop("rationale", None) or "No reason supplied."
            value["evidence_refs"] = list(value.get("evidence_refs") or [])[:8]
        return value


class ProtectionAssessmentDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protection_id: str = Field(min_length=1, max_length=80)
    coverage_status: Literal["effective", "ineffective", "unknown"]
    rationale: str = Field(min_length=1, max_length=600)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def normalize_compatible_fields(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            alias = value.pop("status", None)
            value["coverage_status"] = value.get("coverage_status") or alias or "unknown"
            value["rationale"] = value.get("rationale") or value.pop("reason", None) or "No rationale supplied."
            value["evidence_refs"] = list(value.get("evidence_refs") or [])[:8]
        return value


class CallerContractDraft(BaseModel):
    """Structured caller-side extent/argument mapping extracted from SOURCE evidence."""

    model_config = ConfigDict(extra="forbid")

    caller_function: str = Field(min_length=1, max_length=160)
    call_site: str = Field(min_length=1, max_length=300)
    call_site_location: SourceLocation | None = None
    argument_expression: str = Field(min_length=1, max_length=300)
    allocation_expression: str | None = Field(default=None, max_length=500)
    read_length_expression: str | None = Field(default=None, max_length=500)
    required_size_expression: str | None = Field(default=None, max_length=300)
    short_read_guard: Literal["present", "absent", "unknown"] = "unknown"
    extent_relation: Literal["sufficient", "insufficient", "unknown"] = "unknown"
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)


class PremiseUpdateDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    premise_id: str = Field(min_length=1, max_length=80)
    status: Literal["supported", "unknown", "refuted"]
    reason: str = Field(min_length=1, max_length=600)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def normalize_compatible_fields(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            alias = value.pop("assessment", None)
            value["status"] = value.get("status") or alias or "unknown"
            value["reason"] = value.get("reason") or value.pop("rationale", None) or "No reason supplied."
            value["evidence_refs"] = list(value.get("evidence_refs") or [])[:8]
        return value


class FactInvestigationDraft(BaseModel):
    """Evidence interpretation only; it cannot issue a vulnerability verdict."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    investigation_status: Literal["complete", "partial", "inconclusive"]
    outcome_label: str | None = None
    facts: list[FactStatementDraft] = Field(default_factory=list, max_length=12)
    risk_assessments: list[RiskCandidateAssessmentDraft] = Field(default_factory=list, max_length=6)
    candidate_condition_assessments: list[CandidateConditionAssessmentDraft] = Field(default_factory=list, max_length=18)
    protection_assessments: list[ProtectionAssessmentDraft] = Field(default_factory=list, max_length=8)
    caller_contracts: list[CallerContractDraft] = Field(default_factory=list, max_length=8)
    premise_updates: list[PremiseUpdateDraft] = Field(default_factory=list, max_length=8)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=8)
    limitations: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def normalize_compatible_payload(cls, value):
        if not isinstance(value, dict):
            return value
        source = dict(value)
        status = str(source.get("investigation_status") or source.get("status") or "inconclusive").lower()
        status = {
            "completed": "complete", "success": "complete", "successful": "complete",
            "failed": "inconclusive", "unknown": "inconclusive",
        }.get(status, status)
        if status not in {"complete", "partial", "inconclusive"}:
            status = "inconclusive"
        result = {
            "query_id": source.get("query_id") or source.get("query") or "",
            "investigation_status": status,
            "outcome_label": source.get("outcome_label") or source.get("outcome"),
            "facts": source.get("facts") or [],
            "risk_assessments": source.get("risk_assessments") or source.get("candidate_assessments") or [],
            "candidate_condition_assessments": source.get("candidate_condition_assessments") or source.get("condition_assessments") or [],
            "protection_assessments": source.get("protection_assessments") or [],
            "caller_contracts": source.get("caller_contracts") or source.get("caller_evidence") or [],
            "premise_updates": source.get("premise_updates") or source.get("updates") or [],
            "unresolved_questions": source.get("unresolved_questions") or [],
            "limitations": source.get("limitations") or [],
        }
        for key, limit in (
            ("facts", 12), ("risk_assessments", 6), ("candidate_condition_assessments", 18),
            ("protection_assessments", 8), ("caller_contracts", 8),
            ("premise_updates", 8), ("unresolved_questions", 8), ("limitations", 8),
        ):
            result[key] = list(result[key])[:limit] if isinstance(result[key], list) else []
        return result


__all__ = [
    "AgentAnchorDraft",
    "ClaimDraft",
    "DisputeDraft",
    "DisputeFactDraft",
    "DisputeClaimRelationDraft",
    "DisputePremiseDraft",
    "InvestigationTargetDraft",
    "PremiseDraft",
    "PremiseAnalysisDraft",
    "ProtectionDraft",
    "ProtectionMechanismDraft",
    "CallerContractDraft",
    "RiskCandidateDraft",
    "ProtectionAnalysisDraft",
    "QueryCandidateDraft",
    "QueryOutcomeDraft",
    "QueryPlanDraft",
    "FactInvestigationDraft",
    "FactStatementDraft",
    "RiskCandidateAssessmentDraft",
    "CandidateConditionAssessmentDraft",
    "ProtectionAssessmentDraft",
    "PremiseUpdateDraft",
    "RiskAnalysisDraft",
    "RiskPathStepDraft",
]
