import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from mutil_agent_detection.fact_investigator import FactInvestigator
from mutil_agent_detection.protocols import (
    AnalysisInput, Backend, Claim, ClaimKind, Evidence, EvidenceStatus,
    DiagnosisRiskCandidate, FactInvestigationInput, ProgramScope, QueryOutcome, QuerySpec,
    SourceLocation, SourceSnapshot,
)


class FakeModel:
    def __init__(self, response):
        self.response = response

    def with_structured_output(self, _schema, method="function_calling", include_raw=False):
        return RunnableLambda(lambda _value: {
            "parsed": self.response, "raw": AIMessage(content=json.dumps(self.response)),
        })


def case(response):
    scope = ProgramScope(snapshot=SourceSnapshot(version="test"), entrypoint="f")
    query = QuerySpec(
        id="q1", target_claim_ids=["risk"], target_premise_ids=["p1"], scope=scope,
        kind="cpg", backend=Backend.JOERN, question="alignment?",
        outcomes=[
            QueryOutcome(label="yes", meaning="aligned", supports_premise_ids=["p1"]),
            QueryOutcome(label="no", meaning="not aligned", negates_premise_ids=["p1"]),
            QueryOutcome(label="unknown", meaning="unknown", leaves_unknown_premise_ids=["p1"]),
        ], evidence_requirement="exact anchor",
    )
    evidence = Evidence(
        id="e1", status=EvidenceStatus.CANDIDATE, summary="found cast",
        scope=scope, query_id="q1", tool=Backend.JOERN,
        source_anchors=[SourceLocation(file="sample.c", line=8)],
    )
    request = AnalysisInput(sample_id="s", function_name="f", source="int f(){}", scope=scope)
    value = FactInvestigationInput(query=query, evidence=evidence, scope=scope)
    return FactInvestigator(FakeModel(response))(value)


def test_fact_investigator_maps_evidence_to_declared_premise():
    result = case({
        "query_id": "q1", "investigation_status": "complete", "outcome_label": "no",
        "facts": [{"id": "f1", "statement": "No alignment constraint was found.",
                   "evidence_refs": ["e1"], "source_anchor_refs": ["sample.c:8"]}],
        "premise_updates": [{"premise_id": "p1", "status": "supported",
                              "reason": "The CPG contains no alignment operation.", "evidence_refs": ["e1"]}],
    })
    assert result.outcome_label == "no"
    assert result.premise_updates[0].premise_id == "p1"
    assert result.facts[0].source_anchors[0].line == 8


def test_fact_investigator_rejects_undeclared_outcome_conservatively():
    result = case({
        "query_id": "q1", "investigation_status": "complete", "outcome_label": "verdict",
        "facts": [], "premise_updates": [],
    })
    assert result.investigation_status == "inconclusive"
    assert not result.premise_updates


def test_query_id_evidence_alias_is_normalized_for_ledger():
    result = case({
        "query_id": "q1", "investigation_status": "complete", "outcome_label": "yes",
        "facts": [],
        "premise_updates": [{"premise_id": "p1", "status": "supported",
                              "reason": "query evidence supports premise", "evidence_refs": ["q1"]}],
    })
    assert result.premise_updates[0].evidence_refs == ["e1"]


def test_fact_investigator_maps_caller_contract_fields():
    result = case({
        "query_id": "q1", "investigation_status": "complete", "outcome_label": "yes",
        "facts": [],
        "caller_contracts": [{
            "caller_function": "caller", "call_site": "f(n)",
            "argument_expression": "pixels", "allocation_expression": "AcquireMemory(row_size)",
            "read_length_expression": "ReadBlob(row_size)",
            "required_size_expression": "columns * packet_size",
            "short_read_guard": "present", "extent_relation": "sufficient",
            "evidence_refs": ["e1"],
        }],
        "premise_updates": [],
    })
    assert result.caller_contracts[0].extent_relation == "sufficient"
    assert result.caller_contracts[0].short_read_guard == "present"


def test_fact_investigator_normalizes_compatible_fields_and_bounds_anchor_refs():
    result = case({
        "query_id": "q1", "status": "partial", "outcome": "yes",
        "facts": [{
            "id": "f1", "statement": "bounded fact", "evidence_refs": ["e1"],
            "source_anchor_refs": ["sample.c:8"] * 12,
        }],
        "risk_assessments": [{
            "candidate_id": "missing", "assessment": "unknown",
            "reason": "compatible alias", "evidence_refs": ["e1"] * 12,
        }],
        "updates": [{
            "premise_id": "p1", "assessment": "supported",
            "rationale": "compatible aliases", "evidence_refs": ["e1"] * 12,
        }],
    })
    assert result.investigation_status == "partial"
    assert result.facts[0].source_anchors[0].line == 8
    assert result.premise_updates[0].status == "supported"


def test_invalid_secondary_assessment_does_not_discard_valid_premise_update():
    result = case({
        "query_id": "q1", "investigation_status": "partial", "outcome_label": "yes",
        "facts": [],
        "risk_assessments": [{"candidate_id": "missing", "status": "supported",
                               "rationale": "bad reference", "evidence_refs": ["e1"]}],
        "premise_updates": [{"premise_id": "p1", "status": "supported",
                              "reason": "direct evidence", "evidence_refs": ["e1"]}],
    })
    assert result.premise_updates[0].status == "supported"
    assert not result.risk_assessments
    assert result.limitations


def test_fact_investigator_downgrades_partial_candidate_support():
    scope = ProgramScope(snapshot=SourceSnapshot(version="test"), entrypoint="f")
    query = QuerySpec(
        id="q1", target_claim_ids=["risk"], target_premise_ids=["width"], scope=scope,
        kind="source_search", backend=Backend.SOURCE, question="type width?",
        outcomes=[QueryOutcome(label="yes", meaning="width established")],
        evidence_requirement="exact definition",
    )
    evidence = Evidence(
        id="e1", status=EvidenceStatus.CANDIDATE, summary="width found",
        scope=scope, query_id="q1", tool=Backend.SOURCE,
    )
    value = FactInvestigationInput(
        query=query, evidence=evidence, scope=scope,
        risk_candidates=[DiagnosisRiskCandidate(
            candidate_id="risk-1", summary="conditional read", candidate_class="conditional_interprocedural",
            required_conditions=[{"id": "width"}, {"id": "caller_length"}],
        )],
    )
    result = FactInvestigator(FakeModel({
        "query_id": "q1", "investigation_status": "partial", "facts": [],
        "risk_assessments": [{"candidate_id": "risk-1", "status": "supported",
                              "rationale": "width is established", "evidence_refs": ["e1"]}],
        "candidate_condition_assessments": [{
            "candidate_id": "risk-1", "premise_id": "width", "status": "supported",
            "reason": "definition", "evidence_refs": ["e1"],
        }],
        "premise_updates": [],
    }))(value)
    assert result.risk_assessments[0].status == "unknown"


def test_fallback_preserves_explicit_executor_premise_mapping():
    scope = ProgramScope(snapshot=SourceSnapshot(version="test"), entrypoint="f")
    query = QuerySpec(
        id="q1", target_premise_ids=["p1"], scope=scope, kind="cpg", backend=Backend.JOERN,
        question="alignment?", outcomes=[QueryOutcome(label="yes", meaning="yes")],
        evidence_requirement="exact anchor",
    )
    evidence = Evidence(
        id="e1", status=EvidenceStatus.SUPPORTS, summary="mapped", scope=scope,
        query_id="q1", tool=Backend.JOERN, supports_premise_ids=["p1"],
    )
    value = FactInvestigationInput(
        query=query, evidence=evidence,
        scope=scope,
    )
    result = FactInvestigator._fallback(value, "invalid response")
    assert result.premise_updates[0].evidence_refs == ["e1"]
