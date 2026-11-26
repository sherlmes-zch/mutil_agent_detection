from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from mutil_agent_detection.protocols import (
    AnalysisInput,
    AuditState,
    Backend,
    Budget,
    Claim,
    ClaimKind,
    Dispute,
    Evidence,
    EvidenceStatus,
    FinalReport,
    MemoryRecord,
    Prediction,
    ProgramScope,
    QueryKind,
    QueryOutcome,
    QuerySpec,
    SourceLocation,
    SourceSnapshot,
)


def scope() -> ProgramScope:
    return ProgramScope(
        snapshot=SourceSnapshot(version="primevul-test", commit="abc123"),
        entrypoint="process_frame",
        target_operation="memcpy",
        target_object="out.data",
        locations=[SourceLocation(file="frame.c", function="write_items", line=12)],
    )


def test_protocol_round_trip_is_json_serializable():
    request = AnalysisInput(
        sample_id="sample-1",
        function_name="process_frame",
        source="int process_frame(void) { return 0; }",
        scope=scope(),
        budget=Budget(token_limit=1000, tool_call_limit=5),
        dataset_split="test",
    )
    state = AuditState(sample_id=request.sample_id, input=request)
    restored = AuditState.model_validate_json(state.model_dump_json())

    assert restored.input.scope.target_operation == "memcpy"
    assert restored.input.budget.token_limit == 1000


def test_query_declares_effects_for_each_result_branch():
    query = QuerySpec(
        target_premise_ids=["p1"],
        scope=scope(),
        kind=QueryKind.ALLOCATION,
        backend=Backend.JOERN,
        question="Does the allocation include the metadata items?",
        outcomes=[
            QueryOutcome(
                label="included",
                meaning="capacity includes metadata",
                supports_premise_ids=["p1"],
            ),
            QueryOutcome(
                label="not_included",
                meaning="capacity excludes metadata",
                negates_premise_ids=["p1"],
            ),
        ],
        evidence_requirement="Allocation expression and capacity assignment",
    )

    assert query.outcomes[1].negates_premise_ids == ["p1"]


def test_evidence_keeps_scope_and_tool_boundary():
    evidence = Evidence(
        status=EvidenceStatus.SUPPORTS,
        summary="The allocation uses requested * sizeof(Item).",
        scope=scope(),
        tool=Backend.JOERN,
        tool_version="1.2.0",
        analysis_boundary="Only process_frame call path was resolved.",
        created_at=datetime.now(timezone.utc),
    )
    assert evidence.tool == "joern"
    assert evidence.analysis_boundary is not None


def test_invalid_source_anchor_and_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        SourceLocation(file="", line=0)

    with pytest.raises(ValidationError):
        Claim(
            kind=ClaimKind.RISK,
            text="possible overflow",
            scope=scope(),
            created_by="risk_analyst",
            unexpected="must fail",
        )


def test_final_report_preserves_unknown_prediction_and_unresolved_premises():
    report = FinalReport(
        sample_id="sample-1",
        prediction=Prediction.UNKNOWN,
        conclusion_scope=scope(),
        explanation="The relevant allocation path is unresolved.",
        unresolved_premises=["p-allocation"],
    )
    assert report.prediction == "unknown"
    assert report.unresolved_premises == ["p-allocation"]


def test_memory_record_is_strategy_only_and_has_provenance_boundary():
    record = MemoryRecord(
        dispute_pattern="capacity and emitted length may use different units",
        description="Investigate allocation and transformed length before verification.",
        recommended_query_kinds=[QueryKind.ALLOCATION, QueryKind.NUMERIC_RANGE],
        applicability_conditions=["The target object and allocation path are locatable."],
        source_refs=["train-case-17"],
        source_split="training",
    )

    assert record.frozen is True
    assert record.source_split == "training"
    assert QueryKind.ALLOCATION.value in record.model_dump()["recommended_query_kinds"]
