from mutil_agent_detection.context_compaction import (
    compact_claim, compact_evidence, compact_key_operations, compact_skeleton,
)
from mutil_agent_detection.protocols import (
    Backend, Claim, ClaimKind, Evidence, EvidenceStatus, ProgramScope, SourceSnapshot,
)


def scope():
    return ProgramScope(snapshot=SourceSnapshot(version="test"), entrypoint="f")


def test_claim_compaction_removes_recursive_interaction_transport():
    claim = Claim(
        kind=ClaimKind.RISK, text="risk", scope=scope(), created_by="test",
        analysis={"candidates": [{"candidate_id": "r1"}], "_interaction": {"source": "x" * 10000}},
    )

    compact = compact_claim(claim)
    summary = compact_claim(claim, summary_only=True)

    assert "_interaction" not in compact.analysis
    assert compact.analysis["candidates"][0]["candidate_id"] == "r1"
    assert summary.analysis == {}


def test_skeleton_and_evidence_are_bounded_without_losing_provenance():
    skeleton = {"status": "ready", "structural_operations": [{"line": i} for i in range(100)]}
    compact = compact_skeleton(skeleton)
    evidence = Evidence(
        status=EvidenceStatus.CANDIDATE, summary="result", raw_result="x" * 30000,
        scope=scope(), tool=Backend.JOERN,
    )
    bounded = compact_evidence(evidence)

    assert compact["status"] == "ready"
    assert len(compact["structural_operations"]) == 24
    assert len(bounded.raw_result) < 25000
    assert bounded.id == evidence.id


def test_key_operation_compaction_prioritizes_memory_operations():
    values = [{"line": i, "categories": ["condition"]} for i in range(100)]
    values.append({"line": 200, "categories": ["cast", "dereference"]})

    compact = compact_key_operations(values, limit=10)

    assert any(item["line"] == 200 for item in compact)
    assert len(compact) == 10
