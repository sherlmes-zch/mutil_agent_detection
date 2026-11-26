from mutil_agent_detection import WorkflowHandlers, build_audit_graph
from mutil_agent_detection.protocols import (
    AnalysisInput,
    Backend,
    Claim,
    ClaimKind,
    ClaimStatus,
    Dispute,
    Evidence,
    EvidenceStatus,
    ProgramScope,
    QueryKind,
    QueryOutcome,
    QuerySpec,
    SourceSnapshot,
)


def request() -> AnalysisInput:
    return AnalysisInput(
        sample_id="workflow-sample",
        function_name="copy_items",
        source="void copy_items(void) {}",
        scope=ProgramScope(
            snapshot=SourceSnapshot(version="test-snapshot"),
            entrypoint="copy_items",
            target_operation="memcpy",
            target_object="out.data",
        ),
    )


def test_default_graph_reaches_conservative_final_report():
    result = build_audit_graph().invoke({"input": request()})

    assert result["report"].prediction == "unknown"
    assert result["audit_state"].phase == "completed"
    assert result["program_skeleton"]["status"] == "not_extracted"
    assert any(event.actor == "finalizer" for event in result["audit_state"].events)


def test_injected_query_and_evidence_flow_updates_risk_claim():
    def risk(req, _skeleton):
        return Claim(
            kind=ClaimKind.RISK,
            text="copy can exceed allocation",
            status=ClaimStatus.CANDIDATE,
            scope=req.scope,
            created_by="test-risk",
        )

    def plan(dispute, _memories, scope, _budget):
        return [
            QuerySpec(
                target_claim_ids=[dispute.risk_claim_id],
                scope=scope,
                kind=QueryKind.ALLOCATION,
                backend=Backend.JOERN,
                question="Does the allocation cover the copied length?",
                outcomes=[QueryOutcome(label="confirmed", meaning="allocation is insufficient")],
                evidence_requirement="Allocation expression and copy length",
            )
        ]

    def execute(query, req):
        return Evidence(
            status=EvidenceStatus.SUPPORTS,
            summary="The copied length exceeds the allocation.",
            scope=query.scope,
            query_id=query.id,
            supports_claim_ids=query.target_claim_ids,
            tool=Backend.JOERN,
            analysis_boundary=f"Test evidence for {req.sample_id}",
        )

    def diagnose(value):
        return Dispute(
            scope=value.scope, risk_claim_id=value.risk_claim_id,
            protection_claim_id=value.protection_claim_id,
            diagnosis_status="complete", alignment_status="aligned",
            summary="No additional decisive premise was identified.",
        )

    result = build_audit_graph(
        WorkflowHandlers(risk=risk, diagnose=diagnose, plan_queries=plan, execute_query=execute)
    ).invoke({"input": request()})

    assert result["report"].prediction == "vulnerable"
    assert len(result["audit_state"].evidence) == 1
    risk_claim = next(c for c in result["audit_state"].claims.values() if c.kind == "risk")
    assert risk_claim.status == "supported"
    assert result["report"].evidence_refs == list(result["audit_state"].evidence)
    assert result["report"].risk_assessment == ["copy can exceed allocation"]


def test_final_decision_stays_unknown_when_decisive_premise_lacks_evidence():
    from mutil_agent_detection.protocols import Dispute, DisputePremise

    def risk(req, _skeleton):
        return Claim(kind=ClaimKind.RISK, text="unsafe dereference", status=ClaimStatus.CANDIDATE,
                     scope=req.scope, created_by="risk")

    def diagnose(value):
        return Dispute(
            scope=value.scope, risk_claim_id=value.risk_claim_id,
            protection_claim_id=value.protection_claim_id, diagnosis_status="partial",
            alignment_status="unresolved", summary="alignment premise remains uncertain",
            premises=[DisputePremise(id="prem-1", description="address may be misaligned", decisive=True)],
            unresolved_premises=["prem-1"], decisive_premises=["prem-1"],
        )

    def plan(dispute, _memories, scope, _budget):
        return [QuerySpec(
            id="q-risk", target_claim_ids=[dispute.risk_claim_id], target_premise_ids=["prem-1"],
            scope=scope, kind=QueryKind.CPG, backend=Backend.JOERN, question="check alignment",
            outcomes=[QueryOutcome(label="yes", meaning="yes", supports_premise_ids=["prem-1"])],
            evidence_requirement="exact alignment evidence",
        )]

    def execute(query, _request):
        return Evidence(
            status=EvidenceStatus.SUPPORTS, summary="unsafe access exists", scope=query.scope,
            query_id=query.id, supports_claim_ids=query.target_claim_ids, tool=Backend.JOERN,
        )

    result = build_audit_graph(WorkflowHandlers(
        risk=risk, diagnose=diagnose, plan_queries=plan, execute_query=execute,
    )).invoke({"input": request()})

    assert result["report"].prediction == "unknown"
    assert "prem-1" in result["report"].unresolved_premises


def test_fact_investigation_updates_premise_relation_and_status():
    from mutil_agent_detection.protocols import Dispute, DisputePremise, FactInvestigationResult, PremiseUpdate

    def diagnose(value):
        return Dispute(
            scope=value.scope, risk_claim_id=value.risk_claim_id,
            protection_claim_id=value.protection_claim_id,
            diagnosis_status="partial", alignment_status="unresolved", summary="needs evidence",
            premises=[DisputePremise(id="p1", description="condition", decisive=True)],
            unresolved_premises=["p1"], decisive_premises=["p1"],
        )

    def plan(dispute, _memories, scope, _budget):
        return [QuerySpec(
            id="q-premise", target_claim_ids=[dispute.risk_claim_id], target_premise_ids=["p1"],
            scope=scope, kind=QueryKind.SOURCE_SEARCH, backend=Backend.SOURCE,
            question="condition?", outcomes=[
                QueryOutcome(label="yes", meaning="yes", supports_premise_ids=["p1"]),
                QueryOutcome(label="no", meaning="no", negates_premise_ids=["p1"]),
            ], evidence_requirement="source",
        )]

    def execute(query, _request):
        return Evidence(status=EvidenceStatus.CANDIDATE, summary="fact", scope=query.scope,
                        query_id=query.id, tool=Backend.SOURCE)

    def investigate(value):
        return FactInvestigationResult(
            query_id=value.query.id, investigation_status="complete", outcome_label="yes",
            premise_updates=[PremiseUpdate(premise_id="p1", status="supported", reason="fact", evidence_refs=[value.evidence.id])],
        )

    result = build_audit_graph(WorkflowHandlers(
        diagnose=diagnose, plan_queries=plan, execute_query=execute,
        investigate_evidence=investigate,
    )).invoke({"input": request()})

    assert result["dispute"].premises[0].status == "supported"
    assert result["dispute"].premises[0].relation == "supports"
    assert result["dispute"].unresolved_premises == []
    assert "q-premise" in result["audit_state"].fact_investigations


def test_resolved_repository_snapshot_is_used_by_later_rounds_and_report():
    from mutil_agent_detection.protocols import FactInvestigationResult

    declared_scope = request().scope.model_copy(update={
        "snapshot": request().scope.snapshot.model_copy(update={
            "version": "declared-commit", "commit": "declared-commit",
        }),
    })
    input_request = request().model_copy(update={"scope": declared_scope})
    resolved_scope = declared_scope.model_copy(update={
        "snapshot": declared_scope.snapshot.model_copy(update={
            "version": "ancestor-commit", "commit": "ancestor-commit", "root": "/repo/worktree",
        }),
        "unresolved_dependencies": [],
    })
    planned_scopes = []
    investigated_scopes = []

    def plan(_dispute, _memories, scope, _budget):
        planned_scopes.append(scope.snapshot.commit)
        return [QuerySpec(
            id=f"q-{len(planned_scopes)}", scope=scope, kind=QueryKind.SOURCE_SEARCH,
            backend=Backend.SOURCE, question="inspect source", outcomes=[
                QueryOutcome(label="found", meaning="fact found"),
            ], evidence_requirement="matching source",
        )]

    def execute(query, _request):
        return Evidence(
            status=EvidenceStatus.CANDIDATE, summary="source candidate", scope=resolved_scope,
            query_id=query.id, tool=Backend.SOURCE,
            analysis_boundary="Exact-commit repository search was performed.",
        )

    def investigate(value):
        investigated_scopes.append(value.scope.snapshot.commit)
        return FactInvestigationResult(
            query_id=value.query.id, investigation_status="inconclusive",
        )

    result = build_audit_graph(WorkflowHandlers(
        plan_queries=plan, execute_query=execute, investigate_evidence=investigate,
    ), max_query_rounds=2).invoke({"input": input_request})

    assert planned_scopes == ["declared-commit", "ancestor-commit"]
    assert investigated_scopes == ["ancestor-commit", "ancestor-commit"]
    assert result["audit_state"].input.scope.snapshot.commit == "ancestor-commit"
    assert result["report"].conclusion_scope.snapshot.commit == "ancestor-commit"
    assert result["audit_state"].input.metadata["repository_version_status"] == "matched_first_parent"


def test_successful_first_investigation_continues_for_unresolved_decisive_premise():
    """A complete answer to one query must not suppress the next decisive query."""
    from mutil_agent_detection.protocols import DisputePremise, FactInvestigationResult, PremiseUpdate

    def diagnose(value):
        return Dispute(
            scope=value.scope, risk_claim_id=value.risk_claim_id,
            protection_claim_id=value.protection_claim_id,
            diagnosis_status="partial", alignment_status="unresolved", summary="two premises",
            premises=[
                DisputePremise(id="p1", description="first", decisive=True),
                DisputePremise(id="p2", description="second", decisive=True),
            ], unresolved_premises=["p1", "p2"], decisive_premises=["p1", "p2"],
        )

    def plan(dispute, _memories, scope, _budget):
        unresolved = set(dispute.unresolved_premises)
        premise_id = "p1" if "p1" in unresolved else "p2"
        return [QuerySpec(
            id=f"q-{premise_id}", target_premise_ids=[premise_id], scope=scope,
            kind=QueryKind.SOURCE_SEARCH, backend=Backend.SOURCE, question=premise_id,
            outcomes=[QueryOutcome(label="yes", meaning="confirmed", supports_premise_ids=[premise_id])],
            evidence_requirement="source",
        )]

    def execute(query, _request):
        return Evidence(
            status=EvidenceStatus.CANDIDATE, summary="fact", scope=query.scope,
            query_id=query.id, tool=Backend.SOURCE,
        )

    def investigate(value):
        premise_id = value.query.target_premise_ids[0]
        return FactInvestigationResult(
            query_id=value.query.id, investigation_status="complete",
            premise_updates=[PremiseUpdate(
                premise_id=premise_id, status="supported", reason="fact",
                evidence_refs=[value.evidence.id],
            )],
        )

    result = build_audit_graph(
        WorkflowHandlers(
            diagnose=diagnose, plan_queries=plan, execute_query=execute,
            investigate_evidence=investigate,
        ), max_query_rounds=2,
    ).invoke({"input": request()})

    assert result["audit_state"].query_round == 2
    assert set(result["audit_state"].fact_investigations) == {"q-p1", "q-p2"}
