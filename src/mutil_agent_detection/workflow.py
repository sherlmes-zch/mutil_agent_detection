"""Minimal LangGraph orchestration skeleton.

This module owns control flow only.  Model calls, Joern, GraphRAG and
symbolic execution are injected through ``WorkflowHandlers`` so that the
graph can be tested before those integrations exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .diagnosis_context import build_diagnosis_input, normalize_claim_entity_ids
from .final_decision import synthesize_final_report
from .context_compaction import (
    compact_claim, compact_evidence, compact_planning_skeleton,
)
from .protocols import (
    AnalysisInput,
    AuditEvent,
    AuditPhase,
    AuditState,
    Backend,
    Budget,
    Claim,
    ClaimKind,
    ClaimStatus,
    DiagnosisInput,
    Dispute,
    Evidence,
    EvidenceStatus,
    FinalReport,
    MemoryRecord,
    Prediction,
    ProgramScope,
    QueryDecision,
    QuerySpec,
    QueryPlanningInput,
    FactInvestigationInput,
    FactInvestigationResult,
    FactStatement,
    PremiseUpdate,
)


class GraphState(TypedDict, total=False):
    """Keys exchanged by graph nodes.

    ``audit_state`` is the durable protocol object.  The other keys are
    short-lived node outputs kept separate so parallel branches never write
    the same dictionary key.
    """

    input: AnalysisInput
    audit_state: AuditState
    program_scope: ProgramScope
    program_skeleton: dict[str, Any]
    risk_claim: Claim
    protection_claim: Claim
    diagnosis_input: DiagnosisInput
    dispute: Dispute
    memories: list[MemoryRecord]
    query_candidates: list[QuerySpec]
    selected_query: QuerySpec | None
    last_evidence: Evidence | None
    fact_investigation: FactInvestigationResult | None
    report: FinalReport
    query_round: int
    max_query_rounds: int


@dataclass(frozen=True)
class WorkflowHandlers:
    """Optional integrations for later LangChain/tool implementations."""

    skeleton: Callable[[AnalysisInput], dict[str, Any]] | None = None
    risk: Callable[[AnalysisInput, dict[str, Any]], Claim] | None = None
    protection: Callable[[AnalysisInput, dict[str, Any]], Claim] | None = None
    diagnose: Callable[[DiagnosisInput], Dispute] | None = None
    retrieve_memory: Callable[[Dispute, AnalysisInput], list[MemoryRecord]] | None = None
    plan_queries: Callable[[Dispute, list[MemoryRecord], ProgramScope, Budget], list[QuerySpec]] | None = None
    query_planner: Callable[[QueryPlanningInput], list[QuerySpec]] | None = None
    execute_query: Callable[[QuerySpec, AnalysisInput], Evidence] | None = None
    investigate_evidence: Callable[[FactInvestigationInput], FactInvestigationResult] | None = None
    finalize: Callable[[AuditState], FinalReport] | None = None


def _event(phase: AuditPhase, actor: str, message: str, *ids: str) -> AuditEvent:
    return AuditEvent(phase=phase, actor=actor, message=message, related_ids=list(ids))


def _with_audit(state: GraphState, *, phase: AuditPhase | None = None,
                event: AuditEvent | None = None, **updates: Any) -> AuditState:
    audit = state["audit_state"].model_copy(deep=True)
    if phase is not None:
        audit.phase = phase
    for key, value in updates.items():
        setattr(audit, key, value)
    if event is not None:
        audit.events.append(event)
    return audit


def _default_skeleton(request: AnalysisInput) -> dict[str, Any]:
    return {
        "entrypoint": request.function_name,
        "source_length": len(request.source),
        "status": "not_extracted",
        "structural_operations": [],
        "source_candidates": [],
        "candidate_dataflows": [],
        "evidence_semantics": "Joern CPG facts are source-located review clues, not vulnerability conclusions.",
        "unresolved_dependencies": ["Joern handler is not configured"],
    }


def _default_claim(request: AnalysisInput, kind: ClaimKind, actor: str) -> Claim:
    return Claim(
        kind=kind,
        text=f"{actor} is not configured; no {kind.value} conclusion was produced.",
        status=ClaimStatus.UNKNOWN,
        scope=request.scope,
        created_by=actor,
    )


def _default_dispute(request: DiagnosisInput) -> Dispute:
    premise_id = "diagnosis-unavailable"
    return Dispute(
        scope=request.scope,
        risk_claim_id=request.risk_claim_id,
        protection_claim_id=request.protection_claim_id,
        diagnosis_status="insufficient_evidence",
        alignment_status="unresolved",
        summary="No disagreement diagnostician is configured; the claim relationship remains unresolved.",
        premises=[
            {
                "id": premise_id,
                "description": "A reliable alignment of risk candidates and protections is unavailable.",
                "relation": "unresolved",
                "status": "unknown",
                "claim_refs": [request.risk_claim_id, request.protection_claim_id],
                "decisive": True,
            }
        ],
        unresolved_premises=[premise_id],
        decisive_premises=[premise_id],
        alignment_notes="No semantic alignment was asserted.",
    )


def _default_report(audit: AuditState, decision_policy: str = "strict") -> FinalReport:
    return synthesize_final_report(audit, decision_policy=decision_policy)


def _initialize(state: GraphState) -> GraphState:
    request = state["input"]
    audit = AuditState(
        sample_id=request.sample_id,
        input=request,
        remaining_budget=request.budget,
        phase=AuditPhase.INITIALIZED,
    )
    return {
        "audit_state": audit,
        "program_scope": request.scope,
        "query_round": 0,
        "max_query_rounds": state.get("max_query_rounds", 2),
    }


def build_audit_graph(
    handlers: WorkflowHandlers | None = None, *, max_query_rounds: int = 2,
    decision_policy: str = "balanced",
):
    """Build and compile the minimal audit graph.

    The returned object is a normal LangGraph compiled graph.  Supplying
    handlers progressively replaces defaults without changing graph topology.
    """

    handlers = handlers or WorkflowHandlers()
    graph = StateGraph(GraphState)

    def initialize(state: GraphState) -> GraphState:
        state = {**state, "max_query_rounds": max_query_rounds}
        return _initialize(state)

    def prepare_skeleton(state: GraphState) -> GraphState:
        request = state["input"]
        skeleton = (handlers.skeleton or _default_skeleton)(request)
        audit = _with_audit(
            state,
            phase=AuditPhase.ANALYZING,
            event=_event(AuditPhase.ANALYZING, "program_skeleton", "Program skeleton prepared"),
        )
        return {"program_skeleton": skeleton, "audit_state": audit}

    def risk_analysis(state: GraphState) -> GraphState:
        request = state["input"]
        claim = (handlers.risk or (lambda r, _: _default_claim(r, ClaimKind.RISK, "risk_analyst")))(
            request, deepcopy(state["program_skeleton"])
        )
        return {"risk_claim": claim}

    def protection_analysis(state: GraphState) -> GraphState:
        request = state["input"]
        claim = (handlers.protection or (lambda r, _: _default_claim(r, ClaimKind.PROTECTION, "protection_analyst")))(
            request, deepcopy(state["program_skeleton"])
        )
        return {"protection_claim": claim}

    def merge_analyses(state: GraphState) -> GraphState:
        claims = dict(state["audit_state"].claims)
        # Normalize provider-generated entity IDs once, before diagnosis and
        # investigation. This makes retries/models address the same candidates.
        risk_claim = normalize_claim_entity_ids(state["risk_claim"])
        protection_claim = normalize_claim_entity_ids(state["protection_claim"])
        claims[risk_claim.id] = risk_claim
        claims[protection_claim.id] = protection_claim
        audit = _with_audit(
            state,
            phase=AuditPhase.DIAGNOSING,
            claims=claims,
            event=_event(
                AuditPhase.DIAGNOSING,
                "analysis_merger",
                "Independent risk and protection analyses merged",
                risk_claim.id,
                protection_claim.id,
            ),
        )
        return {"audit_state": audit, "risk_claim": risk_claim, "protection_claim": protection_claim}

    def diagnose(state: GraphState) -> GraphState:
        diagnosis_input = build_diagnosis_input(
            sample_id=state["input"].sample_id,
            risk_claim=state["risk_claim"],
            protection_claim=state["protection_claim"],
            scope=state["program_scope"],
            skeleton=state["program_skeleton"],
        )
        if handlers.diagnose is None:
            dispute = _default_dispute(diagnosis_input)
        else:
            dispute = handlers.diagnose(diagnosis_input)
        disputes = dict(state["audit_state"].disputes)
        disputes[dispute.id] = dispute
        related_ids = [state["risk_claim"].id, state["protection_claim"].id]
        related_ids.extend(candidate.candidate_id for candidate in diagnosis_input.risk_candidates)
        related_ids.extend(protection.protection_id for protection in diagnosis_input.protections)
        audit = _with_audit(
            state,
            phase=AuditPhase.PLANNING,
            disputes=disputes,
            unresolved_premises=dispute.unresolved_premises,
            event=_event(
                AuditPhase.DIAGNOSING,
                "disagreement_diagnoser",
                "Risk/protection claims aligned and conclusion-dependent premises recorded",
                dispute.id,
                *related_ids,
            ),
        )
        return {"diagnosis_input": diagnosis_input, "dispute": dispute, "audit_state": audit}

    def retrieve_memory(state: GraphState) -> GraphState:
        memories = (handlers.retrieve_memory or (lambda _d, _r: []))(state["dispute"], state["input"])
        return {"memories": memories}

    def plan_queries(state: GraphState) -> GraphState:
        if handlers.query_planner is not None:
            planning = QueryPlanningInput(
                sample_id=state["input"].sample_id,
                risk_claim=compact_claim(state["risk_claim"]),
                protection_claim=compact_claim(state["protection_claim"]),
                dispute=state["dispute"], scope=state["program_scope"],
                program_skeleton=compact_planning_skeleton(deepcopy(state.get("program_skeleton", {}))),
                budget=state["audit_state"].remaining_budget,
                prior_queries=list(state["audit_state"].queries.values()),
                # Only advertise backends with an installed executor. SMT/KLEE remain
                # protocol options but must not consume an investigation round yet.
                enabled_backends=[Backend.SOURCE, Backend.JOERN],
            )
            candidates = handlers.query_planner(planning)
        else:
            candidates = (handlers.plan_queries or (lambda _d, _m, _s, _b: []))(
                state["dispute"], state.get("memories", []), state["program_scope"], state["audit_state"].remaining_budget
            )
        known = set(state["audit_state"].queries)
        candidates = [query for query in candidates if query.id not in known]
        queries = dict(state["audit_state"].queries)
        for query in candidates:
            queries[query.id] = query
        audit = _with_audit(
            state,
            phase=AuditPhase.INVESTIGATING,
            queries=queries,
            event=_event(AuditPhase.INVESTIGATING, "query_planner", f"Planned {len(candidates)} query(s)"),
        )
        return {"query_candidates": candidates, "audit_state": audit}

    def select_query(state: GraphState) -> GraphState:
        candidates = state.get("query_candidates", [])
        selected = candidates[0] if candidates else None
        if selected is not None:
            selected = selected.model_copy(update={"decision": QueryDecision.SELECTED})
        return {"selected_query": selected}

    def execute_query(state: GraphState) -> GraphState:
        query = state["selected_query"]
        if query is None:
            return {"last_evidence": None}
        evidence = (handlers.execute_query or _default_execute_query)(query, state["input"])
        # Repository-aware executors may resolve the PrimeVul fragment to an
        # ancestor snapshot. Make that verified scope canonical for subsequent
        # investigation rounds and the final report.
        current_input = state["input"]
        current_scope = state["program_scope"]
        witness = evidence.witness or {}
        metadata = dict(current_input.metadata)
        metadata.update(witness)
        current_input = current_input.model_copy(update={"metadata": metadata})
        if (
            evidence.scope.snapshot.commit
            and evidence.scope.snapshot.root
            and (
                evidence.scope.snapshot.commit != current_scope.snapshot.commit
                or evidence.scope.snapshot.root != current_scope.snapshot.root
            )
        ):
            current_scope = evidence.scope
            metadata = dict(current_input.metadata)
            metadata.update({
                "repository_root": evidence.scope.snapshot.root,
                "repository_snapshot_commit": evidence.scope.snapshot.commit,
                "repository_version_status": (
                    "matched_first_parent"
                    if metadata.get("declared_commit") != evidence.scope.snapshot.commit
                    else metadata.get("repository_version_status") or "declared_commit"
                ),
                "context_mode": "repository",
            })
            current_input = current_input.model_copy(update={
                "scope": current_scope,
                "metadata": metadata,
            })
        if evidence.scope.snapshot.commit and (
            query.scope.snapshot.commit != evidence.scope.snapshot.commit
            or query.scope.snapshot.root != evidence.scope.snapshot.root
        ):
            query = query.model_copy(update={"scope": evidence.scope})
        queries = dict(state["audit_state"].queries)
        queries[query.id] = query.model_copy(update={"decision": QueryDecision.EXECUTED})
        claims = {
            claim_id: claim.model_copy(update={"scope": current_scope})
            for claim_id, claim in state["audit_state"].claims.items()
        }
        disputes = {
            dispute_id: dispute.model_copy(update={"scope": current_scope})
            for dispute_id, dispute in state["audit_state"].disputes.items()
        }
        audit_updates: dict[str, Any] = {"queries": queries, "claims": claims, "disputes": disputes}
        if current_input != state["input"]:
            audit_updates["input"] = current_input
        audit = _with_audit(
            state,
            phase=AuditPhase.UPDATING,
            **audit_updates,
            event=_event(AuditPhase.UPDATING, "query_executor", "Query executed", query.id),
        )
        return {
            "last_evidence": evidence,
            "audit_state": audit,
            "input": current_input,
            "program_scope": current_scope,
            "selected_query": query,
            "dispute": disputes.get(state["dispute"].id, state["dispute"]),
            "risk_claim": claims.get(state["risk_claim"].id, state["risk_claim"]),
            "protection_claim": claims.get(state["protection_claim"].id, state["protection_claim"]),
        }

    def investigate_evidence(state: GraphState) -> GraphState:
        evidence = state.get("last_evidence")
        query = state.get("selected_query")
        if evidence is None or query is None:
            return {"fact_investigation": None}
        investigation_input = FactInvestigationInput(
            query=query,
            evidence=compact_evidence(evidence),
            premises=list(state["dispute"].premises),
            claims=[
                compact_claim(state["risk_claim"], summary_only=True),
                compact_claim(state["protection_claim"], summary_only=True),
            ],
            risk_candidates=(state.get("diagnosis_input").risk_candidates if state.get("diagnosis_input") else []),
            protections=(state.get("diagnosis_input").protections if state.get("diagnosis_input") else []),
            scope=state["program_scope"],
        )
        if handlers.investigate_evidence is None:
            result = _default_fact_investigation(investigation_input)
        else:
            result = handlers.investigate_evidence(investigation_input)
        return {"fact_investigation": result}

    def update_evidence(state: GraphState) -> GraphState:
        evidence = state.get("last_evidence")
        if evidence is None:
            return {"query_round": state.get("query_round", 0) + 1}
        evidence_map = dict(state["audit_state"].evidence)
        evidence_map[evidence.id] = evidence
        claims = dict(state["audit_state"].claims)
        for claim_id in evidence.supports_claim_ids:
            if claim_id in claims:
                claims[claim_id] = claims[claim_id].model_copy(update={"status": ClaimStatus.SUPPORTED})
        for claim_id in evidence.negates_claim_ids:
            if claim_id in claims:
                claims[claim_id] = claims[claim_id].model_copy(update={"status": ClaimStatus.REFUTED})
        investigation = state.get("fact_investigation")
        disputes = dict(state["audit_state"].disputes)
        investigations = dict(state["audit_state"].fact_investigations)
        dispute = state.get("dispute")
        if investigation is not None and dispute is not None:
            investigations[investigation.query_id] = investigation
            premise_updates = {item.premise_id: item for item in investigation.premise_updates}
            if premise_updates:
                valid_evidence_refs = set(evidence_map)
                updated_premises = [
                    premise.model_copy(update={
                        "status": (
                            premise_updates[premise.id].status
                            if premise_updates[premise.id].evidence_refs
                            and set(premise_updates[premise.id].evidence_refs) <= valid_evidence_refs
                            else "unknown"
                        ),
                        "relation": (
                            "supports" if premise_updates[premise.id].status == "supported"
                            else "contradicts" if premise_updates[premise.id].status == "refuted"
                            else "unresolved"
                        ) if premise_updates[premise.id].evidence_refs
                        and set(premise_updates[premise.id].evidence_refs) <= valid_evidence_refs else "unresolved",
                        "evidence_refs": list(dict.fromkeys(
                            [*premise.evidence_refs, *premise_updates[premise.id].evidence_refs]
                        )) if set(premise_updates[premise.id].evidence_refs) <= valid_evidence_refs else premise.evidence_refs,
                    })
                    if premise.id in premise_updates else premise
                    for premise in dispute.premises
                ]
                unresolved = [
                    premise.id for premise in updated_premises
                    if premise.status == "unknown" or premise.relation == "unresolved"
                ]
                updated_dispute = dispute.model_copy(update={
                    "premises": updated_premises, "unresolved_premises": unresolved,
                })
                disputes[updated_dispute.id] = updated_dispute
        audit = _with_audit(
            state,
            phase=AuditPhase.UPDATING,
            evidence=evidence_map, disputes=disputes, fact_investigations=investigations,
            claims=claims,
            query_round=state.get("query_round", 0) + 1,
            event=_event(AuditPhase.UPDATING, "evidence_ledger", "Evidence recorded and premise updates applied", evidence.id),
        )
        return {"audit_state": audit, "dispute": disputes.get(dispute.id, dispute), "query_round": state.get("query_round", 0) + 1}

    def should_continue(state: GraphState) -> str:
        evidence = state.get("last_evidence")
        investigation = state.get("fact_investigation")
        investigation_inconclusive = (
            investigation is not None and investigation.investigation_status == "inconclusive"
        )
        # A successful interpretation of one query does not mean that the
        # sample is fully investigated. Continue while decisive premises from
        # the diagnosis remain unresolved, so a query answering (for example)
        # reader width still allows a second query for alignment or range.
        dispute = state.get("dispute")
        unresolved_decisive = bool(
            dispute
            and set(dispute.unresolved_premises)
            & set(dispute.decisive_premises or dispute.unresolved_premises)
        )
        if (
            evidence is not None
            and (
                evidence.status == EvidenceStatus.INCONCLUSIVE
                or investigation_inconclusive
                or unresolved_decisive
            )
            and state.get("query_round", 0) < state.get("max_query_rounds", max_query_rounds)
        ):
            return "plan_queries"
        return "finalize"

    def finalize(state: GraphState) -> GraphState:
        audit = state["audit_state"].model_copy(deep=True)
        audit.phase = AuditPhase.COMPLETED
        audit.events.append(_event(AuditPhase.COMPLETED, "finalizer", "Audit report finalized"))
        report = (handlers.finalize or (lambda value: _default_report(value, decision_policy)))(audit)
        return {"audit_state": audit, "report": report}

    def no_query_finalize(state: GraphState) -> GraphState:
        return finalize(state)

    graph.add_node("initialize", initialize)
    graph.add_node("prepare_skeleton", prepare_skeleton)
    graph.add_node("risk_analysis", risk_analysis)
    graph.add_node("protection_analysis", protection_analysis)
    graph.add_node("merge_analyses", merge_analyses)
    graph.add_node("diagnose", diagnose)
    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("plan_queries", plan_queries)
    graph.add_node("select_query", select_query)
    graph.add_node("execute_query", execute_query)
    graph.add_node("investigate_evidence", investigate_evidence)
    graph.add_node("update_evidence", update_evidence)
    graph.add_node("finalize", finalize)
    graph.add_node("no_query_finalize", no_query_finalize)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "prepare_skeleton")
    graph.add_edge("prepare_skeleton", "risk_analysis")
    graph.add_edge("prepare_skeleton", "protection_analysis")
    graph.add_edge("risk_analysis", "merge_analyses")
    graph.add_edge("protection_analysis", "merge_analyses")
    graph.add_edge("merge_analyses", "diagnose")
    graph.add_edge("diagnose", "retrieve_memory")
    graph.add_edge("retrieve_memory", "plan_queries")
    graph.add_edge("plan_queries", "select_query")
    graph.add_conditional_edges(
        "select_query",
        lambda state: "execute_query" if state.get("selected_query") is not None else "no_query_finalize",
        {"execute_query": "execute_query", "no_query_finalize": "no_query_finalize"},
    )
    graph.add_edge("execute_query", "investigate_evidence")
    graph.add_edge("investigate_evidence", "update_evidence")
    graph.add_conditional_edges(
        "update_evidence",
        should_continue,
        {"plan_queries": "plan_queries", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("no_query_finalize", END)
    return graph.compile()


def _default_execute_query(query: QuerySpec, request: AnalysisInput) -> Evidence:
    """Conservative placeholder: no backend result is claimed."""

    return Evidence(
        status=EvidenceStatus.INCONCLUSIVE,
        summary="Query backend is not configured; no program fact was established.",
        scope=query.scope,
        query_id=query.id,
        tool=Backend.OTHER,
        analysis_boundary=f"Requested for sample {request.sample_id}; execution was a skeleton placeholder.",
    )


def _default_fact_investigation(value: FactInvestigationInput) -> FactInvestigationResult:
    """Preserve legacy evidence mappings when no investigator is configured."""
    evidence = value.evidence
    updates = []
    for outcome in value.query.outcomes:
        if set(outcome.supports_premise_ids) & set(evidence.supports_premise_ids):
            updates.extend(PremiseUpdate(
                premise_id=premise_id, status="supported", reason="Evidence declared support for this premise.",
                evidence_refs=[evidence.id]
            ) for premise_id in outcome.supports_premise_ids)
        if set(outcome.negates_premise_ids) & set(evidence.negates_premise_ids):
            updates.extend(PremiseUpdate(
                premise_id=premise_id, status="refuted", reason="Evidence declared negation of this premise.",
                evidence_refs=[evidence.id]
            ) for premise_id in outcome.negates_premise_ids)
    return FactInvestigationResult(
        query_id=value.query.id,
        investigation_status="inconclusive" if evidence.status == EvidenceStatus.INCONCLUSIVE else "partial",
        premise_updates=updates,
        limitations=["No fact investigator handler was configured; only explicit evidence mappings were applied."],
    )


__all__ = ["GraphState", "WorkflowHandlers", "build_audit_graph"]
