"""Batch execution of normalized PrimeVul samples through LangGraph."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
import subprocess
from typing import Any
from pathlib import Path
from threading import Lock
import json

from pydantic import BaseModel, ConfigDict, Field

from .dataset import PrimeVulSample, iter_primevul
from .base_agent import FirstPassAgent, StructuredAgentError
from .repository import RepositoryManager
from .source_evidence import function_matches_repository, normalized_function_hash
from .protocols import (
    AuditState, Claim, ClaimKind, ClaimStatus, FinalReport,
    FactInvestigationResult,
)
from .workflow import WorkflowHandlers, build_audit_graph, _default_dispute


class SampleRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    idx: int
    target: int
    report: FinalReport | None = None
    audit_state: AuditState | None = None
    agent_interactions: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class BatchRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    requested: int
    completed: int
    failed: int
    results: list[SampleRunResult] = Field(default_factory=list)

    @property
    def predictions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            if result.report is not None:
                key = str(result.report.prediction)
                counts[key] = counts.get(key, 0) + 1
        return counts


def run_sample(
    sample: PrimeVulSample,
    *,
    handlers: WorkflowHandlers | None = None,
    max_query_rounds: int = 2,
    repository_root: str | Path | None = None,
    context_mode: str = "function_only",
    repository_manager: RepositoryManager | None = None,
    decision_policy: str = "balanced",
    agent_retries: int = 0,
) -> SampleRunResult:
    """Run one sample in an isolated graph invocation."""

    interactions: list[dict[str, Any]] = []
    interaction_lock = Lock()
    try:
        repository_commit = None
        repository_match_status = None
        repository_match_note = None
        repository_function_hash = None
        repository_checked_commits: list[str] = []
        effective_context_mode = context_mode
        if context_mode in {"repository", "repository_context"}:
            if repository_manager is not None:
                resolution = repository_manager.prepare_matching_snapshot(sample)
                repository_root = resolution.root
                repository_commit = resolution.matched_commit
                repository_match_status = resolution.status
                repository_match_note = resolution.note
                repository_function_hash = resolution.repository_function_hash
                repository_checked_commits = list(resolution.checked_commits)
            elif repository_root is not None:
                actual = subprocess.run(
                    ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
                    check=False, capture_output=True, text=True,
                )
                repository_commit = actual.stdout.strip() if actual.returncode == 0 else None
                matched, note = function_matches_repository(
                    sample.func, sample.file_name, repository_root,
                    expected_commit=repository_commit or "",
                ) if repository_commit else (False, "cannot verify repository HEAD")
                repository_match_status = "declared_commit" if matched and repository_commit == sample.commit_id else (
                    "matched_first_parent" if matched else "repository_version_mismatch"
                )
                if not matched:
                    repository_match_note = note
                    repository_root = None
                    repository_commit = None
                else:
                    repository_function_hash = normalized_function_hash(sample.func)
                    repository_checked_commits = [repository_commit]
            else:
                raise ValueError("repository mode requires a repository manager or repository_root")
            if repository_root is None:
                effective_context_mode = "function_only"
        tracked_handlers = handlers
        if handlers is not None:
            def interaction_status(agent: Any) -> str | None:
                interaction = getattr(agent, "last_interaction", None)
                return interaction.get("status") if isinstance(interaction, dict) else None

            def interaction_error(agent: Any) -> str | None:
                interaction = getattr(agent, "last_interaction", None)
                return interaction.get("error") if isinstance(interaction, dict) else None

            def attach_retry_history(agent: Any, history: list[dict[str, Any]]) -> None:
                if not history:
                    return
                interaction = getattr(agent, "last_interaction", None)
                if isinstance(interaction, dict):
                    interaction["retries"] = history

            def track(agent, role):
                if agent is None:
                    return None
                def invoke(request, skeleton):
                    interaction_index = None
                    if isinstance(agent, FirstPassAgent):
                        with interaction_lock:
                            interaction_index = len(interactions)
                            interactions.append({"agent": role, "status": "running"})
                    attempt = 0
                    last_error: StructuredAgentError | None = None
                    result = None
                    retry_history: list[dict[str, Any]] = []
                    while True:
                        try:
                            result = agent(request, skeleton)
                            break
                        except StructuredAgentError as exc:
                            last_error = exc
                            if attempt < max(0, agent_retries):
                                attempt += 1
                                retry_history.append({
                                    "status": "retrying", "error": str(exc),
                                    "retry_index": attempt,
                                })
                                continue
                            break
                    if last_error is not None and result is None:
                        exc = last_error
                        if retry_history:
                            exc.interaction["retries"] = retry_history
                        with interaction_lock:
                            if interaction_index is None:
                                interactions.append(exc.interaction)
                            else:
                                interactions[interaction_index] = exc.interaction
                        # A malformed provider response from one analyst must
                        # not discard the independent analyst's usable result.
                        # Continue conservatively with an explicit unknown claim.
                        if isinstance(agent, FirstPassAgent):
                            exc.interaction["fallback"] = "unknown_claim"
                            kind = getattr(agent, "kind", ClaimKind.RISK)
                            actor = getattr(agent, "actor", role)
                            return Claim(
                                kind=kind,
                                text=f"{actor} returned no validated structured answer.",
                                status=ClaimStatus.UNKNOWN,
                                scope=request.scope,
                                created_by=actor,
                                analysis={"agent_error": str(exc)},
                            )
                        raise
                    interaction = getattr(agent, "last_interaction", None)
                    if interaction is not None:
                        if retry_history:
                            interaction["retries"] = retry_history
                        with interaction_lock:
                            if interaction_index is None:
                                interactions.append(interaction)
                            else:
                                interactions[interaction_index] = interaction
                    return result
                return invoke

            def track_diagnostician(agent):
                if agent is None:
                    return None

                def invoke(diagnosis_input):
                    retry_history: list[dict[str, Any]] = []
                    attempt = 0
                    while True:
                        try:
                            result = agent(diagnosis_input)
                        except Exception as exc:
                            result = None
                            error = f"{type(exc).__name__}: {exc}"
                        else:
                            error = None
                        failed = interaction_status(agent) == "failed" or result is None
                        if not failed or attempt >= max(0, agent_retries):
                            break
                        attempt += 1
                        retry_history.append({
                            "status": "retrying", "retry_index": attempt,
                            "error": error or interaction_error(agent),
                        })
                    attach_retry_history(agent, retry_history)
                    interaction = getattr(agent, "last_interaction", None)
                    if interaction is not None:
                        with interaction_lock:
                            interactions.append(interaction)
                    if result is None:
                        # Keep the graph running with an explicit unresolved
                        # dispute instead of turning a provider outage into a
                        # sample-level batch failure.
                        result = _default_dispute(diagnosis_input)
                        if isinstance(interaction, dict):
                            interaction["fallback"] = "diagnosis_unavailable"
                    return result

                return invoke

            def track_query_planner(agent):
                if agent is None:
                    return None
                def invoke(planning):
                    retry_history: list[dict[str, Any]] = []
                    attempt = 0
                    result = None
                    while True:
                        try:
                            result = agent(planning)
                            error = None
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            result = []
                        failed = interaction_status(agent) in {"failed", "fallback"} or error is not None
                        if not failed or attempt >= max(0, agent_retries):
                            break
                        attempt += 1
                        retry_history.append({
                            "status": "retrying", "retry_index": attempt,
                            "error": error or interaction_error(agent),
                        })
                    attach_retry_history(agent, retry_history)
                    interaction = getattr(agent, "last_interaction", None)
                    if interaction is not None:
                        with interaction_lock:
                            interactions.append(interaction)
                    return result
                return invoke

            def track_fact_investigator(agent):
                if agent is None:
                    return None
                def invoke(value):
                    retry_history: list[dict[str, Any]] = []
                    attempt = 0
                    result = None
                    while True:
                        try:
                            result = agent(value)
                            error = None
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            result = FactInvestigationResult(
                                query_id=value.query.id,
                                investigation_status="inconclusive",
                                unresolved_questions=["Fact investigator invocation failed."],
                                limitations=[error],
                            )
                        failed = interaction_status(agent) == "failed" or error is not None
                        if not failed or attempt >= max(0, agent_retries):
                            break
                        attempt += 1
                        retry_history.append({
                            "status": "retrying", "retry_index": attempt,
                            "error": error or interaction_error(agent),
                        })
                    attach_retry_history(agent, retry_history)
                    interaction = getattr(agent, "last_interaction", None)
                    if interaction is not None:
                        with interaction_lock:
                            interactions.append(interaction)
                    return result
                return invoke

            def track_query_executor(executor):
                """Persist the exact selected query and bounded tool result in the trace."""
                if executor is None:
                    return None
                def invoke(query, request):
                    try:
                        evidence = executor(query, request)
                    except Exception as exc:
                        with interaction_lock:
                            interactions.append({
                                "agent": "query_executor", "status": "failed",
                                "selected_query": query.model_dump(mode="json"),
                                "input": {"query": query.model_dump(mode="json")},
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                        raise
                    with interaction_lock:
                        interactions.append({
                            "agent": "query_executor", "status": "executed",
                            # This is the post-selection QuerySpec, not the
                            # planner's unfiltered candidate list.
                            "selected_query": query.model_dump(mode="json"),
                            "input": {"query": query.model_dump(mode="json")},
                            "output": {
                                "evidence_id": evidence.id,
                                "status": evidence.status.value if hasattr(evidence.status, "value") else evidence.status,
                                "summary": evidence.summary,
                                "structured_facts": evidence.structured_facts,
                                "tool": evidence.tool.value if hasattr(evidence.tool, "value") else evidence.tool,
                                "query_id": evidence.query_id,
                                "source_anchors": [anchor.model_dump(mode="json") for anchor in evidence.source_anchors[:30]],
                                "scope": evidence.scope.model_dump(mode="json"),
                                "analysis_boundary": evidence.analysis_boundary,
                                "witness": evidence.witness,
                                "raw_result": (evidence.raw_result[:12000] if isinstance(evidence.raw_result, str) else evidence.raw_result),
                            },
                        })
                    return evidence
                return invoke

            tracked_handlers = replace(
                handlers,
                risk=track(handlers.risk, "risk_analyst"),
                protection=track(handlers.protection, "protection_analyst"),
                diagnose=track_diagnostician(handlers.diagnose),
                query_planner=track_query_planner(handlers.query_planner),
                execute_query=track_query_executor(handlers.execute_query),
                investigate_evidence=track_fact_investigator(handlers.investigate_evidence),
            )
        graph = build_audit_graph(
            tracked_handlers, max_query_rounds=max_query_rounds,
            decision_policy=decision_policy,
        )
        result = graph.invoke(
            {
                "input": sample.to_analysis_input(
                    context_mode=effective_context_mode,
                    repository_root=str(repository_root) if repository_root else None,
                    repository_commit=repository_commit,
                    repository_match_status=repository_match_status,
                    repository_match_note=repository_match_note,
                    repository_function_hash=repository_function_hash,
                    repository_checked_commits=repository_checked_commits,
                )
            }
        )
        return SampleRunResult(
            sample_id=sample.sample_id,
            idx=sample.idx,
            target=sample.target,
            report=result.get("report"),
            audit_state=result.get("audit_state"),
            agent_interactions=interactions,
        )
    except Exception as exc:  # batch processing must preserve other samples
        return SampleRunResult(
            sample_id=sample.sample_id,
            idx=sample.idx,
            target=sample.target,
            agent_interactions=interactions,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_samples(
    samples: Iterable[PrimeVulSample],
    *,
    handlers: WorkflowHandlers | None = None,
    max_query_rounds: int = 2,
    stop_after: int | None = None,
    repository_root_resolver: Callable[[PrimeVulSample], str | Path | None] | None = None,
    context_mode: str = "function_only",
    repository_manager: RepositoryManager | None = None,
    decision_policy: str = "balanced",
    agent_retries: int = 0,
    progress: bool = False,
) -> BatchRunReport:
    """Run samples sequentially, retaining per-sample state and failures."""

    rows = list(samples)
    if stop_after is not None:
        rows = rows[:stop_after]
    results = []
    total_rows = len(rows)
    for position, row in enumerate(rows, start=1):
        result = run_sample(
            row,
            handlers=handlers,
            max_query_rounds=max_query_rounds,
            repository_root=repository_root_resolver(row) if repository_root_resolver else None,
            context_mode=context_mode,
            repository_manager=repository_manager,
            decision_policy=decision_policy,
            agent_retries=agent_retries,
        )
        results.append(result)
        if progress:
            report = result.report
            payload = {
                "progress": f"{position}/{total_rows}",
                "event": "sample_completed" if result.error is None else "sample_failed",
                "idx": result.idx,
                "sample_id": result.sample_id,
                "target": result.target,
                "prediction": (
                    getattr(report.prediction, "value", report.prediction)
                    if report is not None else None
                ),
                "confidence": report.decision_confidence if report is not None else None,
                "evidence_grade": report.decision_evidence_grade if report is not None else None,
                "reason": (
                    (report.explanation.splitlines()[0][:300] if report and report.explanation else None)
                    if result.error is None else result.error
                ),
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)
    completed = sum(result.error is None for result in results)
    return BatchRunReport(
        dataset_name=rows[0].dataset_name if rows else "unknown",
        requested=len(rows),
        completed=completed,
        failed=len(rows) - completed,
        results=results,
    )


def run_jsonl(
    path: str,
    *,
    handlers: WorkflowHandlers | None = None,
    max_query_rounds: int = 2,
    limit: int | None = None,
    strict: bool = True,
    repository_root_resolver: Callable[[PrimeVulSample], str | Path | None] | None = None,
    context_mode: str = "function_only",
    repository_manager: RepositoryManager | None = None,
    decision_policy: str = "balanced",
    agent_retries: int = 0,
    progress: bool = False,
) -> BatchRunReport:
    """Read a PrimeVul JSONL file and run its samples through the graph."""

    return run_samples(
        iter_primevul(path, limit=limit, strict=strict),
        handlers=handlers,
        max_query_rounds=max_query_rounds,
        repository_root_resolver=repository_root_resolver,
        context_mode=context_mode,
        repository_manager=repository_manager,
        decision_policy=decision_policy,
        agent_retries=agent_retries,
        progress=progress,
    )


__all__ = ["BatchRunReport", "SampleRunResult", "run_jsonl", "run_sample", "run_samples"]
