"""Bounded model-facing views of durable workflow objects.

The audit ledger retains full objects. These helpers remove transport/debug material and cap
repeated structural clues before the next Agent call.
"""
from __future__ import annotations

from typing import Any

from .protocols import Claim, Evidence


_SKELETON_LIMITS = {
    "methods": 4,
    "parameters": 10,
    "structural_operations": 24,
    "calls": 16,
    "controls": 12,
    "candidate_dataflows": 16,
    "source_candidates": 10,
}


def compact_skeleton(skeleton: dict[str, Any]) -> dict[str, Any]:
    """Keep repository/Joern semantics and the highest-priority bounded clues."""
    result = {
        key: _bound_value(value)
        for key, value in skeleton.items()
        if key not in _SKELETON_LIMITS
        and key not in {"raw_output", "query_script", "command"}
    }
    for key, limit in _SKELETON_LIMITS.items():
        values = skeleton.get(key)
        if isinstance(values, list):
            result[key] = [_bound_value(item) for item in values[:limit]]
    return result


def compact_claim(claim: Claim, *, summary_only: bool = False) -> Claim:
    """Remove recursive Agent transport logs while retaining analysis semantics."""
    analysis = {} if summary_only else {
        key: _bound_value(value)
        for key, value in claim.analysis.items()
        if key != "_interaction"
    }
    return claim.model_copy(update={"analysis": analysis})


def compact_planning_skeleton(skeleton: dict[str, Any]) -> dict[str, Any]:
    """A smaller topology view for query selection after diagnosis already extracted facts."""
    compact = compact_skeleton(skeleton)
    for key in _SKELETON_LIMITS:
        if isinstance(compact.get(key), list):
            compact[key] = compact[key][:8]
    return compact


def compact_evidence(evidence: Evidence, *, raw_limit: int = 24000) -> Evidence:
    """Bound raw tool output for interpretation; anchors and provenance stay intact."""
    raw = evidence.raw_result
    if isinstance(raw, str) and len(raw) > raw_limit:
        raw = raw[:raw_limit] + f"\n... [tool output truncated; original_chars={len(evidence.raw_result)}]"
    return evidence.model_copy(update={"raw_result": raw})


def compact_key_operations(values: Any, *, limit: int = 72) -> list[dict[str, Any]]:
    """Prefer operations most relevant to memory-safety review."""
    if not isinstance(values, list):
        return []
    priority = {
        "narrow_integer_conversion": 0, "integer_conversion": 1,
        "alignment_sensitive_cast": 2, "typed_dereference": 3,
        "cast": 4, "dereference": 5, "index": 6, "allocation": 7,
        "pointer_step": 8, "arithmetic": 9, "condition": 10,
    }
    rows = [item for item in values if isinstance(item, dict)]
    rows.sort(key=lambda item: (
        min((priority.get(str(category), 99) for category in item.get("categories", [])), default=99),
        int(item.get("line") or 0),
    ))
    selected = rows[:limit]
    return sorted(selected, key=lambda item: int(item.get("line") or 0))


def _bound_value(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 1000 else value[:1000] + "... [truncated]"
    if isinstance(value, list):
        return [_bound_value(item) for item in value[:40]]
    if isinstance(value, dict):
        return {str(key): _bound_value(item) for key, item in value.items()}
    return value


__all__ = [
    "compact_claim", "compact_evidence", "compact_key_operations",
    "compact_planning_skeleton", "compact_skeleton",
]
