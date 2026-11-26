"""Build bounded, provenance-preserving disagreement-diagnosis inputs."""

from __future__ import annotations

from typing import Any

from .protocols import (
    Claim, DiagnosisInput, DiagnosisProtection, DiagnosisRiskCandidate,
    DiagnosisSourceRef, ProgramScope, SourceLocation,
)


def normalize_claim_entity_ids(claim: Claim) -> Claim:
    """Give model entities deterministic IDs before they cross agent boundaries.

    Providers often invent UUID-like candidate IDs on each call.  The rest of the
    workflow needs IDs that survive retries and can be referenced by the
    diagnostician and fact investigator, so the program owns this namespace.
    """
    analysis = dict(claim.analysis)
    kind = getattr(claim.kind, "value", claim.kind)
    if kind == "risk":
        key, prefix, limit = "candidates", "risk", 3
    elif kind == "protection":
        key, prefix, limit = "protections", "protection", 8
    else:
        return claim
    rows = analysis.get(key)
    if not isinstance(rows, list):
        return claim
    normalized = []
    for index, row in enumerate(rows[:limit], start=1):
        if not isinstance(row, dict):
            continue
        item = dict(row)
        old_id = item.get("candidate_id") if prefix == "risk" else item.get("id")
        stable_id = f"{prefix}-{index}"
        item["candidate_id" if prefix == "risk" else "id"] = stable_id
        if old_id and old_id != stable_id:
            item["model_entity_id"] = str(old_id)
        normalized.append(item)
    analysis[key] = normalized
    return claim.model_copy(update={"analysis": analysis})


def build_diagnosis_input(
    sample_id: str,
    risk_claim: Claim,
    protection_claim: Claim,
    scope: ProgramScope,
    skeleton: dict[str, Any],
) -> DiagnosisInput:
    refs: list[DiagnosisSourceRef] = []
    ref_keys: dict[tuple[Any, ...], str] = {}

    def add_ref(side: str, anchor: Any, trusted_location: bool = False) -> str | None:
        if not isinstance(anchor, dict):
            if not trusted_location:
                return None
            anchor = anchor.model_dump(mode="json")
        excerpt = anchor.get("excerpt")
        if anchor.get("validated") is False or (not trusted_location and not excerpt):
            return None
        location_values = {
            key: anchor.get(key)
            for key in ("file", "function", "line", "column", "end_line", "code_hash")
            if anchor.get(key) is not None
        }
        if not location_values.get("file"):
            return None
        try:
            location = SourceLocation.model_validate(location_values)
        except Exception:
            return None
        key = side, location.file, location.line, excerpt, anchor.get("basis")
        if key in ref_keys:
            return ref_keys[key]
        ref_id = f"{side}-anchor-{1 + sum(ref.side == side for ref in refs)}"
        ref_keys[key] = ref_id
        refs.append(
            DiagnosisSourceRef(
                ref_id=ref_id,
                side=side,
                excerpt=excerpt,
                location=location,
                basis=anchor.get("basis"),
            )
        )
        return ref_id

    def collect_anchors(value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                if isinstance(item.get("excerpt"), str) and item.get("basis") in {
                    "function_source", "joern_skeleton"
                }:
                    found.append(item)
                    return
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
        return found

    def claim_ref_ids(claim: Claim, side: str) -> list[str]:
        result = [
            ref for anchor in collect_anchors(claim.analysis)
            if (ref := add_ref(side, anchor))
        ]
        result.extend(
            ref for anchor in claim.source_anchors
            if (ref := add_ref(side, anchor, trusted_location=True))
        )
        return list(dict.fromkeys(result))

    def dict_list(value: Any, limit: int) -> list[dict[str, Any]]:
        return [item for item in value[:limit] if isinstance(item, dict)] if isinstance(value, list) else []

    def string_list(value: Any, limit: int) -> list[str]:
        return [str(item)[:500] for item in value[:limit] if isinstance(item, str)] if isinstance(value, list) else []

    risk_ref_ids = claim_ref_ids(risk_claim, "risk")
    protection_ref_ids = claim_ref_ids(protection_claim, "protection")
    risk_analysis = risk_claim.analysis
    candidates: list[DiagnosisRiskCandidate] = []
    raw_candidates = risk_analysis.get("candidates")
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    for index, item in enumerate(raw_candidates[:3], start=1):
        if not isinstance(item, dict):
            continue
        candidate_refs = [
            ref for anchor in collect_anchors(item)
            if (ref := add_ref("risk", anchor))
        ]
        candidates.append(
            DiagnosisRiskCandidate(
                candidate_id=str(item.get("candidate_id") or f"risk-{index}"),
                model_entity_id=str(item.get("model_entity_id")) if item.get("model_entity_id") else None,
                candidate_class=item.get("candidate_class", "speculative"),
                priority=item.get("priority"),
                summary=str(item.get("summary") or ""),
                security_property=item.get("security_property"),
                risk_operation=item.get("risk_operation"),
                target_object=item.get("target_object"),
                address_source=item.get("address_source"),
                access_type=item.get("access_type"),
                alignment_requirement=item.get("alignment_requirement"),
                conversion_source_type=item.get("conversion_source_type"),
                conversion_target_type=item.get("conversion_target_type"),
                representation_requirement=item.get("representation_requirement"),
                risk_path=dict_list(item.get("risk_path"), 6),
                required_conditions=dict_list(item.get("required_conditions"), 6),
                unresolved_questions=string_list(item.get("unresolved_questions"), 6),
                source_ref_ids=list(dict.fromkeys(candidate_refs)),
            )
        )
    if not candidates and risk_analysis.get("risk_operation"):
        candidates.append(
            DiagnosisRiskCandidate(
                candidate_id="risk-primary",
                priority=1,
                summary=risk_claim.text,
                security_property=risk_analysis.get("security_property"),
                risk_operation=risk_analysis.get("risk_operation"),
                target_object=risk_analysis.get("target_object"),
                address_source=risk_analysis.get("address_source"),
                access_type=risk_analysis.get("access_type"),
                alignment_requirement=risk_analysis.get("alignment_requirement"),
                risk_path=dict_list(risk_analysis.get("risk_path"), 6),
                required_conditions=dict_list(risk_analysis.get("required_conditions"), 6),
                unresolved_questions=string_list(risk_analysis.get("unresolved_questions"), 6),
                source_ref_ids=risk_ref_ids,
            )
        )

    protection_analysis = protection_claim.analysis
    protections: list[DiagnosisProtection] = []
    raw_protections = protection_analysis.get("protections")
    if not isinstance(raw_protections, list):
        raw_protections = []
    for index, item in enumerate(raw_protections[:8], start=1):
        if not isinstance(item, dict):
            continue
        protection_refs = [
            ref for anchor in collect_anchors(item)
            if (ref := add_ref("protection", anchor))
        ]
        protections.append(
            DiagnosisProtection(
                protection_id=str(item.get("id") or f"protection-{index}"),
                model_entity_id=str(item.get("model_entity_id")) if item.get("model_entity_id") else None,
                kind=item.get("kind"),
                description=str(item.get("description") or ""),
                protected_object=item.get("protected_object"),
                path_condition=item.get("path_condition"),
                coverage=item.get("coverage", "unknown"),
                blocks_conditions=string_list(item.get("blocks_conditions"), 8),
                bypass_paths=string_list(item.get("bypass_paths"), 8),
                source_ref_ids=list(dict.fromkeys(protection_refs)),
            )
        )

    skeleton_view = _compact_skeleton(skeleton)
    for field in ("structural_operations", "calls", "controls", "candidate_dataflows", "source_candidates"):
        for index, item in enumerate(skeleton_view.get(field, []), start=1):
            code = item.get("code") or item.get("call_code") or item.get("source_code")
            if not code:
                continue
            location = None
            if item.get("file") and isinstance(item.get("line"), int) and item["line"] > 0:
                location = SourceLocation(
                    file=str(item["file"]), function=scope.entrypoint, line=item["line"]
                )
            ref_id = f"joern-{field}-{index}"
            refs.append(
                DiagnosisSourceRef(
                    ref_id=ref_id,
                    side="joern",
                    excerpt=str(code)[:400],
                    location=location,
                    basis="joern_skeleton",
                )
            )
            item["source_ref_id"] = ref_id

    return DiagnosisInput(
        sample_id=sample_id,
        scope=scope,
        risk_claim_id=risk_claim.id,
        risk_claim_status=risk_claim.status,
        risk_claim_text=risk_claim.text[:1200],
        risk_candidates=candidates,
        risk_source_ref_ids=risk_ref_ids,
        protection_claim_id=protection_claim.id,
        protection_claim_status=protection_claim.status,
        protection_claim_text=protection_claim.text[:1200],
        protections=protections,
        protection_invariants=dict_list(protection_analysis.get("invariants"), 8),
        protection_source_ref_ids=protection_ref_ids,
        source_refs=refs[:60],
        relevant_skeleton=skeleton_view,
    )


def _compact_skeleton(skeleton: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: skeleton.get(key)
        for key in (
            "status", "mode", "scope", "entrypoint", "evidence_semantics",
            "node_counts", "truncated", "unresolved_dependencies",
        )
        if key in skeleton
    }
    for key, limit in (
        ("methods", 4), ("structural_operations", 24), ("calls", 16),
        ("controls", 12), ("candidate_dataflows", 16),
        ("source_candidates", 10), ("parameters", 10),
    ):
        values = skeleton.get(key)
        if isinstance(values, list):
            result[key] = [item for item in values[:limit] if isinstance(item, dict)]
    return result


__all__ = ["build_diagnosis_input", "normalize_claim_entity_ids"]
