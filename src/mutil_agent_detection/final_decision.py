"""Evidence-led, conservative synthesis for the final sample-level prediction."""

from __future__ import annotations

from .protocols import (
    AuditState,
    Claim,
    ClaimKind,
    EvidenceStatus,
    FinalReport,
    Prediction,
    SourceLocation,
)


def synthesize_final_report(
    audit: AuditState, *, decision_policy: str = "strict"
) -> FinalReport:
    """Synthesize a verdict under strict, balanced, or recall-first evidence rules."""
    if decision_policy not in {"strict", "balanced", "recall_first"}:
        raise ValueError(f"unsupported decision policy: {decision_policy}")
    risks = [claim for claim in audit.claims.values() if claim.kind == ClaimKind.RISK]
    protections = [claim for claim in audit.claims.values() if claim.kind == ClaimKind.PROTECTION]
    valid_evidence = audit.evidence
    disputes = list(audit.disputes.values())
    dispute = disputes[-1] if disputes else None
    premises = dispute.premises if dispute else []
    candidate_rows = {
        str(item.get("candidate_id")): item
        for claim in risks
        for item in claim.analysis.get("candidates", [])
        if isinstance(item, dict) and item.get("candidate_id")
    }
    # Program-derived facts from the first-pass source anchors. These are kept
    # separate from model conclusions so later query failures do not erase
    # already verified code facts.
    initial_evidence_summary: list[str] = []
    for candidate in candidate_rows.values():
        anchors = [
            item for item in candidate.get("source_anchors", [])
            if isinstance(item, dict) and item.get("validated") is True
        ]
        operation = candidate.get("risk_operation")
        target = candidate.get("target_object")
        if operation and target and anchors:
            initial_evidence_summary.append(
                f"{candidate.get('candidate_id')} [{candidate.get('candidate_class', 'speculative')}]: "
                f"validated source shows {operation} on {target} "
                f"({len(anchors)} anchor(s)); this is a code fact, not an exploitability proof."
            )
    decisive_ids = {premise.id for premise in premises if premise.decisive}
    # Include explicit risk preconditions even when the diagnostician omitted the decisive flag.
    decisive_ids.update(
        str(condition.get("id"))
        for item in candidate_rows.values()
        for condition in item.get("required_conditions", [])
        if isinstance(condition, dict) and condition.get("id") in {p.id for p in premises}
        and condition.get("status") != "supported"
    )
    decisive = [premise for premise in premises if premise.id in decisive_ids]
    evidence_ids = set(valid_evidence)
    valid_refs: set[str] = set()
    questions: list[str] = []
    investigated_facts = []
    risk_assessments = []
    candidate_condition_assessments = []
    protection_assessments = []
    investigated_premise_statuses: dict[str, set[str]] = {}
    for investigation in audit.fact_investigations.values():
        questions.extend(investigation.unresolved_questions)
        questions.extend(investigation.limitations)
        for update in investigation.premise_updates:
            update_refs = set(update.evidence_refs)
            if (
                update.status in {"supported", "refuted"}
                and update_refs
                and update_refs <= set(valid_evidence)
                and all(valid_evidence[eid].status != EvidenceStatus.INCONCLUSIVE for eid in update_refs)
            ):
                valid_refs.update(update_refs)
                investigated_premise_statuses.setdefault(update.premise_id, set()).add(update.status)
        for fact in investigation.facts:
            if fact.evidence_refs and set(fact.evidence_refs) <= set(valid_evidence):
                valid_refs.update(fact.evidence_refs)
                investigated_facts.append(fact)
        for assessment in investigation.risk_assessments:
            evidence_refs = set(assessment.evidence_refs)
            if evidence_refs and evidence_refs <= evidence_ids and all(
                valid_evidence[eid].status != EvidenceStatus.INCONCLUSIVE for eid in evidence_refs
            ):
                risk_assessments.append(assessment)
                valid_refs.update(evidence_refs)
        for assessment in investigation.candidate_condition_assessments:
            evidence_refs = set(assessment.evidence_refs)
            if evidence_refs and evidence_refs <= evidence_ids and all(
                valid_evidence[eid].status != EvidenceStatus.INCONCLUSIVE for eid in evidence_refs
            ):
                candidate_condition_assessments.append(assessment)
                valid_refs.update(evidence_refs)
        for assessment in investigation.protection_assessments:
            evidence_refs = set(assessment.evidence_refs)
            if evidence_refs and evidence_refs <= evidence_ids and all(
                valid_evidence[eid].status != EvidenceStatus.INCONCLUSIVE for eid in evidence_refs
            ):
                protection_assessments.append(assessment)
                valid_refs.update(evidence_refs)

    # A valid investigator update is authoritative for the queried premise,
    # even if a retry/state merge left the durable Dispute status unchanged.
    supported = [premise for premise in premises if (
        (premise.status == "supported" and premise.evidence_refs and set(premise.evidence_refs) <= evidence_ids)
        or investigated_premise_statuses.get(premise.id) == {"supported"}
    )]
    refuted = [premise for premise in premises if (
        (premise.status == "refuted" and premise.evidence_refs and set(premise.evidence_refs) <= evidence_ids)
        or investigated_premise_statuses.get(premise.id) == {"refuted"}
    )]
    supported_ids = {premise.id for premise in supported}
    refuted_ids = {premise.id for premise in refuted}
    resolved_ids = supported_ids | refuted_ids
    unresolved = [premise for premise in premises if premise.id not in resolved_ids]

    risk_support: set[str] = set()
    risk_refutation: set[str] = set()
    for risk in risks:
        for evidence_id, evidence in valid_evidence.items():
            if evidence.status == EvidenceStatus.SUPPORTS and risk.id in evidence.supports_claim_ids:
                risk_support.add(evidence_id)
            if evidence.status == EvidenceStatus.NEGATES and risk.id in evidence.negates_claim_ids:
                risk_refutation.add(evidence_id)

    candidate_status: dict[str, str] = {}
    candidate_refs: dict[str, set[str]] = {}
    assessment_statuses: dict[str, set[str]] = {}
    condition_assessment_statuses: dict[tuple[str, str], set[str]] = {}
    for item in risk_assessments:
        assessment_statuses.setdefault(item.candidate_id, set()).add(item.status)
        candidate_refs.setdefault(item.candidate_id, set()).update(item.evidence_refs)
    for investigation in audit.fact_investigations.values():
        for item in investigation.candidate_condition_assessments:
            refs = set(item.evidence_refs)
            if not refs or not refs <= evidence_ids:
                continue
            if any(valid_evidence[eid].status == EvidenceStatus.INCONCLUSIVE for eid in refs):
                continue
            condition_assessment_statuses.setdefault(
                (item.candidate_id, item.premise_id), set()
            ).add(item.status)
    for candidate_id, statuses in assessment_statuses.items():
        # A supported result from one query must not override an unknown
        # result from another query. Candidate evidence is conjunctive.
        if "refuted" in statuses and "supported" not in statuses:
            candidate_status[candidate_id] = "refuted"
        elif statuses == {"supported"}:
            candidate_status[candidate_id] = "supported"
        else:
            candidate_status[candidate_id] = "unknown"
    # A candidate can be resolved without a separate candidate assessment when every
    # declared required condition has been evidence-resolved. This is a mechanical
    # projection, not a new model conclusion.
    for candidate_id, candidate in candidate_rows.items():
        if candidate_id in candidate_status:
            continue
        condition_ids = [
            str(condition.get("id")) for condition in candidate.get("required_conditions", [])
            if isinstance(condition, dict) and condition.get("id")
        ]
        condition_map = {premise.id: premise for premise in premises}
        if condition_ids and all(condition_id in condition_map for condition_id in condition_ids):
            condition_rows = [condition_map[condition_id] for condition_id in condition_ids]
            if all(row.id in supported_ids for row in condition_rows):
                candidate_status[candidate_id] = "supported"
                candidate_refs[candidate_id] = {
                    ref for row in condition_rows for ref in row.evidence_refs
                }
            elif any(row.id in refuted_ids for row in condition_rows):
                candidate_status[candidate_id] = "refuted"
                candidate_refs[candidate_id] = {
                    ref for row in condition_rows for ref in row.evidence_refs
                }
    # Reconcile model assessments with the diagnostician's candidate-to-premise
    # links. A conditional candidate is supported only when every linked
    # necessary premise is supported; one supported premise is only partial
    # evidence and remains unknown.
    for candidate_id, current_status in list(candidate_status.items()):
        candidate = candidate_rows.get(candidate_id, {})
        candidate_class = str(candidate.get("candidate_class") or "direct_violation")
        # Direct local violations may use the balanced minimum-evidence rule;
        # interprocedural/speculative candidates require conjunction of all
        # diagnostician-linked premises.
        if candidate_class == "direct_violation":
            continue
        linked = [premise for premise in premises if candidate_id in premise.claim_refs]
        if not linked:
            continue
        linked_statuses = {}
        for premise in linked:
            statuses = condition_assessment_statuses.get((candidate_id, premise.id), set())
            if premise.status in {"supported", "refuted"}:
                linked_statuses[premise.id] = premise.status
            elif statuses == {"supported"}:
                linked_statuses[premise.id] = "supported"
            elif statuses == {"refuted"}:
                linked_statuses[premise.id] = "refuted"
            else:
                linked_statuses[premise.id] = "unknown"
        if any(status == "refuted" for status in linked_statuses.values()):
            candidate_status[candidate_id] = "refuted"
        elif all(status == "supported" for status in linked_statuses.values()):
            candidate_status[candidate_id] = "supported"
        else:
            candidate_status[candidate_id] = "unknown"
    effective_protections = {
        item.protection_id for item in protection_assessments if item.coverage_status == "effective"
    }
    relations = dispute.claim_relations if dispute else []
    relations_by_candidate: dict[str, list] = {}
    for item in relations:
        relations_by_candidate.setdefault(item.risk_candidate_id, []).append(item)

    def candidate_has_matching_effective_protection(candidate_id: str) -> bool:
        """Require an evidence-backed protection tied to this exact risk property/path."""
        return any(
            bool(set(relation.protection_ids) & effective_protections)
            and relation.relation in {"supports", "contradicts", "limits"}
            and relation.object_relation == "same"
            and relation.property_relation == "matches"
            and relation.path_coverage == "complete"
            for relation in relations_by_candidate.get(candidate_id, [])
        )

    protected_candidates = {
        candidate_id for candidate_id in candidate_rows
        if candidate_has_matching_effective_protection(candidate_id)
    }
    decisive_complete = all(premise.id in supported_ids for premise in decisive)
    decisive_refuted = all(premise.id in refuted_ids for premise in decisive)
    unresolved_decisive = any(premise.id not in resolved_ids for premise in decisive)
    has_effective_protection = bool(protected_candidates)
    direct_risk_support = bool(risk_support) and not candidate_rows and not decisive
    decision_basis: list[str] = []
    decision_anchors = []
    decision_confidence = "low"
    decision_evidence_grade = "none"
    supported_candidates = {
        key for key, status in candidate_status.items() if status == "supported"
    }

    def candidate_required_conditions_supported(candidate_id: str) -> bool:
        """Check the candidate's own necessary conditions, not just one query result.

        A conditional candidate can be mentioned as ``supported`` by an
        investigator that answered only one premise.  That is useful partial
        evidence, but it must not become a sample-level vulnerability verdict.
        The candidate declaration is the authoritative list of conditions that
        still have to be closed.
        """
        candidate = candidate_rows.get(candidate_id, {})
        conditions = [
            item for item in candidate.get("required_conditions", [])
            if isinstance(item, dict) and item.get("id")
        ]
        if not conditions:
            return False
        for condition in conditions:
            condition_id = str(condition["id"])
            # The risk model's initial ``status`` is a hypothesis, not an
            # executed fact.  Only a durable premise update or a validated
            # condition assessment can close this requirement.
            if condition_id in supported_ids:
                continue
            statuses = condition_assessment_statuses.get((candidate_id, condition_id), set())
            if statuses != {"supported"}:
                return False
        return True

    # Only a fully closed conditional/speculative candidate may enter the
    # evidence-complete branch.  Direct local violations remain eligible for
    # the regular balanced minimum-evidence rule below.
    supported_candidates = {
        candidate_id for candidate_id in supported_candidates
        if str(candidate_rows.get(candidate_id, {}).get("candidate_class") or "direct_violation")
        == "direct_violation"
        or candidate_required_conditions_supported(candidate_id)
    }
    if ((supported_candidates or direct_risk_support)
            and decisive_complete and not unresolved_decisive
            and not (supported_candidates & protected_candidates)):
        prediction = Prediction.VULNERABLE
        reason = "At least one risk candidate is supported by query evidence, its decisive premises are supported, and no effective protection is established."
        decision_confidence = "high"
        decision_evidence_grade = "A" if any(
            valid_evidence.get(ref) is not None
            and valid_evidence[ref].status in {EvidenceStatus.SUPPORTS, EvidenceStatus.NEGATES}
            for ref in valid_refs
        ) else "B"
        decision_basis.append("Strict: query evidence supports a risk candidate and all decisive premises are supported.")
        for candidate_id, status in candidate_status.items():
            if status == "supported":
                valid_refs.update(candidate_refs.get(candidate_id, set()))
        if direct_risk_support:
            valid_refs.update(risk_support)
    elif (
        bool(candidate_rows)
        and all(
            candidate_status.get(candidate_id) == "refuted"
            or candidate_id in protected_candidates
            for candidate_id in candidate_rows
        )
        and not unresolved_decisive
    ):
        prediction = Prediction.BENIGN
        reason = "Every risk candidate is evidence-refuted or covered by an evidence-backed protection matching the same object, property, and complete path."
        decision_confidence = "high"
        decision_evidence_grade = "A" if any(
            valid_evidence.get(ref) is not None
            and valid_evidence[ref].status in {EvidenceStatus.SUPPORTS, EvidenceStatus.NEGATES}
            for ref in valid_refs
        ) else "B"
        decision_basis.append("Candidate-level query evidence refutes each risk or establishes a property-matched complete protection.")
        for candidate_id, status in candidate_status.items():
            if status == "refuted":
                valid_refs.update(candidate_refs.get(candidate_id, set()))
        valid_refs.update(
            ref for item in protection_assessments
            if item.coverage_status == "effective" and item.protection_id in effective_protections
            for ref in item.evidence_refs
        )
    else:
        # Balanced policies can use independently analyzed, program-validated
        # source anchors when query tools have not established platform/build
        # details. This is deliberately limited to structurally complete risk
        # candidates whose property is explicitly not addressed by the supplied
        # protections. Mere candidate presence or model confidence is not enough.
        if decision_policy in {"balanced", "recall_first"}:
            refuted_candidates = {key for key, status in candidate_status.items() if status == "refuted"}
            eligible: list[tuple[dict, Claim, list]] = []
            for risk in risks:
                for candidate in risk.analysis.get("candidates", []):
                    candidate_id = candidate.get("candidate_id") if isinstance(candidate, dict) else None
                    if not candidate_id or candidate_id in refuted_candidates:
                        continue
                    candidate_relations = relations_by_candidate.get(candidate_id, [])
                    relation = next(
                        (
                            item for item in candidate_relations
                            if item.relation in {"does_not_address", "limits", "unresolved"}
                            and (
                                item.property_relation == "mismatches"
                                or (
                                    item.relation == "does_not_address"
                                    and item.path_coverage == "none"
                                )
                            )
                        ),
                        None,
                    )
                    if decision_policy == "balanced" and not (
                        relation is not None
                    ):
                        continue
                    if candidate_id in protected_candidates:
                        continue
                    anchors = [
                        item for item in candidate.get("source_anchors", [])
                        if isinstance(item, dict) and item.get("validated") is True
                        and item.get("file") and item.get("line")
                    ]
                    # Balanced uses a minimum sufficient code-fact set rather
                    # than requiring every environmental premise to resolve.
                    # Strict still follows the decisive-premise branch above.
                    complete = all((
                        bool(candidate.get("risk_operation")),
                        bool(candidate.get("target_object")),
                        bool(candidate.get("security_property")),
                        len(anchors) >= 2,
                        bool(candidate.get("risk_path") or candidate.get("address_source")),
                    ))
                    # A protection mismatch says only that safety was not
                    # established. Balanced also requires positive risk
                    # evidence: either a query-supported candidate or a
                    # source-verifiable typed-access/alignment violation.
                    query_supported = candidate_status.get(candidate_id) == "supported"
                    property_name = str(candidate.get("security_property") or "").lower()
                    # A supported semantic conversion is not automatically a
                    # security violation. Conditional candidates need a
                    # concrete memory-safety/undefined-behavior property
                    # before query support can promote them.
                    query_support_eligible = property_name in {
                        "buffer_overflow", "out_of_bounds_read", "out_of_bounds_write",
                        "use_after_free", "double_free", "invalid_free", "null_deref",
                        "uninitialized_read", "misaligned_access", "alignment",
                        "integer_overflow", "signedness", "undefined_behavior",
                    }
                    alignment_text = str(candidate.get("alignment_requirement") or "").lower()
                    candidate_class = str(candidate.get("candidate_class") or "direct_violation")
                    explicit_property_violation = all((
                        property_name in {"misaligned_access", "alignment", "type_safety"},
                        property_name != "type_safety" or candidate_class == "direct_violation",
                        bool(candidate.get("address_source")),
                        bool(candidate.get("access_type")),
                        bool(candidate.get("alignment_requirement")),
                        any(token in alignment_text for token in (
                            "not established", "not guaranteed", "unknown", "unaligned",
                            "no alignment", "not proven",
                        )),
                        relation is not None and relation.property_relation == "mismatches",
                    ))
                    representation_text = str(candidate.get("representation_requirement") or "").lower()
                    explicit_conversion_violation = all((
                        str(candidate.get("candidate_class") or "") == "direct_violation",
                        property_name in {
                            "integer_conversion", "integer_overflow", "numeric_range",
                            "type_safety", "undefined_behavior", "signedness",
                        },
                        bool(candidate.get("conversion_target_type")),
                        bool(candidate.get("conversion_source_type")),
                        bool(candidate.get("representation_requirement")),
                        any(token in representation_text for token in (
                            "not representable", "out of range", "outside", "undefined",
                            "not established", "not guaranteed",
                        )),
                    ))
                    if complete and (
                        decision_policy == "recall_first"
                        or (query_supported and query_support_eligible)
                        or explicit_property_violation
                        or explicit_conversion_violation
                    ):
                        eligible.append((candidate, risk, anchors))
            if eligible:
                candidate, _risk, anchors = eligible[0]
                prediction = Prediction.VULNERABLE
                decision_confidence = "medium" if decision_policy == "balanced" else "low"
                decision_evidence_grade = "B" if decision_policy == "balanced" else "C"
                label = "Balanced" if decision_policy == "balanced" else "Recall-first"
                reason = (
                    f"{label}: a structurally complete risk candidate has multiple program-validated source anchors, "
                    "positive risk evidence establishes its safety-property conflict, and the diagnosed protection "
                    "relation does not cover or match that property. "
                    "Some environmental premises remain unresolved."
                )
                decision_basis.extend([
                    "B-level combined evidence: program-validated anchors confirm the cited code fragments; causal path and trigger interpretation comes from structured agent analysis, not an independently executed path witness.",
                    "The diagnosed protection relation does not address the candidate property or covers only part of its path.",
                    "Environmental/build premises remain unresolved; confidence is therefore below strict confirmation.",
                    f"Decision anchor line basis: {audit.input.metadata.get('source_line_basis', 'function_local')}.",
                ])
                decision_basis.append(
                    "Minimum sufficient evidence: the operation and source anchors are program-validated, "
                    "and either query evidence supports the candidate or a typed-access/alignment or integer-conversion "
                    "property conflict is explicit in the validated source facts."
                )
                if candidate.get("unverified_anchor_count"):
                    decision_basis.append(
                        f"{candidate['unverified_anchor_count']} supplementary candidate anchor(s) were not uniquely verified; they were not used as proof."
                    )
                decision_anchors = [SourceLocation(
                    file=item["file"], line=item["line"], function=audit.input.function_name,
                ) for item in anchors]
                for anchor in anchors:
                    valid_refs.update(
                        ref for ref, evidence in valid_evidence.items()
                        if any(
                            source.file == anchor["file"] and source.line == anchor["line"]
                            for source in evidence.source_anchors
                        )
                    )
            elif (
                decision_policy in {"balanced", "recall_first"}
                and candidate_status
                and set(candidate_status) >= set(candidate_rows)
                and all(status == "refuted" for status in candidate_status.values())
                and all(candidate_refs.get(key) for key in candidate_rows)
            ):
                prediction = Prediction.BENIGN
                decision_confidence = "medium"
                decision_evidence_grade = "B"
                reason = f"{decision_policy.title()}: every assessed risk candidate is refuted by cited query evidence."
                decision_basis.append("All assessed risk candidates have evidence-backed refutation.")
                for refs in candidate_refs.values():
                    valid_refs.update(refs)
            elif (
                decision_policy == "balanced"
                and candidate_rows
                and all(
                    (
                        candidate_status.get(candidate_id) == "refuted"
                        or candidate_id in protected_candidates
                        or (
                            str(candidate.get("candidate_class") or "speculative")
                            in {"conditional_interprocedural", "speculative"}
                            and candidate_status.get(candidate_id) == "unknown"
                            and bool(candidate_refs.get(candidate_id))
                        )
                    )
                    for candidate_id, candidate in candidate_rows.items()
                )
                and not any(
                    str(candidate.get("candidate_class") or "speculative") == "direct_violation"
                    and candidate_status.get(candidate_id) not in {"refuted"}
                    and candidate_id not in protected_candidates
                    for candidate_id, candidate in candidate_rows.items()
                )
            ):
                prediction = Prediction.BENIGN
                decision_confidence = "medium"
                decision_evidence_grade = "B"
                reason = (
                    "Balanced candidate closure: every direct violation was refuted or completely protected, "
                    "and every conditional/speculative candidate was investigated without establishing its "
                    "necessary condition."
                )
                decision_basis.extend([
                    "Candidate-level closure was applied; an unqueried hypothesis cannot be silently discarded.",
                    "Conditional/speculative candidates were closed only after evidence-backed investigation left their necessary condition unestablished.",
                ])
                for refs in candidate_refs.values():
                    valid_refs.update(refs)
            else:
                prediction = Prediction.UNKNOWN
                reason = "Conclusion remains unknown because evidence does not meet the selected policy's minimum risk/protection criteria."
                decision_basis.append("No decision threshold was met; unresolved or non-discriminating evidence remains.")
        else:
            prediction = Prediction.UNKNOWN
            gaps = []
            if not candidate_status:
                gaps.append("no evidence-backed risk-candidate assessment was produced")
            elif not any(status in {"supported", "refuted"} for status in candidate_status.values()):
                gaps.append("risk candidates remain unknown after investigation")
            if has_effective_protection and any(status == "supported" for status in candidate_status.values()):
                gaps.append("risk and effective-protection assessments conflict")
            if unresolved_decisive:
                gaps.append("decisive premises remain unresolved: " + ", ".join(p.id for p in decisive if p.id not in resolved_ids))
            elif decisive and not decisive_complete and not decisive_refuted:
                gaps.append("the decisive premises do not form a complete supported or refuted chain")
            reason = "Conclusion remains unknown because " + ("; ".join(gaps) or "evidence is conflicting or incomplete") + "."
            decision_basis.append("Strict: unresolved decisive premises prevent a binary verdict.")

    retained = []
    if prediction == Prediction.VULNERABLE:
        retained = [claim.id for claim in risks]
    elif prediction == Prediction.BENIGN:
        retained = [claim.id for claim in risks]
    explanation_parts = [reason]
    if dispute and dispute.summary:
        explanation_parts.append("Agent alignment: " + dispute.summary)
    limitations = list(dict.fromkeys(
        [
            dependency for dependency in audit.input.scope.unresolved_dependencies
            if not (
                dependency == "repository context not loaded"
                and any(
                    evidence.analysis_boundary
                    and "exact-commit repository" in evidence.analysis_boundary.lower()
                    for evidence in valid_evidence.values()
                )
            )
        ]
        + [e.analysis_boundary for e in valid_evidence.values() if e.analysis_boundary]
    ))
    if not audit.fact_investigations:
        limitations.append("No fact-investigation result was recorded.")

    conclusion_scope = audit.input.scope
    if any(
        evidence.analysis_boundary
        and "exact-commit repository" in evidence.analysis_boundary.lower()
        for evidence in valid_evidence.values()
    ):
        conclusion_scope = audit.input.scope.model_copy(update={
            "unresolved_dependencies": [
                dependency for dependency in audit.input.scope.unresolved_dependencies
                if dependency != "repository context not loaded"
            ]
        })

    return FinalReport(
        sample_id=audit.sample_id,
        prediction=prediction,
        decision_policy=decision_policy,
        decision_confidence=decision_confidence,
        decision_evidence_grade=decision_evidence_grade,
        decision_basis=decision_basis,
        decision_source_anchors=decision_anchors,
        initial_evidence_summary=list(dict.fromkeys(initial_evidence_summary)),
        retained_claim_ids=retained,
        risk_claim_ids=[claim.id for claim in risks],
        protection_claim_ids=[claim.id for claim in protections],
        supported_premise_ids=sorted(supported_ids),
        refuted_premise_ids=sorted(refuted_ids),
        evidence_refs=sorted(valid_refs),
        unresolved_questions=list(dict.fromkeys(questions)),
        risk_assessment=[claim.text for claim in risks],
        protection_assessment=[claim.text for claim in protections],
        disagreement_summary=dispute.summary if dispute else None,
        investigated_facts=investigated_facts,
        risk_assessments=risk_assessments,
        candidate_condition_assessments=candidate_condition_assessments,
        protection_assessments=protection_assessments,
        evidence_ids=list(valid_evidence),
        unresolved_premises=[premise.id for premise in unresolved],
        conclusion_scope=conclusion_scope,
        explanation="\n".join(explanation_parts),
        limitations=list(dict.fromkeys(limitations)),
        audit_event_ids=[event.id for event in audit.events],
    )


__all__ = ["synthesize_final_report"]
