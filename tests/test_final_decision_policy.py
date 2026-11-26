from mutil_agent_detection.final_decision import synthesize_final_report
from mutil_agent_detection.protocols import (
    AnalysisInput, AuditState, Claim, ClaimKind, ClaimStatus, Dispute,
    DisputeClaimRelation, DisputePremise, ProgramScope, SourceSnapshot,
    Evidence, EvidenceStatus, Backend, FactInvestigationResult, RiskCandidateAssessment,
    ProtectionAssessment,
)


def audit_with_unresolved_anchored_risk(*, verified=True, relation=True):
    scope = ProgramScope(
        snapshot=SourceSnapshot(version="sample", commit="sample"),
        entrypoint="parse_value",
    )
    anchors = [
        {"file": "parser.c", "line": 20, "excerpt": "*(double *) p", "validated": verified},
        {"file": "parser.c", "line": 21, "excerpt": "p = base + offset", "validated": verified},
    ]
    candidate = {
        "candidate_id": "risk-alignment",
        "security_property": "misaligned_access",
        "risk_operation": "dereference double pointer",
        "target_object": "p",
        "address_source": "base + input_offset",
        "access_type": "double",
        "alignment_requirement": "double alignment is not established",
        "risk_path": [{"order": 1, "operation": "dereference", "object": "p"}],
        "required_conditions": [{"id": "p1", "description": "input-derived offset"}],
        "source_anchors": anchors,
        "evidence_status": "anchors_verified" if verified else "unverified",
    }
    risk = Claim(
        id="risk-claim", kind=ClaimKind.RISK, text="conditional alignment risk",
        status=ClaimStatus.CANDIDATE, scope=scope, created_by="risk",
        analysis={"candidates": [candidate]},
        source_anchors=[],
    )
    protection = Claim(
        id="protection-claim", kind=ClaimKind.PROTECTION, text="range checks only",
        status=ClaimStatus.CANDIDATE, scope=scope, created_by="protection",
    )
    premises = [DisputePremise(
        id="p1", description="alignment condition", status="unknown",
        relation="unresolved", claim_refs=["risk-claim", "protection-claim"], decisive=True,
    )]
    relations = [DisputeClaimRelation(
        risk_candidate_id="risk-alignment", protection_ids=["protection-range"],
        relation="does_not_address", object_relation="same", property_relation="mismatches",
        path_coverage="none", explanation="Range checks do not prove alignment.",
    )] if relation else []
    dispute = Dispute(
        scope=scope, risk_claim_id=risk.id, protection_claim_id=protection.id,
        diagnosis_status="partial", alignment_status="partially_aligned",
        summary="Alignment protection mismatch.", premises=premises,
        unresolved_premises=["p1"], decisive_premises=["p1"], claim_relations=relations,
    )
    analysis_input = AnalysisInput(
        sample_id="sample-1", function_name="parse_value", source="void f() {}", scope=scope,
    )
    return AuditState(
        sample_id="sample-1", input=analysis_input,
        claims={risk.id: risk, protection.id: protection},
        disputes={dispute.id: dispute},
    )


def test_strict_preserves_unknown_for_unresolved_decisive_premise():
    report = synthesize_final_report(audit_with_unresolved_anchored_risk(), decision_policy="strict")
    assert report.prediction == "unknown"
    assert report.decision_policy == "strict"


def test_premise_update_is_used_when_dispute_status_was_not_merged():
    audit = audit_with_unresolved_anchored_risk()
    evidence = Evidence(
        id="e-premise", status=EvidenceStatus.SUPPORTS,
        summary="the decisive condition is established", scope=audit.input.scope,
        query_id="q-premise", tool=Backend.SOURCE,
    )
    audit.evidence[evidence.id] = evidence
    audit.fact_investigations["q-premise"] = FactInvestigationResult(
        query_id="q-premise", investigation_status="complete",
        premise_updates=[{
            "premise_id": "p1", "status": "supported", "reason": "source fact",
            "evidence_refs": [evidence.id],
        }],
    )
    report = synthesize_final_report(audit, decision_policy="strict")
    assert report.prediction == "vulnerable"
    assert report.supported_premise_ids == ["p1"]


def test_balanced_uses_verified_risk_anchors_and_property_mismatch():
    report = synthesize_final_report(audit_with_unresolved_anchored_risk(), decision_policy="balanced")
    assert report.prediction == "vulnerable"
    assert report.decision_confidence == "medium"
    assert report.decision_source_anchors
    assert report.decision_evidence_grade == "B"
    assert any("property" in item.lower() or "protection" in item.lower() for item in report.decision_basis)


def test_balanced_does_not_promote_unverified_or_unaligned_candidate():
    unverified = synthesize_final_report(
        audit_with_unresolved_anchored_risk(verified=False), decision_policy="balanced"
    )
    unaligned = synthesize_final_report(
        audit_with_unresolved_anchored_risk(relation=False), decision_policy="balanced"
    )
    assert unverified.prediction == "unknown"
    assert unaligned.prediction == "unknown"


def test_balanced_ignores_invalid_supplementary_anchor_when_core_anchors_are_verified():
    audit = audit_with_unresolved_anchored_risk()
    risk = next(item for item in audit.claims.values() if item.kind == ClaimKind.RISK)
    candidate = risk.analysis["candidates"][0]
    candidate["source_anchors"].append({
        "file": "parser.c", "line": None, "excerpt": "ambiguous repeated line", "validated": False,
    })
    candidate["evidence_status"] = "unverified"
    candidate["unverified_anchor_count"] = 1
    report = synthesize_final_report(audit, decision_policy="balanced")
    assert report.prediction == "vulnerable"
    assert any("supplementary" in item for item in report.decision_basis)


def test_balanced_does_not_promote_partial_protection_relation_without_property_mismatch():
    audit = audit_with_unresolved_anchored_risk()
    dispute_id, dispute = next(iter(audit.disputes.items()))
    relation = dispute.claim_relations[0].model_copy(update={
        "relation": "limits", "property_relation": "unknown", "path_coverage": "partial",
    })
    audit.disputes[dispute_id] = dispute.model_copy(update={"claim_relations": [relation]})
    report = synthesize_final_report(audit, decision_policy="balanced")
    assert report.prediction == "unknown"


def test_balanced_keeps_mismatch_when_later_relation_is_unrelated():
    audit = audit_with_unresolved_anchored_risk()
    dispute_id, dispute = next(iter(audit.disputes.items()))
    first = dispute.claim_relations[0]
    later = first.model_copy(update={
        "protection_ids": ["protection-header"],
        "relation": "unresolved", "property_relation": "not_applicable",
        "path_coverage": "none",
    })
    audit.disputes[dispute_id] = dispute.model_copy(update={
        "claim_relations": [first, later],
    })
    report = synthesize_final_report(audit, decision_policy="balanced")
    assert report.prediction == "vulnerable"


def test_balanced_does_not_promote_unproven_buffer_extent_from_mismatch_alone():
    audit = audit_with_unresolved_anchored_risk()
    risk = next(item for item in audit.claims.values() if item.kind == ClaimKind.RISK)
    candidate = risk.analysis["candidates"][0]
    candidate.update({
        "security_property": "out_of_bounds_read",
        "risk_operation": "read p[i]",
        "address_source": "caller supplied pixels",
        "access_type": "byte",
        "alignment_requirement": None,
    })
    report = synthesize_final_report(audit, decision_policy="balanced")
    assert report.prediction == "unknown"


def test_balanced_does_not_promote_supported_semantic_conversion_to_vulnerability():
    audit = audit_with_unresolved_anchored_risk()
    risk = next(item for item in audit.claims.values() if item.kind == ClaimKind.RISK)
    candidate = risk.analysis["candidates"][0]
    candidate.update({
        "candidate_class": "conditional_interprocedural",
        "security_property": "type_safety",
    })
    evidence = Evidence(
        id="e-conversion", status=EvidenceStatus.CANDIDATE,
        summary="conversion is wider than the target", scope=audit.input.scope,
        query_id="q-conversion", tool=Backend.SOURCE,
    )
    audit.evidence[evidence.id] = evidence
    audit.fact_investigations["q-conversion"] = FactInvestigationResult(
        query_id="q-conversion", investigation_status="complete",
        risk_assessments=[RiskCandidateAssessment(
            candidate_id="risk-alignment", status="supported",
            rationale="semantic conversion is present", evidence_refs=[evidence.id],
        )],
    )
    report = synthesize_final_report(audit, decision_policy="balanced")
    assert report.prediction == "unknown"


def test_balanced_does_not_promote_partial_conditional_candidate_support():
    """One supported premise cannot close a candidate with another unknown prerequisite."""
    audit = audit_with_unresolved_anchored_risk()
    risk = next(item for item in audit.claims.values() if item.kind == ClaimKind.RISK)
    candidate = risk.analysis["candidates"][0]
    candidate["candidate_class"] = "conditional_interprocedural"
    candidate["security_property"] = "type_safety"
    candidate["required_conditions"] = [
        {"id": "p-supported", "description": "type width", "status": "unknown"},
        {"id": "p-unresolved", "description": "caller range", "status": "unknown"},
    ]
    evidence = Evidence(
        id="e-partial", status=EvidenceStatus.CANDIDATE,
        summary="only the type width was found", scope=audit.input.scope,
        query_id="q-partial", tool=Backend.SOURCE,
    )
    audit.evidence[evidence.id] = evidence
    audit.fact_investigations["q-partial"] = FactInvestigationResult(
        query_id="q-partial", investigation_status="partial",
        risk_assessments=[RiskCandidateAssessment(
            candidate_id="risk-alignment", status="supported",
            rationale="one premise is supported", evidence_refs=[evidence.id],
        )],
        premise_updates=[{
            "premise_id": "p-supported", "status": "supported",
            "reason": "type definition", "evidence_refs": [evidence.id],
        }],
    )
    report = synthesize_final_report(audit, decision_policy="balanced")
    assert report.prediction == "unknown"


def test_unrelated_effective_protection_does_not_make_sample_benign():
    audit = audit_with_unresolved_anchored_risk()
    evidence = Evidence(
        id="e-protection", status=EvidenceStatus.CANDIDATE,
        summary="q is non-null", scope=audit.input.scope,
        query_id="q-protection", tool=Backend.SOURCE,
    )
    audit.evidence[evidence.id] = evidence
    audit.fact_investigations["q-protection"] = FactInvestigationResult(
        query_id="q-protection", investigation_status="complete",
        protection_assessments=[ProtectionAssessment(
            protection_id="protection-range", coverage_status="effective",
            rationale="A different object is protected.", evidence_refs=[evidence.id],
        )],
    )
    report = synthesize_final_report(audit, decision_policy="balanced")
    assert report.prediction != "benign"


def test_effective_property_matched_protection_blocks_mismatch_promotion():
    audit = audit_with_unresolved_anchored_risk()
    dispute_id, dispute = next(iter(audit.disputes.items()))
    matched = dispute.claim_relations[0].model_copy(update={
        "protection_ids": ["protection-range"],
        "relation": "limits", "object_relation": "same",
        "property_relation": "matches", "path_coverage": "complete",
    })
    audit.disputes[dispute_id] = dispute.model_copy(update={
        "claim_relations": [dispute.claim_relations[0], matched],
    })
    evidence = Evidence(
        id="e-protection", status=EvidenceStatus.CANDIDATE,
        summary="same property and path protected", scope=audit.input.scope,
        query_id="q-protection", tool=Backend.SOURCE,
    )
    audit.evidence[evidence.id] = evidence
    audit.fact_investigations["q-protection"] = FactInvestigationResult(
        query_id="q-protection", investigation_status="complete",
        protection_assessments=[ProtectionAssessment(
            protection_id="protection-range", coverage_status="effective",
            rationale="The same typed access is covered on the complete path.",
            evidence_refs=[evidence.id],
        )],
    )
    report = synthesize_final_report(audit, decision_policy="balanced")
    assert report.prediction != "vulnerable"


def test_recall_first_can_use_complete_verified_candidate_without_diagnosis_relation():
    report = synthesize_final_report(
        audit_with_unresolved_anchored_risk(relation=False), decision_policy="recall_first"
    )
    assert report.prediction == "vulnerable"
    assert report.decision_confidence == "low"
    assert report.decision_evidence_grade == "C"


def test_neither_policy_calls_sample_benign_when_only_one_of_multiple_candidates_is_refuted():
    audit = audit_with_unresolved_anchored_risk(relation=False)
    risk = next(item for item in audit.claims.values() if item.kind == ClaimKind.RISK)
    risk.analysis["candidates"].append({"candidate_id": "risk-second"})
    evidence = Evidence(
        id="e-refute", status=EvidenceStatus.CANDIDATE, summary="one candidate checked",
        scope=risk.scope, query_id="q1", tool=Backend.SOURCE,
    )
    audit.evidence[evidence.id] = evidence
    audit.fact_investigations["q1"] = FactInvestigationResult(
        query_id="q1", investigation_status="partial",
        risk_assessments=[RiskCandidateAssessment(
            candidate_id="risk-alignment", status="refuted", rationale="refuted by evidence",
            evidence_refs=[evidence.id],
        )],
    )
    for policy in ("strict", "balanced", "recall_first"):
        report = synthesize_final_report(audit, decision_policy=policy)
        assert report.prediction != "benign"


def test_balanced_closes_investigated_unestablished_conditional_candidate_as_benign():
    audit = audit_with_unresolved_anchored_risk(relation=False)
    risk = next(item for item in audit.claims.values() if item.kind == ClaimKind.RISK)
    risk.analysis["candidates"][0]["candidate_class"] = "conditional_interprocedural"
    evidence = Evidence(
        id="e-investigated", status=EvidenceStatus.CANDIDATE,
        summary="caller contract was investigated but trigger was not established",
        scope=risk.scope, query_id="q1", tool=Backend.SOURCE,
    )
    audit.evidence[evidence.id] = evidence
    audit.fact_investigations["q1"] = FactInvestigationResult(
        query_id="q1", investigation_status="partial",
        risk_assessments=[RiskCandidateAssessment(
            candidate_id="risk-alignment", status="unknown",
            rationale="Necessary external condition was not established.",
            evidence_refs=[evidence.id],
        )],
    )
    report = synthesize_final_report(audit, decision_policy="balanced")
    assert report.prediction == "benign"
    assert "candidate closure" in report.explanation
