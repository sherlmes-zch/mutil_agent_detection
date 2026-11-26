import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from mutil_agent_detection.base_agent import agent_io_trace
from mutil_agent_detection.disagreement_diagnostician import DisagreementDiagnostician
from mutil_agent_detection.protocols import (
    AnalysisInput,
    Claim,
    ClaimKind,
    ClaimStatus,
    DiagnosisInput,
    Dispute,
    ProgramScope,
    SourceSnapshot,
)
from mutil_agent_detection.workflow import WorkflowHandlers, build_audit_graph


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response

    def with_structured_output(self, _schema, method="function_calling", include_raw=False):
        def invoke(_messages):
            return {
                "raw": AIMessage(content=json.dumps(self.response)),
                "parsed": self.response,
                "parsing_error": None,
            }

        return RunnableLambda(invoke)


class FallbackStructuredModel:
    """Simulate an OpenAI-compatible endpoint with broken function calling."""

    def __init__(self, response, *, prose=False):
        self.response = response
        self.prose = prose

    def with_structured_output(self, _schema, method="function_calling", include_raw=False):
        def invoke(_messages):
            if method == "function_calling":
                return {"raw": AIMessage(content=""), "parsed": None, "parsing_error": None}
            content = json.dumps(self.response)
            if self.prose:
                content = "Here is the diagnosis:\n" + content + "\nEnd."
            return {"raw": AIMessage(content=content), "parsed": None, "parsing_error": None}

        return RunnableLambda(invoke)


def diagnosis_input():
    scope = ProgramScope(
        snapshot=SourceSnapshot(version="test"), entrypoint="inspect"
    )
    return DiagnosisInput(
        sample_id="diagnosis-test",
        scope=scope,
        risk_claim_id="risk-claim",
        risk_claim_status=ClaimStatus.CANDIDATE,
        risk_claim_text="A pointer may be dereferenced after release.",
        risk_candidates=[
            {
                "candidate_id": "risk-uaf",
                "summary": "Potential use after free",
                "risk_operation": "read *ptr",
                "target_object": "ptr",
            }
        ],
        protection_claim_id="protection-claim",
        protection_claim_status=ClaimStatus.CANDIDATE,
        protection_claim_text="An extra reference may preserve the object.",
        protections=[
            {
                "protection_id": "refcount-protection",
                "description": "An extra reference may keep the object alive.",
                "protected_object": "ptr target",
                "coverage": "unknown",
            }
        ],
    )


def test_diagnostician_maps_relations_premises_and_targets():
    response = {
        "diagnosis_status": "partial",
        "alignment_status": "partially_aligned",
        "summary": "The protection concerns the same object but its path coverage is unknown.",
        "claim_relations": [
            {
                "risk_candidate_id": "risk-uaf",
                "protection_ids": ["refcount-protection"],
                "relation": "limits",
                "object_relation": "same",
                "property_relation": "matches",
                "path_coverage": "unknown",
                "explanation": "The reference-count claim may mitigate the same lifetime risk, but coverage is unverified.",
            }
        ],
        "premises": [
            {
                "id": "p-refcount",
                "description": "The extra reference is acquired on the path reaching the later read.",
                "relation": "unresolved",
                "status": "unknown",
                "claim_refs": ["risk-uaf", "refcount-protection"],
                "decisive": True,
            }
        ],
        "investigation_targets": [
            {
                "premise_id": "p-refcount",
                "question": "Does the relevant path acquire an independent reference before release?",
                "discriminating_outcomes": ["reference acquired", "no reference acquired"],
                "suggested_query_kinds": ["source_search", "path_reachability"],
                "evidence_requirement": "Locate ref acquisition and prove it dominates the release/read path.",
            }
        ],
    }
    diagnostician = DisagreementDiagnostician(FakeStructuredModel(response))

    dispute = diagnostician(diagnosis_input())

    assert dispute.alignment_status == "partially_aligned"
    assert dispute.claim_relations[0].path_coverage == "unknown"
    assert dispute.unresolved_premises == ["p-refcount"]
    assert dispute.investigation_targets[0].premise_id == "p-refcount"
    assert diagnostician.last_interaction["status"] == "parsed"


def test_diagnostician_rejects_unknown_references_and_falls_back_conservatively():
    response = {
        "diagnosis_status": "complete",
        "alignment_status": "aligned",
        "summary": "Claims align.",
        "claim_relations": [
            {
                "risk_candidate_id": "invented-risk",
                "protection_ids": [],
                "relation": "supports",
                "explanation": "Unsupported reference.",
            }
        ],
    }
    diagnostician = DisagreementDiagnostician(FakeStructuredModel(response))

    dispute = diagnostician(diagnosis_input())

    assert dispute.diagnosis_status == "insufficient_evidence"
    assert dispute.alignment_status == "unresolved"
    assert dispute.aligned is False
    assert "unknown risk candidate" in dispute.diagnosis_error
    assert dispute.unresolved_premises == ["diagnosis-unavailable"]
    assert diagnostician.last_interaction["status"] == "failed"


def test_diagnostician_retries_json_mode_when_function_calling_is_empty():
    response = {
        "diagnosis_status": "partial",
        "alignment_status": "unresolved",
        "summary": "The available evidence is insufficient to align the claims.",
        "facts": [],
        "claim_relations": [],
        "premises": [],
        "investigation_targets": [],
    }
    diagnostician = DisagreementDiagnostician(FallbackStructuredModel(response))

    dispute = diagnostician(diagnosis_input())

    assert dispute.diagnosis_status == "partial"
    assert diagnostician.last_interaction["status"] == "parsed"
    assert diagnostician.last_interaction["attempts"][0].startswith("function_calling:")
    assert diagnostician.last_interaction["attempts"][1] == "json_mode"


def test_diagnostician_extracts_json_wrapped_in_provider_prose():
    response = {
        "diagnosis_status": "partial",
        "alignment_status": "unresolved",
        "summary": "No reliable relation is established.",
        "facts": [],
        "claim_relations": [],
        "premises": [],
        "investigation_targets": [],
    }
    diagnostician = DisagreementDiagnostician(
        FallbackStructuredModel(response, prose=True)
    )

    dispute = diagnostician(diagnosis_input())

    assert dispute.alignment_status == "unresolved"
    assert diagnostician.last_interaction["status"] == "parsed"


def test_diagnostician_fills_missing_claim_refs_and_query_kinds():
    response = {
        "diagnosis_status": "partial",
        "alignment_status": "partially_aligned",
        "summary": "Alignment remains uncertain.",
        "facts": [],
        "claim_relations": [],
        "premises": [
            {
                "id": "p-alignment",
                "description": "The pointer alignment for the typed dereference is unknown.",
                "status": "unknown",
                "decisive": True,
            }
        ],
        "investigation_targets": [
            {
                "premise_id": "p-alignment",
                "question": "Is the cast target aligned on every reachable path?",
                "discriminating_outcomes": ["aligned", "misaligned"],
                "suggested_query_kinds": [],
                "evidence_requirement": "Type cast and address provenance",
            }
        ],
    }
    diagnostician = DisagreementDiagnostician(FakeStructuredModel(response))

    dispute = diagnostician(diagnosis_input())

    assert dispute.premises[0].claim_refs == ["risk-claim", "protection-claim"]
    assert dispute.investigation_targets[0].suggested_query_kinds == [
        "source_search", "path_reachability", "cpg"
    ]
    parsed = diagnostician.last_interaction["parsed_output"]
    assert parsed["premises"][0]["claim_refs"] == ["risk-claim", "protection-claim"]
    assert parsed["investigation_targets"][0]["suggested_query_kinds"] == [
        "source_search", "path_reachability", "cpg"
    ]


def test_diagnostician_drops_unknown_optional_claim_refs_instead_of_falling_back():
    response = {
        "diagnosis_status": "partial",
        "alignment_status": "partially_aligned",
        "summary": "The range check does not establish alignment.",
        "facts": [],
        "claim_relations": [{
            "risk_candidate_id": "risk-uaf",
            "protection_ids": ["made-up-protection", "refcount-protection"],
            "relation": "does_not_address",
            "object_relation": "same", "property_relation": "mismatches",
            "path_coverage": "none",
            "explanation": "The supplied protection is about lifetime, not alignment.",
            "evidence_refs": ["made-up-ref"],
        }],
        "premises": [{
            "id": "p-alignment", "description": "Whether the address is aligned remains unresolved.",
            "relation": "unresolved", "status": "unknown",
            "claim_refs": ["risk-uaf", "invented-claim"],
            "evidence_refs": ["invented-evidence"], "decisive": True,
        }],
        "investigation_targets": [{
            "premise_id": "p-alignment", "question": "Can the address be misaligned?",
            "discriminating_outcomes": ["aligned", "misaligned"],
            "suggested_query_kinds": ["source_search"], "evidence_requirement": "address expression",
        }],
    }
    diagnostician = DisagreementDiagnostician(FakeStructuredModel(response))

    dispute = diagnostician(diagnosis_input())

    assert diagnostician.last_interaction["status"] == "parsed"
    assert dispute.claim_relations[0].protection_ids == ["refcount-protection"]
    assert dispute.claim_relations[0].evidence_refs == []
    assert dispute.premises[0].claim_refs == ["risk-uaf"]
    assert dispute.premises[0].evidence_refs == []
    assert diagnostician.last_interaction["reference_repairs"]


def test_diagnostician_recovers_transient_claim_candidate_id():
    response = {
        "diagnosis_status": "partial",
        "alignment_status": "partially_aligned",
        "summary": "The transient candidate ID maps to the sole supplied risk candidate.",
        "claim_relations": [{
            "risk_candidate_id": "claim_transient_12345678901234567890",
            "protection_ids": ["refcount-protection"],
            "relation": "limits", "object_relation": "same",
            "property_relation": "matches", "path_coverage": "partial",
            "explanation": "Coverage remains partial.",
        }],
        "premises": [], "investigation_targets": [],
    }
    diagnostician = DisagreementDiagnostician(FakeStructuredModel(response))

    dispute = diagnostician(diagnosis_input())

    assert dispute.claim_relations[0].risk_candidate_id == "risk-uaf"
    assert any("mapped transient" in item for item in diagnostician.last_interaction["reference_repairs"])


def test_diagnostician_expands_summary_only_response_into_queryable_ledger():
    response = {
        "diagnosis_status": "partial",
        "alignment_status": "partially_aligned",
        "summary": "Bounds checks do not establish alignment.",
        "facts": [], "claim_relations": [], "premises": [], "investigation_targets": [],
    }
    diagnostician = DisagreementDiagnostician(FakeStructuredModel(response))

    dispute = diagnostician(diagnosis_input())

    assert dispute.claim_relations
    assert dispute.unresolved_premises
    assert dispute.investigation_targets
    assert set(dispute.unresolved_premises) <= {
        target.premise_id for target in dispute.investigation_targets
    }


def test_diagnostician_repairs_empty_protection_ids_in_existing_relations():
    response = {
        "diagnosis_status": "partial",
        "alignment_status": "partially_aligned",
        "summary": "Protection coverage is uncertain.",
        "facts": [],
        "claim_relations": [{
            "risk_candidate_id": "risk-uaf", "protection_ids": [],
            "relation": "limits", "explanation": "The protection may limit lifetime risk.",
        }],
        "premises": [{
            "id": "p1", "description": "Protection covers the relevant path.",
            "status": "unknown", "relation": "unresolved", "decisive": True,
        }],
        "investigation_targets": [{
            "premise_id": "p1", "question": "Does protection cover the path?",
            "discriminating_outcomes": ["yes", "no"],
            "suggested_query_kinds": ["source_search"], "evidence_requirement": "source",
        }],
    }
    diagnostician = DisagreementDiagnostician(FakeStructuredModel(response))

    dispute = diagnostician(diagnosis_input())

    assert dispute.claim_relations[0].protection_ids == ["refcount-protection"]
    assert diagnostician.last_interaction["parsed_output"]["claim_relations"][0]["protection_ids"] == [
        "refcount-protection"
    ]


def test_graph_passes_bounded_input_to_diagnostician_and_records_claim_ids():
    request = AnalysisInput(
        sample_id="graph-diagnosis",
        function_name="inspect",
        source="void inspect(void) {}",
        scope=ProgramScope(
            snapshot=SourceSnapshot(version="test"), entrypoint="inspect"
        ),
    )
    received = {}

    def make_claim(kind, actor):
        def invoke(req, _skeleton):
            return Claim(
                kind=kind,
                text=f"{actor} observation",
                status=ClaimStatus.CANDIDATE,
                scope=req.scope,
                created_by=actor,
                analysis={"summary": "bounded analysis"},
            )

        return invoke

    def diagnose(value):
        received["input"] = value
        return Dispute(
            scope=value.scope,
            risk_claim_id=value.risk_claim_id,
            protection_claim_id=value.protection_claim_id,
            diagnosis_status="partial",
            alignment_status="unresolved",
            summary="Need evidence.",
            premises=[
                {
                    "id": "p-unknown",
                    "description": "Object identity is unknown.",
                    "status": "unknown",
                    "decisive": True,
                }
            ],
            unresolved_premises=["p-unknown"],
        )

    result = build_audit_graph(
        WorkflowHandlers(
            risk=make_claim(ClaimKind.RISK, "risk"),
            protection=make_claim(ClaimKind.PROTECTION, "protection"),
            diagnose=diagnose,
        )
    ).invoke({"input": request})

    assert isinstance(received["input"], DiagnosisInput)
    assert received["input"].sample_id == request.sample_id
    assert result["diagnosis_input"] == received["input"]
    assert result["audit_state"].unresolved_premises == ["p-unknown"]
    event = next(event for event in result["audit_state"].events if event.actor == "disagreement_diagnoser")
    assert result["dispute"].id in event.related_ids
    assert result["dispute"].risk_claim_id in event.related_ids
    assert result["dispute"].protection_claim_id in event.related_ids


def test_default_diagnosis_does_not_claim_alignment():
    result = build_audit_graph().invoke(
        {
            "input": AnalysisInput(
                sample_id="default-diagnosis",
                function_name="noop",
                source="void noop(void) {}",
                scope=ProgramScope(snapshot=SourceSnapshot(version="test"), entrypoint="noop"),
            )
        }
    )

    assert result["dispute"].alignment_status == "unresolved"
    assert result["dispute"].aligned is False
    assert result["audit_state"].unresolved_premises == ["diagnosis-unavailable"]


def test_agent_trace_includes_diagnostician_input_and_parsed_output():
    trace = agent_io_trace(
        [
            {
                "agent": "disagreement_diagnostician",
                "status": "parsed",
                "input": {
                    "instruction": "Align claims",
                    "response_schema": "DisputeDraft",
                    "output_contract": "Structured relation only",
                    "diagnosis_input": {"sample_id": "s1", "risk_claim_text": "risk"},
                },
                "parsed_output": {"alignment_status": "unresolved"},
            }
        ]
    )

    assert trace[0]["agent"] == "disagreement_diagnostician"
    assert trace[0]["input"]["diagnosis_input"]["sample_id"] == "s1"
    assert trace[0]["output"] == {"alignment_status": "unresolved"}
