from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
import json
import pytest

from mutil_agent_detection.agents import make_first_pass_handlers
from mutil_agent_detection.dataset import normalize_record
from mutil_agent_detection.protection_agent import ProtectionEvidenceAgent
from mutil_agent_detection.risk_agent import RiskPathAgent
from mutil_agent_detection.base_agent import StructuredAgentError
from mutil_agent_detection.base_agent import agent_io_trace, compact_interaction


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response

    def with_structured_output(self, _schema, method="function_calling", include_raw=False):
        def invoke(_messages):
            if not include_raw:
                return self.response
            raw_content = json.dumps(self.response) if self.response is not None else ""
            return {
                "raw": AIMessage(content=raw_content),
                "parsed": self.response,
                "parsing_error": None,
            }

        return RunnableLambda(invoke)


class SequenceStructuredModel:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def with_structured_output(self, _schema, method="function_calling", include_raw=False):
        def invoke(_messages):
            response = self.responses[min(self.calls, len(self.responses) - 1)]
            self.calls += 1
            if not include_raw:
                return response
            return {
                "raw": AIMessage(content=json.dumps(response)),
                "parsed": response,
                "parsing_error": None,
            }

        return RunnableLambda(invoke)


def sample():
    return normalize_record(
        {
            "idx": 1,
            "target": 1,
            "project": "demo",
            "commit_id": "a" * 40,
            "func": "int copy(char *dst, const char *src) { strcpy(dst, src); return 0; }",
            "file_name": "demo.c",
        },
        dataset_name="primevul_test",
        source_line=1,
    ).to_analysis_input()


def test_risk_agent_preserves_structured_path_and_premises():
    anchor = {
        "file": "demo.c",
        "line": 1,
        "excerpt": "strcpy(dst, src);",
        "basis": "function_source",
    }
    model = FakeStructuredModel(
        {
            "candidate_found": True,
            "summary": "The copy may overflow dst if src exceeds its capacity.",
            "source_anchors": [anchor],
            "confidence": 0.8,
            "security_property": "buffer_overflow",
            "risk_operation": "strcpy(dst, src)",
            "target_object": "dst",
            "risk_path": [
                {
                    "order": 1,
                    "operation": "strcpy(dst, src)",
                    "object": "dst",
                    "effect": "Copies source bytes without a destination capacity argument.",
                    "source_anchors": [anchor],
                }
            ],
            "required_conditions": json.dumps(
                [
                    {
                        "id": "risk-premise-1",
                        "description": "src length exceeds dst capacity",
                        "status": "unknown",
                        "evidence_refs": [anchor],
                    }
                ]
            ),
        }
    )
    risk, _ = make_first_pass_handlers(model)

    claim = risk(sample(), {"calls": [{"name": "strcpy", "line": 1}]})

    assert claim.kind == "risk"
    assert claim.analysis["security_property"] == "buffer_overflow"
    assert claim.analysis["risk_path"][0]["order"] == 1
    assert claim.premise_ids == ["risk-premise-1"]
    assert claim.source_anchors[0].line == 1
    assert claim.analysis["anchor_validation"]["valid_count"] == 3
    assert "required_conditions" in claim.analysis["_interaction"]["attempts"][0]["repairs"]


def test_agents_have_explicit_role_specific_output_contracts():
    model = FakeStructuredModel({"text": "unknown"})
    risk = RiskPathAgent(model)
    protection = ProtectionEvidenceAgent(model)

    assert '"candidates"' in risk.output_contract
    assert '"priority"' in risk.output_contract
    assert "at most 3" in risk.instruction
    assert '"risk_path"' in risk.output_contract
    assert '"required_conditions"' in risk.output_contract
    assert '"alignment_requirement"' in risk.output_contract
    assert '"path_condition"' not in risk.output_contract
    assert '"protections"' in protection.output_contract
    assert '"bypass_paths"' in protection.output_contract
    assert RiskPathAgent.schema.__name__ == "RiskAnalysisDraft"
    assert ProtectionEvidenceAgent.schema.__name__ == "ProtectionAnalysisDraft"


def test_query_executor_interaction_is_preserved_in_agent_trace():
    interaction = {
        "agent": "query_executor", "status": "executed",
        "input": {"query": {"id": "q1", "question": "Find callers"}},
        "output": {"evidence_id": "e1", "status": "candidate", "raw_result": "facts"},
    }
    trace = agent_io_trace([interaction])
    assert trace[0]["agent"] == "query_executor"
    assert trace[0]["input"]["query"]["id"] == "q1"
    assert trace[0]["output"]["evidence_id"] == "e1"


def test_risk_agent_preserves_ranked_candidates_and_projects_primary_claim():
    valid_anchor = {
        "file": "demo.c", "line": 1, "excerpt": "strcpy(dst, src);", "basis": "function_source"
    }
    invalid_anchor = {
        "file": "demo.c", "line": 99, "excerpt": "memcpy(dst, src, n);", "basis": "function_source"
    }

    def candidate(candidate_id, priority, operation, target, anchor, premise_id, status="unknown"):
        return {
            "candidate_id": candidate_id,
            "priority": priority,
            "summary": f"Candidate {priority}: {operation} may violate a safety condition.",
            "security_property": "buffer_overflow" if priority == 1 else "out_of_bounds_read",
            "risk_operation": operation,
            "target_object": target,
            "risk_path": [{
                "order": 1,
                "operation": operation,
                "object": target,
                "effect": "Access may exceed the valid object extent.",
                "source_anchors": [anchor],
            }],
            "required_conditions": [{
                "id": premise_id,
                "description": "The computed length exceeds the object extent.",
                "status": status,
                "evidence_refs": [anchor],
            }],
            "unresolved_questions": [],
            "source_anchors": [anchor],
            "confidence": 0.7 - (priority - 1) * 0.2,
        }

    model = FakeStructuredModel({
        "candidate_found": True,
        "summary": "Two distinct, ranked memory-safety hypotheses.",
        "candidates": [
            candidate("risk-1", 1, "strcpy(dst, src)", "dst", valid_anchor, "p1"),
            candidate("risk-2", 2, "memcpy(dst, src, n)", "dst", invalid_anchor, "p2", "supported"),
        ],
    })
    risk, _ = make_first_pass_handlers(model)

    claim = risk(sample(), {"calls": []})

    assert [item["candidate_id"] for item in claim.analysis["candidates"]] == ["risk-1", "risk-2"]
    assert claim.analysis["risk_operation"] == "strcpy(dst, src)"
    assert claim.analysis["target_object"] == "dst"
    assert claim.analysis["summary"] == "Two distinct, ranked memory-safety hypotheses."
    assert claim.premise_ids == ["p1", "p2"]
    assert claim.analysis["candidates"][1]["required_conditions"][0]["status"] == "unknown"
    assert claim.analysis["anchor_validation"]["valid_count"] == 3
    assert claim.analysis["anchor_validation"]["invalid_count"] == 3


def test_risk_agent_injects_missing_narrow_conversion_candidate():
    """A generated conversion index must not disappear from the risk protocol."""
    request = normalize_record(
        {
            "idx": 9,
            "target": 1,
            "project": "demo",
            "commit_id": "a" * 40,
            "func": (
                "unsigned char decode(Quantum pixel) {\n"
                "  return ((unsigned char) pixel) & 7;\n"
                "}\n"
            ),
            "file_name": "demo.c",
        },
        dataset_name="primevul_test",
        source_line=1,
    ).to_analysis_input()
    anchor = {
        "file": "demo.c", "line": 2, "excerpt": "return ((unsigned char) pixel) & 7;",
        "basis": "function_source",
    }
    model = FakeStructuredModel({
        "candidate_found": True,
        "summary": "Only a buffer hypothesis was returned.",
        "candidates": [{
            "candidate_id": "risk-buffer",
            "candidate_class": "speculative",
            "priority": 1,
            "summary": "A separate buffer concern.",
            "risk_operation": "pixel",
            "target_object": "pixel",
            "risk_path": [{
                "order": 1, "operation": "pixel", "object": "pixel", "effect": "read",
                "source_anchors": [anchor],
            }],
            "required_conditions": [{
                "id": "buffer-premise", "description": "buffer contract is insufficient",
                "status": "unknown", "evidence_refs": [],
            }],
            "source_anchors": [anchor],
            "confidence": 0.2,
        }],
    })

    claim = RiskPathAgent(model)(request, {"key_operation_index": request.metadata["key_operation_index"]})

    conversion = next(
        item for item in claim.analysis["candidates"]
        if item["security_property"] == "integer_conversion"
    )
    assert conversion["candidate_class"] == "conditional_interprocedural"
    assert conversion["conversion_target_type"] == "unsigned char"
    assert conversion["source_anchors"][0]["line"] == 2
    assert conversion["required_conditions"][0]["status"] == "unknown"


def test_agent_prompt_and_anchor_validation_use_declared_repository_line_numbers():
    request = sample()
    location = request.scope.locations[0].model_copy(update={"line": 50, "end_line": 50})
    request = request.model_copy(
        update={
            "scope": request.scope.model_copy(update={"locations": [location]}),
            "metadata": {
                **request.metadata,
                "source_line_basis": "repository_absolute",
                "source_line_start": 50,
                "key_operation_index": [{"line": 50, "categories": ["dereference"], "code": "strcpy(dst, src);"}],
            },
        }
    )
    anchor = {
        "file": "demo.c", "line": 999, "excerpt": "strcpy(dst, src);", "basis": "function_source"
    }
    model = FakeStructuredModel({
        "candidate_found": True,
        "summary": "Potential overflow.",
        "security_property": "buffer_overflow",
        "risk_operation": "strcpy(dst, src)",
        "target_object": "dst",
        "source_anchors": [anchor],
        "risk_path": [{
            "order": 1,
            "operation": "strcpy(dst, src)",
            "object": "dst",
            "effect": "Unbounded copy.",
            "source_anchors": [anchor],
        }],
        "required_conditions": [{"id": "p1", "description": "Input exceeds destination.", "evidence_refs": []}],
    })
    risk, _ = make_first_pass_handlers(model)

    claim = risk(request, {"status": "ready"})
    prompt_input = claim.analysis["_interaction"]["input"]

    assert prompt_input["source_line_basis"] == "repository_absolute"
    assert prompt_input["source"].startswith("    50 | ")
    assert json.loads(prompt_input["key_operation_index"])[0]["line"] == 50
    assert claim.source_anchors[0].line == 50
    assert claim.analysis["source_anchors"][0]["line"] == 50
    assert claim.analysis["anchor_validation"]["valid_count"] == 2


def test_joern_anchor_line_is_resolved_from_exact_skeleton_excerpt():
    risk = RiskPathAgent(FakeStructuredModel({}))
    details = {
        "source_anchors": [{
            "excerpt": "*(double *)raw",
            "basis": "joern_skeleton",
        }]
    }
    report = risk._validate_anchors(
        details,
        sample(),
        {
            "structural_operations": [{
                "file": "demo.c",
                "line": 18,
                "code": "*(double *)raw",
                "category": "dereference",
            }]
        },
    )

    assert report["valid_count"] == 1
    assert details["source_anchors"][0]["line"] == 18
    assert details["source_anchors"][0]["file"] == "demo.c"


def test_compact_interaction_keeps_only_agent_io_for_workflow_trace():
    compact = compact_interaction(
        {
            "agent": "risk_analyst",
            "kind": "risk",
            "status": "parsed",
            "input": {
                "instruction": "Find one concrete risk.",
                "output_contract": "large schema contract",
                "sample_id": "sample-1",
                "function_name": "copy",
                "file_name": "copy.c",
                "line_range": "1-3",
                "commit_id": "abc",
                "context_mode": "function_only",
                "source": "void copy() {}",
                "key_operation_index": "[{\"line\": 2}]",
                "skeleton": '{"calls": []}',
                "scope": '{"duplicate": true}',
            },
            "parsed_output": {"summary": "possible overflow"},
            "validation": {"verbose": "omitted"},
            "attempts": [{"raw": {"large": "payload"}}],
            "anchor_repair": {
                "input": {"task": "Correct invalid excerpts", "feedback": {"invalid_anchors": []}},
                "output": {"summary": "corrected"},
                "status": "parsed",
                "messages": [{"content": "large duplicated prompt"}],
            },
        }
    )

    assert compact["input"]["task"] == "Find one concrete risk."
    assert compact["input"]["response_schema"] == "RiskAnalysisDraft"
    assert compact["input"]["joern_skeleton"] == {"calls": []}
    assert compact["input"]["key_operation_index"] == [{"line": 2}]
    assert compact["output"] == {"summary": "possible overflow"}
    assert "validation" not in compact
    assert "attempts" not in compact
    assert compact["anchor_repair"]["status"] == "parsed"
    assert "messages" not in compact["anchor_repair"]


def test_agent_io_trace_records_order_and_only_agent_io_fields():
    trace = agent_io_trace(
        [
            {
                "agent": "risk_analyst",
                "kind": "risk",
                "status": "parsed",
                "input": {"instruction": "Find risks", "source": "int f() {}", "skeleton": "{}"},
                "parsed_output": {"summary": "possible issue"},
                "validation": {"verbose": True},
            },
            {
                "agent": "protection_analyst",
                "kind": "protection",
                "status": "failed",
                "input": {"instruction": "Find protections"},
                "parsed_output": None,
                "error": "invalid response",
                "fallback": "unknown_claim",
            },
        ]
    )

    assert [item["order"] for item in trace] == [1, 2]
    assert [item["agent"] for item in trace] == ["risk_analyst", "protection_analyst"]
    assert trace[0]["output"] == {"summary": "possible issue"}
    assert set(trace[0]) == {"order", "agent", "input", "output", "status"}
    assert trace[1]["output"] is None
    assert trace[1]["error"] == "invalid response"
    assert trace[1]["fallback"] == "unknown_claim"


def test_none_model_response_raises_error_with_captured_interaction():
    risk, _ = make_first_pass_handlers(FakeStructuredModel(None))

    with pytest.raises(StructuredAgentError) as caught:
        risk(sample(), {"status": "ready"})

    assert caught.value.interaction["status"] == "failed"
    assert caught.value.interaction["raw_output"]
    assert caught.value.interaction["messages"]
    assert len(caught.value.interaction["attempts"]) == 2
    assert caught.value.interaction["attempts"][-1]["method"] == "json_mode"


def test_agent_recovers_json_content_when_function_call_parser_returns_none():
    response = {
        "candidate_found": True,
        "summary": "Potential overflow if src exceeds dst capacity.",
        "security_property": "buffer_overflow",
        "risk_operation": "strcpy(dst, src)",
        "target_object": "dst",
        "risk_path": [
            {
                "order": 1,
                "operation": "strcpy(dst, src)",
                "object": "dst",
                "effect": "Copies bytes without a destination length.",
                "source_anchors": [{"file": "demo.c", "line": 1, "excerpt": "strcpy(dst, src);", "basis": "function_source"}],
            }
        ],
        "required_conditions": [
            {"id": "p1", "description": "src exceeds dst capacity", "status": "unknown", "evidence_refs": []}
        ],
    }

    class JsonTextModel:
        def with_structured_output(self, _schema, method="function_calling", include_raw=False):
            def invoke(_messages):
                return {
                    "raw": AIMessage(
                        content=json.dumps(response),
                        response_metadata={"finish_reason": "stop"},
                    ),
                    "parsed": None if method == "function_calling" else response,
                    "parsing_error": None,
                }

            return RunnableLambda(invoke)

    risk, _ = make_first_pass_handlers(JsonTextModel())

    claim = risk(sample(), {"status": "ready"})

    assert claim.analysis["security_property"] == "buffer_overflow"
    assert claim.analysis["_interaction"]["attempts"][0]["recovered_by"] == "raw_json_content"
    assert len(claim.analysis["_interaction"]["attempts"]) == 1


def test_length_truncated_response_is_reported_without_redundant_json_retry():
    class TruncatedModel:
        def with_structured_output(self, _schema, method="function_calling", include_raw=False):
            def invoke(_messages):
                return {
                    "raw": AIMessage(content="", response_metadata={"finish_reason": "length"}),
                    "parsed": None,
                    "parsing_error": None,
                }

            return RunnableLambda(invoke)

    risk, _ = make_first_pass_handlers(TruncatedModel())

    with pytest.raises(StructuredAgentError, match="completion token limit") as caught:
        risk(sample(), {"status": "ready"})

    assert len(caught.value.interaction["attempts"]) == 1
    assert caught.value.interaction["attempts"][0]["metadata"]["response_metadata"]["finish_reason"] == "length"


def test_protection_agent_preserves_coverage_and_bypass_paths():
    anchor = {
        "file": "demo.c",
        "line": 1,
        "excerpt": "strcpy(dst, src);",
        "basis": "function_source",
    }
    model = FakeStructuredModel(
        {
            "summary": "The check only protects the guarded path.",
            "source_anchors": [anchor],
            "confidence": 0.55,
            "protections": [
                {
                    "id": "protection-1",
                    "kind": "bounds_check",
                    "description": "Destination capacity is checked before copying.",
                    "protected_object": "dst",
                    "path_condition": "The copy is reached through the checked branch.",
                    "coverage": "complete",
                    "blocks_conditions": ["src length is within dst capacity"],
                    "bypass_paths": [],
                    "source_anchors": [anchor],
                }
            ],
            "invariants": [
                {
                    "id": "protection-premise-1",
                    "description": "dst has enough capacity on the guarded path",
                    "status": "supported",
                    "evidence_refs": [],
                }
            ],
        }
    )
    _, protection = make_first_pass_handlers(model)

    claim = protection(sample(), {"controls": [{"type": "if", "line": 1}]})

    assert claim.kind == "protection"
    assert claim.analysis["protections"][0]["coverage"] == "partial"
    assert "downgraded" in claim.analysis["protections"][0]["validation_note"]
    assert "protection-premise-1" in claim.premise_ids
    assert claim.analysis["invariants"][0]["status"] == "unknown"
    assert claim.analysis["invariants"][0]["validation_note"]
    assert claim.analysis["_interaction"]["validation"]["protection_coverage"]["protection-1"] == "partial"


def test_agent_rejects_or_downgrades_unsupported_source_anchors():
    model = FakeStructuredModel(
        {
            "candidate_found": True,
            "summary": "A possible overflow depends on the input length.",
            "security_property": "buffer_overflow",
            "risk_operation": "strcpy(dst, src)",
            "target_object": "dst",
            "risk_path": [
                {
                    "order": 1,
                    "operation": "strcpy(dst, src)",
                    "object": "dst",
                    "effect": "May exceed the destination capacity.",
                    "source_anchors": [
                        {"file": "demo.c", "line": 999, "excerpt": "not_present(dst, src);", "basis": "function_source"}
                    ],
                }
            ],
            "required_conditions": [
                {
                    "id": "risk-premise-1",
                    "description": "src exceeds dst capacity",
                    "status": "supported",
                    "evidence_refs": [
                        {"file": "demo.c", "line": 999, "excerpt": "not_present(dst, src);", "basis": "function_source"}
                    ],
                }
            ],
        }
    )
    risk, _ = make_first_pass_handlers(model)

    claim = risk(sample(), {"status": "ready"})

    assert claim.source_anchors == []
    assert claim.analysis["required_conditions"][0]["status"] == "unknown"
    assert claim.analysis["anchor_validation"]["invalid_count"] == 2


@pytest.mark.parametrize(
    ("repair_excerpt", "expected_status", "expected_confidence"),
    [
        ("strcpy(dst, src);", "anchors_verified", 0.8),
        ("still_not_in_source();", "unverified", None),
    ],
)
def test_invalid_anchor_triggers_one_targeted_repair_then_gates_evidence(
    repair_excerpt, expected_status, expected_confidence
):
    def response(excerpt):
        anchor = {"excerpt": excerpt, "basis": "function_source"}
        return {
            "candidate_found": True,
            "summary": "Potential overflow if input exceeds destination capacity.",
            "security_property": "buffer_overflow",
            "risk_operation": "strcpy(dst, src)",
            "target_object": "dst",
            "source_anchors": [anchor],
            "confidence": 0.8,
            "risk_path": [{
                "order": 1,
                "operation": "strcpy(dst, src)",
                "object": "dst",
                "effect": "Copies input without a capacity argument.",
                "source_anchors": [anchor],
            }],
            "required_conditions": [{
                "id": "p1",
                "description": "Input exceeds destination capacity.",
                "status": "supported",
                "evidence_refs": [anchor],
            }],
        }

    model = SequenceStructuredModel([response("invented_call();"), response(repair_excerpt)])
    risk, _ = make_first_pass_handlers(model)

    claim = risk(sample(), {"status": "ready"})

    assert model.calls == 2
    assert claim.analysis["_interaction"]["anchor_repair"]["status"] == "parsed"
    assert claim.analysis["_interaction"]["anchor_repair"]["input"]["feedback"]["invalid_anchors"]
    assert claim.analysis["evidence_status"] == expected_status
    assert claim.confidence == expected_confidence
    if expected_status == "anchors_verified":
        assert claim.analysis["source_anchors"][0]["line"] == 1
        assert claim.analysis["required_conditions"][0]["status"] == "supported"
    else:
        assert claim.analysis["source_anchors"][0]["line"] is None
        assert claim.analysis["required_conditions"][0]["status"] == "unknown"
