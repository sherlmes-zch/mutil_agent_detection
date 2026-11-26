import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from mutil_agent_detection.protocols import (
    Claim, ClaimKind, Dispute, DisputePremise, ProgramScope, QueryPlanningInput,
    SourceSnapshot, QuerySpec, QueryOutcome,
)
from mutil_agent_detection.query_planner import QueryPlanner


class FakeModel:
    def __init__(self, response):
        self.response = response

    def with_structured_output(self, _schema, method="function_calling", include_raw=False):
        return RunnableLambda(lambda _value: {
            "parsed": self.response,
            "raw": AIMessage(content=json.dumps(self.response)),
        })


def planning_input():
    scope = ProgramScope(snapshot=SourceSnapshot(version="test"), entrypoint="f")
    risk = Claim(id="risk", kind=ClaimKind.RISK, text="risk", scope=scope, created_by="risk")
    protection = Claim(id="protection", kind=ClaimKind.PROTECTION, text="protection", scope=scope, created_by="protection")
    dispute = Dispute(
        scope=scope, risk_claim_id="risk", protection_claim_id="protection",
        premises=[DisputePremise(id="p1", description="object identity", decisive=True)],
        unresolved_premises=["p1"],
    )
    return QueryPlanningInput(sample_id="s", risk_claim=risk, protection_claim=protection,
                              dispute=dispute, scope=scope)


def test_planner_rejects_invalid_candidate_and_uses_valid_premise_template():
    response = {"summary": "proposals", "candidates": [
        {"candidate_id": "bad", "target_claim_ids": ["risk"], "target_premise_ids": ["missing"],
         "kind": "source_search", "backend": "source", "question": "bad", "outcomes": [
             {"label": "a", "meaning": "a", "leaves_unknown_premise_ids": ["missing"]},
             {"label": "b", "meaning": "b", "leaves_unknown_premise_ids": ["missing"]}],
         "evidence_requirement": "source"}
    ]}
    planner = QueryPlanner(FakeModel(response))
    queries = planner(planning_input())
    assert len(queries) == 1
    assert queries[0].proposed_by == "premise_template"
    assert planner.last_interaction["status"] == "parsed"
    assert planner.last_interaction["planned_candidates"] == [queries[0].id]
    assert planner.last_interaction["fallback"] == "premise_templates"


def test_planner_validates_outcomes_and_ranks_valid_query():
    response = {"summary": "proposals", "candidates": [
        {"candidate_id": "q1", "target_claim_ids": ["risk"], "target_premise_ids": ["p1"],
         "kind": "source_search", "backend": "source", "question": "Is object identity preserved?", "outcomes": [
             {"label": "same", "meaning": "same object", "supports_premise_ids": ["p1"]},
             {"label": "different", "meaning": "different object", "negates_premise_ids": ["p1"]}],
         "evidence_requirement": "exact source location"}
    ]}
    planner = QueryPlanner(FakeModel(response))
    queries = planner(planning_input())
    assert [query.id for query in queries] == ["q1"]
    assert queries[0].separation == 1
    assert queries[0].target_premise_ids == ["p1"]


def test_planner_uses_premise_template_when_model_returns_summary_only():
    planner = QueryPlanner(FakeModel({"summary": "Investigate alignment.", "candidates": []}))

    queries = planner(planning_input())

    assert len(queries) == 1
    assert queries[0].target_premise_ids == ["p1"]
    assert queries[0].proposed_by == "premise_template"
    assert planner.last_interaction["fallback"] == "premise_templates"


def test_planner_normalizes_compatible_endpoint_aliases_and_long_meanings():
    response = {
        "summary": "alignment query",
        "candidates": [{
            "rank": 1,
            "id": "q-alignment",
            "target_claims": ["risk"],
            "target_premises": ["p1"],
            "query": "Does the helper preserve alignment?",
            "kind": "cpg",
            "backend": "joern",
            "outcomes": {
                "support": {"premises": ["p1"], "meaning": "x" * 1000},
                "negate": {"premises": ["p1"], "meaning": "not established"},
                "unknown": {"premises": ["p1"], "meaning": "unknown"},
            },
            "evidence_required": "helper implementation",
        }],
    }
    planner = QueryPlanner(FakeModel(response))

    queries = planner(planning_input())

    assert [query.id for query in queries] == ["q-alignment"]
    assert queries[0].required_context == "repository"
    assert max(len(outcome.meaning) for outcome in queries[0].outcomes) <= 300


class BrokenModel(FakeModel):
    def with_structured_output(self, _schema, method="function_calling", include_raw=False):
        return RunnableLambda(lambda _value: {"parsed": None, "raw": AIMessage(content="not json")})


def test_planner_uses_premise_templates_when_both_provider_formats_fail():
    planner = QueryPlanner(BrokenModel({}))

    queries = planner(planning_input())

    assert queries
    assert queries[0].proposed_by == "premise_template"
    assert planner.last_interaction["status"] == "fallback"


def test_planner_replaces_unavailable_smt_candidates_with_executable_template():
    response = {"summary": "use SMT", "candidates": [{
        "candidate_id": "q-smt", "target_claim_ids": ["risk"], "target_premise_ids": ["p1"],
        "kind": "numeric_range", "backend": "smt", "question": "Can arithmetic overflow?",
        "outcomes": [
            {"label": "yes", "meaning": "overflow", "supports_premise_ids": ["p1"]},
            {"label": "no", "meaning": "bounded", "negates_premise_ids": ["p1"]},
        ], "evidence_requirement": "exact range proof",
    }]}
    planning = planning_input().model_copy(update={"enabled_backends": ["source", "joern"]})
    planner = QueryPlanner(FakeModel(response))

    queries = planner(planning)

    assert queries
    assert all(query.backend in {"source", "joern"} for query in queries)
    assert queries[0].proposed_by == "premise_template"
    assert planner.last_interaction["fallback"] == "premise_templates"


def test_template_for_reader_implementation_uses_repository_joern():
    planning = planning_input().model_copy(update={
        "dispute": Dispute(
            scope=planning_input().scope,
            risk_claim_id="risk", protection_claim_id="protection",
            premises=[DisputePremise(
                id="p1", description="Inspect ReadPropertySignedLong reader implementation and bounds checks.",
                decisive=True,
            )],
            unresolved_premises=["p1"], decisive_premises=["p1"],
        ),
        "enabled_backends": ["source", "joern"],
    })

    queries = QueryPlanner._template_queries(planning)

    assert queries[0].backend == "joern"
    assert queries[0].required_context == "repository"


def test_template_for_component_span_uses_repository_context():
    base = planning_input()
    planning = base.model_copy(update={
        "dispute": Dispute(
            scope=base.scope,
            risk_claim_id="risk", protection_claim_id="protection",
            premises=[DisputePremise(
                id="p1",
                description=(
                    "A macro-formatted EXIF profile entry can have components>1 and the final read "
                    "can exceed the validated span or inline record."
                ),
                decisive=True,
            )],
            unresolved_premises=["p1"],
            decisive_premises=["p1"],
        ),
        "enabled_backends": ["source", "joern"],
    })

    queries = QueryPlanner._template_queries(planning)

    assert len(queries) == 1
    assert queries[0].backend == "joern"
    assert queries[0].required_context == "repository"


def test_planner_does_not_repeat_same_premise_backend_and_kind():
    planning = planning_input()
    prior = QuerySpec(
        id="prior", target_claim_ids=["risk"], target_premise_ids=["p1"],
        scope=planning.scope, kind="source_search", backend="source",
        question="First wording for object identity", outcomes=[
            QueryOutcome(label="yes", meaning="yes", supports_premise_ids=["p1"]),
            QueryOutcome(label="no", meaning="no", negates_premise_ids=["p1"]),
        ], evidence_requirement="source",
    )
    planning = planning.model_copy(update={"prior_queries": [prior]})
    response = {"summary": "rephrase", "candidates": [{
        "candidate_id": "again", "target_claim_ids": ["risk"],
        "target_premise_ids": ["p1"], "kind": "source_search", "backend": "source",
        "question": "Different wording for the same object identity", "outcomes": [
            {"label": "yes", "meaning": "yes", "supports_premise_ids": ["p1"]},
            {"label": "no", "meaning": "no", "negates_premise_ids": ["p1"]},
        ], "evidence_requirement": "same source",
    }]}
    queries = QueryPlanner(FakeModel(response))(planning)
    assert all(query.id != "again" for query in queries)


def test_planner_routes_caller_queries_to_repository_source_search():
    response = {"summary": "caller", "candidates": [{
        "candidate_id": "q-call", "target_claim_ids": ["risk"], "target_premise_ids": ["p1"],
        "kind": "allocation_relation", "backend": "joern",
        "question": "At every caller call site, what allocation length is passed to f?",
        "outcomes": [
            {"label": "covered", "meaning": "covered", "supports_premise_ids": ["p1"]},
            {"label": "short", "meaning": "short", "negates_premise_ids": ["p1"]},
        ], "evidence_requirement": "caller allocation expression",
    }]}
    planning = planning_input().model_copy(update={"enabled_backends": ["source", "joern"]})

    queries = QueryPlanner(FakeModel(response))(planning)

    assert queries[0].backend == "source"
    assert queries[0].kind == "source_search"
    assert queries[0].required_context == "repository"
