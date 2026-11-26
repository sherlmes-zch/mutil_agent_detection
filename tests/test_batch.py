from mutil_agent_detection.batch import run_jsonl, run_samples
from mutil_agent_detection.dataset import normalize_record


def sample(idx: int):
    return normalize_record(
        {
            "idx": idx,
            "project": "demo",
            "commit_id": "commit",
            "target": idx % 2,
            "func": "int copy_items(int n) { return n; }",
            "file_name": "demo.c",
            "func_hash": str(idx),
        },
        dataset_name="primevul_test",
        source_line=idx,
    )


def test_batch_runs_each_sample_in_isolated_graph():
    report = run_samples([sample(1), sample(2)])

    assert report.requested == 2
    assert report.completed == 2
    assert report.failed == 0
    assert len(report.results) == 2
    assert report.results[0].audit_state.sample_id != report.results[1].audit_state.sample_id
    assert all(result.report.prediction == "unknown" for result in report.results)


def test_batch_failure_isolated_from_other_samples():
    class BadSample:
        sample_id = "bad"
        idx = 99
        target = 0
        dataset_name = "primevul_test"

        def to_analysis_input(self, **_kwargs):
            raise RuntimeError("bad sample")

    report = run_samples([BadSample(), sample(2)])

    assert report.completed == 1
    assert report.failed == 1
    assert "RuntimeError" in report.results[0].error
    assert report.results[1].report is not None


def test_jsonl_entrypoint_uses_streaming_dataset_reader(tmp_path):
    path = tmp_path / "primevul_test.jsonl"
    path.write_text(
        '{"idx": 1, "target": 0, "func": "int f() { return 0; }", "project": "demo"}\n'
        '{"idx": 2, "target": 1, "func": "int g() { return 1; }", "project": "demo"}\n',
        encoding="utf-8",
    )
    report = run_jsonl(str(path), limit=1)
    assert report.requested == 1
    assert report.completed == 1


def test_repository_mode_requires_repository_manager():
    report = run_samples([sample(1)], context_mode="repository")
    assert report.failed == 1
    assert "repository mode requires" in report.results[0].error


def test_repository_mode_passes_verified_root_into_analysis_input(tmp_path):
    observed = {}

    class Manager:
        def prepare_matching_snapshot(self, sample):
            from types import SimpleNamespace
            return SimpleNamespace(
                root=tmp_path, declared_commit=sample.commit_id,
                matched_commit=sample.commit_id, status="declared_commit",
                note=None, sample_function_hash="hash", repository_function_hash="hash",
                checked_commits=(sample.commit_id,),
            )

    from mutil_agent_detection.workflow import WorkflowHandlers

    def skeleton(request):
        observed.update(request.metadata)
        return {"status": "test"}

    report = run_samples(
        [sample(1)],
        context_mode="repository",
        repository_manager=Manager(),
        handlers=WorkflowHandlers(skeleton=skeleton),
    )

    assert report.completed == 1
    assert observed["context_mode"] == "repository"
    assert observed["repository_root"] == str(tmp_path)


def test_failed_agent_interaction_is_included_in_sample_result():
    from mutil_agent_detection.base_agent import StructuredAgentError
    from mutil_agent_detection.workflow import WorkflowHandlers

    interaction = {
        "agent": "risk_analyst",
        "status": "failed",
        "input": {"sample_id": "sample"},
        "raw_output": None,
        "error": "structured model returned None",
    }

    def failing_agent(_request, _skeleton):
        raise StructuredAgentError("structured model returned None", interaction)

    result = run_samples(
        [sample(1)],
        handlers=WorkflowHandlers(risk=failing_agent),
    ).results[0]

    assert result.error is not None
    assert result.agent_interactions == [interaction]


def test_diagnostician_interaction_is_captured_after_independent_analysts():
    from mutil_agent_detection.protocols import Claim, ClaimKind, ClaimStatus, Dispute
    from mutil_agent_detection.workflow import WorkflowHandlers

    class Analyst:
        def __init__(self, kind, actor):
            self.kind = kind
            self.actor = actor
            self.last_interaction = None

        def __call__(self, request, _skeleton):
            self.last_interaction = {
                "agent": self.actor,
                "status": "parsed",
                "input": {"function_name": request.function_name},
                "parsed_output": {"summary": self.actor},
            }
            return Claim(
                kind=self.kind,
                text=self.actor,
                status=ClaimStatus.CANDIDATE,
                scope=request.scope,
                created_by=self.actor,
            )

    class Diagnostician:
        last_interaction = None

        def __call__(self, value):
            self.last_interaction = {
                "agent": "disagreement_diagnostician",
                "status": "parsed",
                "input": {
                    "instruction": "Align claims",
                    "response_schema": "DisputeDraft",
                    "diagnosis_input": value.model_dump(mode="json"),
                },
                "parsed_output": {"alignment_status": "unresolved"},
            }
            return Dispute(
                scope=value.scope,
                risk_claim_id=value.risk_claim_id,
                protection_claim_id=value.protection_claim_id,
                alignment_status="unresolved",
                summary="Unknown relation.",
            )

    result = run_samples(
        [sample(1)],
        handlers=WorkflowHandlers(
            risk=Analyst(ClaimKind.RISK, "risk_analyst"),
            protection=Analyst(ClaimKind.PROTECTION, "protection_analyst"),
            diagnose=Diagnostician(),
        ),
    ).results[0]

    assert result.error is None
    assert {item["agent"] for item in result.agent_interactions[:2]} == {
        "risk_analyst", "protection_analyst"
    }
    assert result.agent_interactions[2]["agent"] == "disagreement_diagnostician"
