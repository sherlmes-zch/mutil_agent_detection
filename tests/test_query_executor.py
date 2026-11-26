from mutil_agent_detection.protocols import (
    AnalysisInput, Backend, ProgramScope, QueryKind, QueryOutcome, QuerySpec, SourceSnapshot,
)
from mutil_agent_detection.query_executor import QueryExecutor
import subprocess
from types import SimpleNamespace


def matching_repository(root, samples=None):
    class FakeRepository:
        def prepare_matching_snapshot(self, sample):
            if samples is not None:
                samples.append(sample)
            commit = sample.commit_id
            return SimpleNamespace(
                root=root, declared_commit=commit, matched_commit=commit,
                status="declared_commit", sample_function_hash="sample-hash",
                repository_function_hash="sample-hash", checked_commits=(commit,), note=None,
            )

    return FakeRepository()


def request():
    return AnalysisInput(
        sample_id="executor-sample", function_name="f",
        source="int f(void) {\n  int value = 1;\n  return value;\n}\n",
        scope=ProgramScope(snapshot=SourceSnapshot(version="test"), entrypoint="f"),
        metadata={"file_name": "sample.c"},
    )


def query(backend, question):
    return QuerySpec(
        target_claim_ids=["risk"], target_premise_ids=["p1"], scope=request().scope,
        kind=QueryKind.SOURCE_SEARCH, backend=backend, question=question,
        outcomes=[QueryOutcome(label="yes", meaning="supports", supports_premise_ids=["p1"]),
                  QueryOutcome(label="no", meaning="negates", negates_premise_ids=["p1"])],
        evidence_requirement="exact source anchor",
    )


def test_source_executor_returns_anchored_candidate_evidence():
    evidence = QueryExecutor()(query(Backend.SOURCE, "Find `value = 1`"), request())

    assert evidence.status == "candidate"
    assert evidence.tool == "source"
    assert evidence.source_anchors[0].file == "sample.c"
    assert evidence.source_anchors[0].line == 2
    assert evidence.supports_claim_ids == []


def test_unconfigured_smt_is_explicitly_inconclusive():
    class MustNotPrepareRepository:
        def prepare(self, _sample):
            raise AssertionError("unavailable backends must be rejected before repository preparation")

    request_value = request().model_copy(update={"metadata": {
        "file_name": "sample.c", "project": "demo", "project_url": "https://github.com/demo/demo",
    }})
    smt_query = query(Backend.SMT, "Can constraints be solved?").model_copy(update={"required_context": "repository"})
    evidence = QueryExecutor(repository_manager=MustNotPrepareRepository())(smt_query, request_value)

    assert evidence.status == "inconclusive"
    assert "not configured" in evidence.summary
    assert "repository context was not prepared" in evidence.summary


def test_joern_executor_maps_bounded_skeleton_clues():
    class FakeJoern:
        def analyze(self, _request):
            return {
                "status": "ready", "entrypoint": "f",
                "structural_operations": [{"code": "*(double *) p", "line": 8, "file": "sample.c", "evidence_kind": "cast"}],
                "candidate_dataflows": [], "calls": [], "controls": [], "source_candidates": [],
            }

    evidence = QueryExecutor(joern=FakeJoern())(query(Backend.JOERN, "Check `double` pointer cast alignment"), request())

    assert evidence.status == "candidate"
    assert evidence.source_anchors[0].line == 8
    assert evidence.tool == "joern"


def test_repository_context_is_prepared_lazily_and_source_searches_exact_commit(tmp_path):
    repo_file = tmp_path / "sample.c"
    repo_file.write_text("int before(void) { return 0; }\n\nint f(void) {\n  return 7;\n}\n", encoding="utf-8")
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "sample.c"], check=True, capture_output=True)

    samples = []
    manager = matching_repository(tmp_path, samples)
    request_value = request().model_copy(update={
        "scope": ProgramScope(snapshot=SourceSnapshot(version="abc", commit="a" * 40), entrypoint="f"),
        "metadata": {
            "file_name": "sample.c", "project": "demo", "project_url": "https://github.com/demo/demo",
            "source_line": 1, "idx": 3,
        },
    })
    repository_query = query(Backend.SOURCE, "Find `return 7` ").model_copy(update={"required_context": "repository"})

    evidence = QueryExecutor(repository_manager=manager)(repository_query, request_value)

    assert evidence.status == "candidate"
    assert evidence.source_anchors[0].line == 4
    assert samples[0].commit_id == "a" * 40
    assert evidence.scope.snapshot.root == str(tmp_path)
    assert "Exact-commit repository-wide search" in evidence.analysis_boundary


def test_repository_required_query_without_manager_is_inconclusive():
    repository_query = query(Backend.SOURCE, "Find callers").model_copy(update={"required_context": "repository"})

    evidence = QueryExecutor()(repository_query, request())

    assert evidence.status == "inconclusive"
    assert "repository context is required" in evidence.summary


def test_repository_source_query_does_not_fall_back_to_function_fragment(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    manager = matching_repository(tmp_path)

    request_value = request().model_copy(update={
        "scope": ProgramScope(snapshot=SourceSnapshot(version="a" * 40, commit="a" * 40), entrypoint="f"),
        "metadata": {
            "file_name": "missing.c", "project": "demo", "project_url": "https://github.com/demo/demo",
        },
    })
    repository_query = query(Backend.SOURCE, "Find callers").model_copy(update={"required_context": "repository"})

    evidence = QueryExecutor(repository_manager=manager)(repository_query, request_value)

    assert evidence.status == "inconclusive"
    assert "no exact code symbol" in evidence.summary
    assert "repository-wide search" in evidence.analysis_boundary


def test_repository_source_query_searches_tracked_files_across_repository(tmp_path):
    caller = tmp_path / "src" / "caller.c"
    caller.parent.mkdir()
    caller.write_text("int caller(void) {\n  return target_func();\n}\n", encoding="utf-8")
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "src/caller.c"], check=True, capture_output=True)

    manager = matching_repository(tmp_path)

    request_value = request().model_copy(update={
        "scope": ProgramScope(snapshot=SourceSnapshot(version="a" * 40, commit="a" * 40), entrypoint="f"),
        "metadata": {
            "file_name": "sample.c", "project": "demo", "project_url": "https://github.com/demo/demo",
        },
    })
    repository_query = query(Backend.SOURCE, "Find callers of `target_func`").model_copy(
        update={"required_context": "repository"}
    )

    evidence = QueryExecutor(repository_manager=manager)(repository_query, request_value)

    assert evidence.status == "candidate"
    assert evidence.source_anchors[0].file == "src/caller.c"


def test_repository_caller_query_includes_enclosing_function_slice(tmp_path):
    caller = tmp_path / "src" / "caller.c"
    caller.parent.mkdir()
    caller.write_text(
        "int caller(void) {\n"
        "  int n = 16;\n"
        "  return f(n);\n"
        "}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "src/caller.c"], check=True, capture_output=True)

    manager = matching_repository(tmp_path)
    request_value = request().model_copy(update={
        "scope": ProgramScope(snapshot=SourceSnapshot(version="a" * 40, commit="a" * 40), entrypoint="f"),
        "metadata": {
            "file_name": "sample.c", "project": "demo", "project_url": "https://github.com/demo/demo",
        },
    })
    repository_query = query(Backend.SOURCE, "Find caller allocation and call site of `f`").model_copy(
        update={"required_context": "repository"}
    )

    evidence = QueryExecutor(repository_manager=manager)(repository_query, request_value)

    assert evidence.status == "candidate"
    assert "caller_enclosing_function" in (evidence.raw_result or "")
    assert "int caller(void)" in (evidence.raw_result or "")
    assert evidence.structured_facts["result_kind"] == "caller_contract"
    assert evidence.structured_facts["caller_contracts"][0]["caller_function"] == "caller"
    assert evidence.structured_facts["caller_contracts"][0]["argument_expression"] == "n"
    assert evidence.structured_facts["caller_contracts"][0]["call_site_location"]["line"] == 3


def test_function_context_query_does_not_search_other_functions_in_same_repository_file(tmp_path):
    target = tmp_path / "sample.c"
    target.write_text(
        "int before(void) {\n"
        "  int value = 99;\n"
        "  return value;\n"
        "}\n"
        "\n"
        "int f(void) {\n"
        "  int value = 1;\n"
        "  return value;\n"
        "}\n"
        "\n"
        "int after(void) {\n"
        "  int value = 99;\n"
        "  return value;\n"
        "}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "sample.c"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "fixture"],
        check=True, capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    manager = matching_repository(tmp_path)
    function_source = "int f(void) {\n  int value = 1;\n  return value;\n}"
    request_value = request().model_copy(update={
        "source": function_source,
        "scope": ProgramScope(
            snapshot=SourceSnapshot(version=commit, commit=commit), entrypoint="f",
            locations=[{"file": "sample.c", "function": "f", "line": 6, "end_line": 9}],
        ),
        "metadata": {
            "file_name": "sample.c", "project": "demo", "project_url": "https://github.com/demo/demo",
            "repository_root": str(tmp_path), "repository_version_status": "declared_commit",
            "source_line_start": 6, "source_line_end": 9,
        },
    })
    function_query = query(Backend.SOURCE, "Find `value = 1`")

    evidence = QueryExecutor(repository_manager=manager)(function_query, request_value)

    assert evidence.status == "candidate"
    assert [(anchor.line, anchor.function) for anchor in evidence.source_anchors] == [(7, "f")]
    assert "repository lines 6-9" in evidence.analysis_boundary


def test_repository_search_prefers_exact_symbols_over_prose_word_matches(tmp_path):
    target = tmp_path / "coders" / "property.c"
    target.parent.mkdir()
    target.write_text(
        "void GetEXIFProperty(void) {\n"
        "  p1 = p;\n"
        "  value = ReadPropertyUnsignedShort(p1);\n"
        "  p1 += tag_bytes[format];\n"
        "  if (number_bytes <= 4) p = q+8;\n"
        "}\n",
        encoding="utf-8",
    )
    helper = tmp_path / "magick" / "property-reader.c"
    helper.parent.mkdir()
    helper.write_text(
        "unsigned short ReadPropertyUnsignedShort(const unsigned char *p) {\n"
        "  return *p;\n"
        "}\n",
        encoding="utf-8",
    )
    distractor_dir = tmp_path / "Magick++"
    distractor_dir.mkdir()
    for index in range(35):
        (distractor_dir / f"notes-{index:02}.c").write_text(
            "void note(void) { size_t number_bytes; number_bytes = 0; }\n",
            encoding="utf-8",
        )
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)

    manager = matching_repository(tmp_path)

    request_value = request().model_copy(update={
        "function_name": "GetEXIFProperty",
        "scope": ProgramScope(snapshot=SourceSnapshot(version="a" * 40, commit="a" * 40), entrypoint="GetEXIFProperty"),
        "metadata": {
            "file_name": "property.c", "project": "demo",
            "project_url": "https://github.com/demo/demo",
        },
    })
    repository_query = query(
        Backend.SOURCE,
        "In the supplied GetEXIFProperty source, do paths call ReadPropertyUnsignedShort and advance p1 per component? "
        "For inline USHORT/components=2, do number_bytes<=4 and p=q+8 hold while tag_bytes[format] is checked?",
    ).model_copy(update={"required_context": "repository"})

    evidence = QueryExecutor(repository_manager=manager)(repository_query, request_value)

    assert evidence.status == "candidate"
    assert evidence.source_anchors[0].file == "coders/property.c"
    helper_positions = [
        index for index, anchor in enumerate(evidence.source_anchors)
        if anchor.file == "magick/property-reader.c"
    ]
    assert helper_positions and helper_positions[0] < 10
    assert "ReadPropertyUnsignedShort" in evidence.raw_result


def test_repository_search_ignores_queries_with_only_prose_terms():
    symbols = QueryExecutor._repository_search_symbols("Find callers and allocation details", None)

    assert symbols == []


def test_repository_context_block_returns_complete_function():
    lines = [
        "static int helper(int value)", "{", "  int converted=value+1;",
        "  return converted;", "}", "", "int caller(void) { return helper(1); }",
    ]
    start, end, kind = QueryExecutor._enclosing_source_block(lines, 3)
    assert kind == "enclosing_function"
    assert start <= 1
    assert end == 5


def test_repository_context_block_recovers_multiline_definition_from_parameter_match():
    lines = [
        "static inline void SetPixel(Image *image,", "  const size_t channels,",
        "  PixelPacket *q)", "{", "  q->red=0;", "}",
    ]
    start, end, kind = QueryExecutor._enclosing_source_block(lines, 2)
    assert kind == "enclosing_function"
    assert start == 1
    assert end == 6


def test_repository_definition_query_prioritizes_requested_helper_body(tmp_path):
    source = tmp_path / "coders" / "psd.c"
    source.parent.mkdir()
    source.write_text(
        "static inline void SetPSDPixel(Image *image,\n"
        "  PixelPacket *q, ssize_t x)\n"
        "{\n"
        "  q[x].red=0;\n"
        "}\n",
        encoding="utf-8",
    )
    # Add many unrelated definitions containing generic type words.  They
    # must not crowd the requested helper out of the bounded context blocks.
    for index in range(12):
        path = tmp_path / f"helper-{index:02}.c"
        path.write_text(
            f"static inline Quantum helper_{index}(Quantum value) {{ return value; }}\n",
            encoding="utf-8",
        )
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    manager = matching_repository(tmp_path)
    request_value = request().model_copy(update={
        "function_name": "ReadPSDChannelPixels",
        "metadata": {
            "file_name": "coders/psd.c", "project": "demo",
            "project_url": "https://github.com/demo/demo",
        },
    })
    repository_query = query(
        Backend.SOURCE,
        "Find the complete definition and implementation body of `SetPSDPixel`, including q and x semantics.",
    ).model_copy(update={"required_context": "repository"})
    evidence = QueryExecutor(repository_manager=manager)(repository_query, request_value)
    blocks = evidence.structured_facts["definition_blocks"]
    assert blocks
    assert "SetPSDPixel" in blocks[0]
    assert "q[x].red=0" in blocks[0]
    assert evidence.structured_facts["definition_completeness"] == "complete_function_or_macro_block"


def test_repository_version_mismatch_falls_back_to_authoritative_function_fragment():
    class MismatchedRepository:
        def prepare_matching_snapshot(self, _sample):
            return SimpleNamespace(
                root=None, declared_commit="a" * 40, matched_commit=None,
                status="repository_version_mismatch", sample_function_hash="sample-hash",
                repository_function_hash=None, checked_commits=("a" * 40,),
                note="normalized function hash was not found",
            )

    request_value = request().model_copy(update={
        "scope": ProgramScope(snapshot=SourceSnapshot(version="a" * 40, commit="a" * 40), entrypoint="f"),
        "metadata": {
            "file_name": "sample.c", "project": "demo",
            "project_url": "https://github.com/demo/demo", "idx": 3,
        },
    })
    repository_query = query(Backend.SOURCE, "Find `value = 1`").model_copy(
        update={"required_context": "repository"}
    )

    evidence = QueryExecutor(repository_manager=MismatchedRepository())(repository_query, request_value)

    assert evidence.status == "candidate"
    assert "Function-fragment fallback" in evidence.summary
    assert "Repository snapshot mismatch" in evidence.analysis_boundary
    assert evidence.source_anchors[0].line == 2
