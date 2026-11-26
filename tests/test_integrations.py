import base64

from mutil_agent_detection.config import BailianConfig, JoernConfig, OpenAICompatibleConfig
from mutil_agent_detection.integrations import create_bailian_chat_model, create_live_handlers, create_openai_compatible_chat_model
from mutil_agent_detection.joern import JoernAnalyzer
from mutil_agent_detection.protocols import AnalysisInput, ProgramScope, SourceSnapshot


def test_bailian_client_configuration_allows_empty_key():
    config = BailianConfig(api_key="")
    model = create_bailian_chat_model(config)

    assert model.model_name == "qwen3.8-27b"
    assert str(model.openai_api_base).rstrip("/") == config.base_url
    assert model.openai_api_key.get_secret_value() == ""
    assert model.extra_body == {"enable_thinking": False}


def test_xem8k5_openai_compatible_client_configuration():
    config = OpenAICompatibleConfig(api_key="", model="deepseek-v4.1-flash")
    model = create_openai_compatible_chat_model(config)

    assert model.model_name == "deepseek-v4.1-flash"
    assert str(model.openai_api_base).rstrip("/") == "https://ai.xem8k5.top/v1"
    assert model.temperature == 0.1
    assert model.extra_body is None


def test_kimi_k3_client_omits_unsupported_temperature():
    config = BailianConfig(api_key="", model="kimi-k3")
    model = create_bailian_chat_model(config)

    assert model.model_name == "kimi-k3"
    assert model.temperature is None


def test_kimi_k31_pro_client_omits_unsupported_temperature():
    config = BailianConfig(api_key="", model="kimi-k3.1-pro")
    model = create_bailian_chat_model(config)

    assert model.model_name == "kimi-k3.1-pro"
    assert model.temperature is None


def test_qwen_24t_a95b_forces_required_thinking_mode():
    config = BailianConfig(api_key="", model="qwen3.8-2.4t-a95b")
    model = create_bailian_chat_model(config)

    assert model.model_name == "qwen3.8-2.4t-a95b"
    assert model.extra_body == {"enable_thinking": True}


def test_live_handlers_connect_agent_callbacks_and_joern():
    model = create_bailian_chat_model(BailianConfig(api_key=""))
    handlers = create_live_handlers(model=model, joern=JoernConfig())

    assert handlers.risk is not None
    assert handlers.protection is not None
    assert handlers.diagnose is not None
    assert isinstance(handlers.skeleton, JoernAnalyzer)


def test_repository_context_never_silently_falls_back_to_function_only():
    request = AnalysisInput(
        sample_id="repo-mode",
        function_name="copy",
        source="int copy(void) { return 0; }",
        scope=ProgramScope(snapshot=SourceSnapshot(version="abc", commit="abc"), entrypoint="copy"),
        metadata={"context_mode": "repository"},
    )

    result = JoernAnalyzer().analyze(request)
    assert result["status"] == "repository_invalid"
    assert result["mode"] == "repository"


def test_repository_cpg_cache_key_uses_analyzer_version(tmp_path):
    request = AnalysisInput(
        sample_id="repo-cache-key",
        function_name="copy",
        source="int copy(void) { return 0; }",
        scope=ProgramScope(
            snapshot=SourceSnapshot(version="a" * 40, commit="a" * 40),
            entrypoint="copy",
        ),
    )

    first = JoernAnalyzer(JoernConfig(version="test-version-1"))
    second = JoernAnalyzer(JoernConfig(version="test-version-2"))

    first_key = first._repository_key(request, tmp_path, "c")
    second_key = second._repository_key(request, tmp_path, "c")

    assert len(first_key) == 64
    assert first_key != second_key


def test_joern_query_parser_preserves_structural_types_and_exact_source_locations():
    enc = lambda value: base64.b64encode(value.encode()).decode()
    output = "\n".join(
        [
            f"METHOD\t{enc('inspect')}\t{enc('inspect')}\t{enc('src/sample.c')}\t10",
            f"PARAM\t{enc('raw')}\t{enc('raw')}\t{enc('char *')}\t11\t{enc('src/sample.c')}",
            f"OP\t{enc('dereference')}\t{enc('<operator>.indirection')}\t{enc('*value')}\t{enc('double')}\t14\t{enc('src/sample.c')}",
            f"OP\t{enc('array_or_pointer_index')}\t{enc('<operator>.indexAccess')}\t{enc('items[i]')}\t{enc('double')}\t15\t{enc('src/sample.c')}",
            f"SOURCE\t{enc('recv')}\t{enc('recv(fd, raw, n, 0)')}\t{enc('long')}\t12\t{enc('src/sample.c')}",
            f"OPFLOW\tparameter\t{enc('dereference')}\t{enc('<operator>.indirection')}\t{enc('*value')}\t{enc('raw')}\t11\t14\t{enc('src/sample.c')}",
            f"TYPEOP\t{enc('dereference')}\t{enc('<operator>.indirection')}\t{enc('*(double *) raw')}\t{enc('raw')}\t{enc('char *')}\t{enc('double')}\t{enc('alignment-sensitive scalar access')}\t14\t{enc('src/sample.c')}",
            f"TYPEOP\t{enc('integer_conversion')}\t{enc('<operator>.cast')}\t{enc('(unsigned char) pixel')}\t{enc('pixel')}\t{enc('double')}\t{enc('unsigned char')}\t{enc('')}\t{enc('check source range and target width')}\t16\t{enc('src/sample.c')}",
        ]
    )

    parsed = JoernAnalyzer._parse_query_output(output)

    assert parsed["structural_operations"][0]["category"] == "dereference"
    assert parsed["structural_operations"][0]["result_type"] == "double"
    assert parsed["structural_operations"][0]["file"] == "src/sample.c"
    assert parsed["structural_operations"][0]["line"] == 14
    assert parsed["source_candidates"][0]["line"] == 12
    assert parsed["parameters"][0]["kind"] == "function_parameter_source"
    assert parsed["candidate_dataflows"][0]["source_line"] == 11
    assert parsed["candidate_dataflows"][0]["sink_line"] == 14
    typed = next(item for item in parsed["structural_operations"] if item["evidence_kind"].startswith("joern_type"))
    assert typed["operand_type"] == "char *"
    assert typed["result_type"] == "double"
    assert typed["alignment_requirement"] == "alignment-sensitive scalar access"
    conversion = next(
        item for item in parsed["structural_operations"]
        if item["evidence_kind"] == "joern_integer_conversion_candidate"
    )
    assert conversion["result_type"] == "unsigned char"
    assert "source range" in conversion["representation_requirement"]


def test_joern_bound_result_prioritizes_type_alignment_operations():
    analyzer = JoernAnalyzer(JoernConfig(max_structural_operations=2))
    result = {
        "methods": [], "calls": [], "controls": [], "parameters": [], "candidate_dataflows": [],
        "source_candidates": [],
        "structural_operations": [
            {"code": "ordinary()", "line": 1, "evidence_kind": "joern_candidate_structure"},
            {"code": "*(double *) p", "line": 20, "evidence_kind": "joern_type_and_alignment_candidate"},
            {"code": "*(float *) q", "line": 21, "evidence_kind": "joern_type_and_alignment_candidate"},
        ],
    }

    analyzer._bound_result(result)

    assert [item["line"] for item in result["structural_operations"]] == [20, 21]
