"""Factories that connect provider configuration to workflow handlers."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from .agents import make_first_pass_handlers
from .config import BailianConfig, JoernConfig, OpenAICompatibleConfig
from .disagreement_diagnostician import DisagreementDiagnostician
from .joern import JoernAnalyzer
from .query_planner import QueryPlanner
from .query_executor import QueryExecutor
from .fact_investigator import FactInvestigator
from .repository import RepositoryManager
from .workflow import WorkflowHandlers


def _effective_thinking(model: str, requested: bool) -> bool:
    """Apply provider-required thinking settings for known Bailian model families."""
    normalized = model.strip().lower()
    # qwen3.8-2.4t-a95b currently rejects enable_thinking=False.
    if normalized.startswith("qwen3.8-2.4t-a95b"):
        return True
    return requested


def create_bailian_chat_model(config: BailianConfig | None = None) -> ChatOpenAI:
    """Create a LangChain OpenAI-compatible client for Aliyun Bailian."""

    config = config or BailianConfig.from_env()
    options = {
        "api_key": config.api_key,
        "base_url": config.base_url,
        "model": config.model,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout_seconds,
        "max_retries": config.max_retries,
        "extra_body": {"enable_thinking": _effective_thinking(config.model, config.enable_thinking)},
    }
    # Bailian's Kimi K3 family rejects the temperature parameter entirely.
    # Match the family prefix so variants such as kimi-k3.1-pro are covered.
    if not config.model.lower().startswith("kimi-k3"):
        options["temperature"] = config.temperature
    return ChatOpenAI(**options)


def create_openai_compatible_chat_model(
    config: OpenAICompatibleConfig | None = None,
) -> ChatOpenAI:
    """Create a client for gateways exposing /v1/chat/completions."""

    config = config or OpenAICompatibleConfig.from_env()
    return ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def create_live_handlers(
    *,
    model: Any | None = None,
    bailian: BailianConfig | None = None,
    openai_compatible: OpenAICompatibleConfig | None = None,
    joern: JoernConfig | None = None,
    repository_manager: RepositoryManager | None = None,
) -> WorkflowHandlers:
    """Wire real Joern and first-pass model agents into the existing graph."""

    model = model or create_bailian_chat_model(bailian)
    risk_agent, protection_agent = make_first_pass_handlers(model)
    diagnostician = DisagreementDiagnostician(model)
    query_planner = QueryPlanner(model)
    joern_analyzer = JoernAnalyzer(joern)
    query_executor = QueryExecutor(joern=joern_analyzer, repository_manager=repository_manager)
    fact_investigator = FactInvestigator(model)
    return WorkflowHandlers(
        skeleton=joern_analyzer,
        risk=risk_agent,
        protection=protection_agent,
        diagnose=diagnostician,
        query_planner=query_planner,
        execute_query=query_executor,
        investigate_evidence=fact_investigator,
    )


__all__ = [
    "create_bailian_chat_model", "create_openai_compatible_chat_model", "create_live_handlers",
]
