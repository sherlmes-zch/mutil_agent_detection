"""Runtime configuration for Aliyun Bailian and Joern integrations."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class BailianConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.8-27b"
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=8192, gt=0)
    enable_thinking: bool = False
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=2, ge=0)

    @classmethod
    def from_env(cls, **overrides: object) -> "BailianConfig":
        """Read the key from the environment; never place credentials in source."""

        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY") or ""
        return cls(api_key=api_key, **overrides)


class OpenAICompatibleConfig(BaseModel):
    """Configuration for generic OpenAI-compatible chat-completion gateways."""

    model_config = ConfigDict(extra="forbid")

    api_key: str = ""
    base_url: str = "https://ai.xem8k5.top/v1"
    model: str = "deepseek-v4.1-flash"
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=8192, gt=0)
    timeout_seconds: float = Field(default=180, gt=0)
    max_retries: int = Field(default=3, ge=0)

    @classmethod
    def from_env(cls, **overrides: object) -> "OpenAICompatibleConfig":
        api_key = os.getenv("XEM8K5_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        return cls(api_key=api_key, **overrides)


class JoernConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executable: Path = Path("/home/zch/joern/joern-cli/joern")
    parse_executable: Path | None = None
    version: str = "4.0.497"
    cache_dir: Path = Path(".cache/joern")
    language: str = "auto"
    timeout_seconds: float = Field(default=900, gt=0)
    max_calls: int = Field(default=80, ge=0)
    max_controls: int = Field(default=50, ge=0)
    max_parameters: int = Field(default=32, ge=0)
    max_dataflows: int = Field(default=40, ge=0)
    max_structural_operations: int = Field(default=120, ge=0)
    max_source_candidates: int = Field(default=40, ge=0)
    max_code_chars: int = Field(default=400, ge=32)
    repository_root_field: str = "repository_root"
    require_matching_commit: bool = True

    def resolved_parse_executable(self) -> Path:
        return self.parse_executable or self.executable.with_name("joern-parse")


class RepositoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cache_dir: Path = Path(".cache/primevul_repositories")
    source_cache_dir: Path | None = None
    git_executable: str = "git"
    timeout_seconds: float = Field(default=1800, gt=0)
    allowed_schemes: list[str] = Field(default_factory=lambda: ["https"])
    allowed_hosts: list[str] | None = Field(default_factory=lambda: ["github.com", "www.github.com"])
    lock_timeout_seconds: float = Field(default=300, gt=0)
    max_first_parent_commits: int = Field(default=50, ge=0, le=500)


__all__ = ["BailianConfig", "OpenAICompatibleConfig", "JoernConfig", "RepositoryConfig"]
