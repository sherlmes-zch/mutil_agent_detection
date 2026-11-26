"""Command-line entry point for a bounded PrimeVul model/Joern smoke run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .batch import run_jsonl
from .config import BailianConfig, JoernConfig, OpenAICompatibleConfig
from .integrations import create_bailian_chat_model, create_openai_compatible_chat_model, create_live_handlers
from .repository import RepositoryManager
from .config import RepositoryConfig
from .base_agent import agent_io_trace, compact_interaction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PrimeVul samples through Qwen and Joern")
    parser.add_argument("dataset", type=Path, help="PrimeVul JSONL path")
    parser.add_argument("--limit", type=int, default=2, help="Maximum samples (default: 2)")
    parser.add_argument(
        "--max-query-rounds", type=int, default=2,
        help="Maximum executed evidence-investigation rounds per sample (default: 2)",
    )
    parser.add_argument("--model", help="Model name (defaults per selected API platform)")
    parser.add_argument("--api-platform", choices=("bailian", "xem8k5"), default="bailian")
    parser.add_argument("--base-url", help="Override the selected API platform base URL")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum completion tokens per Agent")
    parser.add_argument(
        "--agent-retries", type=int, default=1,
        help="Bounded retry count for model-agent failures across all workflow agents (default: 1)",
    )
    parser.add_argument(
        "--quiet-progress", action="store_true",
        help="Do not print one-line progress information after each sample",
    )
    parser.add_argument(
        "--decision-policy", choices=("strict", "balanced", "recall_first"), default="balanced",
        help="Final evidence policy: balanced (default), strict, or recall_first",
    )
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen reasoning mode (uses completion budget)")
    parser.add_argument("--api-key-env", help="Override the selected platform's API-key environment variable")
    parser.add_argument("--joern", type=Path, default=Path("/home/zch/joern/joern-cli/joern"))
    parser.add_argument("--joern-cache", type=Path, default=Path(".cache/joern"))
    parser.add_argument("--joern-timeout", type=float, default=900)
    parser.add_argument(
        "--context-mode",
        choices=("function_only", "repository"),
        default="function_only",
        help="Analyze only the function or automatically prepare its exact repository commit",
    )
    parser.add_argument("--repository-cache", type=Path, default=Path(".cache/primevul_repositories"))
    parser.add_argument(
        "--repository-source-cache",
        type=Path,
        help="Optional existing clone cache organized as PROJECT or PROJECT/COMMIT",
    )
    parser.add_argument("--output", type=Path, help="Optional primary experiment result output")
    parser.add_argument(
        "--agent-trace-output",
        type=Path,
        help="Optional ordered Agent input/output trace (defaults to <output-stem>.agent_io.json)",
    )
    parser.add_argument(
        "--output-format",
        choices=("jsonl", "json"),
        help="Experiment trace format (default: inferred from --output suffix)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_api_key_env = "DASHSCOPE_API_KEY" if args.api_platform == "bailian" else "XEM8K5_API_KEY"
    api_key_env = args.api_key_env or default_api_key_env
    api_key = os.getenv(api_key_env, "")
    model_name = args.model or (
        "qwen3.8-27b" if args.api_platform == "bailian" else "deepseek-v4.1-flash"
    )
    if not api_key:
        print(
            f"Missing API key. Export {api_key_env} before making model calls.",
            file=sys.stderr,
        )
        return 2
    if not args.dataset.is_file():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    bailian = None
    if args.api_platform == "bailian":
        bailian = BailianConfig(
            api_key=api_key,
            base_url=args.base_url or BailianConfig.model_fields["base_url"].default,
            model=model_name,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
        )
        model = create_bailian_chat_model(bailian)
    else:
        compatible = OpenAICompatibleConfig(
            api_key=api_key,
            base_url=args.base_url or OpenAICompatibleConfig.model_fields["base_url"].default,
            model=model_name,
            max_tokens=args.max_tokens,
        )
        model = create_openai_compatible_chat_model(compatible)
    joern = JoernConfig(
        executable=args.joern,
        cache_dir=args.joern_cache,
        timeout_seconds=args.joern_timeout,
    )
    # Keep repository acquisition lazy: function-only first-pass agents do not clone anything,
    # while repository-required investigation queries can prepare the exact commit on demand.
    repository_manager = RepositoryManager(
        RepositoryConfig(
            cache_dir=args.repository_cache,
            source_cache_dir=args.repository_source_cache,
        )
    )
    handlers = create_live_handlers(
        model=model,
        bailian=bailian,
        joern=joern,
        repository_manager=repository_manager,
    )

    report = run_jsonl(
        args.dataset,
        handlers=handlers,
        limit=args.limit,
        max_query_rounds=max(1, args.max_query_rounds),
        context_mode=args.context_mode,
        repository_manager=repository_manager,
        decision_policy=args.decision_policy,
        agent_retries=max(0, args.agent_retries),
        progress=not args.quiet_progress,
    )
    if args.output or args.agent_trace_output:
        trace_path = args.agent_trace_output
        if trace_path is None and args.output is not None:
            trace_path = args.output.with_name(f"{args.output.stem}.agent_io.json")
        primary_format = args.output_format or (
            "json" if args.output and args.output.suffix.lower() == ".json" else "jsonl"
        )
        trace_format = args.output_format or (
            "json" if trace_path and trace_path.suffix.lower() == ".json" else "jsonl"
        )
        records = [
            {
                "sample_id": result.sample_id,
                "idx": result.idx,
                "report": result.report.model_dump(mode="json") if result.report is not None else None,
                "workflow": [compact_interaction(item) for item in result.agent_interactions],
                "error": result.error,
            }
            for result in report.results
        ]
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as output:
                if primary_format == "json":
                    json.dump({"dataset": report.dataset_name, "samples": records}, output, ensure_ascii=False, indent=2)
                    output.write("\n")
                else:
                    for record in records:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        if trace_path is not None:
            if args.output is not None and trace_path.resolve() == args.output.resolve():
                raise ValueError("--agent-trace-output must differ from --output")
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_records = [
                {
                    "sample_id": result.sample_id,
                    "idx": result.idx,
                    "agent_sequence": agent_io_trace(result.agent_interactions),
                }
                for result in report.results
            ]
            with trace_path.open("w", encoding="utf-8") as output:
                if trace_format == "json":
                    json.dump({"dataset": report.dataset_name, "samples": trace_records}, output, ensure_ascii=False, indent=2)
                    output.write("\n")
                else:
                    for record in trace_records:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "dataset": report.dataset_name,
                "requested": report.requested,
                "completed": report.completed,
                "failed": report.failed,
                "predictions": report.predictions,
                "output": str(args.output) if args.output else None,
                "agent_trace_output": str(trace_path) if trace_path else None,
            },
            ensure_ascii=False,
        )
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
