"""Joern CLI integration for cached repository or function-local CPGs."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .config import JoernConfig
from .protocols import AnalysisInput


class JoernError(RuntimeError):
    pass


_QUERY_SCRIPT = r'''@main def exec(cpgFile: String, target: String, targetFile: String): Unit = {
  importCpg(cpgFile)
  def enc(value: String): String = java.util.Base64.getEncoder.encodeToString(value.getBytes("UTF-8"))
  val allMethods = cpg.method.l
  val shortTarget = target.split("::").last
  val namedMethods = if (target == "<unknown>") allMethods.filter(_.filename.contains("primevul")) else allMethods.filter(m => m.name == target || m.name == shortTarget || m.fullName.endsWith(target) || m.fullName.endsWith("." + shortTarget))
  val methods = namedMethods.filter(m => m.filename != "<empty>" && m.lineNumber.isDefined && (targetFile == "" || m.filename.endsWith(targetFile)))
  val riskyCalls = Set("memcpy", "memmove", "strcpy", "strncpy", "strcat", "sprintf", "snprintf", "gets", "read", "recv", "free", "realloc", "system")
  val inputSources = Set("read", "recv", "gets", "fread", "fgets", "scanf", "fscanf", "getenv", "SSL_read", "BIO_read")
  val structuralOperators = Map(
    "<operator>.cast" -> "pointer_or_type_cast",
    "<operator>.indirection" -> "dereference",
    "<operator>.indexAccess" -> "array_or_pointer_index",
    "<operator>.indirectIndexAccess" -> "array_or_pointer_index",
    "<operator>.addition" -> "arithmetic_addition",
    "<operator>.subtraction" -> "arithmetic_subtraction",
    "<operator>.multiplication" -> "arithmetic_multiplication",
    "<operator>.division" -> "arithmetic_division",
    "<operator>.modulo" -> "arithmetic_modulo",
    "<operator>.shiftLeft" -> "arithmetic_shift",
    "<operator>.shiftRight" -> "arithmetic_shift",
    "<operator>.lessThan" -> "comparison",
    "<operator>.lessEqualsThan" -> "comparison",
    "<operator>.greaterThan" -> "comparison",
    "<operator>.greaterEqualsThan" -> "comparison"
  )
  methods.foreach { method =>
    // Keep all evidence inside the selected method.  In particular, a
    // reachableBy traversal can otherwise walk through a caller/callee and
    // make an interprocedural fact look like a local one.
    val methodStart = method.lineNumber.getOrElse(0)
    val methodEnd = method.lineNumberEnd.getOrElse(Int.MaxValue)
    def localLine(line: Option[Int]): Boolean = line.exists(value => value >= methodStart && value <= methodEnd)
    println("METHOD\t" + enc(method.name) + "\t" + enc(method.fullName) + "\t" + enc(method.filename) + "\t" + method.lineNumber.getOrElse(-1))
    method.parameter.l.foreach { parameter =>
      if (parameter.lineNumber.isDefined) {
        println("PARAM\t" + enc(parameter.name) + "\t" + enc(parameter.code) + "\t" + enc(parameter.typeFullName) + "\t" + parameter.lineNumber.get + "\t" + enc(method.filename))
      }
    }
    method.ast.isCall.l.foreach { call =>
      println("CALL\t" + enc(call.name) + "\t" + enc(call.code) + "\t" + enc(call.methodFullName) + "\t" + call.lineNumber.getOrElse(-1))
      if (riskyCalls.contains(call.name)) {
        call.argument.reachableBy(method.parameter).filter(source => localLine(source.lineNumber)).l.foreach { source =>
          println("FLOW\t" + enc(call.name) + "\t" + enc(call.code) + "\t" + enc(source.code) + "\t" + source.lineNumber.getOrElse(-1) + "\t" + call.lineNumber.getOrElse(-1))
        }
      }
      if (inputSources.contains(call.name) && call.lineNumber.isDefined) {
        println("SOURCE\t" + enc(call.name) + "\t" + enc(call.code) + "\t" + enc(call.typeFullName) + "\t" + call.lineNumber.get + "\t" + enc(method.filename))
      }
    }
    method.ast.isCall.filter(call => structuralOperators.contains(call.name) && call.lineNumber.isDefined).l.foreach { op =>
      val resultTypeLower = op.typeFullName.toLowerCase
      val integerType = Seq("char", "short", "int", "long", "size_t", "ssize_t").exists(resultTypeLower.contains)
      val narrowIntegerType = Seq("char", "short", "int8", "uint8", "int16", "uint16").exists(resultTypeLower.contains)
      val category = if (op.name == "<operator>.cast" && narrowIntegerType) "narrow_integer_conversion" else if (op.name == "<operator>.cast" && integerType) "integer_conversion" else structuralOperators(op.name)
      println("OP\t" + enc(category) + "\t" + enc(op.name) + "\t" + enc(op.code) + "\t" + enc(op.typeFullName) + "\t" + op.lineNumber.get + "\t" + enc(method.filename))
      if (op.name == "<operator>.cast" || op.name == "<operator>.indirection") {
        val operand = op.argument.l.lastOption
        val operandCode = operand.map(_.code).getOrElse("")
        // Expression does not expose typeFullName in all Joern versions. Keep the
        // operand source and use the operation result type, which is portable.
        val operandType = ""
        val accessType = op.typeFullName
        val alignment = if (accessType.toLowerCase.contains("double") || accessType.toLowerCase.contains("float")) "alignment-sensitive scalar access" else ""
        val representation = if (op.name == "<operator>.cast" && integerType) "check source range, signedness, target width, and nested casts" else ""
        println("TYPEOP\t" + enc(category) + "\t" + enc(op.name) + "\t" + enc(op.code) + "\t" + enc(operandCode) + "\t" + enc(operandType) + "\t" + enc(accessType) + "\t" + enc(alignment) + "\t" + enc(representation) + "\t" + op.lineNumber.get + "\t" + enc(method.filename))
      }
      op.argument.reachableBy(method.parameter).filter(source => localLine(source.lineNumber)).l.foreach { source =>
        if (source.lineNumber.isDefined) {
          println("OPFLOW\tparameter\t" + enc(category) + "\t" + enc(op.name) + "\t" + enc(op.code) + "\t" + enc(source.code) + "\t" + source.lineNumber.get + "\t" + op.lineNumber.get + "\t" + enc(method.filename))
        }
      }
    }
    method.controlStructure.l.foreach { control =>
      println("CONTROL\t" + enc(control.controlStructureType) + "\t" + enc(control.code) + "\t" + control.lineNumber.getOrElse(-1))
    }
  }
}
'''
_QUALIFIED_METHOD = re.compile(r"(?P<qual>(?:[A-Za-z_]\w*::)+)(?P<method>[~A-Za-z_]\w*)\s*\(")
_RISKY_CALLS = {
    "memcpy", "memmove", "strcpy", "strncpy", "strcat", "sprintf", "snprintf",
    "gets", "read", "recv", "free", "realloc", "system",
}


def _decode(value: str) -> str:
    return base64.b64decode(value.encode("ascii")).decode("utf-8", errors="replace") if value else ""


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


class JoernAnalyzer:
    """Build/reuse CPGs and extract a bounded, source-anchored function skeleton."""

    def __init__(self, config: JoernConfig | None = None):
        self.config = config or JoernConfig()

    def __call__(self, request: AnalysisInput) -> dict[str, Any]:
        return self.analyze(request)

    def analyze(self, request: AnalysisInput) -> dict[str, Any]:
        repository_root, mode, scope_note = self._resolve_source(request)
        if mode == "repository_error":
            return {
                "status": "repository_invalid",
                "mode": "repository",
                "scope": scope_note,
                "entrypoint": request.function_name,
                "methods": [],
                "calls": [],
                "controls": [],
                "parameters": [],
                "candidate_dataflows": [],
                "structural_operations": [],
                "source_candidates": [],
                "unresolved_dependencies": [scope_note],
            }
        language = self._language_for(request, repository=mode == "repository")
        if mode == "repository" and repository_root is not None:
            source_path = repository_root
            key = self._repository_key(request, repository_root, language)
            parse_note = None
        else:
            source_path, parse_note = self._write_function_source(request, language)
            key = hashlib.sha256(
                f"{language}|{self.config.version}|{source_path.read_text(encoding='utf-8')}".encode()
            ).hexdigest()
        cpg_path = self.config.cache_dir / f"{key}.bin"
        try:
            built = self._build_cpg(source_path, cpg_path, language)
            target_file = str(request.metadata.get("file_name") or "") if mode == "repository" else ""
            result = self._query_cpg(cpg_path, request.function_name, target_file)
            if mode == "function_only" and request.metadata.get("file_name"):
                self._remap_function_local_files(result, str(request.metadata["file_name"]))
            result.update(
                {
                    "status": "ready" if result["methods"] else "target_not_found",
                    "mode": mode,
                    "scope": scope_note,
                    "cpg_path": str(cpg_path),
                    "cpg_cached": not built,
                    "language": language,
                    "joern_executable": str(self.config.executable),
                    "query_scope": "function_cfg_local_dataflow",
                    "local_dataflow_only": True,
                    "unresolved_dependencies": [],
                    "evidence_semantics": "Joern CPG facts are source-located review clues, not vulnerability conclusions.",
                }
            )
            if parse_note:
                result["unresolved_dependencies"].append(parse_note)
            self._bound_result(result)
            if mode == "function_only":
                result["unresolved_dependencies"].append(
                    "function-local CPG does not include repository callers, headers, or external contracts"
                )
            if result["status"] == "target_not_found":
                result["unresolved_dependencies"].append(
                    f"no method matched function name {request.function_name!r}"
                )
            return result
        except (OSError, subprocess.SubprocessError, JoernError) as exc:
            return {
                "status": "failed",
                "mode": mode,
                "scope": scope_note,
                "entrypoint": request.function_name,
                "methods": [],
                "calls": [],
                "controls": [],
                "parameters": [],
                "candidate_dataflows": [],
                "structural_operations": [],
                "source_candidates": [],
                "unresolved_dependencies": [f"Joern analysis failed: {exc}"],
            }

    def _resolve_source(self, request: AnalysisInput) -> tuple[Path | None, str, str]:
        metadata = request.metadata
        candidate = metadata.get(self.config.repository_root_field)
        if candidate:
            root = Path(str(candidate)).expanduser().resolve()
            if root.is_dir():
                expected = request.scope.snapshot.commit
                if expected and self.config.require_matching_commit:
                    actual = self._git_commit(root)
                    if actual != expected:
                        return None, "repository_error", f"repository commit mismatch: expected {expected}, found {actual}"
                return root, "repository", f"repository snapshot at {root}"
        requested_mode = metadata.get("context_mode", "function_only")
        if requested_mode in {"repository", "repository_context"}:
            note = "repository context requested but no validated local repository_root was supplied"
            return None, "repository_error", note
        else:
            note = "analyzing only the PrimeVul function text"
        return None, "function_only", note

    def _language_for(self, request: AnalysisInput, *, repository: bool) -> str:
        configured = self.config.language.lower()
        if configured != "auto":
            return configured
        if repository:
            return "newc"
        file_name = str(request.metadata.get("file_name") or "").lower()
        if (
            file_name.endswith((".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"))
            or "::" in request.function_name
            or "std::" in request.source
            or "class " in request.source
        ):
            return "newc"
        return "c"

    @staticmethod
    def _git_commit(root: Path) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    def _repository_key(self, request: AnalysisInput, root: Path, language: str) -> str:
        material = "|".join(
            [
                str(root),
                request.scope.snapshot.commit or request.scope.snapshot.version,
                request.metadata.get("joern_version", self.config.version),
                language,
            ]
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def _write_function_source(self, request: AnalysisInput, language: str) -> tuple[Path, str | None]:
        suffix = ".cpp" if language == "newc" else ".c"
        temp_root = self.config.cache_dir / "function_sources" / _safe_key(request.sample_id)
        temp_root.mkdir(parents=True, exist_ok=True)
        source_path = temp_root / f"primevul_function{suffix}"
        source = request.source
        parse_note = None
        if language == "newc" and "::" in request.function_name:
            detached, count = _QUALIFIED_METHOD.subn(r"\g<method>(", source, count=1)
            if count:
                source = detached
                parse_note = (
                    "detached C++ member definition for function-local parsing; class and header context are absent"
                )
        source_path.write_text(source, encoding="utf-8")
        return source_path, parse_note

    def _build_cpg(self, source_path: Path, cpg_path: Path, language: str) -> bool:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        if cpg_path.is_file() and cpg_path.stat().st_size > 0:
            return False
        parser = self.config.resolved_parse_executable()
        if not parser.is_file():
            raise JoernError(f"joern-parse executable not found: {parser}")
        command = [str(parser), str(source_path), "--language", language, "--output", str(cpg_path)]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
        )
        if result.returncode != 0 or not cpg_path.is_file() or cpg_path.stat().st_size == 0:
            detail = (result.stderr or result.stdout)[-4000:]
            raise JoernError(f"joern-parse exited {result.returncode}: {detail}")
        return True

    def _query_cpg(self, cpg_path: Path, function_name: str, target_file: str = "") -> dict[str, Any]:
        with tempfile.NamedTemporaryFile("w", suffix=".sc", encoding="utf-8", delete=False) as script:
            script.write(_QUERY_SCRIPT)
            script_path = Path(script.name)
        try:
            result = subprocess.run(
                [
                    str(self.config.executable),
                    str(cpg_path),
                    "--script",
                    str(script_path),
                    "--param",
                    f"cpgFile={cpg_path}",
                    "--param",
                    f"target={function_name}",
                    "--param",
                    f"targetFile={target_file}",
                    "--nocolors",
                ],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            if result.returncode != 0:
                raise JoernError(f"joern query exited {result.returncode}: {(result.stderr or result.stdout)[-4000:]}")
            return self._parse_query_output(result.stdout)
        finally:
            script_path.unlink(missing_ok=True)

    @staticmethod
    def _parse_query_output(output: str) -> dict[str, Any]:
        methods: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        parameters: list[dict[str, Any]] = []
        candidate_dataflows: list[dict[str, Any]] = []
        structural_operations: list[dict[str, Any]] = []
        source_candidates: list[dict[str, Any]] = []
        current_method: dict[str, Any] | None = None
        for line in output.splitlines():
            fields = line.strip().split("\t")
            try:
                if fields[0] == "METHOD" and len(fields) == 5:
                    current_method = {
                        "name": _decode(fields[1]),
                        "full_name": _decode(fields[2]),
                        "filename": _decode(fields[3]),
                        "line": int(fields[4]),
                    }
                    methods.append(current_method)
                elif fields[0] == "CALL" and len(fields) == 5:
                    calls.append(
                        {
                            "method": current_method["full_name"] if current_method else None,
                            "name": _decode(fields[1]),
                            "code": _decode(fields[2]),
                            "target": _decode(fields[3]),
                            "line": int(fields[4]),
                            "file": current_method["filename"] if current_method else None,
                        }
                    )
                elif fields[0] == "CONTROL" and len(fields) == 4:
                    controls.append(
                        {
                            "method": current_method["full_name"] if current_method else None,
                            "type": _decode(fields[1]),
                            "code": _decode(fields[2]),
                            "line": int(fields[3]),
                            "file": current_method["filename"] if current_method else None,
                        }
                    )
                elif fields[0] == "PARAM" and len(fields) == 6:
                    parameters.append(
                        {
                            "method": current_method["full_name"] if current_method else None,
                            "kind": "function_parameter_source",
                            "name": _decode(fields[1]),
                            "code": _decode(fields[2]),
                            "type": _decode(fields[3]),
                            "line": int(fields[4]),
                            "file": _decode(fields[5]),
                        }
                    )
                elif fields[0] == "PARAM" and len(fields) == 4:
                    parameters.append(
                        {
                            "method": current_method["full_name"] if current_method else None,
                            "kind": "function_parameter_source",
                            "name": _decode(fields[1]),
                            "code": _decode(fields[2]),
                            "type": None,
                            "line": int(fields[3]),
                            "file": current_method["filename"] if current_method else None,
                        }
                    )
                elif fields[0] == "FLOW" and len(fields) == 6:
                    candidate_dataflows.append(
                        {
                            "method": current_method["full_name"] if current_method else None,
                            "sink_call": _decode(fields[1]),
                            "call_code": _decode(fields[2]),
                            "source_code": _decode(fields[3]),
                            "source_line": int(fields[4]),
                            "sink_line": int(fields[5]),
                            "file": current_method["filename"] if current_method else None,
                            "source_kind": "parameter",
                            "sink_kind": "risky_call_candidate",
                            "kind": "joern_parameter_to_sink_argument_candidate",
                        }
                    )
                elif fields[0] == "OP" and len(fields) == 7:
                    structural_operations.append(
                        {
                            "method": current_method["full_name"] if current_method else None,
                            "category": _decode(fields[1]),
                            "operator": _decode(fields[2]),
                            "code": _decode(fields[3]),
                            "result_type": _decode(fields[4]),
                            "line": int(fields[5]),
                            "file": _decode(fields[6]),
                            "evidence_kind": "joern_candidate_structure",
                        }
                    )
                elif fields[0] == "TYPEOP" and len(fields) in {10, 11}:
                    extended = len(fields) == 11
                    structural_operations.append(
                        {
                            "method": current_method["full_name"] if current_method else None,
                            "category": _decode(fields[1]),
                            "operator": _decode(fields[2]),
                            "code": _decode(fields[3]),
                            "operand_code": _decode(fields[4]),
                            "operand_type": _decode(fields[5]) or "unknown",
                            "result_type": _decode(fields[6]),
                            "alignment_requirement": _decode(fields[7]) or None,
                            "representation_requirement": _decode(fields[8]) if extended else None,
                            "line": int(fields[9] if extended else fields[8]),
                            "file": _decode(fields[10] if extended else fields[9]),
                            "evidence_kind": (
                                "joern_integer_conversion_candidate"
                                if _decode(fields[1]) in {"integer_conversion", "narrow_integer_conversion"}
                                else "joern_type_and_alignment_candidate"
                            ),
                        }
                    )
                elif fields[0] == "SOURCE" and len(fields) == 6:
                    source_candidates.append(
                        {
                            "method": current_method["full_name"] if current_method else None,
                            "kind": "input_api_candidate",
                            "name": _decode(fields[1]),
                            "code": _decode(fields[2]),
                            "return_type": _decode(fields[3]),
                            "line": int(fields[4]),
                            "file": _decode(fields[5]),
                            "evidence_kind": "joern_candidate_source",
                        }
                    )
                elif fields[0] == "OPFLOW" and len(fields) == 9:
                    candidate_dataflows.append(
                        {
                            "method": current_method["full_name"] if current_method else None,
                            "source_kind": fields[1],
                            "sink_kind": _decode(fields[2]),
                            "sink_operator": _decode(fields[3]),
                            "call_code": _decode(fields[4]),
                            "source_code": _decode(fields[5]),
                            "source_line": int(fields[6]),
                            "sink_line": int(fields[7]),
                            "file": _decode(fields[8]),
                            "kind": "joern_parameter_to_structural_operation_candidate",
                        }
                    )
            except (ValueError, IndexError):
                continue
        return {
            "methods": methods,
            "calls": calls,
            "controls": controls,
            "parameters": parameters,
            "candidate_dataflows": candidate_dataflows,
            "structural_operations": structural_operations,
            "source_candidates": source_candidates,
        }

    def _bound_result(self, result: dict[str, Any]) -> None:
        for call in result["calls"]:
            call["code"] = self._clip(call["code"])
        result["calls"].sort(key=lambda call: (call["name"] not in _RISKY_CALLS, call["line"]))
        for control in result["controls"]:
            control["code"] = self._clip(control["code"])
        for parameter in result["parameters"]:
            parameter["code"] = self._clip(parameter["code"])
        for flow in result["candidate_dataflows"]:
            flow["call_code"] = self._clip(flow["call_code"])
            flow["source_code"] = self._clip(flow["source_code"])
        for operation in result.get("structural_operations", []):
            operation["code"] = self._clip(operation["code"])
        for source in result.get("source_candidates", []):
            source["code"] = self._clip(source["code"])
        result["node_counts"] = {
            "methods": len(result["methods"]),
            "calls": len(result["calls"]),
            "controls": len(result["controls"]),
            "parameters": len(result["parameters"]),
            "candidate_dataflows": len(result["candidate_dataflows"]),
            "structural_operations": len(result.get("structural_operations", [])),
            "source_candidates": len(result.get("source_candidates", [])),
        }
        result["truncated"] = {
            "calls": len(result["calls"]) > self.config.max_calls,
            "controls": len(result["controls"]) > self.config.max_controls,
            "parameters": len(result["parameters"]) > self.config.max_parameters,
            "candidate_dataflows": len(result["candidate_dataflows"]) > self.config.max_dataflows,
            "structural_operations": len(result.get("structural_operations", [])) > self.config.max_structural_operations,
            "source_candidates": len(result.get("source_candidates", [])) > self.config.max_source_candidates,
        }
        result["calls"] = result["calls"][: self.config.max_calls]
        result["controls"] = result["controls"][: self.config.max_controls]
        result["parameters"] = result["parameters"][: self.config.max_parameters]
        unique_flows: dict[tuple[Any, ...], dict[str, Any]] = {}
        for flow in result["candidate_dataflows"]:
            key = (
                flow["method"],
                flow.get("sink_call") or flow.get("sink_operator"),
                flow.get("call_code") or flow.get("sink_code"),
                flow["source_code"],
                flow["sink_line"],
            )
            unique_flows.setdefault(key, flow)
        result["candidate_dataflows"] = list(unique_flows.values())[: self.config.max_dataflows]
        # Preserve type/alignment-sensitive operations before generic AST noise so that
        # a bounded skeleton still exposes float/double casts and dereferences.
        structural = result.get("structural_operations", [])
        structural.sort(
            key=lambda item: (
                item.get("evidence_kind") not in {
                    "joern_integer_conversion_candidate", "joern_type_and_alignment_candidate"
                },
                not any(
                    token in str(item.get("code", "")).lower()
                    for token in ("float", "double")
                ),
                item.get("line", 0),
            )
        )
        result["structural_operations"] = structural[: self.config.max_structural_operations]
        result["source_candidates"] = result.get("source_candidates", [])[: self.config.max_source_candidates]

    @staticmethod
    def _remap_function_local_files(result: dict[str, Any], file_name: str) -> None:
        """Replace Joern's temporary parse filename with the PrimeVul source path."""
        for method in result.get("methods", []):
            method["filename"] = file_name
        for field in (
            "calls", "controls", "parameters", "candidate_dataflows",
            "structural_operations", "source_candidates",
        ):
            for item in result.get(field, []):
                if "file" in item:
                    item["file"] = file_name

    def _clip(self, value: str) -> str:
        if len(value) <= self.config.max_code_chars:
            return value
        return value[: self.config.max_code_chars] + "...<truncated>"


__all__ = ["JoernAnalyzer", "JoernError"]
