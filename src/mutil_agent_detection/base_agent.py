"""Common LangChain plumbing for first-pass audit agents."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from .agent_models import ClaimDraft
from .context_compaction import compact_key_operations, compact_skeleton
from .protocols import AnalysisInput, Claim, ClaimKind, ClaimStatus


class StructuredAgentError(RuntimeError):
    """Structured-output invocation failed with an attached audit trace."""

    def __init__(self, message: str, interaction: dict[str, Any]):
        super().__init__(message)
        self.interaction = interaction


class FirstPassAgent:
    """Build a role-specific structured-output chain and map it to a Claim."""

    kind: ClaimKind
    actor: str
    instruction: str
    output_contract: str
    schema = ClaimDraft

    def __init__(self, model: Any):
        self.last_interaction: dict[str, Any] | None = None
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{instruction}\n\n{output_contract}\n\n"
                    "General rules:\n"
                    "- Return exactly one JSON object matching the schema.\n"
                    "- Use empty arrays or null when a field is not applicable.\n"
                    "- For function_source anchors, use source_line_basis: repository_absolute means exact commit file line numbers; function_local means lines relative to the supplied function.\n"
                    "- The key-operation index is a program-generated navigation aid, not a vulnerability conclusion; review the complete function and cite the exact source line.\n"
                    "- Joern structural operations, source candidates, and dataflows are candidate evidence only; independently verify what they mean in the supplied code.\n"
                    "- Copy a short exact excerpt into each anchor; unsupported citations become unknown.\n"
                    "- Return only anchor excerpt and basis; do not guess line numbers or filenames. The program resolves them.\n"
                    "- Never invent repository facts, callers, types, or evidence.\n"
                    "- Unknown is a valid result and must be used for missing context.\n"
                    "- Do not output chain-of-thought, self-correction, or rejected hypotheses.",
                ),
                (
                    "human",
                    "INPUT\n"
                    "sample_id: {sample_id}\n"
                    "function_name: {function_name}\n"
                    "file_name: {file_name}\n"
                    "function_line_range: {line_range}\n"
                    "source_line_basis: {source_line_basis}\n"
                    "source_line_mapping_note: {source_line_mapping_note}\n"
                    "commit_id: {commit_id}\n"
                    "context_mode: {context_mode}\n"
                    "complete_line_numbered_source (prefix is metadata; copy only code after '|'):\n```c\n{source}\n```\n"
                    "program_generated_key_operation_index_json:\n{key_operation_index}\n"
                    "shared_joern_skeleton_json:\n{skeleton}\n"
                    "anchor_repair_feedback_json:\n{anchor_repair_feedback}\n"
                    "scope_json:\n{scope}\n",
                ),
            ]
        )
        self.function_calling_chain = self.prompt | model.with_structured_output(
            self.schema, method="function_calling", include_raw=True
        )
        self.json_mode_chain = self.prompt | model.with_structured_output(
            self.schema, method="json_mode", include_raw=True
        )
        self.chain = self.function_calling_chain

    def __call__(self, request: AnalysisInput, skeleton: dict[str, Any]) -> Claim:
        prompt_values = self._prompt_values(request, skeleton)
        interaction: dict[str, Any] = {
            "agent": self.actor,
            "kind": getattr(self.kind, "value", str(self.kind)),
            "input": prompt_values,
            "messages": [],
            "raw_output": None,
            "status": "started",
            "error": None,
        }
        self.last_interaction = interaction
        try:
            prompt_value = self.prompt.invoke(prompt_values)
            interaction["messages"] = [
                {"type": message.type, "content": self._jsonable(message.content)}
                for message in prompt_value.to_messages()
            ]
            result = self.function_calling_chain.invoke(prompt_values)
        except Exception as exc:
            interaction.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            raise StructuredAgentError(str(exc), interaction) from exc
        interaction["attempts"] = []
        draft = self._extract_draft(result, interaction, method="function_calling")
        if draft is None:
            # Some OpenAI-compatible endpoints return ordinary JSON content instead
            # of a tool call. Prefer parsing that response before spending a retry.
            raw = self._unwrap_response(result).get("raw")
            draft = self._parse_raw_content(raw)
            if draft is not None:
                interaction["attempts"][-1]["recovered_by"] = "raw_json_content"
            elif self._finish_reason(raw) == "length":
                message = (
                    "model response reached the completion token limit before producing a structured result; "
                    "Qwen thinking may be consuming the budget. Disable thinking or increase --max-tokens."
                )
                interaction.update({"status": "failed", "error": message})
                raise StructuredAgentError(message, interaction)
        if draft is None:
            try:
                fallback = self.json_mode_chain.invoke(prompt_values)
                draft = self._extract_draft(fallback, interaction, method="json_mode")
                if draft is None:
                    # LangChain may reject an oversized parsed object before we
                    # can apply our bounded repair. Parse the provider's JSON
                    # content directly and validate the repaired payload.
                    raw = self._unwrap_response(fallback).get("raw")
                    draft = self._parse_raw_content(raw)
                    if draft is not None and interaction["attempts"]:
                        interaction["attempts"][-1]["recovered_by"] = "bounded_raw_json"
            except Exception as exc:
                interaction.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                raise StructuredAgentError(str(exc), interaction) from exc
        if draft is None:
            last_attempt = interaction["attempts"][-1] if interaction["attempts"] else {}
            message = (
                "model response contained neither a parsed ClaimDraft nor valid JSON; "
                f"last attempt metadata={last_attempt.get('metadata')}"
            )
            interaction.update({"status": "failed", "error": message})
            raise StructuredAgentError(message, interaction)
        details = draft.model_dump(mode="json")
        self._project_primary_risk_candidate(details)
        anchor_report = self._validate_anchors(details, request, skeleton)
        repair_feedback = self._anchor_repair_feedback(details, anchor_report)
        if repair_feedback:
            repaired_draft = self._retry_invalid_anchors(
                prompt_values, repair_feedback, interaction
            )
            if repaired_draft is not None:
                draft = repaired_draft
                details = draft.model_dump(mode="json")
                self._project_primary_risk_candidate(details)
                anchor_report = self._validate_anchors(details, request, skeleton)
        structure_feedback = self._structured_repair_feedback(
            details, request=request, skeleton=skeleton
        )
        if structure_feedback:
            repaired_draft = self._retry_invalid_anchors(
                prompt_values, structure_feedback, interaction
            )
            if repaired_draft is not None:
                draft = repaired_draft
                details = draft.model_dump(mode="json")
                self._project_primary_risk_candidate(details)
                anchor_report = self._validate_anchors(details, request, skeleton)
        self._mark_unverified_evidence(details)
        self._downgrade_unverified_supported_premises(details)
        self._downgrade_unverified_protection_coverage(details)
        details["anchor_validation"] = anchor_report
        interaction["parsed_output"] = self._jsonable(
            {key: value for key, value in details.items() if key != "_interaction"}
        )
        premise_ids = list(dict.fromkeys(self._collect_premise_ids(details)))
        interaction.update({"status": "parsed", "error": None})
        text = getattr(draft, "summary", None) or getattr(draft, "text", "")
        interaction["validation"] = {
            "anchor_validation": anchor_report,
            "premise_statuses": {
                premise["id"]: premise["status"]
                for field in ("required_conditions", "invariants")
                for premise in details.get(field, [])
            },
            "protection_coverage": {
                protection["id"]: protection["coverage"]
                for protection in details.get("protections", [])
            },
            "normalized_summary": text,
        }
        details["_interaction"] = interaction
        return Claim(
            kind=self.kind,
            text=text,
            status=ClaimStatus.CANDIDATE,
            scope=request.scope,
            premise_ids=premise_ids,
            source_anchors=self._validated_source_locations(details, request),
            created_by=self.actor,
            confidence=(
                (
                    (
                        draft.candidates[0].confidence
                        if details.get("candidates", [{}])[0].get("evidence_status") != "unverified"
                        else None
                    )
                    if draft.candidates
                    else (None if details.get("evidence_status") == "unverified" else draft.confidence)
                )
                if self.kind == ClaimKind.RISK else draft.confidence
            ),
            analysis=details,
        )

    def _structured_repair_feedback(
        self,
        details: dict[str, Any],
        *,
        request: AnalysisInput | None = None,
        skeleton: dict[str, Any] | None = None,
    ) -> str:
        """Optional role-specific completeness check before accepting a response."""
        return ""

    @staticmethod
    def _project_primary_risk_candidate(details: dict[str, Any]) -> None:
        """Expose the highest-ranked candidate through the legacy claim fields."""
        candidates = details.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return
        primary = candidates[0]
        for field in (
            "security_property", "risk_operation", "target_object", "address_source",
            "access_type", "alignment_requirement", "candidate_class",
            "conversion_source_type", "conversion_target_type", "representation_requirement", "risk_path",
            "required_conditions", "unresolved_questions", "source_anchors", "confidence",
        ):
            details[field] = primary.get(field)

    def _anchor_repair_feedback(self, details: dict[str, Any], report: dict[str, Any]) -> str:
        invalid = [anchor for anchor in report.get("anchors", []) if not anchor.get("valid")]
        if not invalid:
            return ""
        affected_indices: set[int] = set()
        for anchor in invalid:
            match = re.search(r"analysis\.candidates\[(\d+)\]", anchor.get("path", ""))
            if match and int(match.group(1)) < len(details.get("candidates", [])):
                affected_indices.add(int(match.group(1)))
        payload = {
            "instruction": (
                "Repair only the affected candidate/protection evidence. Use an exact unique excerpt from the supplied source or Joern skeleton. "
                "Do not return file or line. If a risk candidate has no exact support, remove it; renumber remaining priorities consecutively and update candidate_found. Never invent an excerpt."
            ),
            "invalid_anchors": [
                {key: anchor.get(key) for key in ("path", "excerpt", "basis", "reason")}
                for anchor in invalid[:12]
            ],
            "candidate_ids": [
                details["candidates"][index].get("candidate_id")
                for index in sorted(affected_indices)
            ],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _retry_invalid_anchors(
        self, prompt_values: dict[str, str], feedback: str, interaction: dict[str, Any]
    ) -> Any | None:
        task = (
            "Targeted structured-answer correction: revise only the fields identified by the feedback, "
            "preserve valid analysis, and return the full response schema. Never invent evidence; use "
            "exact unique excerpts from the supplied source or Joern skeleton."
        )
        repair_input = {
            "task": task,
            "feedback": json.loads(feedback),
            "response_schema": self.schema.__name__,
        }
        repair_record: dict[str, Any] = {"input": repair_input, "output": None, "status": "started"}
        interaction["anchor_repair"] = repair_record
        repair_values = dict(prompt_values)
        repair_values["instruction"] = f"{prompt_values['instruction']}\n\n{task}"
        repair_values["anchor_repair_feedback"] = feedback

        try:
            prompt_value = self.prompt.invoke(repair_values)
            repair_record["messages"] = [
                {"type": message.type, "content": self._jsonable(message.content)}
                for message in prompt_value.to_messages()
            ]
            result = self.function_calling_chain.invoke(repair_values)
            draft = self._extract_draft(result, interaction, method="anchor_repair_function_calling")
            if draft is None:
                raw = self._unwrap_response(result).get("raw")
                draft = self._parse_raw_content(raw)
            if draft is None:
                result = self.json_mode_chain.invoke(repair_values)
                draft = self._extract_draft(result, interaction, method="anchor_repair_json_mode")
                if draft is None:
                    raw = self._unwrap_response(result).get("raw")
                    draft = self._parse_raw_content(raw)
            if draft is None:
                repair_record.update({"status": "failed", "error": "anchor repair response was not parseable"})
                return None
            repair_record.update({"status": "parsed", "output": self._jsonable(draft)})
            return draft
        except Exception as exc:
            repair_record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            return None

    @staticmethod
    def _anchors_in(value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []

        def walk(item: Any) -> None:
            if isinstance(item, dict):
                if isinstance(item.get("excerpt"), str) and item.get("basis") in {"function_source", "joern_skeleton"}:
                    found.append(item)
                else:
                    for child in item.values():
                        walk(child)
            elif isinstance(item, list):
                for child in item:
                    walk(child)

        walk(value)
        return found

    @classmethod
    def _mark_unverified_evidence(cls, details: dict[str, Any]) -> None:
        for candidate in details.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            anchors = cls._anchors_in(candidate)
            invalid = [anchor for anchor in anchors if not anchor.get("validated")]
            candidate["evidence_status"] = "unverified" if not anchors or invalid else "anchors_verified"
            if invalid:
                candidate["unverified_anchor_count"] = len(invalid)
        if details.get("candidates"):
            primary = details["candidates"][0]
            details["evidence_status"] = primary["evidence_status"]
            if primary.get("unverified_anchor_count"):
                details["unverified_anchor_count"] = primary["unverified_anchor_count"]
        elif details.get("candidate_found"):
            anchors = cls._anchors_in(details)
            invalid = [anchor for anchor in anchors if not anchor.get("validated")]
            details["evidence_status"] = "unverified" if not anchors or invalid else "anchors_verified"
            if invalid:
                details["unverified_anchor_count"] = len(invalid)
        for protection in details.get("protections", []):
            anchors = cls._anchors_in(protection)
            invalid = [anchor for anchor in anchors if not anchor.get("validated")]
            protection["evidence_status"] = "unverified" if not anchors or invalid else "anchors_verified"

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, BaseException):
            return f"{type(value).__name__}: {value}"
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(key): FirstPassAgent._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [FirstPassAgent._jsonable(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _extract_draft(self, result: Any, interaction: dict[str, Any], *, method: str) -> Any | None:
        wrapped = self._unwrap_response(result)
        raw = wrapped.get("raw")
        parsed = wrapped.get("parsed", result if not wrapped else None)
        attempt = {
            "method": method,
            "raw": self._jsonable(raw),
            "parsed": self._jsonable(parsed),
            "parsing_error": self._jsonable(wrapped.get("parsing_error")),
            "metadata": self._response_metadata(raw),
        }
        interaction["attempts"].append(attempt)
        interaction["raw_output"] = attempt["raw"] if raw is not None else self._jsonable(result)
        if parsed is None:
            return None
        if isinstance(parsed, self.schema):
            return parsed
        if isinstance(parsed, dict):
            try:
                normalized, repairs = self._normalize_json_encoded_arrays(parsed)
                normalized, bound_repairs = self._bound_payload(normalized)
                repairs.extend(bound_repairs)
                if repairs:
                    attempt["repairs"] = repairs
                return self.schema.model_validate(normalized)
            except Exception as exc:
                attempt["parsing_error"] = f"{type(exc).__name__}: {exc}"
                return None
        attempt["parsing_error"] = f"unsupported parsed type: {type(parsed)!r}"
        return None

    @staticmethod
    def _unwrap_response(result: Any) -> dict[str, Any]:
        if isinstance(result, dict) and "raw" in result and "parsed" in result:
            return result
        return {}

    def _parse_raw_content(self, raw: Any) -> Any | None:
        content = getattr(raw, "content", None)
        if not isinstance(content, str) or not content.strip():
            return None
        candidate = content.strip()
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
        try:
            normalized = json.loads(candidate)
            normalized, _ = self._normalize_json_encoded_arrays(normalized)
            normalized, _ = self._bound_payload(normalized)
            return self.schema.model_validate(normalized)
        except Exception:
            return None

    @staticmethod
    def _normalize_json_encoded_arrays(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Repair only schema-list fields encoded as JSON strings by tool-call providers."""
        normalized = dict(payload)
        repairs: list[str] = []
        for field in ("risk_path", "required_conditions", "protections", "invariants", "source_anchors", "unresolved_questions", "candidates"):
            value = normalized.get(field)
            if isinstance(value, str):
                decoded = FirstPassAgent._decode_embedded_json_list(value)
                if isinstance(decoded, list):
                    normalized[field] = decoded
                    repairs.extend((field, f"{field}:decoded_string"))
        return normalized, repairs

    @staticmethod
    def _decode_embedded_json_list(value: str) -> list[Any] | None:
        """Decode a list even when a provider appends harmless JSON tail text."""
        text = value.strip()
        try:
            decoded = json.loads(text)
            return decoded if isinstance(decoded, list) else None
        except json.JSONDecodeError:
            pass
        start = text.find("[")
        if start < 0:
            return None
        try:
            decoded, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, list) else None

    @staticmethod
    def _bound_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Bound repetitive model fields before strict validation.

        Models occasionally return every check in a long function despite the
        requested concise schema. Keeping the first items preserves a usable,
        auditable answer while avoiding a whole-sample failure.
        """
        normalized = dict(payload)
        repairs: list[str] = []
        limits = {
            "candidates": 3,
            "risk_path": 8,
            "required_conditions": 8,
            "protections": 8,
            "invariants": 8,
            "source_anchors": 8,
            "unresolved_questions": 8,
        }
        for field, limit in limits.items():
            value = normalized.get(field)
            if isinstance(value, list) and len(value) > limit:
                normalized[field] = value[:limit]
                repairs.append(f"{field}:truncated_to_{limit}")
        for field, limits_for_items in {
            "candidates": {
                "risk_path": 6,
                "required_conditions": 6,
                "unresolved_questions": 6,
                "source_anchors": 8,
            },
            "risk_path": {"source_anchors": 3},
            "required_conditions": {"evidence_refs": 5},
            "invariants": {"evidence_refs": 5},
            "protections": {"source_anchors": 5, "blocks_conditions": 8, "bypass_paths": 8},
        }.items():
            items = normalized.get(field)
            if not isinstance(items, list):
                continue
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                for child, limit in limits_for_items.items():
                    value = item.get(child)
                    if isinstance(value, list) and len(value) > limit:
                        item[child] = value[:limit]
                        repairs.append(f"{field}[{index}].{child}:truncated_to_{limit}")
                if field == "candidates":
                    for step_index, step in enumerate(item.get("risk_path", [])):
                        if not isinstance(step, dict):
                            continue
                        anchors = step.get("source_anchors")
                        if isinstance(anchors, list) and len(anchors) > 3:
                            step["source_anchors"] = anchors[:3]
                            repairs.append(f"candidates[{index}].risk_path[{step_index}].source_anchors:truncated_to_3")
                    for condition_index, condition in enumerate(item.get("required_conditions", [])):
                        if not isinstance(condition, dict):
                            continue
                        refs = condition.get("evidence_refs")
                        if isinstance(refs, list) and len(refs) > 5:
                            condition["evidence_refs"] = refs[:5]
                            repairs.append(f"candidates[{index}].required_conditions[{condition_index}].evidence_refs:truncated_to_5")
                items[index] = item
            normalized[field] = items
        return normalized, repairs

    def _validate_anchors(
        self, details: dict[str, Any], request: AnalysisInput, skeleton: dict[str, Any]
    ) -> dict[str, Any]:
        requested_file = str(request.metadata.get("file_name") or "<function-local>")
        skeleton_evidence = self._skeleton_source_locations(skeleton, requested_file)
        checked: list[dict[str, Any]] = []

        def walk(value: Any, path: str) -> None:
            if isinstance(value, dict):
                if isinstance(value.get("excerpt"), str) and value.get("basis") in {"function_source", "joern_skeleton"}:
                    valid, reason = self._resolve_anchor(
                        value,
                        requested_file=requested_file,
                        source=request.source,
                        source_line_start=request.metadata.get("source_line_start"),
                        skeleton_evidence=skeleton_evidence,
                    )
                    value["validated"] = valid
                    value["validation_note"] = reason
                    checked.append({
                        "path": path,
                        "file": value.get("file"),
                        "line": value.get("line"),
                        "excerpt": value["excerpt"],
                        "basis": value["basis"],
                        "valid": valid,
                        "reason": reason,
                    })
                for key, child in value.items():
                    if key not in {"validation_note"}:
                        walk(child, f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, f"{path}[{index}]")

        if details.get("candidates"):
            canonical = {key: value for key, value in details.items() if key not in {
                "security_property", "risk_operation", "target_object", "risk_path",
                "required_conditions", "unresolved_questions", "source_anchors", "confidence",
            }}
            walk(canonical, "analysis")
        else:
            walk(details, "analysis")
        return {
            "basis": "program-resolved from exact excerpt in function source or Joern skeleton",
            "verification_scope": "Confirms excerpt existence and resolves its source location; it does not prove semantic entailment.",
            "valid_count": sum(item["valid"] for item in checked),
            "invalid_count": sum(not item["valid"] for item in checked),
            "anchors": checked,
        }

    def _resolve_anchor(
        self,
        anchor: dict[str, Any],
        *,
        requested_file: str,
        source: str,
        source_line_start: int | None,
        skeleton_evidence: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        excerpt = self._normalize_code(anchor["excerpt"])
        reported_file = anchor.get("file")
        if anchor["basis"] == "function_source":
            matches = self._source_excerpt_lines(source, excerpt)
            if len(matches) != 1:
                anchor["file"] = requested_file
                anchor["line"] = None
                if not matches:
                    return False, "exact excerpt was not found in the supplied function source"
                return False, "excerpt occurs on multiple function lines; use a longer unique excerpt"
            local_line = matches[0]
            resolved_line = source_line_start + local_line - 1 if source_line_start else local_line
            anchor["file"] = requested_file
            anchor["line"] = resolved_line
            if reported_file and not self._same_file(str(reported_file), requested_file):
                return False, "reported filename does not match the supplied target file"
            return True, "exact excerpt located in supplied function source"

        matches = [
            item for item in skeleton_evidence
            if excerpt in self._normalize_code(item["code"])
        ]
        if reported_file:
            matching_file = [item for item in matches if self._same_file(str(reported_file), item["file"])]
            if matching_file:
                matches = matching_file
        locations = {(item["file"], item["line"]) for item in matches}
        if len(locations) != 1:
            anchor["file"] = requested_file
            anchor["line"] = None
            if not locations:
                return False, "exact excerpt was not found in the supplied Joern skeleton"
            return False, "excerpt maps to multiple Joern source locations; use a longer unique excerpt"
        resolved_file, resolved_line = next(iter(locations))
        anchor["file"] = resolved_file
        anchor["line"] = resolved_line
        if reported_file and not self._same_file(str(reported_file), resolved_file):
            return False, "reported filename does not match the Joern source location"
        return True, "exact excerpt located in Joern skeleton"

    @staticmethod
    def _same_file(left: str, right: str) -> bool:
        return left == right or left.rsplit("/", 1)[-1] == right.rsplit("/", 1)[-1]

    @classmethod
    def _source_excerpt_lines(cls, source: str, excerpt: str) -> list[int]:
        normalized: list[str] = []
        offsets: list[int] = []
        pending_space = False
        for index, char in enumerate(source):
            if char.isspace():
                pending_space = bool(normalized)
                continue
            if pending_space:
                normalized.append(" ")
                offsets.append(index)
                pending_space = False
            normalized.append(char)
            offsets.append(index)
        text = "".join(normalized)
        lines: set[int] = set()
        position = text.find(excerpt)
        while position >= 0:
            original_offset = offsets[position]
            lines.add(source.count("\n", 0, original_offset) + 1)
            position = text.find(excerpt, position + 1)
        return sorted(lines)

    @staticmethod
    def _normalize_code(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _skeleton_source_locations(skeleton: dict[str, Any], fallback_file: str) -> list[dict[str, Any]]:
        locations: list[dict[str, Any]] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                code = value.get("code")
                line = value.get("line")
                file_name = value.get("file") or value.get("filename") or fallback_file
                if isinstance(code, str) and code and isinstance(line, int) and line > 0:
                    locations.append({"code": code, "line": line, "file": str(file_name)})
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(skeleton)
        return locations

    @staticmethod
    def _downgrade_unverified_supported_premises(details: dict[str, Any]) -> None:
        premise_groups = [details.get(field, []) for field in ("required_conditions", "invariants")]
        premise_groups.extend(
            candidate.get("required_conditions", [])
            for candidate in details.get("candidates", [])
            if isinstance(candidate, dict)
        )
        for group in premise_groups:
            for premise in group:
                refs = premise.get("evidence_refs", [])
                if premise.get("status") == "supported" and not any(ref.get("validated") for ref in refs):
                    premise["status"] = "unknown"
                    premise["validation_note"] = "No source anchor was validated for this supported premise."

    @staticmethod
    def _downgrade_unverified_protection_coverage(details: dict[str, Any]) -> None:
        for protection in details.get("protections", []):
            anchors = FirstPassAgent._anchors_in(protection)
            valid_anchors = {
                (anchor.get("file"), anchor.get("line"))
                for anchor in anchors
                if anchor.get("validated")
            }
            if not valid_anchors:
                protection["coverage"] = "unknown"
                protection["validation_note"] = "No source anchor was validated for this protection."
            elif protection.get("coverage") == "complete" and (
                any(not anchor.get("validated") for anchor in anchors)
                or len(valid_anchors) < 2
                or protection.get("bypass_paths")
                or not protection.get("blocks_conditions")
            ):
                protection["coverage"] = "partial"
                protection["validation_note"] = (
                    "Complete coverage was downgraded: cite both the guard and protected operation, "
                    "state the blocked condition, and account for bypass paths."
                )

    @staticmethod
    def _collect_premise_ids(details: dict[str, Any]) -> list[str]:
        ids = []
        for field in ("required_conditions", "invariants"):
            ids.extend(
                premise["id"] for premise in details.get(field, [])
                if isinstance(premise, dict) and isinstance(premise.get("id"), str)
            )
        for candidate in details.get("candidates", []):
            if isinstance(candidate, dict):
                ids.extend(
                    premise["id"] for premise in candidate.get("required_conditions", [])
                    if isinstance(premise, dict) and isinstance(premise.get("id"), str)
                )
        return ids

    @staticmethod
    def _validated_source_locations(details: dict[str, Any], request: AnalysisInput):
        from .protocols import SourceLocation

        locations = []
        seen = set()
        for anchor in FirstPassAgent._anchors_in(details):
            key = (anchor.get("file"), anchor.get("line"))
            if anchor.get("validated") and key not in seen:
                seen.add(key)
                locations.append(SourceLocation(file=key[0], line=key[1], function=request.function_name))
        return locations

    @staticmethod
    def _response_metadata(raw: Any) -> dict[str, Any] | None:
        if raw is None:
            return None
        response_metadata = getattr(raw, "response_metadata", None) or {}
        usage_metadata = getattr(raw, "usage_metadata", None)
        return FirstPassAgent._jsonable(
            {
                "response_metadata": response_metadata,
                "usage_metadata": usage_metadata,
                "id": getattr(raw, "id", None),
            }
        )

    @staticmethod
    def _finish_reason(raw: Any) -> str | None:
        response_metadata = getattr(raw, "response_metadata", None) or {}
        return response_metadata.get("finish_reason")

    def _prompt_values(self, request: AnalysisInput, skeleton: dict[str, Any]) -> dict[str, str]:
        location = request.scope.locations[0] if request.scope.locations else None
        source_start = request.metadata.get("source_line_start")
        if not isinstance(source_start, int):
            source_start = 1
        return {
            "instruction": self.instruction,
            "output_contract": self.output_contract,
            "sample_id": request.sample_id,
            "function_name": request.function_name,
            "file_name": str(request.metadata.get("file_name") or "<function-local>"),
            "line_range": f"{location.line or 1}-{location.end_line or 1}" if location else "1-unknown",
            "source_line_basis": str(request.metadata.get("source_line_basis", "function_local")),
            "source_line_mapping_note": str(request.metadata.get("source_line_mapping_note") or "verified function-local coordinates"),
            "commit_id": request.scope.snapshot.commit or "unknown",
            "context_mode": str(request.metadata.get("context_mode", "function_only")),
            "source": self._number_source(request.source, source_start),
            "key_operation_index": json.dumps(
                compact_key_operations(request.metadata.get("key_operation_index", [])),
                ensure_ascii=False, sort_keys=True
            ),
            "anchor_repair_feedback": "",
            "skeleton": json.dumps(compact_skeleton(skeleton), ensure_ascii=False, sort_keys=True),
            "scope": request.scope.model_dump_json(),
        }

    @staticmethod
    def _number_source(source: str, start_line: int) -> str:
        return "\n".join(
            f"{line_number:>6} | {line}"
            for line_number, line in enumerate(source.splitlines(), start=start_line)
        )


def compact_interaction(interaction: dict[str, Any]) -> dict[str, Any]:
    """Return only the useful Agent input/output pair for experiment review."""
    prompt_values = interaction.get("input") or {}
    trace_output = _compact_trace_output(interaction.get("parsed_output"))
    if interaction.get("agent") == "query_executor" or "query" in prompt_values:
        output = interaction.get("output")
        if isinstance(output, dict):
            output = dict(output)
            raw = output.get("raw_result")
            if isinstance(raw, str) and len(raw) > 12000:
                output["raw_result"] = raw[:12000] + "\n... [trace truncated]"
        return {
            "agent": interaction.get("agent", "query_executor"),
            "status": interaction.get("status"),
            "selected_query": interaction.get("selected_query") or prompt_values.get("query"),
            "input": {"query": prompt_values.get("query")},
            "output": output,
            **({"error": interaction["error"]} if interaction.get("error") else {}),
        }
    diagnosis_input = prompt_values.get("diagnosis_input")
    if diagnosis_input is not None:
        if isinstance(diagnosis_input, str):
            diagnosis_input = _parse_json_value(diagnosis_input)
        input_record = {
            "task": prompt_values.get("instruction"),
            "response_schema": prompt_values.get("response_schema", "DisputeDraft"),
            "response_contract": prompt_values.get("output_contract"),
            "diagnosis_input": _compact_trace_diagnosis(diagnosis_input),
        }
        record = {
            "agent": interaction.get("agent"),
            "status": interaction.get("status"),
            "input": input_record,
            "output": trace_output,
        }
        error = interaction.get("error")
        if error:
            record["error"] = error[:1200] + "... [truncated]" if len(error) > 1200 else error
        return record
    planning_input = prompt_values.get("planning_input")
    if planning_input is not None:
        if isinstance(planning_input, str):
            planning_input = _parse_json_value(planning_input)
        input_record = {
            "task": prompt_values.get("instruction"),
            "response_schema": prompt_values.get("response_schema", "QueryPlanDraft"),
            "response_contract": prompt_values.get("output_contract"),
            "planning_input": _compact_trace_planning(planning_input),
        }
        record = {
            "agent": interaction.get("agent"),
            "status": interaction.get("status"),
            "input": input_record,
            "output": trace_output,
        }
        record["planning_metadata"] = {
            key: interaction.get(key)
            for key in ("planned_candidates", "errors", "candidate_repair")
            if interaction.get(key) is not None
        }
        for key in ("error", "fallback"):
            if interaction.get(key):
                value = interaction[key]
                record[key] = value[:1200] + "... [truncated]" if key == "error" and len(value) > 1200 else value
        return record
    investigation_input = prompt_values.get("investigation_input")
    if investigation_input is not None:
        if isinstance(investigation_input, str):
            investigation_input = _parse_json_value(investigation_input)
        input_record = {
            "task": prompt_values.get("instruction"),
            "response_schema": prompt_values.get("response_schema", "FactInvestigationDraft"),
            "response_contract": prompt_values.get("output_contract"),
            "investigation_input": _compact_trace_investigation(investigation_input),
        }
        record = {"agent": interaction.get("agent"), "status": interaction.get("status"),
                  "input": input_record, "output": trace_output}
        if interaction.get("error"):
            error = interaction["error"]
            record["error"] = error[:1200] + "... [truncated]" if len(error) > 1200 else error
        return record
    skeleton = prompt_values.get("skeleton")
    if isinstance(skeleton, str):
        try:
            skeleton = json.loads(skeleton)
        except json.JSONDecodeError:
            pass
    input_record = {
        "task": prompt_values.get("instruction"),
        "response_schema": "RiskAnalysisDraft" if interaction.get("kind") == "risk" else "ProtectionAnalysisDraft",
        "response_contract": prompt_values.get("output_contract"),
        "sample": {
            key: prompt_values.get(key)
            for key in ("sample_id", "function_name", "file_name", "line_range", "source_line_basis", "source_line_mapping_note", "commit_id", "context_mode")
            if key in prompt_values
        },
        "source": _source_trace_summary(prompt_values.get("source"), interaction.get("parsed_output")),
        "key_operation_index": (_parse_json_value(prompt_values.get("key_operation_index")) or [])[:30],
        "joern_skeleton": _compact_trace_skeleton(skeleton),
    }
    record = {
        "agent": interaction.get("agent"),
        "status": interaction.get("status"),
        "input": input_record,
        "output": trace_output,
    }
    error = interaction.get("error")
    if error:
        record["error"] = error[:1200] + "... [truncated]" if len(error) > 1200 else error
    fallback = interaction.get("fallback")
    if fallback:
        record["fallback"] = fallback
    repair = interaction.get("anchor_repair")
    if repair:
        record["anchor_repair"] = {
            key: repair.get(key)
            for key in ("input", "output", "status", "error")
            if key in repair
        }
    return record


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _source_trace_summary(source: Any, output: Any) -> dict[str, Any]:
    text = source if isinstance(source, str) else ""
    excerpts: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            excerpt = value.get("excerpt")
            if isinstance(excerpt, str):
                key = (value.get("file"), value.get("line"), excerpt)
                if key not in seen and len(excerpts) < 24:
                    seen.add(key)
                    excerpts.append({k: value.get(k) for k in ("file", "line", "excerpt", "basis") if value.get(k) is not None})
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(output)
    return {
        "supplied_to_agent": True,
        "line_count": len(text.splitlines()),
        "character_count": len(text),
        "evidence_excerpts_used_in_output": excerpts,
        "note": "Full numbered source was supplied to the Agent but omitted from this compact trace.",
    }


def _compact_trace_output(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    if isinstance(result.get("candidates"), list) and result["candidates"]:
        for key in (
            "security_property", "risk_operation", "target_object", "address_source",
            "access_type", "alignment_requirement", "risk_path", "required_conditions",
            "unresolved_questions", "source_anchors", "confidence",
        ):
            result.pop(key, None)
    result.pop("_interaction", None)
    return result


def _compact_trace_skeleton(skeleton: Any) -> Any:
    if not isinstance(skeleton, dict):
        return skeleton
    result = {key: skeleton.get(key) for key in ("status", "mode", "scope", "entrypoint", "node_counts", "unresolved_dependencies") if key in skeleton}
    for key in ("structural_operations", "candidate_dataflows", "calls", "controls", "source_candidates"):
        if isinstance(skeleton.get(key), list):
            result[key] = skeleton[key][:10]
    return result


def _compact_trace_diagnosis(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    result["source_refs"] = result.get("source_refs", [])[:32]
    result["relevant_skeleton"] = _compact_trace_skeleton(result.get("relevant_skeleton"))
    return result


def _compact_trace_planning(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    for key in ("risk_claim", "protection_claim"):
        claim = result.get(key)
        if isinstance(claim, dict):
            result[key] = {field: claim.get(field) for field in ("id", "kind", "text", "status", "premise_ids", "confidence") if field in claim}
    result["program_skeleton"] = _compact_trace_skeleton(result.get("program_skeleton"))
    return result


def _compact_trace_investigation(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    result["claims"] = [
        {field: claim.get(field) for field in ("id", "kind", "text", "status") if field in claim}
        for claim in result.get("claims", []) if isinstance(claim, dict)
    ]
    evidence = result.get("evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("raw_result"), str) and len(evidence["raw_result"]) > 8000:
        evidence = dict(evidence)
        evidence["raw_result"] = evidence["raw_result"][:8000] + "\n... [trace truncated]"
        result["evidence"] = evidence
    return result


def agent_io_trace(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project interactions into an ordered, input/output-only trace."""
    trace = []
    for order, interaction in enumerate(interactions, start=1):
        compact = compact_interaction(interaction)
        item = {
            "order": order,
            "agent": compact.get("agent"),
            "input": compact.get("input"),
            "output": compact.get("output"),
            "status": compact.get("status"),
        }
        if compact.get("selected_query") is not None:
            item["selected_query"] = compact["selected_query"]
        for key in ("error", "fallback"):
            if key in compact:
                item[key] = compact[key]
        trace.append(item)
    return trace


__all__ = ["FirstPassAgent", "StructuredAgentError", "agent_io_trace", "compact_interaction"]
