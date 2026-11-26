"""Risk-path analyst LangChain agent."""

from __future__ import annotations

import json
from typing import Any

from .base_agent import FirstPassAgent
from .agent_models import RiskAnalysisDraft
from .protocols import AnalysisInput, Claim, ClaimKind, SourceLocation


RISK_INSTRUCTION = """You are the Risk Path Analyst. Analyze only the supplied target function and its
shared Joern skeleton. Work independently: do not infer or invent what another analyst would
say. Do not stop at the first plausible issue. Scan the whole function for materially different
security failure modes, then rank at most 3 concrete candidates. Prefer distinct root causes over
multiple descriptions of the same issue. For every candidate, identify the exact dangerous
operation, affected object, security property, causal path, and necessary trigger conditions.
Return at most 4 path steps, 4 trigger conditions, and 5 source anchors per candidate. Keep each
candidate independently falsifiable and distinguish code facts from assumptions.
Classify every candidate as direct_violation when supplied function facts establish the unsafe
operation and property conflict, conditional_interprocedural when a caller/callee/type/allocation
contract is necessary, or speculative when even the local trigger remains hypothetical. This
classification controls investigation priority, not the final verdict.
Consider, when applicable: out-of-bounds access, integer overflow/sign errors, lifetime/ownership,
initialization, pointer casts and dereferences, alignment, type width/representation, state/error
handling, and concurrency. These are review categories, not evidence that a bug exists.
For every pointer cast or typed dereference, explicitly inspect the source address expression,
the target access type, whether the address is proven suitably aligned, and whether the operation
can be undefined even when the address remains inside the allocation. A bounds check is not an
alignment proof. When this pattern exists, consider a distinct misaligned_access or type_safety
candidate instead of collapsing it into an out-of-bounds candidate.
For every explicit integer conversion, especially to char/short/unsigned char and before bitwise
operations or shifts, identify the source expression/type when available, target type, signedness,
representable-range requirement, and whether local code proves the value is representable. Preserve
nested casts. The program-generated key-operation index is an exhaustive navigation aid: if it contains
`narrow_integer_conversion` or an integer conversion to char/short, you MUST emit a distinct candidate
for that conversion (unless the complete function contains an exact range proof, which must be cited).
Do not hide a conversion under a buffer or pointer candidate. Use direct_violation only when local
language/operation facts establish the conflict; otherwise classify it conditional_interprocedural
and state the exact range, type-width, or external-contract premise to investigate.
Do not treat a risky API or a dataset label as evidence. If the source contradicts a hypothesis,
omit that candidate or state the specific unresolved condition; do not preserve it as a confirmed
claim. Do not include private reasoning, self-dialogue, or a final vulnerable/benign verdict.
Facts absent from this function/skeleton remain unknown; do not assert that absent protections
or callers do not exist."""

RISK_OUTPUT_CONTRACT = """Return one concise object matching RiskAnalysisDraft exactly:
{
  "candidate_found": true,
  "summary": "brief synthesis of the ranked candidates, maximum 1200 characters",
  "candidates": [
    {
      "candidate_id": "risk-1",
      "candidate_class": "direct_violation|conditional_interprocedural|speculative",
      "priority": 1,
      "summary": "one conditional risk claim, maximum 800 characters",
      "security_property": "buffer_overflow|use_after_free|misaligned_access|... or null",
      "risk_operation": "exact operation and expression",
      "target_object": "object/variable/resource",
      "address_source": "expression producing the address, or null",
      "access_type": "type used for the load/store, or null",
      "alignment_requirement": "required alignment and whether it is established, or null",
      "conversion_source_type": "source expression/type for a numeric conversion, or null",
      "conversion_target_type": "target numeric type, or null",
      "representation_requirement": "required representable range and whether established, or null",
      "risk_path": [
        {"order": 1, "operation": "operation", "object": "same affected object", "effect": "state/data effect", "source_anchors": [ANCHOR]}
      ],
      "required_conditions": [
        {"id": "risk-1-premise-1", "description": "necessary condition", "status": "supported|unknown|refuted", "evidence_refs": [ANCHOR]}
      ],
      "unresolved_questions": ["facts requiring later repository/Joern queries"],
      "source_anchors": [ANCHOR],
      "confidence": 0.0
    }
  ]
}
ANCHOR is {"excerpt": "exact short source text copied verbatim", "basis": "function_source|joern_skeleton"}.
Do not provide a file or line number; the program resolves both from the exact excerpt. Choose a
short but unique snippet from the cited source. If no exact supporting snippet exists, remove that
candidate rather than inventing an anchor.
Sort candidates by evidence strength and likely impact, but do not inflate confidence to fill all
3 slots. Each candidate must have a causal path, trigger conditions, and exact source anchors.
Every supported condition must cite at least one anchor; otherwise mark it unknown. If no concrete
candidate is found, return candidate_found=false, candidates=[], and explain briefly in summary.
Do not put rejected hypotheses or chain-of-thought in any field."""


class RiskPathAgent(FirstPassAgent):
    schema = RiskAnalysisDraft
    kind = ClaimKind.RISK
    actor = "risk_analyst"
    instruction = RISK_INSTRUCTION
    output_contract = RISK_OUTPUT_CONTRACT

    @staticmethod
    def _conversion_operations(
        skeleton: dict[str, Any] | None,
        request: AnalysisInput | None = None,
    ) -> list[dict[str, Any]]:
        rows = (skeleton or {}).get("key_operation_index", [])
        if not rows and request is not None:
            rows = request.metadata.get("key_operation_index", [])
        if not isinstance(rows, list):
            return []
        result = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            categories = {str(value) for value in row.get("categories", [])}
            target = str(row.get("conversion_target_type") or "").lower()
            if "narrow_integer_conversion" in categories or (
                "integer_conversion" in categories
                and any(token in target for token in ("char", "short"))
            ):
                result.append(row)
        return result

    def _structured_repair_feedback(
        self, details: dict[str, Any], *, request: AnalysisInput | None = None,
        skeleton: dict[str, Any] | None = None,
    ) -> str:
        conversions = self._conversion_operations(skeleton, request)
        if not conversions:
            return ""
        candidates = details.get("candidates") or []
        covered = any(
            isinstance(candidate, dict)
            and (
                candidate.get("conversion_target_type")
                or str(candidate.get("security_property") or "").lower()
                in {"integer_conversion", "numeric_range", "signedness", "type_safety"}
            )
            for candidate in candidates
        )
        if covered:
            return ""
        return json.dumps({
            "instruction": (
                "The program-generated operation index contains a narrow integer conversion that is "
                "missing from candidates. Add one independently falsifiable candidate for the conversion. "
                "Preserve any existing candidates, use candidate_class=conditional_interprocedural when "
                "type/range/build facts are external, and cite the exact conversion line. Do not merge "
                "this into a buffer or pointer candidate."
            ),
            "required_conversion_operations": [
                {
                    "line": row.get("line"),
                    "code": row.get("code"),
                    "conversion_source_expression": row.get("conversion_source_expression"),
                    "conversion_target_type": row.get("conversion_target_type"),
                    "representation_requirement": row.get("representation_requirement"),
                }
                for row in conversions[:4]
            ],
        }, ensure_ascii=False)

    def __call__(self, request: AnalysisInput, skeleton: dict[str, Any]) -> Claim:
        claim = super().__call__(request, skeleton)
        conversions = self._conversion_operations(skeleton, request)
        if not conversions:
            return claim
        analysis = dict(claim.analysis)
        candidates = [
            dict(candidate) for candidate in (analysis.get("candidates") or [])
            if isinstance(candidate, dict)
        ]
        has_conversion = any(
            candidate.get("conversion_target_type")
            or str(candidate.get("security_property") or "").lower()
            in {"integer_conversion", "numeric_range", "signedness", "type_safety"}
            for candidate in candidates
        )
        if has_conversion:
            return claim

        # Bounded protocol repair: a program-generated narrow-conversion row
        # cannot disappear merely because the model ranked other risks higher.
        row = conversions[0]
        line = row.get("line")
        code = str(row.get("code") or "conversion operation")
        target_type = str(row.get("conversion_target_type") or "integer type")
        source_expr = str(row.get("conversion_source_expression") or "converted expression")
        file_name = str(request.metadata.get("file_name") or "<function>")
        anchor = {
            "file": file_name,
            "line": line,
            "excerpt": code,
            "basis": "function_source",
            "validated": True,
            "validation_note": "program-generated key-operation index line",
        }
        candidate = {
            "candidate_id": f"risk-conversion-{line or 'unknown'}",
            "candidate_class": "conditional_interprocedural",
            "priority": 1,
            "summary": (
                f"The function narrows {source_expr} to {target_type}; the source range and effective "
                "type width must be established before the value is used in bitwise or indexing logic."
            ),
            "security_property": "integer_conversion",
            "risk_operation": code,
            "target_object": source_expr,
            "address_source": None,
            "access_type": target_type,
            "alignment_requirement": None,
            "conversion_source_type": source_expr,
            "conversion_target_type": target_type,
            "representation_requirement": row.get("representation_requirement") or (
                "source value must be representable in the target integer type; inspect signedness, width, "
                "nested casts, and subsequent shifts/masks"
            ),
            "risk_path": [{
                "order": 1,
                "operation": code,
                "object": source_expr,
                "effect": "narrowed representation may change before the subsequent bitwise operation",
                "source_anchors": [anchor],
            }],
            "required_conditions": [{
                "id": f"risk-conversion-{line or 'unknown'}-premise-1",
                "description": "The source value is not proven representable in the target integer type.",
                "status": "unknown",
                "evidence_refs": [],
            }],
            "unresolved_questions": [
                "What are the source and target integer widths and signedness for this conversion?",
                "Does the surrounding operation require the original value range to be preserved?",
            ],
            "source_anchors": [anchor],
            "confidence": 0.25,
            "evidence_status": "anchors_verified",
        }
        candidates.append(candidate)
        candidates.sort(key=lambda item: (
            0 if item.get("candidate_id") == candidate["candidate_id"] else 1,
            int(item.get("priority") or 99),
        ))
        candidates = candidates[:3]
        for priority, item in enumerate(candidates, start=1):
            item["priority"] = priority
        analysis["candidates"] = candidates
        analysis["candidate_found"] = True
        analysis["evidence_status"] = "anchors_verified"
        premise_ids = list(claim.premise_ids)
        for item in candidates:
            for premise in item.get("required_conditions", []):
                if premise.get("id") not in premise_ids:
                    premise_ids.append(premise["id"])
        source_anchors = list(claim.source_anchors)
        if isinstance(line, int) and not any(anchor.line == line and anchor.file == file_name for anchor in source_anchors):
            source_anchors.append(SourceLocation(
                file=file_name, function=request.function_name, line=line,
            ))
        return claim.model_copy(update={
            "analysis": analysis, "premise_ids": premise_ids,
            "source_anchors": source_anchors,
            "text": str(analysis.get("summary") or claim.text),
        })


__all__ = ["RISK_INSTRUCTION", "RISK_OUTPUT_CONTRACT", "RiskPathAgent"]
