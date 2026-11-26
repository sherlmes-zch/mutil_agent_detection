"""Protection-evidence analyst LangChain agent."""

from __future__ import annotations

import json

from .base_agent import FirstPassAgent
from .agent_models import ProtectionAnalysisDraft
from .protocols import ClaimKind


PROTECTION_INSTRUCTION = """You are the Protection Evidence Analyst. Analyze only the supplied target
function and the same shared Joern skeleton, independently of the risk analyst. Find concrete
checks, invariants, initialization, ownership/lifetime rules, or contracts that may block a
security violation. For each, state which object it protects, on which path it applies, and
which triggering conditions it blocks. Distinguish a check's existence from proof that it
covers the complete operation. Do not invent callers, external contracts, or absent protections.
Also inspect type and alignment protections separately: a range check does not prove that a
pointer is correctly aligned for a cast target such as float or double. Report an alignment or
type-safety protection only when an exact source anchor establishes it; otherwise record the
missing alignment premise as unknown.
Return no more than 4 protections, 4 invariants, and 4 source anchors; consolidate repetitive
guards and retain only mechanisms relevant to the strongest risk.
If scope is incomplete, mark coverage unknown/partial and list the missing fact. Do not force a
safe conclusion or issue a final vulnerable/benign verdict. Do not include private reasoning,
self-dialogue, or speculative protections without a source anchor."""

PROTECTION_OUTPUT_CONTRACT = """Return one concise object matching ProtectionAnalysisDraft exactly:
{
  "summary": "what is protected and what remains uncertain, maximum 800 characters",
  "protections": [
    {
      "id": "protection-1",
      "kind": "bounds_check|validation|alignment|type_safety|ownership|lifetime|initialization|error_handling|caller_contract|other",
      "description": "what the mechanism does",
      "protected_object": "object/variable/resource",
      "path_condition": "branch and state under which it applies",
      "coverage": "complete|partial|unknown",
      "blocks_conditions": ["specific triggering conditions blocked"],
      "bypass_paths": ["uncovered branch or failure mode"],
      "source_anchors": [ANCHOR]
    }
  ],
  "invariants": [
    {"id": "protection-premise-1", "description": "invariant", "status": "supported|unknown|refuted", "evidence_refs": [ANCHOR]}
  ],
  "unresolved_questions": ["facts requiring later repository/Joern queries"],
  "source_anchors": [ANCHOR],
  "confidence": 0.0
}
ANCHOR is {"excerpt": "exact short source text copied verbatim", "basis": "function_source|joern_skeleton"}.
Do not provide a file or line number; the program resolves both from the exact excerpt. Choose a
short but unique snippet. If no exact supporting snippet exists, omit that protection or mark the
related invariant unknown rather than inventing an anchor.
Every protection must cite at least one anchor. Mark coverage=complete only when the provided
evidence covers the whole relevant operation and path; otherwise use partial or unknown. Since
you have not seen another analyst's premise IDs, describe blocked conditions in plain terms rather
than inventing IDs. Use empty arrays when no protection is evidenced; absence from supplied scope
is not evidence that no protection exists."""


class ProtectionEvidenceAgent(FirstPassAgent):
    schema = ProtectionAnalysisDraft
    kind = ClaimKind.PROTECTION
    actor = "protection_analyst"
    instruction = PROTECTION_INSTRUCTION
    output_contract = PROTECTION_OUTPUT_CONTRACT

    def _structured_repair_feedback(self, details, **_context):
        """Require concrete mechanisms when the summary claims checks exist."""
        protections = details.get("protections") or []
        summary = str(details.get("summary") or "").lower()
        cues = ("check", "guard", "bounds", "overflow", "validate", "alignment", "range")
        if protections or not any(cue in summary for cue in cues):
            return ""
        return json.dumps({
            "instruction": (
                "The summary names concrete checks or invariants but protections is empty. "
                "Convert each distinct evidenced mechanism into a ProtectionMechanismDraft. "
                "Each protection must have a unique exact source/Joern excerpt, protected object, "
                "path condition, blocked condition, and partial/unknown coverage. Do not claim "
                "alignment/type/lifetime protection unless the supplied evidence proves it."
            ),
            "missing": "protections",
            "summary": details.get("summary"),
        }, ensure_ascii=False)


__all__ = ["PROTECTION_INSTRUCTION", "PROTECTION_OUTPUT_CONTRACT", "ProtectionEvidenceAgent"]
