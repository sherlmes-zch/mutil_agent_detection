"""Compatibility facade for the independent first-pass analyst agents."""

from __future__ import annotations

from typing import Any

from .agent_models import ClaimDraft, PremiseDraft, ProtectionDraft
from .base_agent import FirstPassAgent
from .protection_agent import (
    PROTECTION_INSTRUCTION,
    PROTECTION_OUTPUT_CONTRACT,
    ProtectionEvidenceAgent,
)
from .protocols import ClaimKind
from .risk_agent import RISK_INSTRUCTION, RISK_OUTPUT_CONTRACT, RiskPathAgent


class StructuredClaimAgent(FirstPassAgent):
    """Backward-compatible role selector for the original public API."""

    def __init__(self, model: Any, *, kind: ClaimKind):
        selected = RiskPathAgent if kind == ClaimKind.RISK else ProtectionEvidenceAgent
        self._selected = selected(model)
        self.kind = self._selected.kind
        self.actor = self._selected.actor
        self.instruction = self._selected.instruction
        self.output_contract = self._selected.output_contract
        self.prompt = self._selected.prompt
        self.chain = self._selected.chain

    def __call__(self, request, skeleton):
        return self._selected(request, skeleton)


def make_first_pass_handlers(model: Any):
    """Return independent risk and protection LangChain handlers."""

    return RiskPathAgent(model), ProtectionEvidenceAgent(model)


__all__ = [
    "ClaimDraft",
    "PremiseDraft",
    "ProtectionDraft",
    "StructuredClaimAgent",
    "RiskPathAgent",
    "ProtectionEvidenceAgent",
    "RISK_INSTRUCTION",
    "RISK_OUTPUT_CONTRACT",
    "PROTECTION_INSTRUCTION",
    "PROTECTION_OUTPUT_CONTRACT",
    "make_first_pass_handlers",
]
