"""Structured LangChain disagreement diagnosis."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from .agent_models import (
    DisputeClaimRelationDraft,
    DisputeDraft,
    DisputePremiseDraft,
    InvestigationTargetDraft,
)
from .protocols import (
    DiagnosisInput, Dispute, DisputeClaimRelation, DisputeFact,
    DisputePremise, InvestigationTarget, QueryKind,
)

DIAGNOSIS_INSTRUCTION = """Align risk candidates with protection mechanisms and identify
conclusion-dependent premises. Do not issue a vulnerable/benign verdict, create new claims, or
execute a query. A protection addresses a risk only when object, security property, and path
match the candidate's necessary conditions. Bounds, alignment, type, lifetime, initialization,
and authorization are distinct properties. Cite only supplied reference IDs. Separate facts
from assumptions and unknowns. Investigation targets must ask answerable questions whose
outcomes distinguish competing explanations. Do not include private reasoning.
Respect candidate_class: direct_violation should focus on falsifying the explicit local property
conflict; conditional_interprocedural should identify the exact external contract needed;
speculative should expose the first unsupported trigger rather than multiplying downstream queries.

Return JSON only. Do not use Markdown fences, prose before/after JSON, or chain-of-thought.
Keep the response compact: at most 5 facts, 8 relations, 10 premises, and 5 investigation
targets. Use [] for empty collections. Every investigation target must reference a premise ID
declared in the same response. Every unresolved premise must include claim_refs, and every
investigation target must include suggested_query_kinds (use source_search when uncertain).
If any risk candidate is present and alignment is not aligned, emit at least one claim_relation,
one unknown/unresolved premise, and one investigation_target; never use summary as a substitute
for these arrays. If protections is empty, represent the absence of a concrete protection as a
does_not_address relation and keep coverage unknown. When protections are supplied, every claim
relation must list the concrete protection IDs it considered; an empty protection_ids array is
invalid. If several protections could apply, list all of them and use unknown/partial coverage.
Relation IDs and evidence IDs must be copied exactly from input."""


class DisagreementDiagnostician:
    actor = "disagreement_diagnostician"
    output_contract = (
        "Return exactly one JSON object matching this contract; no Markdown or explanatory text:\n"
        '{"diagnosis_status":"complete|partial|insufficient_evidence",'
        '"alignment_status":"aligned|partially_aligned|misaligned|unresolved",'
        '"summary":"short summary", "facts":[], "claim_relations":[], "premises":[], '
        '"investigation_targets":[]}\n'
        "Each relation must use an input risk candidate ID and protection ID. Each evidence_refs "
        "and claim_refs value must be an input ID. Investigation targets must reference a premise "
        "declared in premises. Every unresolved premise must list at least one claim_refs ID, and "
        "every investigation target must list at least one suggested_query_kinds value. Do not "
        "add fields or issue a vulnerable/benign verdict."
    )

    def __init__(self, model: Any):
        self.last_interaction: dict[str, Any] | None = None
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{instruction}\n\nOutput contract: {output_contract}"),
                ("human", "Diagnosis input JSON:\n{diagnosis_input}"),
            ]
        )
        self.chain = prompt | model.with_structured_output(
            DisputeDraft, method="function_calling", include_raw=True
        )
        self.json_mode_chain = prompt | model.with_structured_output(
            DisputeDraft, method="json_mode", include_raw=True
        )

    def __call__(self, request: DiagnosisInput) -> Dispute:
        payload = request.model_dump(mode="json")
        interaction = {
            "agent": self.actor,
            "status": "started",
            "input": {
                "instruction": DIAGNOSIS_INSTRUCTION,
                "response_schema": "DisputeDraft",
                "output_contract": self.output_contract,
                "diagnosis_input": payload,
            },
            "parsed_output": None,
            "error": None,
        }
        self.last_interaction = interaction
        values = {
            "instruction": DIAGNOSIS_INSTRUCTION,
            "output_contract": self.output_contract,
            "diagnosis_input": json.dumps(payload, ensure_ascii=False),
        }
        errors: list[str] = []
        for attempt, chain in (("function_calling", self.chain), ("json_mode", self.json_mode_chain)):
            try:
                result = chain.invoke(values)
                draft = self._extract_draft(result)
                draft, reference_repairs = self._sanitize_references(draft, request)
                draft = self._normalize_investigation_contract(draft, request)
                draft = self._ensure_diagnostic_completeness(draft, request)
                dispute = self._to_dispute(draft, request)
                interaction.update(
                    {
                        "status": "parsed",
                        "parsed_output": draft.model_dump(mode="json"),
                        "attempts": errors + [attempt],
                        "reference_repairs": reference_repairs,
                    }
                )
                return dispute
            except Exception as exc:
                errors.append(f"{attempt}: {type(exc).__name__}: {exc}")
        error = " | ".join(errors)
        interaction.update({"status": "failed", "error": error, "attempts": ["function_calling", "json_mode"]})
        return self._fallback(request, error)

    @staticmethod
    def _extract_draft(result: Any) -> DisputeDraft:
        parsed = result.get("parsed") if isinstance(result, dict) else result
        if isinstance(parsed, DisputeDraft):
            return parsed
        if isinstance(parsed, BaseModel):
            return DisputeDraft.model_validate(parsed.model_dump())
        if isinstance(parsed, dict):
            return DisputeDraft.model_validate(parsed)
        raw = result.get("raw") if isinstance(result, dict) else None
        content = getattr(raw, "content", None)
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if isinstance(content, str):
            content = content.strip()
            fence = chr(96) * 3
            if content.startswith(fence):
                content = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*|\s*" + re.escape(fence) + r"$", "", content)
            try:
                return DisputeDraft.model_validate_json(content)
            except Exception:
                # Some OpenAI-compatible endpoints prepend a short sentence or append a note.
                # Recover only the first complete JSON object; schema validation still applies.
                start, end = content.find("{"), content.rfind("}")
                if start >= 0 and end > start:
                    try:
                        return DisputeDraft.model_validate_json(content[start : end + 1])
                    except Exception as exc:
                        raise ValueError("response did not contain valid DisputeDraft JSON") from exc
                raise ValueError("response did not contain valid DisputeDraft JSON")
        raise ValueError("response did not contain a parsed DisputeDraft")

    @staticmethod
    def _sanitize_references(
        draft: DisputeDraft, request: DiagnosisInput
    ) -> tuple[DisputeDraft, list[str]]:
        """Drop invented optional references without discarding valid diagnosis content.

        A relation to an unknown risk candidate remains a hard error because its
        semantic target is ambiguous. Claim/evidence refs, on the other hand, can
        be safely filtered and reconstructed from the bounded input below.
        """
        candidates = {item.candidate_id for item in request.risk_candidates}
        aliases = {
            str(item.model_entity_id): item.candidate_id
            for item in request.risk_candidates
            if getattr(item, "model_entity_id", None)
        }
        protections = {item.protection_id for item in request.protections}
        allowed = {
            request.risk_claim_id, request.protection_claim_id, *candidates, *protections,
            *(item.ref_id for item in request.source_refs),
        }
        repairs: list[str] = []
        unresolved_candidate_ids: list[str] = []
        relations = []
        for relation in draft.claim_relations:
            candidate_id = relation.risk_candidate_id
            if candidate_id not in candidates:
                # Providers may echo a transient claim_* ID from an earlier
                # pass. Resolve only safe aliases; otherwise discard this
                # relation while retaining the rest of the diagnosis.
                mapped = aliases.get(candidate_id)
                if mapped is None and (candidate_id.startswith(("claim_", "candidate_")) or len(candidate_id) >= 24):
                    if len(request.risk_candidates) == 1:
                        mapped = request.risk_candidates[0].candidate_id
                    else:
                        match = re.search(r"(?:risk|candidate)[-_]?(\d+)$", candidate_id)
                        if match:
                            index = int(match.group(1)) - 1
                            if 0 <= index < len(request.risk_candidates):
                                mapped = request.risk_candidates[index].candidate_id
                if mapped is None:
                    repairs.append(f"dropped unknown risk candidate {candidate_id}")
                    unresolved_candidate_ids.append(candidate_id)
                    continue
                repairs.append(f"mapped transient risk candidate {candidate_id} to {mapped}")
                relation = relation.model_copy(update={"risk_candidate_id": mapped})
            valid_protection_ids = [item for item in relation.protection_ids if item in protections]
            valid_evidence_refs = [item for item in relation.evidence_refs if item in allowed]
            if len(valid_protection_ids) != len(relation.protection_ids):
                repairs.append(f"removed unknown protection IDs from relation {relation.risk_candidate_id}")
            if len(valid_evidence_refs) != len(relation.evidence_refs):
                repairs.append(f"removed unknown evidence refs from relation {relation.risk_candidate_id}")
            relations.append(relation.model_copy(update={
                "protection_ids": valid_protection_ids,
                "evidence_refs": valid_evidence_refs,
            }))
        # Preserve the previous hard failure for a wholly fabricated relation,
        # but do not discard otherwise useful diagnoses containing one bad row.
        if unresolved_candidate_ids and not relations:
            raise ValueError(f"unknown risk candidate: {unresolved_candidate_ids[0]}")

        facts = []
        for fact in draft.facts:
            refs = [item for item in fact.evidence_refs if item in allowed]
            if len(refs) != len(fact.evidence_refs):
                repairs.append(f"removed unknown evidence refs from fact {fact.id}")
            facts.append(fact.model_copy(update={"evidence_refs": refs}))

        premises = []
        for premise in draft.premises:
            claim_refs = [item for item in premise.claim_refs if item in allowed]
            evidence_refs = [item for item in premise.evidence_refs if item in allowed]
            if len(claim_refs) != len(premise.claim_refs):
                repairs.append(f"removed unknown claim refs from premise {premise.id}")
            if len(evidence_refs) != len(premise.evidence_refs):
                repairs.append(f"removed unknown evidence refs from premise {premise.id}")
            premises.append(premise.model_copy(update={
                "claim_refs": claim_refs,
                "evidence_refs": evidence_refs,
            }))
        return draft.model_copy(update={"claim_relations": relations, "facts": facts, "premises": premises}), repairs

    @staticmethod
    def _normalize_investigation_contract(
        draft: DisputeDraft, request: DiagnosisInput
    ) -> DisputeDraft:
        """Fill safe query metadata when a provider omits it."""
        candidate_ids = [item.candidate_id for item in request.risk_candidates]
        protection_ids = [item.protection_id for item in request.protections]
        premises = []
        for premise in draft.premises:
            if premise.claim_refs:
                premises.append(premise)
                continue
            text = f"{premise.id} {premise.description}".lower()
            if "protection" in text or "invariant" in text:
                refs = [request.protection_claim_id, *protection_ids]
            elif "alignment" in text:
                refs = [request.risk_claim_id, request.protection_claim_id]
            elif any(word in text for word in ("risk", "range")):
                refs = [request.risk_claim_id, *candidate_ids]
            else:
                refs = [request.risk_claim_id, request.protection_claim_id]
            premises.append(premise.model_copy(update={"claim_refs": list(dict.fromkeys(refs))}))

        targets = []
        for target in draft.investigation_targets:
            if target.suggested_query_kinds:
                targets.append(target)
                continue
            text = f"{target.question} {target.evidence_requirement}".lower()
            kinds = [QueryKind.SOURCE_SEARCH]
            if any(word in text for word in ("path", "reachable", "dominates", "branch")):
                kinds.append(QueryKind.PATH_REACHABILITY)
            if any(word in text for word in ("align", "cast", "derefer", "type", "cpg", "operator")):
                kinds.append(QueryKind.CPG)
            if any(word in text for word in ("offset", "range", "bound", "size", "overflow", "allocation")):
                kinds.append(QueryKind.NUMERIC_RANGE)
            targets.append(target.model_copy(update={"suggested_query_kinds": list(dict.fromkeys(kinds))[:4]}))
        return draft.model_copy(update={"premises": premises, "investigation_targets": targets})

    @classmethod
    def _ensure_diagnostic_completeness(
        cls, draft: DisputeDraft, request: DiagnosisInput
    ) -> DisputeDraft:
        """Turn a summary-only response into an explicit premise/target ledger.

        Providers often return a useful summary while leaving optional arrays empty.  The
        workflow cannot plan a query from prose, so create only conservative, claim-linked
        unknowns from the already supplied candidate/protection records.
        """
        relations = list(draft.claim_relations)
        premises = list(draft.premises)
        targets = list(draft.investigation_targets)
        premise_ids = {item.id for item in premises}
        relation_ids = {item.risk_candidate_id for item in relations}
        protections = {item.protection_id for item in request.protections}

        # Models sometimes describe the protection comparison correctly but omit the IDs.
        # Repair that protocol loss using transparent lexical matching; if no unique match is
        # possible, retain all supplied protections and leave semantic coverage unchanged.
        if protections:
            protection_records = {item.protection_id: item for item in request.protections}
            normalized_relations = []
            for relation in relations:
                if relation.protection_ids:
                    normalized_relations.append(relation)
                    continue
                candidate = next(
                    (item for item in request.risk_candidates if item.candidate_id == relation.risk_candidate_id),
                    None,
                )
                context = " ".join(str(value or "") for value in (
                    relation.explanation,
                    candidate.security_property if candidate else None,
                    candidate.target_object if candidate else None,
                    candidate.risk_operation if candidate else None,
                )).lower()
                scores = []
                for protection_id, protection in protection_records.items():
                    text = " ".join(str(value or "") for value in (
                        protection.kind, protection.description, protection.protected_object,
                        *protection.blocks_conditions,
                    )).lower()
                    overlap = sum(1 for token in set(context.split()) if len(token) > 3 and token in text)
                    scores.append((overlap, protection_id))
                best = max((score for score, _ in scores), default=0)
                selected = [protection_id for score, protection_id in scores if score == best and best > 0]
                if not selected:
                    selected = list(protections)
                normalized_relations.append(relation.model_copy(update={"protection_ids": selected}))
            relations = normalized_relations

        for candidate in request.risk_candidates[:5]:
            if candidate.candidate_id not in relation_ids:
                relations.append(DisputeClaimRelationDraft(
                    risk_candidate_id=candidate.candidate_id,
                    protection_ids=list(protections),
                    relation="does_not_address" if not protections else "unresolved",
                    object_relation="unknown", property_relation="unknown",
                    path_coverage="unknown",
                    explanation=(
                        "No concrete protection mechanism was supplied for this risk candidate."
                        if not protections else
                        "Protection coverage and security-property alignment remain unverified."
                    ),
                ))
            # Preserve an explicit model premise ledger; synthesize candidate-level
            # premises only when the provider omitted the ledger entirely.
            if premises:
                continue
            unknowns = [q for q in candidate.unresolved_questions[:3] if q.strip()]
            if not unknowns:
                unknowns = [
                    f"Whether the required condition for {candidate.candidate_id} holds on every reachable path."
                ]
            for index, question in enumerate(unknowns):
                premise_id = f"{candidate.candidate_id}-premise-{index + 1}"
                if premise_id not in premise_ids:
                    premise = DisputePremiseDraft(
                        id=premise_id,
                        description=question,
                        relation="unresolved", status="unknown",
                        claim_refs=[request.risk_claim_id, candidate.candidate_id],
                        decisive=index == 0,
                    )
                    premises.append(premise)
                    premise_ids.add(premise_id)

        if not premises:
            premise_id = "diagnosis-alignment-premise"
            premises.append(DisputePremiseDraft(
                id=premise_id,
                description="Whether the supplied protection evidence covers the strongest risk path.",
                relation="unresolved", status="unknown",
                claim_refs=[request.risk_claim_id, request.protection_claim_id], decisive=True,
            ))
            premise_ids.add(premise_id)

        target_ids = {target.premise_id for target in targets}
        for premise in premises:
            if premise.status != "unknown" and premise.relation != "unresolved":
                continue
            if premise.id in target_ids:
                continue
            kinds = cls._query_kinds_for_text(premise.description)
            targets.append(InvestigationTargetDraft(
                premise_id=premise.id,
                question=f"Can the current source and Joern evidence establish: {premise.description}",
                discriminating_outcomes=["condition established", "condition disproved", "condition remains unknown"],
                suggested_query_kinds=kinds,
                evidence_requirement="Exact source/Joern anchors and the path or type relation needed to establish the condition.",
            ))
            target_ids.add(premise.id)

        status = draft.alignment_status
        if not relations and status != "aligned":
            status = "partially_aligned"
        return draft.model_copy(update={
            "alignment_status": status,
            "diagnosis_status": "partial" if draft.diagnosis_status == "complete" and premises else draft.diagnosis_status,
            "claim_relations": relations[:12], "premises": premises[:16], "investigation_targets": targets[:8],
        })

    @staticmethod
    def _query_kinds_for_text(text: str) -> list[QueryKind]:
        lowered = text.lower()
        kinds = [QueryKind.SOURCE_SEARCH]
        if any(word in lowered for word in ("align", "cast", "derefer", "type", "pointer")):
            kinds.append(QueryKind.CPG)
        if any(word in lowered for word in ("path", "reachable", "branch", "dominates")):
            kinds.append(QueryKind.PATH_REACHABILITY)
        if any(word in lowered for word in ("range", "bound", "offset", "size", "overflow", "component")):
            kinds.append(QueryKind.NUMERIC_RANGE)
        return list(dict.fromkeys(kinds))[:4]

    @staticmethod
    def _to_dispute(draft: DisputeDraft, request: DiagnosisInput) -> Dispute:
        unresolved = [
            item.id for item in draft.premises
            if item.status == "unknown" or item.relation == "unresolved"
        ]
        decisive = [item.id for item in draft.premises if item.decisive]
        competing = [
            item.description for item in draft.premises
            if item.relation in {"contradicts", "limits"}
        ]
        return Dispute(
            scope=request.scope,
            risk_claim_id=request.risk_claim_id,
            protection_claim_id=request.protection_claim_id,
            diagnosis_status=draft.diagnosis_status,
            alignment_status=draft.alignment_status,
            summary=draft.summary,
            facts=[DisputeFact.model_validate(item.model_dump()) for item in draft.facts],
            claim_relations=[
                DisputeClaimRelation.model_validate(item.model_dump())
                for item in draft.claim_relations
            ],
            premises=[DisputePremise.model_validate(item.model_dump()) for item in draft.premises],
            investigation_targets=[
                InvestigationTarget.model_validate(item.model_dump())
                for item in draft.investigation_targets
            ],
            shared_facts=[item.statement for item in draft.facts],
            competing_premises=competing,
            unresolved_premises=unresolved,
            decisive_premises=decisive,
            aligned=draft.alignment_status == "aligned",
            alignment_notes=draft.summary,
        )

    @staticmethod
    def _fallback(request: DiagnosisInput, error: str) -> Dispute:
        premise_id = "diagnosis-unavailable"
        return Dispute(
            scope=request.scope,
            risk_claim_id=request.risk_claim_id,
            protection_claim_id=request.protection_claim_id,
            diagnosis_status="insufficient_evidence",
            alignment_status="unresolved",
            summary="Claims could not be reliably aligned; preserve their relation as unresolved.",
            premises=[
                DisputePremise(
                    id=premise_id,
                    description="A reliable alignment of risk candidates and protections is unavailable.",
                    relation="unresolved",
                    status="unknown",
                    claim_refs=[request.risk_claim_id, request.protection_claim_id],
                    decisive=True,
                )
            ],
            unresolved_premises=[premise_id],
            decisive_premises=[premise_id],
            alignment_notes="Diagnosis failed; no semantic relation is asserted.",
            diagnosis_error=error,
        )


__all__ = ["DIAGNOSIS_INSTRUCTION", "DisagreementDiagnostician"]
