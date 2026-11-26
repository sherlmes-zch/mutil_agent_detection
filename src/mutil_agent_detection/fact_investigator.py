"""Independent Agent that interprets executed evidence without issuing a verdict."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from .agent_models import FactInvestigationDraft
from .protocols import (
    FactInvestigationInput, FactInvestigationResult, FactStatement, PremiseUpdate,
    RiskCandidateAssessment, CandidateConditionAssessment, ProtectionAssessment, CallerContract,
)

FACT_INVESTIGATOR_INSTRUCTION = (
    "Interpret only the supplied executed query and evidence. Do not execute tools, invent facts, "
    "or issue a vulnerable/benign verdict. Select outcome_label only from declared outcomes. State "
    "objective facts with supplied evidence IDs. Update only target premise IDs; use unknown when "
    "evidence is absent, approximate, out of scope, or inconclusive. Joern clues are not proofs. "
    "For each supplied risk candidate, assess whether this query's evidence supports, refutes, or "
    "leaves it unknown; do not treat a premise update alone as proof of the whole candidate. "
    "For each risk candidate, also return candidate_condition_assessments for every necessary premise "
    "that this query addresses. A candidate is supported only when all of its necessary premises are supported; "
    "partial support must remain partial/unknown. "
    "When evidence.structured_facts.result_kind is caller_contract (or caller_contract_query is true), "
    "you MUST extract one caller_contracts item per supplied call site whenever its exact source slice permits it. "
    "Each item must include caller_function, call_site, call_site_location when available, argument_expression, "
    "allocation_expression, read_length_expression, required_size_expression, short_read_guard, and extent_relation. "
    "Use unknown for a field that the slice does not establish; do not omit the caller contract merely because "
    "allocation or length is unknown. "
    "For definition queries, signature_or_fragment_only is insufficient to support implementation semantics; "
    "mark the corresponding premise unknown until a complete function or macro block is supplied. "
    "For each supplied protection, assess effective only if evidence shows it covers the same object, operation, "
    "and path; otherwise use ineffective or unknown. Cite only this query's evidence ID. "
    "Return at most 6 objective facts and keep rationales concise. Return JSON only."
)


class FactInvestigator:
    actor = "fact_investigator"
    output_contract = (
        "Return FactInvestigationDraft JSON only. query_id must match input; outcome_label must be "
        "a declared outcome or null. Evidence refs must be supplied IDs and premise updates must "
        "target only query premises. Include risk_assessments for relevant risk candidates and "
        "candidate_condition_assessments for every necessary premise addressed by the evidence; "
        "do not mark a candidate supported when any of its necessary premises remains unknown. "
        "protection_assessments for relevant protections when the evidence bears on them. Every "
        "non-unknown assessment must cite this query's evidence ID. Never return a final verdict."
    )

    def __init__(self, model: Any):
        self.last_interaction: dict[str, Any] | None = None
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{instruction}\n\n{contract}"),
            ("human", "Investigation input JSON:\n{input}"),
        ])
        self.chain = prompt | model.with_structured_output(FactInvestigationDraft, method="function_calling", include_raw=True)
        self.json_chain = prompt | model.with_structured_output(FactInvestigationDraft, method="json_mode", include_raw=True)

    def __call__(self, investigation: FactInvestigationInput) -> FactInvestigationResult:
        payload = investigation.model_dump(mode="json")
        self.last_interaction = {"agent": self.actor, "status": "started", "input": {
            "instruction": FACT_INVESTIGATOR_INSTRUCTION, "response_schema": "FactInvestigationDraft",
            "output_contract": self.output_contract, "investigation_input": payload},
            "parsed_output": None, "error": None}
        values = {"instruction": FACT_INVESTIGATOR_INSTRUCTION, "contract": self.output_contract,
                  "input": json.dumps(payload, ensure_ascii=False)}
        draft = None
        errors: list[str] = []
        for name, chain in (("function_calling", self.chain), ("json_mode", self.json_chain)):
            try:
                draft = self._extract(chain.invoke(values))
                if draft is not None:
                    break
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if draft is None:
            error = " | ".join(errors) or "empty structured response"
            self.last_interaction.update({"status": "failed", "error": error})
            return self._fallback(investigation, error)
        try:
            result = self._validate_and_convert(draft, investigation)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.last_interaction.update({"status": "failed", "error": error,
                                          "parsed_output": draft.model_dump(mode="json")})
            return self._fallback(investigation, error)
        self.last_interaction.update({"status": "parsed", "parsed_output": draft.model_dump(mode="json"), "errors": errors})
        return result

    @staticmethod
    def _extract(result: Any) -> FactInvestigationDraft | None:
        parsed = result.get("parsed") if isinstance(result, dict) else result
        if isinstance(parsed, FactInvestigationDraft):
            return parsed
        if isinstance(parsed, BaseModel):
            return FactInvestigationDraft.model_validate(parsed.model_dump())
        if isinstance(parsed, dict):
            return FactInvestigationDraft.model_validate(parsed)
        raw = result.get("raw") if isinstance(result, dict) else None
        tool_calls = getattr(raw, "tool_calls", None)
        if isinstance(tool_calls, list):
            for call in tool_calls:
                args = call.get("args") if isinstance(call, dict) else None
                if isinstance(args, dict):
                    try:
                        return FactInvestigationDraft.model_validate(args)
                    except Exception:
                        continue
        content = getattr(raw, "content", "")
        if isinstance(content, list):
            content = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        if isinstance(content, str):
            start, end = content.find("{"), content.rfind("}")
            if start >= 0 and end > start:
                return FactInvestigationDraft.model_validate_json(content[start:end + 1])
        return None

    @classmethod
    def _validate_and_convert(cls, draft: FactInvestigationDraft, value: FactInvestigationInput) -> FactInvestigationResult:
        query, evidence = value.query, value.evidence
        premise_ids = set(query.target_premise_ids)
        candidate_ids = {item.candidate_id for item in value.risk_candidates}
        protection_ids = {item.protection_id for item in value.protections}
        candidate_aliases = {
            str(item.model_entity_id): item.candidate_id
            for item in value.risk_candidates if item.model_entity_id
        }
        protection_aliases = {
            str(item.model_entity_id): item.protection_id
            for item in value.protections if item.model_entity_id
        }
        # The ledger owns evidence IDs. Older prompts sometimes caused the
        # model to cite the query ID; normalize that alias before the result
        # reaches update_evidence(), which only accepts recorded evidence IDs.
        allowed_evidence = {evidence.id, query.id}
        allowed_outcomes = {str(outcome.label) for outcome in query.outcomes}
        if draft.query_id != query.id:
            raise ValueError("query_id does not match the executed query")
        conversion_notes: list[str] = []
        outcome_label = draft.outcome_label
        investigation_status = draft.investigation_status
        if outcome_label is not None and outcome_label not in allowed_outcomes:
            conversion_notes.append(f"cleared undeclared outcome label {outcome_label!r}")
            outcome_label = None
            investigation_status = "inconclusive"

        def refs(values: list[str], label: str) -> list[str]:
            result: list[str] = []
            for ref in values:
                if ref == query.id:
                    ref = evidence.id
                if ref not in {evidence.id}:
                    conversion_notes.append(f"dropped unknown {label} reference {ref!r}")
                    continue
                if ref not in result:
                    result.append(ref)
            return result

        facts: list[FactStatement] = []
        for fact in draft.facts:
            fact_refs = refs(fact.evidence_refs, "fact evidence")
            anchors = {f"{anchor.file}:{anchor.line}": anchor for anchor in evidence.source_anchors if anchor.line}
            line_counts: dict[str, int] = {}
            for anchor in evidence.source_anchors:
                if anchor.line:
                    line_counts[str(anchor.line)] = line_counts.get(str(anchor.line), 0) + 1
            anchors.update({
                str(anchor.line): anchor for anchor in evidence.source_anchors
                if anchor.line and line_counts[str(anchor.line)] == 1
            })
            facts.append(FactStatement(
                id=fact.id, statement=fact.statement, evidence_refs=fact_refs,
                source_anchors=[anchors[item] for item in fact.source_anchor_refs if item in anchors],
            ))

        valid_risks = []
        for assessment in draft.risk_assessments:
            if assessment.candidate_id not in candidate_ids and assessment.candidate_id in candidate_aliases:
                conversion_notes.append(
                    f"mapped transient risk candidate {assessment.candidate_id!r} to {candidate_aliases[assessment.candidate_id]}"
                )
                assessment = assessment.model_copy(update={"candidate_id": candidate_aliases[assessment.candidate_id]})
            if assessment.candidate_id not in candidate_ids:
                conversion_notes.append(f"dropped unknown risk candidate {assessment.candidate_id!r}")
                continue
            assessment = assessment.model_copy(update={"evidence_refs": refs(assessment.evidence_refs, "risk")})
            if assessment.status != "unknown" and not assessment.evidence_refs:
                conversion_notes.append(f"downgraded unsupported risk assessment {assessment.candidate_id!r}")
                assessment = assessment.model_copy(update={"status": "unknown"})
            valid_risks.append(RiskCandidateAssessment.model_validate(assessment.model_dump()))

        valid_condition_assessments = []
        for assessment in draft.candidate_condition_assessments:
            if assessment.candidate_id not in candidate_ids and assessment.candidate_id in candidate_aliases:
                assessment = assessment.model_copy(update={"candidate_id": candidate_aliases[assessment.candidate_id]})
            if assessment.candidate_id not in candidate_ids:
                conversion_notes.append(f"dropped unknown candidate condition assessment {assessment.candidate_id!r}")
                continue
            if assessment.premise_id not in premise_ids:
                conversion_notes.append(f"dropped condition assessment outside query targets {assessment.premise_id!r}")
                continue
            assessment = assessment.model_copy(update={"evidence_refs": refs(assessment.evidence_refs, "candidate condition")})
            if assessment.status != "unknown" and not assessment.evidence_refs:
                assessment = assessment.model_copy(update={"status": "unknown"})
            valid_condition_assessments.append(
                CandidateConditionAssessment.model_validate(assessment.model_dump())
            )

        # A risk assessment is a projection over the candidate's necessary
        # conditions.  Do not allow a query that establishes only one premise
        # (for example, Quantum is wider than unsigned char) to mark a
        # buffer/alignment candidate as wholly supported.  The condition
        # assessments are query-local; later rounds can combine their durable
        # premise updates in the final decision node.
        condition_statuses: dict[str, dict[str, set[str]]] = {}
        for item in valid_condition_assessments:
            condition_statuses.setdefault(item.candidate_id, {}).setdefault(
                item.premise_id, set()
            ).add(item.status)
        normalized_risks: list[RiskCandidateAssessment] = []
        for assessment in valid_risks:
            if assessment.status == "supported":
                candidate = next(
                    (item for item in value.risk_candidates if item.candidate_id == assessment.candidate_id),
                    None,
                )
                required = [
                    str(item.get("id")) for item in (candidate.required_conditions if candidate else [])
                    if isinstance(item, dict) and item.get("id")
                ]
                if required and not all(
                    condition_statuses.get(assessment.candidate_id, {}).get(condition_id) == {"supported"}
                    for condition_id in required
                ):
                    conversion_notes.append(
                        f"downgraded partial candidate assessment {assessment.candidate_id!r}; "
                        "not every necessary condition was supported by this query"
                    )
                    assessment = assessment.model_copy(update={"status": "unknown"})
            normalized_risks.append(assessment)
        valid_risks = normalized_risks

        valid_protections = []
        for assessment in draft.protection_assessments:
            if assessment.protection_id not in protection_ids and assessment.protection_id in protection_aliases:
                conversion_notes.append(
                    f"mapped transient protection {assessment.protection_id!r} to {protection_aliases[assessment.protection_id]}"
                )
                assessment = assessment.model_copy(update={"protection_id": protection_aliases[assessment.protection_id]})
            if assessment.protection_id not in protection_ids:
                conversion_notes.append(f"dropped unknown protection {assessment.protection_id!r}")
                continue
            assessment = assessment.model_copy(update={"evidence_refs": refs(assessment.evidence_refs, "protection")})
            if assessment.coverage_status != "unknown" and not assessment.evidence_refs:
                conversion_notes.append(f"downgraded unsupported protection assessment {assessment.protection_id!r}")
                assessment = assessment.model_copy(update={"coverage_status": "unknown"})
            valid_protections.append(ProtectionAssessment.model_validate(assessment.model_dump()))

        valid_contracts = []
        for contract in draft.caller_contracts:
            refs_used = refs(contract.evidence_refs, "caller contract")
            if not refs_used:
                conversion_notes.append(f"dropped caller contract without valid evidence: {contract.caller_function!r}")
                continue
            valid_contracts.append(CallerContract.model_validate({**contract.model_dump(), "evidence_refs": refs_used}))
        # SOURCE caller queries already provide a deterministic, structured
        # call-site projection. If the model omits the contract section, keep
        # that verified mapping available to later premise updates rather than
        # reducing a successful caller query to an unstructured text fact.
        if not valid_contracts and evidence.structured_facts.get("result_kind") == "caller_contract":
            for row in evidence.structured_facts.get("caller_contracts", []):
                if not isinstance(row, dict):
                    continue
                payload = {
                    key: row.get(key)
                    for key in (
                        "caller_function", "call_site", "call_site_location",
                        "argument_expression", "allocation_expression", "read_length_expression",
                        "required_size_expression", "short_read_guard", "extent_relation",
                    )
                    if row.get(key) is not None
                }
                if not payload.get("caller_function") or not payload.get("call_site"):
                    continue
                payload.setdefault("argument_expression", "unknown")
                payload["evidence_refs"] = [evidence.id]
                try:
                    valid_contracts.append(CallerContract.model_validate(payload))
                except Exception as exc:
                    conversion_notes.append(f"dropped generated caller contract: {exc}")
            if valid_contracts:
                conversion_notes.append("preserved executor caller-contract projection because model omitted caller_contracts")

        valid_updates = []
        for update in draft.premise_updates:
            if update.premise_id not in premise_ids:
                conversion_notes.append(f"dropped premise outside query targets {update.premise_id!r}")
                continue
            update = update.model_copy(update={"evidence_refs": refs(update.evidence_refs, "premise")})
            if update.status != "unknown" and not update.evidence_refs:
                conversion_notes.append(f"downgraded unsupported premise {update.premise_id!r}")
                update = update.model_copy(update={"status": "unknown"})
            valid_updates.append(PremiseUpdate.model_validate(update.model_dump()))
        return FactInvestigationResult(
            query_id=draft.query_id, investigation_status=investigation_status,
            outcome_label=outcome_label, facts=facts,
            risk_assessments=valid_risks,
            candidate_condition_assessments=valid_condition_assessments,
            protection_assessments=valid_protections,
            caller_contracts=valid_contracts,
            premise_updates=valid_updates,
            unresolved_questions=draft.unresolved_questions,
            limitations=[*draft.limitations, *conversion_notes],
        )

    @staticmethod
    def _fallback(value: FactInvestigationInput, error: str) -> FactInvestigationResult:
        # Preserve explicit executor mappings even when the model response is
        # malformed. This is intentionally narrower than model interpretation:
        # only IDs declared by the query and already attached to Evidence are
        # allowed to update the ledger.
        target_ids = set(value.query.target_premise_ids)
        updates: list[PremiseUpdate] = []
        for premise_id in target_ids & set(value.evidence.supports_premise_ids):
            updates.append(PremiseUpdate(
                premise_id=premise_id, status="supported",
                reason="The query executor explicitly mapped this evidence to the premise.",
                evidence_refs=[value.evidence.id],
            ))
        for premise_id in target_ids & set(value.evidence.negates_premise_ids):
            updates.append(PremiseUpdate(
                premise_id=premise_id, status="refuted",
                reason="The query executor explicitly mapped this evidence against the premise.",
                evidence_refs=[value.evidence.id],
            ))
        return FactInvestigationResult(
            query_id=value.query.id, investigation_status="inconclusive",
            premise_updates=updates,
            unresolved_questions=["Structured fact interpretation failed."],
            limitations=[error] + ([] if updates else ["No premise was updated because evidence interpretation was not validated."]),
        )


__all__ = ["FACT_INVESTIGATOR_INSTRUCTION", "FactInvestigator"]
