"""Discriminating query planning agent with program-controlled validation/ranking."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from .agent_models import QueryPlanDraft
from .protocols import Backend, QueryKind, QueryOutcome, QueryPlanningInput, QuerySpec

QUERY_PLANNER_INSTRUCTION = (
    "Generate bounded program queries for unresolved premises. Do not decide vulnerability, "
    "execute tools, or invent facts. Each query must target supplied premise/claim IDs, use "
    "the narrowest useful backend, and declare at least two materially different outcomes. "
    "Every outcome must state which target premises it supports, negates, or leaves unknown. "
    "When unresolved premises are supplied, emit at least one candidate; do not put planned "
    "queries only in the summary. Prioritize decisive premises tied to direct_violation candidates, "
    "then conditional_interprocedural candidates, then speculative candidates. A later round must "
    "use a materially different backend/evidence path; do not repeat a prior caller-length or contract "
    "search with merely rephrased wording. Emit at most 3 ranked candidates and keep wording concise. "
    "Return only QueryPlanDraft JSON."
)


class QueryPlanner:
    actor = "query_planner"
    output_contract = (
        "QueryPlanDraft JSON only. For every decisive unresolved premise, emit a candidate unless "
        "the available backend truly cannot answer it. Copy IDs exactly. Each candidate must have "
        "target_claim_ids, target_premise_ids, kind, backend, a concrete evidence question, and "
        "three complete outcome partitions: supports the premise, negates the premise, and leaves "
        "it unknown. Do not put candidates only in summary; do not issue a vulnerability verdict."
    )

    def __init__(self, model: Any):
        self.last_interaction: dict[str, Any] | None = None
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{instruction}\n\n{contract}"),
            ("human", "Planning input JSON:\n{input}"),
        ])
        self.chain = prompt | model.with_structured_output(QueryPlanDraft, method="function_calling", include_raw=True)
        self.json_chain = prompt | model.with_structured_output(QueryPlanDraft, method="json_mode", include_raw=True)

    def __call__(self, planning: QueryPlanningInput) -> list[QuerySpec]:
        payload = planning.model_dump(mode="json")
        self.last_interaction = {"agent": self.actor, "status": "started", "input": {
            "instruction": QUERY_PLANNER_INSTRUCTION, "response_schema": "QueryPlanDraft",
            "output_contract": self.output_contract, "planning_input": payload},
            "parsed_output": None, "error": None}
        values = {"instruction": QUERY_PLANNER_INSTRUCTION, "contract": self.output_contract,
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
            fallback_queries = self._template_queries(planning)
            self.last_interaction.update({
                "status": "fallback" if fallback_queries else "failed",
                "error": " | ".join(errors) or "empty response",
                "fallback": "premise_templates" if fallback_queries else None,
                "planned_candidates": [query.id for query in fallback_queries],
            })
            return fallback_queries
        if not draft.candidates and planning.dispute.unresolved_premises and planning.budget.tool_call_limit != 0:
            repair_values = dict(values)
            repair_values["instruction"] = (
                f"{QUERY_PLANNER_INSTRUCTION}\n\nCorrection required: the previous response omitted candidates. "
                "Return one candidate per decisive unresolved premise. For each candidate, copy the "
                "premise ID exactly and provide three outcomes: supports, negates, and leaves unknown."
            )
            try:
                repaired = self._extract(self.json_chain.invoke(repair_values))
                if repaired is not None and repaired.candidates:
                    draft = repaired
                    self.last_interaction["candidate_repair"] = {
                        "status": "parsed", "output": repaired.model_dump(mode="json")
                    }
                else:
                    self.last_interaction["candidate_repair"] = {"status": "empty"}
            except Exception as exc:
                self.last_interaction["candidate_repair"] = {
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}"
                }
        try:
            queries = self._validate_and_rank(draft, planning)
            if not queries:
                queries = self._template_queries(planning)
            queries = self._split_definition_queries(queries)
        except Exception as exc:
            self.last_interaction.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                          "parsed_output": draft.model_dump(mode="json")})
            return []
        self.last_interaction.update({"status": "parsed", "parsed_output": draft.model_dump(mode="json"),
                                      "planned_candidates": [q.id for q in queries], "errors": errors,
                                      "fallback": "premise_templates" if queries and not self._validate_and_rank(draft, planning) else None})
        return queries

    @classmethod
    def _split_definition_queries(cls, queries: list[QuerySpec]) -> list[QuerySpec]:
        """Split broad Quantum/type questions into independently answerable queries."""
        result: list[QuerySpec] = []
        for query in queries:
            lowered = query.question.lower()
            if query.backend == Backend.SOURCE and "quantum" in lowered and "scalechartoquantum" in lowered:
                for suffix, symbol, focus in (
                    ("quantum-type", "Quantum", "Find only the Quantum typedef, QuantumRange definition, and active depth branches."),
                    ("scale-char", "ScaleCharToQuantum", "Find only the complete ScaleCharToQuantum definition and return expression."),
                ):
                    result.append(query.model_copy(update={
                        "id": f"{query.id}-{suffix}",
                        "question": f"{focus} Symbol focus: `{symbol}`.",
                        "evidence_requirement": f"{query.evidence_requirement}; return complete definition blocks for {symbol}, not unrelated call sites.",
                        "required_context": "repository",
                        "assumptions": [
                            *query.assumptions,
                            f"This is an independent definition subquery derived from: {query.question}",
                        ],
                        "selection_reasons": [*query.selection_reasons, f"split_definition:{symbol}"],
                    }))
            else:
                result.append(query)
        return result

    @staticmethod
    def _extract(result: Any) -> QueryPlanDraft | None:
        parsed = result.get("parsed") if isinstance(result, dict) else result
        if isinstance(parsed, QueryPlanDraft):
            return parsed
        if isinstance(parsed, BaseModel):
            return QueryPlanDraft.model_validate(QueryPlanner._normalize_payload(parsed.model_dump()))
        if isinstance(parsed, dict):
            return QueryPlanDraft.model_validate(QueryPlanner._normalize_payload(parsed))
        raw = result.get("raw") if isinstance(result, dict) else None
        content = getattr(raw, "content", "")
        if isinstance(content, list):
            content = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
        if isinstance(content, str):
            start, end = content.find("{"), content.rfind("}")
            if start >= 0 and end > start:
                import json
                return QueryPlanDraft.model_validate(
                    QueryPlanner._normalize_payload(json.loads(content[start:end + 1]))
                )
        return None

    @staticmethod
    def _normalize_payload(payload: Any) -> dict[str, Any]:
        """Normalize common compatible-endpoint variants before strict validation."""
        if not isinstance(payload, dict):
            raise ValueError("query planner response is not an object")
        result = dict(payload)
        result["summary"] = str(result.get("summary") or result.get("rationale") or "Generated query candidates")[:1000]
        raw_candidates = result.get("candidates") or result.get("queries") or []
        if not isinstance(raw_candidates, list):
            raw_candidates = []
        candidates: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_candidates[:6], start=1):
            if not isinstance(raw, dict):
                continue
            source = dict(raw)
            item: dict[str, Any] = {}
            item["candidate_id"] = str(
                source.get("candidate_id") or source.get("id") or source.get("query_id") or f"model-query-{index}"
            )[:80]
            item["target_claim_ids"] = list(source.get("target_claim_ids") or source.get("target_claims") or [])[:4]
            item["target_premise_ids"] = list(source.get("target_premise_ids") or source.get("target_premises") or [])[:8]
            item["question"] = str(source.get("question") or source.get("query") or source.get("evidence_question") or "")[:700]
            item["evidence_requirement"] = str(source.get("evidence_requirement") or source.get("evidence_required") or "Exact source or structural evidence tied to the target premise.")[:500]
            item["assumptions"] = [str(value)[:300] for value in (source.get("assumptions") or [])[:8]]
            item["kind"] = source.get("kind") or source.get("query_kind") or "source_search"
            item["backend"] = source.get("backend") or ("joern" if item["kind"] in {"cpg", "numeric_range"} else "source")
            item["outcomes"] = QueryPlanner._normalize_outcomes(source.get("outcomes"), item["target_premise_ids"])
            candidates.append(item)
        result["candidates"] = candidates
        return result

    @staticmethod
    def _normalize_outcomes(value: Any, target_ids: list[str]) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            rows = []
            aliases = (("supports", "support"), ("negates", "negate", "refutes", "refute"), ("unknown", "leaves_unknown"))
            for label, *keys in aliases:
                node = next((value[key] for key in keys if key in value), None)
                if node is None:
                    continue
                node = node if isinstance(node, dict) else {"meaning": str(node)}
                ids = node.get("premise_ids") or node.get("premises") or target_ids
                rows.append({
                    "label": label,
                    "meaning": str(node.get("meaning") or node.get("description") or label)[:300],
                    "supports_premise_ids": list(ids)[:8] if label == "supports" else [],
                    "negates_premise_ids": list(ids)[:8] if label == "negates" else [],
                    "leaves_unknown_premise_ids": list(ids)[:8] if label == "unknown" else [],
                })
            return rows
        if not isinstance(value, list):
            return []
        rows = []
        for index, raw in enumerate(value[:4], start=1):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item["label"] = str(item.get("label") or item.get("outcome") or f"outcome-{index}")[:80]
            item["meaning"] = str(item.get("meaning") or item.get("description") or item["label"])[:300]
            rows.append(item)
        return rows

    @classmethod
    def _validate_and_rank(cls, draft: QueryPlanDraft, planning: QueryPlanningInput) -> list[QuerySpec]:
        unresolved = set(planning.dispute.unresolved_premises)
        if not unresolved or planning.budget.tool_call_limit == 0:
            return []
        premises = unresolved
        claims = {planning.risk_claim.id, planning.protection_claim.id}
        decisive = set(planning.dispute.decisive_premises)
        enabled = set(planning.enabled_backends) or {Backend.SOURCE, Backend.JOERN}
        prior = {q.question.strip().lower() for q in planning.prior_queries}
        prior_shapes = {
            (
                frozenset(q.target_premise_ids),
                str(getattr(q.kind, "value", q.kind)),
                str(getattr(q.backend, "value", q.backend)),
            )
            for q in planning.prior_queries
        }
        prior_premise_ids = {
            premise_id for query in planning.prior_queries for premise_id in query.target_premise_ids
        }
        candidate_classes = {
            str(row.get("candidate_id")): str(row.get("candidate_class") or "speculative")
            for row in planning.risk_claim.analysis.get("candidates", [])
            if isinstance(row, dict) and row.get("candidate_id")
        }
        premise_rows = {item.id: item for item in planning.dispute.premises}
        ranked: list[tuple[float, QuerySpec]] = []
        for item in draft.candidates:
            if set(item.target_premise_ids) - premises or set(item.target_claim_ids) - claims:
                continue
            if item.backend not in enabled or item.question.strip().lower() in prior:
                continue
            # Caller allocation/argument questions are answerable by the
            # repository source index and caller slices. The current Joern
            # script extracts one target method, so routing these questions
            # to Joern produces a CPG without caller methods.
            item_backend = item.backend
            item_kind = item.kind
            if (
                item_backend == Backend.JOERN
                and Backend.SOURCE in enabled
                and cls._caller_query_requested(item.question)
            ):
                item_backend = Backend.SOURCE
                item_kind = QueryKind.SOURCE_SEARCH
            target = set(item.target_premise_ids)
            shape = (
                frozenset(target),
                str(getattr(item_kind, "value", item_kind)),
                str(getattr(item_backend, "value", item_backend)),
            )
            if shape in prior_shapes:
                continue
            outcomes = []
            signatures = set()
            for outcome in item.outcomes:
                support, negate, unknown = (set(outcome.supports_premise_ids),
                                             set(outcome.negates_premise_ids),
                                             set(outcome.leaves_unknown_premise_ids))
                if (support | negate | unknown) != target or support & negate:
                    outcomes = []
                    break
                signatures.add((tuple(sorted(support)), tuple(sorted(negate)), tuple(sorted(unknown))))
                outcomes.append(outcome)
            if len(outcomes) < 2 or len(signatures) < 2:
                continue
            spec = QuerySpec(id=item.candidate_id, target_claim_ids=item.target_claim_ids,
                target_premise_ids=item.target_premise_ids, scope=planning.scope, kind=item_kind,
                backend=item_backend, question=item.question,
                outcomes=[outcome.model_dump() for outcome in outcomes], assumptions=item.assumptions,
                evidence_requirement=item.evidence_requirement, budget=planning.budget,
                proposed_by=cls.__name__, required_context=cls._infer_required_context(item))
            if spec.backend == Backend.JOERN:
                # Joern currently emits bounded facts for the target method;
                # caller/allocation and external definition questions are
                # routed to repository SOURCE.
                lowered_query = spec.question.lower()
                external_definition = any(marker in lowered_query for marker in (
                    "helper implementation", "helper body", "helper", "implementation",
                    "typedef", "macro definition",
                    "formatlocalestring", "setpsdpixel", "pushshortpixel", "pushfloatpixel",
                    "readproperty", "quantum typedef", "quantumrange",
                    "allocation", "caller", "call site",
                ))
                if (cls._caller_query_requested(spec.question) or external_definition) and Backend.SOURCE in enabled:
                    spec = spec.model_copy(update={
                        "backend": Backend.SOURCE,
                        "kind": QueryKind.SOURCE_SEARCH,
                        "required_context": "repository",
                    })
                else:
                    spec = spec.model_copy(update={"required_context": "function"})
            impact = min(1.0, len(target) / max(1, len(planning.dispute.unresolved_premises)))
            separation = min(1.0, (len(signatures) - 1) / max(1, len(outcomes) - 1))
            cost = {"source_search": .15, "cpg": .3, "allocation_relation": .35,
                    "numeric_range": .5, "parameter_mapping": .4, "side_effect": .45,
                    "path_reachability": .6, "symbolic_execution": .85, "smt": .9}.get(item.kind.value, .6)
            spec = spec.model_copy(update={"impact": impact, "separation": separation,
                "answerability": 1.0, "estimated_cost": cost,
                "model_risk": min(1.0, len(item.assumptions) / 5)})
            score = impact * separation - .35 * cost - .15 * (spec.model_risk or 0)
            reasons = []
            if target & decisive:
                score += 0.45
                reasons.append("targets_decisive_premise")
            linked_classes = {
                candidate_classes[ref]
                for premise_id in target
                for ref in getattr(premise_rows.get(premise_id), "claim_refs", [])
                if ref in candidate_classes
            }
            if "direct_violation" in linked_classes:
                score += 0.35
                reasons.append("targets_direct_violation")
            elif "conditional_interprocedural" in linked_classes:
                score += 0.15
                reasons.append("targets_conditional_candidate")
            if target & prior_premise_ids:
                score -= 0.30
                reasons.append("premise_previously_queried_requires_new_evidence_path")
            if set(item.target_claim_ids) & {planning.risk_claim.id}:
                score += 0.05
                reasons.append("targets_risk_claim")
            if target == unresolved:
                reasons.append("covers_all_unresolved_premises")
            spec = spec.model_copy(update={"selection_score": score, "selection_reasons": reasons})
            ranked.append((score, spec))
        return [query for _, query in sorted(ranked, key=lambda pair: pair[0], reverse=True)]

    @staticmethod
    def _infer_required_context(item: Any) -> str:
        """Upgrade underspecified model plans when the question needs repository facts."""
        if item.required_context == "build":
            return "build"
        text = " ".join(
            str(value) for value in (
                item.question, item.evidence_requirement, *item.assumptions,
            )
        ).lower()
        repository_markers = (
            "caller", "callers", "allocation", "allocated", "getstringinfodatum",
            "header", "include", "build", "external", "repository", "contract",
            "callee", "invocation", "refcount", "reference count", "helper", "implementation",
            "function body", "platform", "architecture", "compiler", "effective type", "alignment",
            "macro", "component", "components", "profile", "span", "format", "tag_bytes",
            "readproperty", "inline value", "inline-data", "inline data",
            "quantum", "quantumrange", "scalechartoquantum", "typedef", "narrowing",
            "representation", "signedness", "setpsdpixel", "getauthenticpixels",
            "getauthenticindexqueue",
        )
        return "repository" if any(marker in text for marker in repository_markers) else "function"

    @staticmethod
    def _caller_query_requested(question: str) -> bool:
        lowered = question.lower()
        return any(marker in lowered for marker in (
            "caller", "callers", "call site", "call-site", "allocation", "allocated",
            "argument passed", "actual argument", "parameter mapping",
        ))

    @classmethod
    def _template_queries(cls, planning: QueryPlanningInput) -> list[QuerySpec]:
        """Provide a conservative query when the model returned only prose."""
        candidate_classes = {
            str(row.get("candidate_id")): str(row.get("candidate_class") or "speculative")
            for row in planning.risk_claim.analysis.get("candidates", [])
            if isinstance(row, dict) and row.get("candidate_id")
        }
        premise_rows = {item.id: item for item in planning.dispute.premises}
        class_rank = {"direct_violation": 0, "conditional_interprocedural": 1, "speculative": 2}
        decisive = set(planning.dispute.decisive_premises)
        unresolved = sorted(
            planning.dispute.unresolved_premises,
            key=lambda premise_id: (
                premise_id not in decisive,
                min((
                    class_rank.get(candidate_classes.get(ref, "speculative"), 2)
                    for ref in getattr(premise_rows.get(premise_id), "claim_refs", [])
                ), default=2),
            ),
        )
        if not unresolved or planning.budget.tool_call_limit == 0:
            return []
        queries: list[QuerySpec] = []
        prior = {q.question.strip().lower() for q in planning.prior_queries}
        prior_shapes = {
            (
                frozenset(q.target_premise_ids),
                str(getattr(q.kind, "value", q.kind)),
                str(getattr(q.backend, "value", q.backend)),
            )
            for q in planning.prior_queries
        }
        for index, premise_id in enumerate(unresolved[:4], start=1):
            premise = next((p for p in planning.dispute.premises if p.id == premise_id), None)
            description = premise.description if premise else premise_id
            lowered = description.lower()
            external_symbol_query = any(symbol in lowered for symbol in (
                "formatlocalestring", "setpsdpixel", "pushshortpixel", "pushfloatpixel",
                "quantum", "scalechartoquantum",
            ))
            if external_symbol_query:
                kind, backend = QueryKind.SOURCE_SEARCH, Backend.SOURCE
            elif any(word in lowered for word in ("align", "cast", "derefer", "pointer", "type")):
                kind, backend = QueryKind.CPG, Backend.JOERN
            elif any(word in lowered for word in ("range", "bound", "offset", "size", "overflow", "component")):
                kind, backend = QueryKind.NUMERIC_RANGE, Backend.JOERN
            else:
                kind, backend = QueryKind.SOURCE_SEARCH, Backend.SOURCE
            if cls._caller_query_requested(description):
                kind, backend = QueryKind.SOURCE_SEARCH, Backend.SOURCE
            # Template fallback must use the same external-fact context as the
            # model planner. A premise such as "depth selects the float path"
            # may not say "repository", but its linked conditional candidate
            # usually lists caller/header/type facts that cannot be answered
            # from the function fragment.
            linked_questions = []
            risk_candidates = {
                str(candidate.get("candidate_id")): candidate
                for candidate in planning.risk_claim.analysis.get("candidates", [])
                if isinstance(candidate, dict) and candidate.get("candidate_id")
            }
            for claim_ref in getattr(premise, "claim_refs", []):
                candidate = risk_candidates.get(str(claim_ref))
                if candidate:
                    linked_questions.extend(candidate.get("unresolved_questions") or [])
            context_text = " ".join([description, *map(str, linked_questions)]).lower()
            required_context = "repository" if any(word in context_text for word in (
                "caller", "allocation", "getstring", "build", "external", "repository", "header", "contract",
                "helper", "implementation", "function body", "platform", "architecture", "compiler",
                "effective type", "alignment", "readproperty", "reader", "profile provenance",
                "profile passed", "allocator", "datum implementation", "macro", "component",
                "components", "profile", "span", "format", "tag_bytes", "inline value",
                "inline-data", "inline data", "quantum", "quantumrange", "scalechartoquantum",
                "typedef", "narrowing", "representation", "signedness", "setpsdpixel",
                "getauthenticpixels", "getauthenticindexqueue",
            )) else "function"
            enabled = set(planning.enabled_backends) or {Backend.SOURCE, Backend.JOERN}
            if backend not in enabled:
                if Backend.SOURCE in enabled:
                    backend = Backend.SOURCE
                    kind = QueryKind.SOURCE_SEARCH
                elif Backend.JOERN in enabled:
                    backend = Backend.JOERN
                    kind = QueryKind.CPG
                else:
                    continue
            shape = (
                frozenset({premise_id}),
                str(getattr(kind, "value", kind)),
                str(getattr(backend, "value", backend)),
            )
            if shape in prior_shapes:
                enabled = set(planning.enabled_backends) or {Backend.SOURCE, Backend.JOERN}
                if backend == Backend.JOERN and Backend.SOURCE in enabled:
                    kind, backend = QueryKind.SOURCE_SEARCH, Backend.SOURCE
                elif backend == Backend.SOURCE and Backend.JOERN in enabled:
                    kind, backend = QueryKind.CPG, Backend.JOERN
            question = f"Can source/Joern evidence establish whether: {description}"
            if question.lower() in prior or (
                frozenset({premise_id}),
                str(getattr(kind, "value", kind)),
                str(getattr(backend, "value", backend)),
            ) in prior_shapes:
                continue
            queries.append(QuerySpec(
                id=f"template-{index}-{premise_id}",
                target_claim_ids=[planning.risk_claim.id, planning.protection_claim.id],
                target_premise_ids=[premise_id], scope=planning.scope, kind=kind, backend=backend,
                question=question,
                outcomes=[
                    QueryOutcome(label="established", meaning="The condition is established by exact evidence.", supports_premise_ids=[premise_id]),
                    QueryOutcome(label="disproved", meaning="The condition is disproved by exact evidence.", negates_premise_ids=[premise_id]),
                    QueryOutcome(label="unknown", meaning="The available scope cannot establish the condition.", leaves_unknown_premise_ids=[premise_id]),
                ],
                assumptions=[
                    "Use the exact sample snapshot when repository context is required; returned clues remain candidates, not proof."
                ],
                evidence_requirement="Exact source or Joern anchors tied to the premise and relevant path/type relation.",
                budget=planning.budget, impact=1.0,
                separation=1.0, answerability=1.0, estimated_cost=.3,
                model_risk=.2, proposed_by="premise_template",
                required_context=required_context,
                selection_score=1.0 + (0.45 if premise_id in set(planning.dispute.decisive_premises) else 0.0),
                selection_reasons=["template_for_missing_model_candidate"] + (
                    ["targets_decisive_premise"] if premise_id in set(planning.dispute.decisive_premises) else []
                ),
            ))
        return queries


__all__ = ["QueryPlanner", "QUERY_PLANNER_INSTRUCTION"]
