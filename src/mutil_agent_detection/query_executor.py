"""Small, backend-neutral query execution facade.

This layer deliberately returns bounded evidence, not vulnerability conclusions.  A later fact
investigator can interpret the raw result and map it to QueryOutcome premise updates.
"""

from __future__ import annotations

import re
import subprocess
import heapq
from typing import Any
from pathlib import Path

from .joern import JoernAnalyzer
from .repository import RepositoryError, RepositoryManager
from .dataset import PrimeVulSample
from .source_evidence import index_key_operations, locate_function_in_repository
from .protocols import (
    AnalysisInput, Backend, Evidence, EvidenceStatus, QueryKind, QuerySpec,
    SourceLocation,
)


class QueryExecutionError(RuntimeError):
    """Raised only for invalid executor configuration or malformed requests."""


class QueryExecutor:
    """Execute first-version source and Joern queries with explicit evidence boundaries."""

    def __init__(self, *, joern: JoernAnalyzer | None = None, repository_manager: RepositoryManager | None = None,
                 cache_enabled: bool = True):
        self.joern = joern or JoernAnalyzer()
        self.repository_manager = repository_manager
        self.cache_enabled = cache_enabled
        self._cache: dict[tuple[Any, ...], Evidence] = {}

    def __call__(self, query: QuerySpec, request: AnalysisInput) -> Evidence:
        cache_key = self._cache_key(query, request)
        if self.cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached.model_copy(update={"query_id": query.id})
        if query.backend not in {Backend.SOURCE, Backend.JOERN}:
            return self._unavailable(
                query, request,
                f"{self._enum_value(query.backend)} backend is not configured; repository context was not prepared",
            )
        request, context_error = self._upgrade_context(query, request)
        if context_error:
            evidence = self._unavailable(query, request, context_error)
            return evidence.model_copy(update={"witness": self._resolution_witness(request)})
        # The first repository upgrade may keep the declared commit while
        # filling in the validated worktree root. Preserve both pieces of
        # provenance on the query/evidence scope for later rounds.
        if (
            request.scope.snapshot.commit != query.scope.snapshot.commit
            or request.scope.snapshot.root != query.scope.snapshot.root
        ):
            query = query.model_copy(update={"scope": request.scope})
        if query.backend == Backend.SOURCE:
            if query.required_context in {"repository", "build"}:
                if request.metadata.get("repository_version_status") == "repository_version_mismatch":
                    evidence = self._source_search(query, request)
                else:
                    evidence = self._repository_source_search(query, request)
            else:
                evidence = self._source_search(query, request)
        elif query.backend == Backend.JOERN:
            evidence = self._joern_query(query, request)
        else:
            evidence = self._unavailable(query, request, "query backend is not configured")
        evidence = evidence.model_copy(update={"witness": self._resolution_witness(request)})
        if self.cache_enabled and evidence.status != EvidenceStatus.INCONCLUSIVE:
            self._cache[cache_key] = evidence
        return evidence

    @staticmethod
    def _cache_key(query: QuerySpec, request: AnalysisInput) -> tuple[Any, ...]:
        """Key only on bounded, deterministic query inputs.

        Query IDs are intentionally excluded: a retry may create a new ID while
        asking the same question against the same snapshot.
        """
        return (
            str(query.backend), str(query.kind), query.question,
            query.required_context, request.function_name,
            request.scope.snapshot.commit, request.metadata.get("repository_root"),
            request.metadata.get("repository_version_status"),
        )

    @staticmethod
    def _resolution_witness(request: AnalysisInput) -> dict[str, Any]:
        keys = (
            "declared_commit", "repository_snapshot_commit", "repository_version_status",
            "repository_version_note", "sample_function_hash", "repository_function_hash",
            "repository_checked_commits", "repository_root",
        )
        return {key: request.metadata[key] for key in keys if key in request.metadata}

    def _upgrade_context(self, query: QuerySpec, request: AnalysisInput) -> tuple[AnalysisInput, str | None]:
        if query.required_context == "function":
            return request, None
        if self.repository_manager is None:
            return request, "repository context is required but no RepositoryManager is configured"
        metadata = request.metadata
        try:
            sample = PrimeVulSample(
                dataset_name=str(metadata.get("dataset_name") or "primevul"),
                source_line=int(metadata.get("source_line") or 1),
                sample_id=request.sample_id, idx=int(metadata.get("idx") or 0),
                project=metadata.get("project"), commit_id=request.scope.snapshot.commit,
                project_url=metadata.get("project_url"), commit_url=metadata.get("commit_url"),
                commit_message=metadata.get("commit_message"), target=request.expected_label or 0,
                func=request.source, file_name=metadata.get("file_name"),
                func_hash=metadata.get("func_hash"), file_hash=metadata.get("file_hash"),
                cwe=list(metadata.get("cwe") or []), cve=metadata.get("cve"),
                cve_desc=metadata.get("cve_desc"), nvd_url=metadata.get("nvd_url"),
                function_name=request.function_name,
            )
            resolution = self.repository_manager.prepare_matching_snapshot(sample)
            updated = dict(metadata)
            updated.update({
                "context_mode": "repository" if resolution.root else "function_only",
                "repository_root": str(resolution.root) if resolution.root else None,
                "query_context": query.required_context,
                "declared_commit": resolution.declared_commit,
                "repository_snapshot_commit": resolution.matched_commit,
                "repository_version_status": resolution.status,
                "repository_version_note": resolution.note,
                "sample_function_hash": resolution.sample_function_hash,
                "repository_function_hash": resolution.repository_function_hash,
                "repository_checked_commits": list(resolution.checked_commits),
            })
            if resolution.root is None:
                updated["repository_version_status"] = "repository_version_mismatch"
                return request.model_copy(update={"metadata": updated}), None

            snapshot = request.scope.snapshot.model_copy(update={
                "version": resolution.matched_commit,
                "commit": resolution.matched_commit,
                "root": str(resolution.root),
            })
            locations = request.scope.locations
            line_map = locate_function_in_repository(
                request.source, metadata.get("file_name"), resolution.root,
                expected_commit=resolution.matched_commit,
            )
            if line_map["start_line"] is not None and locations:
                locations = [locations[0].model_copy(update={
                    "line": line_map["start_line"], "end_line": line_map["end_line"],
                }), *locations[1:]]
            updated["source_line_basis"] = line_map["basis"]
            updated["source_line_start"] = line_map["start_line"]
            updated["source_line_end"] = line_map["end_line"]
            updated["source_line_mapping_note"] = line_map["note"]
            updated["key_operation_index"] = index_key_operations(
                request.source, start_line=line_map["start_line"] or 1,
            )
            scope = request.scope.model_copy(update={
                "snapshot": snapshot, "locations": locations,
                "unresolved_dependencies": [
                    item for item in request.scope.unresolved_dependencies
                    if item != "repository context not loaded"
                ],
            })
            return request.model_copy(update={"metadata": updated, "scope": scope}), None
        except (RepositoryError, ValueError, TypeError) as exc:
            return request, f"repository context preparation failed: {exc}"

    def _source_search(self, query: QuerySpec, request: AnalysisInput) -> Evidence:
        source_text, source_file, boundary, first_line = self._source_text(request)
        version_mismatch = request.metadata.get("repository_version_status") == "repository_version_mismatch"
        if query.required_context != "function" and not request.metadata.get("repository_root") and not version_mismatch:
            return self._unavailable(query, request, "repository context was required but not loaded")
        if query.required_context != "function" and not version_mismatch and not any(
            marker in boundary for marker in (
                "Exact-commit target function", "Exact-commit repository file was searched",
            )
        ):
            return self._unavailable(query, request, f"repository source file could not be loaded: {boundary}")
        source_lines = source_text.splitlines()
        needles = self._query_needles(query.question)
        matches: list[SourceLocation] = []
        for number, line in enumerate(source_lines, start=first_line):
            normalized = line.strip()
            if not normalized:
                continue
            if not needles or any(needle in normalized for needle in needles):
                matches.append(SourceLocation(
                    file=source_file,
                    function=request.function_name,
                    line=number,
                ))
        # Source search is intentionally a fact candidate. It does not infer which branch
        # supports or negates a premise; that mapping belongs to the fact investigator.
        status = EvidenceStatus.CANDIDATE if matches else EvidenceStatus.INCONCLUSIVE
        summary = (
            f"Source search found {len(matches)} matching line(s) for the bounded query."
            if matches else "Source search found no exact query anchors in the supplied function."
        )
        if version_mismatch:
            boundary = (
                "Repository snapshot mismatch: only the authoritative PrimeVul function fragment was searched; "
                f"{request.metadata.get('repository_version_note') or 'no matching repository version was found'}."
            )
            summary = "Function-fragment fallback after repository version mismatch. " + summary
        return Evidence(
            status=status, summary=summary, raw_result=self._source_result(source_lines, matches),
            source_anchors=matches[:20], scope=query.scope, query_id=query.id, tool=Backend.SOURCE,
            assumptions=query.assumptions,
            analysis_boundary=boundary,
        )

    def _repository_source_search(self, query: QuerySpec, request: AnalysisInput) -> Evidence:
        """Search bounded, tracked source/build files in the exact-commit worktree."""
        root_value = request.metadata.get("repository_root")
        if not root_value:
            return self._unavailable(query, request, "repository context was required but not loaded")
        root = Path(str(root_value)).resolve()
        try:
            tracked = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z"],
                check=True, capture_output=True, timeout=30,
            ).stdout.split(b"\0")
        except (OSError, subprocess.SubprocessError) as exc:
            return self._unavailable(query, request, f"could not enumerate exact-commit repository files: {exc}")

        source_suffixes = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".m", ".mm"}
        build_names = {"CMakeLists.txt", "configure.ac", "configure.in", "Makefile.am", "meson.build"}
        symbols = self._repository_search_symbols(query.question, request.function_name)
        if not symbols and self._caller_query_requested(query.question) and request.function_name:
            symbols = [request.function_name]
        target_file = self._normalized_repo_path(request.metadata.get("file_name"))
        definition_query = self._is_definition_query(query.question)
        if not symbols:
            return Evidence(
                status=EvidenceStatus.INCONCLUSIVE,
                summary="Repository search was not run because the query and sample metadata contain no exact code symbol.",
                raw_result=None, scope=query.scope, query_id=query.id,
                tool=Backend.SOURCE, assumptions=query.assumptions,
                analysis_boundary=(
                    "Exact-commit repository is available, but a repository-wide search without a discriminative "
                    "symbol would produce noisy evidence."
                ),
            )
        candidates: list[tuple[int, int, str, int, str, str, str]] = []
        candidate_limit = 5000
        sequence = 0
        candidate_count = 0
        scanned = 0
        paths = sorted(
            (raw_path.decode("utf-8", errors="replace") for raw_path in tracked if raw_path),
            key=lambda path: (0 if self._same_repo_path(path, target_file) else 1, path),
        )
        for relative in paths:
            path = (root / relative).resolve()
            if root != path and root not in path.parents:
                continue
            if path.suffix.lower() not in source_suffixes and path.name not in build_names:
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            scanned += 1
            for number, line in enumerate(lines, start=1):
                score = self._repository_line_score(relative, line, symbols, target_file)
                if score <= 0:
                    continue
                if definition_query:
                    # Type/macro/contract questions should prefer the header or
                    # implementation that defines the symbol over call sites in
                    # the target function. The previous target-file bonus made
                    # those call sites crowd out the actual definitions.
                    lowered = line.lower()
                    if path.suffix.lower() in {".h", ".hpp", ".hh", ".hxx"}:
                        score += 36
                    # Prefer actual definitions over ordinary uses. Generic
                    # symbols such as Quantum otherwise crowd out typedefs
                    # and macro/function bodies before the result is capped.
                    if re.search(r"^\s*(typedef|#define)\b", line):
                        score += 90
                    if re.search(r"\b(?:static\s+inline\s+)?[A-Za-z_]\w*\s+\w+\s*\([^;]*\)", line):
                        score += 42
                    if any(
                        re.search(rf"\b(?:static\s+inline\s+)?[A-Za-z_]\w*\s+{re.escape(symbol)}\s*\(", line)
                        for symbol in symbols if " " not in symbol
                    ):
                        # A symbol's definition is more useful than its many
                        # call sites when the query explicitly asks for it.
                        score += 150
                    if any(marker in lowered for marker in ("#define", "typedef", "quantum", "scalechartoquantum", "getauthenticpixels", "setpsdpixel")):
                        score += 24
                    if self._same_repo_path(relative, target_file):
                        score += 12
                candidate_count += 1
                candidate = (
                    score, -sequence, relative, number,
                    lines[number - 2] if number > 1 else "", line,
                    lines[number] if number < len(lines) else "",
                )
                sequence += 1
                if len(candidates) < candidate_limit:
                    heapq.heappush(candidates, candidate)
                elif candidate[:2] > candidates[0][:2]:
                    heapq.heapreplace(candidates, candidate)
            if scanned >= 12000:
                break

        candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        selected = candidates[:30]
        matches = [SourceLocation(file=relative, line=number) for _, _, relative, number, *_ in selected]
        excerpts = self._repository_excerpts(selected)
        context_blocks = self._repository_context_blocks(root, selected, symbols=symbols)
        caller_anchors: list[SourceLocation] = []
        caller_blocks: list[str] = []
        if self._caller_query_requested(query.question):
            caller_anchors, caller_blocks = self._repository_callsite_blocks(
                root, request.function_name,
            )
            matches = self._unique_locations([*caller_anchors, *matches])
            context_blocks = [*caller_blocks, *context_blocks]
        structured_facts = self._structured_repository_facts(
            query.question, request.function_name, caller_anchors, caller_blocks, selected, context_blocks,
        )

        status = EvidenceStatus.CANDIDATE if matches else EvidenceStatus.INCONCLUSIVE
        summary = (
            f"Exact-commit repository search scanned {scanned} tracked source/build file(s) and selected "
            f"{len(matches)} relevant line(s) from {candidate_count} symbol/file matches."
            if matches else
            (
                "Repository search was not run because the exact commit contains no tracked source files; no exact code symbol was available."
                if scanned == 0
                else f"Exact-commit repository search scanned {scanned} tracked source/build file(s) and found no exact "
                f"matches for the requested code symbols: {', '.join(symbols) or 'none supplied'}."
            )
        )
        return Evidence(
            status=status, summary=summary, raw_result="\n".join([
                "MATCHED LINES", *excerpts,
                "", "SYMBOL DEFINITIONS AND CALL-SITE FUNCTION SLICES", *context_blocks,
            ]),
            source_anchors=matches[:30], scope=query.scope, query_id=query.id,
            structured_facts=structured_facts,
            tool=Backend.SOURCE, assumptions=query.assumptions,
            analysis_boundary=(
                "Exact-commit repository-wide search over tracked source/build files; a matching line is a candidate fact, "
                "not proof of call reachability, control-flow dominance, or security impact."
                + (" Caller function slices were included for the requested target symbol." if caller_blocks else "")
            ),
        )

    @classmethod
    def _structured_repository_facts(
        cls,
        question: str,
        function_name: str | None,
        caller_anchors: list[SourceLocation],
        caller_blocks: list[str],
        selected: list[tuple[int, int, str, int, str, str, str]],
        context_blocks: list[str],
    ) -> dict[str, Any]:
        """Project SOURCE results into small, role-specific fact records.

        A caller query must not make the Fact Investigator reconstruct a
        contract from a large text dump.  Keep the complete caller slice for
        auditing, but also expose one deterministic record per call site.
        Definition queries get a separate result family so Quantum/type
        definitions cannot be mistaken for ordinary call-site matches.
        """
        is_caller = cls._caller_query_requested(question)
        is_definition = cls._is_definition_query(question)
        facts: dict[str, Any] = {
            "result_kind": "caller_contract" if is_caller else "definition" if is_definition else "source_search",
            "source_basis": "exact_commit_repository",
        }
        if caller_anchors:
            facts["caller_contract_query"] = True
            facts["call_sites"] = [anchor.model_dump(mode="json") for anchor in caller_anchors]
            facts["caller_function_slices"] = caller_blocks[:8]
            facts["caller_slice_count"] = len(caller_blocks)
            facts["caller_contracts"] = cls._caller_contract_records(
                function_name, caller_anchors, caller_blocks,
            )
        if is_definition:
            facts["definition_query"] = True
            facts["definition_matches"] = [
                {"file": relative, "line": number, "code": line}
                for _, _, relative, number, _, line, _ in selected[:30]
            ]
            facts["definition_blocks"] = context_blocks[:8]
            definition_symbols = cls._repository_search_symbols(question, function_name)
            complete_blocks = [
                block for block in context_blocks
                if ("enclosing_function" in block or "macro_definition" in block)
                and any(
                    re.search(rf"\b{re.escape(symbol)}\b", block)
                    for symbol in definition_symbols
                    if " " not in symbol
                )
            ]
            facts["definition_completeness"] = (
                "complete_function_or_macro_block"
                if complete_blocks
                else "signature_or_fragment_only"
            )
        return facts

    @staticmethod
    def _caller_contract_records(
        function_name: str | None,
        anchors: list[SourceLocation],
        blocks: list[str],
    ) -> list[dict[str, Any]]:
        """Extract caller-side expressions without claiming their semantics."""
        if not function_name:
            return []
        call_re = re.compile(rf"\b{re.escape(function_name)}\s*\((?P<args>[^;()]*(?:\([^;()]*\)[^;()]*)*)\)")
        contracts: list[dict[str, Any]] = []
        for block in blocks:
            lines = block.splitlines()
            header = lines[0] if lines else ""
            header_match = re.search(
                r"--- caller_[^ ]+ (?P<file>.+?):\d+-\d+ \(call (?P<line>\d+)\) ---",
                header,
            )
            location = next(
                (
                    anchor for anchor in anchors
                    if header_match
                    and anchor.file == header_match.group("file")
                    and anchor.line == int(header_match.group("line"))
                ),
                None,
            )
            if location is None:
                continue
            call_line = next((line for line in lines if f":{location.line}:" in line), "")
            match = call_re.search(call_line)
            argument = match.group("args").strip() if match else ""
            signature = next(
                (line.split(": ", 1)[-1].strip() for line in lines[1:] if "(" in line and "{" in line),
                "",
            )
            caller_function = "unknown"
            signature_match = re.search(r"\b([A-Za-z_]\w*)\s*\([^)]*\)", signature)
            if signature_match:
                caller_function = signature_match.group(1)
            body = "\n".join(lines)
            allocation = next((line.split(": ", 1)[-1].strip() for line in lines if re.search(
                r"\b(?:malloc|calloc|realloc|Acquire\w*|Allocate\w*|Resize\w*)\s*\(", line,
            )), None)
            read_length = next((line.split(": ", 1)[-1].strip() for line in lines if re.search(
                r"\b(?:read|Read|fread|fgets|ReadBlob|length|size|bytes|extent)\w*\s*\(", line,
            )), None)
            required_size = next((line.split(": ", 1)[-1].strip() for line in lines if re.search(
                r"\b(?:size|length|extent|bytes|count|width|height)\b", line, re.IGNORECASE,
            )), None)
            guard = "present" if any(
                re.search(r"\bif\s*\([^)]*(?:<|<=|>|>=|==|!=)[^)]*\)", line)
                for line in lines
            ) else "unknown"
            contracts.append({
                "caller_function": caller_function,
                "call_site": call_line.split(": ", 1)[-1].strip() if call_line else "",
                "call_site_location": location.model_dump(mode="json"),
                "argument_expression": argument,
                "allocation_expression": allocation,
                "read_length_expression": read_length,
                "required_size_expression": required_size,
                "short_read_guard": guard,
                "extent_relation": "unknown",
                "enclosing_function_slice": body,
            })
        return contracts[:8]

    @staticmethod
    def _source_text(request: AnalysisInput) -> tuple[str, str, str, int]:
        root = request.metadata.get("repository_root")
        file_name = str(request.metadata.get("file_name") or "")
        if root and file_name:
            from pathlib import Path
            root_path = Path(str(root)).resolve()
            path = (root_path / file_name).resolve()
            if not path.is_file():
                # PrimeVul often stores only a basename (for example psd.c),
                # while the repository keeps it under coders/. Resolve a
                # unique tracked basename before falling back to the fragment.
                try:
                    tracked = subprocess.run(
                        ["git", "-C", str(root_path), "ls-files", "-z"],
                        check=True, capture_output=True, timeout=30,
                    ).stdout.split(b"\0")
                    candidates = [
                        (root_path / raw.decode("utf-8", errors="replace")).resolve()
                        for raw in tracked if raw and Path(raw.decode("utf-8", errors="replace")).name == Path(file_name).name
                    ]
                    candidates = [item for item in candidates if item.is_file() and (item == root_path or root_path in item.parents)]
                    if len(candidates) == 1:
                        path = candidates[0]
                        file_name = str(path.relative_to(root_path)).replace("\\", "/")
                except (OSError, subprocess.SubprocessError):
                    pass
            # Dataset paths must stay inside the prepared exact-commit worktree.
            if root_path != path and root_path not in path.parents:
                return request.source, file_name or "<function>", (
                    "Repository file path escaped the prepared worktree; only the supplied function text was searched."
                ), 1
            try:
                repository_source = path.read_text(encoding="utf-8", errors="replace")
                location = next((
                    item for item in request.scope.locations
                    if item.file.replace("\\", "/").endswith(file_name.replace("\\", "/"))
                    and item.line is not None and item.end_line is not None
                ), None)
                if location is not None:
                    lines = repository_source.splitlines()
                    start = max(1, location.line)
                    end = min(len(lines), location.end_line)
                    if start <= end:
                        target_source = "\n".join(lines[start - 1:end])
                        return target_source, file_name, (
                            f"Exact-commit target function {request.function_name} was searched only within "
                            f"repository lines {start}-{end}; anchors retain repository-absolute lines."
                            + (" Build flags/platform configuration were not verified." if request.metadata.get("query_context") == "build" else "")
                        ), start
                return request.source, file_name, (
                    "Exact-commit file was available but target function bounds were not verified; "
                    "only the authoritative PrimeVul function fragment was searched."
                ), 1
            except OSError:
                pass
        return request.source, file_name or "<function>", (
            "Only the supplied PrimeVul function text was searched; repository context was unavailable."
        ), 1

    def _joern_query(self, query: QuerySpec, request: AnalysisInput) -> Evidence:
        skeleton = self.joern.analyze(request)
        if skeleton.get("status") not in {"ready", "target_not_found"}:
            return Evidence(
                status=EvidenceStatus.INCONCLUSIVE,
                summary="Joern query could not obtain a usable CPG result.",
                raw_result=self._json_result(skeleton), scope=query.scope, query_id=query.id,
                tool=Backend.JOERN, assumptions=query.assumptions,
                analysis_boundary=str(skeleton.get("unresolved_dependencies") or skeleton.get("scope") or "Joern failed"),
            )
        anchors = self._relevant_joern_anchors(skeleton, query.question)
        summary = (
            f"Joern returned a bounded skeleton with {len(anchors)} relevant source-located clue(s)."
            if anchors else "Joern returned a skeleton but no clue matched the query wording."
        )
        mode = skeleton.get("mode", request.metadata.get("context_mode", "function_only"))
        boundary = (
            "Joern analyzed the exact-commit repository CPG within the selected function's CFG and local data-flow scope; "
            "structural/data-flow clues are evidence candidates, not conclusions."
            if mode == "repository" else
            "Joern analyzed only the PrimeVul function CPG within its local CFG/data-flow scope; "
            "structural/data-flow clues are candidates, not conclusions."
        )
        if request.metadata.get("repository_version_status") == "repository_version_mismatch":
            boundary += " Repository version mismatch; Joern used only the authoritative PrimeVul function fragment."
        return Evidence(
            status=EvidenceStatus.CANDIDATE if anchors else EvidenceStatus.INCONCLUSIVE,
            # Do not send the complete CPG dump to the investigator. It is
            # frequently larger than the model context and causes relevant
            # control-flow rows to be truncated away. Keep query-focused rows
            # and provenance instead.
            summary=summary, structured_facts={
                "result_kind": "joern_function_local",
                "query_scope": skeleton.get("query_scope", "function_cfg_local_dataflow"),
                "local_dataflow_only": True,
            }, raw_result=self._json_result(
                self._bounded_joern_result(skeleton, query.question)
            ), source_anchors=anchors,
            scope=query.scope, query_id=query.id, tool=Backend.JOERN, assumptions=query.assumptions,
            analysis_boundary=boundary,
        )

    def _unavailable(self, query: QuerySpec, request: AnalysisInput, reason: str) -> Evidence:
        return Evidence(
            status=EvidenceStatus.INCONCLUSIVE,
            summary=f"Query was planned but not executed: {reason}.",
            raw_result=None, scope=query.scope, query_id=query.id,
            tool=query.backend, assumptions=query.assumptions,
            analysis_boundary=(
                f"Sample {request.sample_id}; evidence is inconclusive because the requested context or "
                f"{self._enum_value(query.backend)} executor was unavailable."
            ),
        )

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value))

    @staticmethod
    def _query_needles(question: str) -> list[str]:
        quoted = re.findall(r"[`\"]([^`\"]{3,})[`\"]", question)
        # Prefer exact code fragments, then distinctive identifier tokens.
        if quoted:
            return list(dict.fromkeys(item.strip() for item in quoted if item.strip()))[:8]
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?", question)
        stop = {"whether", "source", "joern", "evidence", "establish", "condition", "target"}
        return list(dict.fromkeys(token for token in tokens if token.lower() not in stop and len(token) > 3))[:8]

    @staticmethod
    def _repository_search_symbols(question: str, function_name: str | None) -> list[str]:
        """Keep code-like identifiers; prose words are too broad for repository-wide matching."""
        identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question)
        quoted = re.findall(r"[`\"]([^`\"]{2,})[`\"]", question)
        quoted_symbols = [fragment.strip() for fragment in quoted if fragment.strip()]
        stop = {
            "find", "whether", "source", "joern", "evidence", "establish", "condition", "target",
            "caller", "callers", "allocation", "allocated", "details", "supplied", "paths", "path",
            "the", "and", "for", "from", "with", "does", "this", "that", "into", "when",
            "return", "void", "unsigned", "signed", "char", "int", "size_t", "const",
        }
        explicit_symbols = [
            token for token in identifiers
            if token.lower() not in stop and len(token) >= 3
            and ("_" in token or any(char.isupper() for char in token) or any(char.isdigit() for char in token))
        ]
        values = [*quoted_symbols, *explicit_symbols]
        if function_name and len(function_name) >= 3:
            values.append(function_name)
        reserved = {"EXIF", "CPG", "SOURCE", "JOERN"}
        return [value for value in dict.fromkeys(values) if value.upper() not in reserved][:24]

    @staticmethod
    def _is_definition_query(question: str) -> bool:
        lowered = question.lower()
        return any(marker in lowered for marker in (
            "definition", "implementation", "typedef", "#define", "macro",
            "quantum", "quantumrange", "scalechartoquantum", "helper body",
            "cache contract", "accessor", "allocation", "type width", "build-time",
            "formatlocalestring", "setpsdpixel", "pushshortpixel", "pushfloatpixel",
            "readproperty",
        ))

    @staticmethod
    def _normalized_repo_path(value: Any) -> str:
        if not value:
            return ""
        return str(value).replace("\\", "/").removeprefix("./").strip("/")

    @staticmethod
    def _same_repo_path(path: str, target_file: str) -> bool:
        normalized = path.replace("\\", "/")
        return bool(target_file) and (
            normalized == target_file or normalized.endswith(f"/{target_file}")
        )

    @classmethod
    def _repository_line_score(
        cls, relative: str, line: str, symbols: list[str], target_file: str
    ) -> int:
        line_symbols = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line))
        matched = [
            symbol for symbol in symbols
            if (symbol in line if " " in symbol else symbol in line_symbols)
        ]
        if not matched:
            return 0
        score = sum(cls._repository_symbol_weight(symbol) for symbol in matched)
        if cls._same_repo_path(relative, target_file):
            score += 30
        if line.lstrip().startswith(("//", "/*", "*")):
            score -= 3
        return max(score, 1)

    @staticmethod
    def _repository_symbol_weight(symbol: str) -> int:
        if " " in symbol:
            return 16
        if any(char.isupper() for char in symbol) or any(char.isdigit() for char in symbol):
            return 18 + min(len(symbol), 32) // 4
        if "_" in symbol:
            return 6
        return 4

    @staticmethod
    def _repository_excerpts(selected: list[tuple[int, int, str, int, str, str, str]]) -> list[str]:
        excerpts: list[str] = []
        seen: set[tuple[str, int]] = set()
        for _, _, relative, line_number, before, line, after in selected:
            for number, text in ((line_number - 1, before), (line_number, line), (line_number + 1, after)):
                key = (relative, number)
                if not text or key in seen:
                    continue
                seen.add(key)
                marker = ">" if number == line_number else " "
                excerpts.append(f"{marker} {relative}:{number}: {text.strip()}")
        return excerpts

    @classmethod
    def _repository_context_blocks(
        cls, root: Path, selected: list[tuple[int, int, str, int, str, str, str]],
        *, symbols: list[str] | None = None,
    ) -> list[str]:
        """Return bounded complete macro/function blocks around high-ranked matches."""
        blocks: list[str] = []
        seen: set[tuple[str, int, int]] = set()
        symbols = symbols or []

        def defines_requested_symbol(line: str) -> bool:
            # A definition block for the requested helper is substantially
            # more useful than an arbitrary function that merely mentions
            # ``Quantum`` or a generic type name.  Keep this check deliberately
            # local; the enclosing-block extractor supplies the full body.
            for symbol in symbols:
                if " " in symbol:
                    continue
                if re.search(
                    rf"\b(?:static\s+|inline\s+|extern\s+|const\s+|unsigned\s+|void\s+|[A-Za-z_]\w*\s+)*"
                    rf"{re.escape(symbol)}\s*\(",
                    line,
                ):
                    return True
                if re.search(rf"\b(?:typedef|#define)\b[^\n]*\b{re.escape(symbol)}\b", line):
                    return True
            return False

        # A capped search may contain many macro call sites. Prefer actual
        # function/typedef definitions when constructing the bounded blocks.
        ordered = sorted(
            selected,
            key=lambda item: (
                0 if defines_requested_symbol(item[5]) else
                1 if re.search(r"\b(?:static\s+inline\s+)?[A-Za-z_]\w*\s+\w+\s*\([^;]*\)", item[5]) else
                2 if re.match(r"^\s*(typedef|#define)\b", item[5]) else 3,
                # Prefer the sample file only after exact symbol matches;
                # this prevents unrelated target-file call sites from
                # crowding out an external helper definition.
                item[2], item[3],
            ),
        )
        for _, _, relative, line_number, *_ in ordered:
            if len(blocks) >= 8:
                break
            path = (root / relative).resolve()
            if root != path and root not in path.parents:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            start, end, kind = cls._enclosing_source_block(lines, line_number)
            key = (relative, start, end)
            if key in seen:
                continue
            seen.add(key)
            body = "\n".join(
                f"{relative}:{number}: {lines[number - 1]}"
                for number in range(start, end + 1)
            )
            blocks.append(f"--- {kind} {relative}:{start}-{end} (match {line_number}) ---\n{body}")
        return blocks

    @staticmethod
    def _caller_query_requested(question: str) -> bool:
        lowered = question.lower()
        return any(marker in lowered for marker in (
            "caller", "callers", "call site", "call-site", "allocation", "allocated",
            "argument passed", "actual argument", "parameter mapping",
        ))

    @classmethod
    def _repository_callsite_blocks(
        cls, root: Path, function_name: str | None,
    ) -> tuple[list[SourceLocation], list[str]]:
        """Return call-site anchors and bounded enclosing caller functions."""
        if not function_name:
            return [], []
        try:
            tracked = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z"],
                check=True, capture_output=True, timeout=30,
            ).stdout.split(b"\0")
        except (OSError, subprocess.SubprocessError):
            return [], []
        call_re = re.compile(rf"\b{re.escape(function_name)}\s*\(")
        anchors: list[SourceLocation] = []
        blocks: list[str] = []
        seen_blocks: set[tuple[str, int, int]] = set()
        suffixes = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".m", ".mm"}
        for raw in tracked:
            if not raw:
                continue
            relative = raw.decode("utf-8", errors="replace")
            path = (root / relative).resolve()
            if path.suffix.lower() not in suffixes or not path.is_file() or root not in path.parents:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, start=1):
                if not call_re.search(line):
                    continue
                # Skip a declaration/definition line; callers are statements
                # inside another function. The target definition is still
                # available through the ordinary symbol search.
                prefix = line[:line.find(function_name)].strip()
                if "{" in line or (prefix and (
                    prefix.startswith(("static ", "inline ", "extern "))
                    or prefix.endswith(("static", "inline", "void", "int", "Type"))
                    or "(" in prefix
                )):
                    continue
                anchors.append(SourceLocation(file=relative, line=number, function=function_name))
                start, end, kind = cls._enclosing_source_block(lines, number)
                key = (relative, start, end)
                if key in seen_blocks:
                    continue
                seen_blocks.add(key)
                body = "\n".join(
                    f"{relative}:{line_number}: {lines[line_number - 1]}"
                    for line_number in range(start, end + 1)
                )
                blocks.append(f"--- caller_{kind} {relative}:{start}-{end} (call {number}) ---\n{body}")
                if len(blocks) >= 8:
                    return cls._unique_locations(anchors)[:20], blocks
        return cls._unique_locations(anchors)[:20], blocks

    @staticmethod
    def _unique_locations(values: list[SourceLocation]) -> list[SourceLocation]:
        result: list[SourceLocation] = []
        seen: set[tuple[str, int | None]] = set()
        for value in values:
            key = (value.file, value.line)
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    @staticmethod
    def _enclosing_source_block(lines: list[str], line_number: int) -> tuple[int, int, str]:
        index = max(0, min(len(lines) - 1, line_number - 1))
        # Preserve complete multi-line macro definitions.
        macro_start = next((
            pos for pos in range(index, max(-1, index - 80), -1)
            if lines[pos].lstrip().startswith("#define ")
        ), None)
        if macro_start is not None:
            macro_end = macro_start
            while macro_end + 1 < len(lines) and lines[macro_end].rstrip().endswith("\\"):
                macro_end += 1
            if macro_start <= index <= macro_end:
                return macro_start + 1, macro_end + 1, "macro_definition"

        # A definition match often lands on a continuation line of a
        # multi-line function signature. Move back to the declaration before
        # looking for its opening brace, otherwise only the tail of the
        # signature is returned as the "definition".
        declaration = next((pos for pos in range(index, max(-1, index - 12), -1)
                            if re.search(
                                r"^\s*(?:static\s+|inline\s+|extern\s+)*"
                                r"[A-Za-z_]\w*\s+[A-Za-z_]\w*\s*\(", lines[pos]
                            )), None)
        if declaration is not None:
            index = declaration

        # A definition match often lands on a multi-line function signature.
        # Start at that signature rather than accidentally selecting the
        # preceding helper whose closing brace is nearest to the match.
        signature_window = lines[index:min(len(lines), index + 5)]
        first_brace = next((pos for pos, text in enumerate(signature_window) if "{" in text), None)
        signature_text = " ".join(signature_window[:first_brace + 1] if first_brace is not None else signature_window)
        is_signature = bool(re.search(r"^\s*(?:static\s+|inline\s+|extern\s+)*[A-Za-z_]\w*\s+[A-Za-z_]\w*\s*\(", lines[index]))
        signature_end = next((
            pos for pos in range(index, min(len(lines), index + 5))
            if "{" in lines[pos]
        ), None) if is_signature else None
        if signature_end is not None:
            balance = 0
            opened = False
            for pos in range(index, min(len(lines), index + 260)):
                balance += lines[pos].count("{") - lines[pos].count("}")
                opened = opened or "{" in lines[pos]
                if opened and balance <= 0 and pos >= signature_end:
                    return index + 1, pos + 1, "enclosing_function"

        # Locate an enclosing brace-balanced function/block and include its full body.
        start = None
        for pos in range(index, max(-1, index - 160), -1):
            text = lines[pos]
            if "{" in text and (
                ")" in text or (pos > 0 and ")" in " ".join(lines[max(0, pos - 4):pos + 1]))
            ):
                prefix = " ".join(lines[max(0, pos - 5):pos + 1])
                if not re.search(r"\b(if|for|while|switch)\s*\([^;]*\)\s*\{?\s*$", prefix):
                    start = max(0, pos - 5)
                    while start < pos and not lines[start].strip():
                        start += 1
                    break
        if start is not None:
            balance = 0
            opened = False
            end = index
            for pos in range(start, min(len(lines), start + 220)):
                balance += lines[pos].count("{") - lines[pos].count("}")
                opened = opened or "{" in lines[pos]
                end = pos
                if opened and balance <= 0 and pos >= index:
                    return start + 1, end + 1, "enclosing_function"

        return max(1, line_number - 12), min(len(lines), line_number + 12), "local_slice"

    @staticmethod
    def _source_result(lines: list[str], anchors: list[SourceLocation]) -> str:
        selected = []
        for anchor in anchors[:20]:
            if anchor.line and anchor.line <= len(lines):
                selected.append(f"{anchor.line}: {lines[anchor.line - 1]}")
        return "\n".join(selected)

    @staticmethod
    def _relevant_joern_anchors(skeleton: dict[str, Any], question: str) -> list[SourceLocation]:
        lowered = question.lower()
        rows: list[dict[str, Any]] = []
        for key in ("structural_operations", "candidate_dataflows", "calls", "controls", "source_candidates"):
            values = skeleton.get(key, [])
            if isinstance(values, list):
                rows.extend(item for item in values if isinstance(item, dict))
        # Query planners often name exact source lines (for example, the
        # guard at 652 and the read at 827). Prefer those rows over generic
        # casts/calls from the beginning of a large CPG listing.
        requested_lines = {
            int(value) for value in re.findall(r"\blines?[- ]+(\d+)", question, flags=re.IGNORECASE)
        }
        exact_rows = []
        if requested_lines:
            for row in rows:
                line = row.get("line") or row.get("sink_line") or row.get("source_line")
                if isinstance(line, int) and line in requested_lines:
                    exact_rows.append(row)
            # Do not fall back to unrelated early rows when a query names
            # exact source locations; that pattern previously made a
            # truncated CPG look like evidence for the requested path.
            rows = exact_rows
        keywords = [word for word in (
            "align", "cast", "conversion", "represent", "signed", "unsigned", "narrow",
            "shift", "derefer", "pointer", "type", "range", "offset", "bound", "path",
            "component", "overflow",
        ) if word in lowered]
        selected = []
        for row in rows:
            text = " ".join(str(row.get(key, "")) for key in ("code", "category", "kind", "operator", "evidence_kind", "source_code"))
            if keywords and not any(word in text.lower() for word in keywords):
                continue
            line = row.get("line") or row.get("sink_line") or row.get("source_line")
            file = row.get("file") or str(skeleton.get("file") or "<joern>")
            if isinstance(line, int) and line > 0:
                selected.append(SourceLocation(file=str(file), function=skeleton.get("entrypoint"), line=line))
        unique = {(item.file, item.line): item for item in selected}
        return list(unique.values())[:20]

    @classmethod
    def _bounded_joern_result(cls, skeleton: dict[str, Any], question: str) -> dict[str, Any]:
        """Return a compact, query-focused CPG view for the fact investigator."""
        requested_lines = {
            int(value) for value in re.findall(r"\blines?[- ]+(\d+)", question, flags=re.IGNORECASE)
        }
        result: dict[str, Any] = {
            key: skeleton.get(key)
            for key in ("status", "mode", "scope", "entrypoint", "file", "node_counts", "unresolved_dependencies")
            if key in skeleton
        }
        for key in ("structural_operations", "candidate_dataflows", "calls", "controls", "source_candidates"):
            values = skeleton.get(key)
            if not isinstance(values, list):
                continue
            rows = []
            for row in values:
                if not isinstance(row, dict):
                    continue
                line = row.get("line") or row.get("sink_line") or row.get("source_line")
                if requested_lines and isinstance(line, int) and line not in requested_lines:
                    continue
                compact = {
                    key_name: value for key_name, value in row.items()
                    if key_name not in {"raw_output", "query_script", "command"}
                }
                for key_name, value in list(compact.items()):
                    if isinstance(value, str) and len(value) > 1200:
                        compact[key_name] = value[:1200] + "... [truncated]"
                rows.append(compact)
            # If exact requested lines are absent, retain a small ranked slice
            # so the investigator can distinguish missing coverage from a huge
            # unrelated dump.
            result[key] = rows[:40] if requested_lines else rows[:24]
        if requested_lines:
            result["requested_source_lines"] = sorted(requested_lines)
            result["line_match_note"] = (
                "Only Joern rows at explicitly requested source lines are included; an empty list means "
                "the CPG did not expose those lines."
            )
        return result

    @staticmethod
    def _json_result(value: Any) -> str:
        import json
        return json.dumps(value, ensure_ascii=False, sort_keys=True)


__all__ = ["QueryExecutionError", "QueryExecutor"]
