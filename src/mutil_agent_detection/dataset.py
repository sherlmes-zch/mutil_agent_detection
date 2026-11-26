"""PrimeVul JSONL ingestion and normalization.

The loader keeps dataset concerns separate from the LangGraph protocol.  A
``PrimeVulSample`` preserves the raw security-dataset metadata, while
``to_analysis_input`` creates the bounded input used by one audit run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .protocols import AnalysisInput, ProgramScope, SourceLocation, SourceSnapshot
from .source_evidence import index_key_operations, locate_function_in_repository


class DatasetFormatError(ValueError):
    """Raised when a JSONL record cannot be converted to a PrimeVul sample."""


class PrimeVulSample(BaseModel):
    """Normalized, provenance-preserving representation of one PrimeVul row."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    dataset_name: str
    source_line: int = Field(ge=1)
    sample_id: str
    idx: int
    project: str | None = None
    commit_id: str | None = None
    project_url: str | None = None
    commit_url: str | None = None
    commit_message: str | None = None
    target: Literal[0, 1]
    func: str
    func_hash: str | None = None
    file_name: str | None = None
    file_hash: str | None = None
    cwe: list[str] = Field(default_factory=list)
    cve: str | None = None
    cve_desc: str | None = None
    nvd_url: str | None = None
    function_name: str = "<unknown>"
    function_name_inferred: bool = False
    pair_id: str | None = None
    pair_role: Literal["vulnerable", "non_vulnerable"] | None = None
    quality_flags: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("func")
    @classmethod
    def function_source_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("PrimeVul func must not be empty")
        return value

    def to_analysis_input(
        self,
        *,
        context_mode: Literal["function_only", "repository", "repository_context"] = "function_only",
        repository_root: str | None = None,
        repository_commit: str | None = None,
        repository_match_status: str | None = None,
        repository_match_note: str | None = None,
        repository_function_hash: str | None = None,
        repository_checked_commits: list[str] | None = None,
    ) -> AnalysisInput:
        """Convert one normalized row into the frozen workflow input contract."""

        unresolved: list[str] = []
        if repository_root:
            context_mode = "repository"
        if repository_match_status == "repository_version_mismatch":
            unresolved.append("repository_version_mismatch: no repository snapshot matched the PrimeVul function")
        elif context_mode == "function_only":
            unresolved.append("repository context not loaded")
        elif repository_root is None:
            unresolved.append("repository context requested but no validated repository root was provided")
        if self.file_name is None:
            unresolved.append("source file path unavailable")

        line_map = locate_function_in_repository(
            self.func, self.file_name, repository_root, expected_commit=repository_commit or self.commit_id
        )
        absolute_start = line_map["start_line"]
        absolute_end = line_map["end_line"]
        if context_mode == "repository" and line_map["note"]:
            unresolved.append(f"repository line mapping unavailable: {line_map['note']}")
        location_start = absolute_start or 1
        location_end = absolute_end or max(1, self.func.count("\n") + 1)
        key_operations = index_key_operations(self.func, start_line=location_start)

        locations: list[SourceLocation] = []
        if self.file_name is not None:
            locations.append(
                SourceLocation(
                    file=self.file_name,
                    function=self.function_name,
                    line=location_start,
                    end_line=location_end,
                    code_hash=self.func_hash,
                )
            )

        scope = ProgramScope(
            snapshot=SourceSnapshot(
                version=repository_commit or self.commit_id or "unknown",
                commit=repository_commit or self.commit_id,
            ),
            entrypoint=self.function_name,
            locations=locations,
            unresolved_dependencies=unresolved,
        )
        metadata = {
            "dataset_name": self.dataset_name,
            "source_line": self.source_line,
            "idx": self.idx,
            "project": self.project,
            "project_url": self.project_url,
            "commit_url": self.commit_url,
            "commit_message": self.commit_message,
            "func_hash": self.func_hash,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "cwe": self.cwe,
            "cve": self.cve,
            "cve_desc": self.cve_desc,
            "nvd_url": self.nvd_url,
            "pair_id": self.pair_id,
            "pair_role": self.pair_role,
            "context_mode": context_mode,
            "repository_root": repository_root,
            "declared_commit": self.commit_id,
            "repository_snapshot_commit": repository_commit,
            "repository_version_status": repository_match_status,
            "repository_version_note": repository_match_note,
            "sample_function_hash": self.func_hash,
            "repository_function_hash": repository_function_hash,
            "repository_checked_commits": repository_checked_commits or [],
            "quality_flags": self.quality_flags,
            "function_name_inferred": self.function_name_inferred,
            "source_line_basis": line_map["basis"],
            "source_line_start": absolute_start,
            "source_line_end": absolute_end,
            "source_line_mapping_note": line_map["note"],
            "key_operation_index": key_operations,
        }
        return AnalysisInput(
            sample_id=self.sample_id,
            function_name=self.function_name,
            source=self.func,
            scope=scope,
            dataset_split=_split_from_name(self.dataset_name),
            expected_label=self.target,
            metadata=metadata,
        )


class PairLink(BaseModel):
    """Explicit relation between a vulnerable and non-vulnerable sample."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    vulnerable_sample_id: str
    non_vulnerable_sample_id: str
    basis: Literal["adjacent_metadata", "metadata_group"]
    confidence: Literal["high", "medium"]
    shared_fields: list[str] = Field(default_factory=list)


class DatasetQualityReport(BaseModel):
    """Auditable summary of ingestion anomalies and label distribution."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    records: int = 0
    labels: dict[str, int] = Field(default_factory=dict)
    projects: int = 0
    cwes: dict[str, int] = Field(default_factory=dict)
    missing_fields: dict[str, int] = Field(default_factory=dict)
    duplicate_idx: list[int] = Field(default_factory=list)
    duplicate_sample_ids: list[str] = Field(default_factory=list)
    function_chars_min: int | None = None
    function_chars_max: int | None = None
    function_chars_average: float | None = None
    paired_links: int = 0
    unpaired_records: int = 0


_FUNCTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[\w:<>,~*&\[\]\\]+\s+)+"
    r"(?P<name>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*"
    r"\([^;{}]*\)\s*(?:const\s*)?(?:noexcept[^\{]*)?\{",
    re.MULTILINE,
)
_INVALID_NAMES = {"if", "for", "while", "switch", "catch"}


def infer_function_name(source: str) -> tuple[str, bool]:
    """Best-effort C/C++ function-name extraction with an explicit fallback."""

    for match in _FUNCTION_PATTERN.finditer(source):
        name = match.group("name")
        if name not in _INVALID_NAMES:
            return name, True
    return "<unknown>", False


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in (_clean_optional(v) for v in value) if item is not None]
    item = _clean_optional(value)
    return [item] if item is not None else []


def _stable_sample_id(dataset_name: str, record: dict[str, Any]) -> str:
    identity = "|".join(
        str(record.get(key, ""))
        for key in ("idx", "project", "commit_id", "file_name", "func_hash", "target")
    )
    digest = hashlib.sha256(f"{dataset_name}|{identity}".encode()).hexdigest()[:20]
    return f"primevul_{digest}"


def _split_from_name(dataset_name: str) -> Literal["train", "validation", "test", "external"]:
    name = dataset_name.lower()
    if "train" in name:
        return "train"
    if "valid" in name or "dev" in name:
        return "validation"
    if "test" in name:
        return "test"
    return "external"


def normalize_record(
    record: dict[str, Any], *, dataset_name: str, source_line: int
) -> PrimeVulSample:
    """Normalize one raw PrimeVul JSON object without losing its original form."""

    if not isinstance(record, dict):
        raise DatasetFormatError(f"line {source_line}: expected a JSON object")
    try:
        idx = int(record["idx"])
        target = int(record["target"])
        if target not in (0, 1):
            raise ValueError("target must be 0 or 1")
        func = str(record["func"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetFormatError(f"line {source_line}: missing or invalid required field: {exc}") from exc

    function_name, inferred = infer_function_name(func)
    file_name = _clean_optional(record.get("file_name"))
    quality_flags: list[str] = []
    if file_name is None:
        quality_flags.append("missing_file_name")
    if _clean_optional(record.get("file_hash")) is None:
        quality_flags.append("missing_file_hash")
    if not inferred:
        quality_flags.append("function_name_not_inferred")
    if _clean_optional(record.get("commit_id")) is None:
        quality_flags.append("missing_commit_id")

    return PrimeVulSample(
        dataset_name=dataset_name,
        source_line=source_line,
        sample_id=_stable_sample_id(dataset_name, record),
        idx=idx,
        project=_clean_optional(record.get("project")),
        commit_id=_clean_optional(record.get("commit_id")),
        project_url=_clean_optional(record.get("project_url")),
        commit_url=_clean_optional(record.get("commit_url")),
        commit_message=_clean_optional(record.get("commit_message")),
        target=target,
        func=func,
        func_hash=_clean_optional(record.get("func_hash")),
        file_name=file_name,
        file_hash=_clean_optional(record.get("file_hash")),
        cwe=_clean_list(record.get("cwe")),
        cve=_clean_optional(record.get("cve")),
        cve_desc=_clean_optional(record.get("cve_desc")),
        nvd_url=_clean_optional(record.get("nvd_url")),
        function_name=function_name,
        function_name_inferred=inferred,
        quality_flags=quality_flags,
        raw=record,
    )


def iter_primevul(
    path: str | Path,
    *,
    dataset_name: str | None = None,
    limit: int | None = None,
    strict: bool = True,
) -> Iterator[PrimeVulSample]:
    """Stream normalized samples from a PrimeVul JSONL file."""

    path = Path(path)
    name = dataset_name or path.stem
    yielded = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sample = normalize_record(record, dataset_name=name, source_line=line_number)
            except (json.JSONDecodeError, DatasetFormatError, ValueError) as exc:
                if strict:
                    raise DatasetFormatError(f"{path}:{line_number}: {exc}") from exc
                continue
            yield sample
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def link_adjacent_pairs(samples: Iterable[PrimeVulSample]) -> tuple[list[PairLink], list[PrimeVulSample]]:
    """Link adjacent opposite-label rows when metadata supports the pairing.

    The paired file is ordered vulnerable/non-vulnerable, but ordering alone is
    not accepted as evidence.  A link requires a shared project plus CVE or
    commit.  Unmatched rows remain explicitly unpaired.
    """

    rows = list(samples)
    links: list[PairLink] = []
    paired_ids: set[str] = set()
    for left, right in zip(rows[::2], rows[1::2]):
        shared: list[str] = []
        if left.project and left.project == right.project:
            shared.append("project")
        if left.cve and left.cve == right.cve:
            shared.append("cve")
        if left.commit_id and left.commit_id == right.commit_id:
            shared.append("commit_id")
        if left.target == right.target or "project" not in shared or not ({left.target, right.target} == {0, 1}):
            continue
        pair_key = "|".join(sorted((left.sample_id, right.sample_id)))
        pair_id = f"pair_{hashlib.sha256(pair_key.encode()).hexdigest()[:20]}"
        vulnerable, benign = (left, right) if left.target == 1 else (right, left)
        links.append(
            PairLink(
                pair_id=pair_id,
                vulnerable_sample_id=vulnerable.sample_id,
                non_vulnerable_sample_id=benign.sample_id,
                basis="adjacent_metadata",
                confidence="high" if {"cve", "commit_id"} & set(shared) else "medium",
                shared_fields=shared,
            )
        )
        paired_ids.update((left.sample_id, right.sample_id))
    return links, [sample for sample in rows if sample.sample_id not in paired_ids]


def apply_pair_links(samples: Iterable[PrimeVulSample], links: Iterable[PairLink]) -> list[PrimeVulSample]:
    """Return samples annotated with explicit pair IDs and roles."""

    by_id = {sample.sample_id: sample for sample in samples}
    for link in links:
        if link.vulnerable_sample_id in by_id:
            by_id[link.vulnerable_sample_id] = by_id[link.vulnerable_sample_id].model_copy(
                update={"pair_id": link.pair_id, "pair_role": "vulnerable"}
            )
        if link.non_vulnerable_sample_id in by_id:
            by_id[link.non_vulnerable_sample_id] = by_id[link.non_vulnerable_sample_id].model_copy(
                update={"pair_id": link.pair_id, "pair_role": "non_vulnerable"}
            )
    return list(by_id.values())


def quality_report(samples: Iterable[PrimeVulSample], *, paired_links: int = 0) -> DatasetQualityReport:
    """Build a deterministic quality summary without loading raw JSON again."""

    rows = list(samples)
    dataset_name = rows[0].dataset_name if rows else "unknown"
    idx_counts = Counter(row.idx for row in rows)
    id_counts = Counter(row.sample_id for row in rows)
    missing = Counter(flag for row in rows for flag in row.quality_flags)
    cwes = Counter(cwe for row in rows for cwe in row.cwe)
    lengths = [len(row.func) for row in rows]
    return DatasetQualityReport(
        dataset_name=dataset_name,
        records=len(rows),
        labels={str(label): count for label, count in Counter(row.target for row in rows).items()},
        projects=len({row.project for row in rows if row.project}),
        cwes=dict(cwes),
        missing_fields=dict(missing),
        duplicate_idx=sorted(idx for idx, count in idx_counts.items() if count > 1),
        duplicate_sample_ids=sorted(sample_id for sample_id, count in id_counts.items() if count > 1),
        function_chars_min=min(lengths) if lengths else None,
        function_chars_max=max(lengths) if lengths else None,
        function_chars_average=(sum(lengths) / len(lengths)) if lengths else None,
        paired_links=paired_links,
        unpaired_records=max(0, len(rows) - (paired_links * 2)),
    )


__all__ = [
    "DatasetFormatError",
    "DatasetQualityReport",
    "PairLink",
    "PrimeVulSample",
    "apply_pair_links",
    "infer_function_name",
    "iter_primevul",
    "link_adjacent_pairs",
    "normalize_record",
    "quality_report",
]
