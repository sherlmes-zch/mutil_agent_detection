"""Precise function-to-repository line mapping and compact operation indexing."""

from __future__ import annotations

import re
import hashlib
import subprocess
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(
    r"//[^\n]*|/\*[\s\S]*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|"
    r"[A-Za-z_]\w*|(?:0[xX][0-9A-Fa-f]+|\d+)|"
    r"->|\+\+|--|\+=|-=|<<=?|>>=?|==|!=|<=|>=|&&|\|\||[^\s]"
)
_COMMENT_RE = re.compile(r"^(?://|/\*)")
_PATTERNS = {
    "cast": re.compile(r"\(\s*(?:(?:const|volatile|unsigned|signed|struct|enum)\s+)*[A-Za-z_]\w*(?:\s*[*&]+|\s*\[[^]]*\])\s*\)"),
    "alignment_sensitive_cast": re.compile(r"\(\s*(?:const\s+|volatile\s+)*(?:float|double)\s*\*\s*\)"),
    "typed_dereference": re.compile(r"\*\s*\(\s*(?:const\s+|volatile\s+)*(?:float|double)\s*\*\s*\)"),
    "integer_conversion": re.compile(
        r"\(\s*(?:(?:const|volatile)\s+)*(?:(?:unsigned|signed)\s+)?"
        r"(?:char|short|int|long(?:\s+long)?|size_t|ssize_t|int\d+_t|uint\d+_t)\s*\)"
    ),
    "narrow_integer_conversion": re.compile(
        r"\(\s*(?:(?:unsigned|signed)\s+)?(?:char|short|int8_t|uint8_t|int16_t|uint16_t)\s*\)"
    ),
    "dereference": re.compile(r"(?<![\w)])\*\s*(?:[A-Za-z_]\w*|\([^\n)]*\))|\b[A-Za-z_]\w*\s*->\s*[A-Za-z_]\w*"),
    "index": re.compile(r"\b[A-Za-z_]\w*(?:\s*(?:->|\.)\s*[A-Za-z_]\w*)*\s*\[[^\]\n]+\]"),
    "pointer_step": re.compile(r"(?:\+\+\s*[A-Za-z_]\w*|--\s*[A-Za-z_]\w*|\b[A-Za-z_]\w*\s*(?:\+\+|--)|\b[A-Za-z_]\w*\s*(?:\+=|-=)\s*[^;,)]+|\b[A-Za-z_]\w*\s*[+-]\s*(?:[A-Za-z_]\w*|\d+))"),
    "condition": re.compile(r"\b(?:if|while|for|switch)\s*\(|\?"),
}


def locate_function_in_repository(
    function_source: str,
    file_name: str | None,
    repository_root: str | Path | None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Locate exact function tokens in the supplied commit snapshot; never guess on ambiguity."""
    if not file_name or not repository_root:
        return {"start_line": None, "end_line": None, "basis": "function_local", "note": "repository file location unavailable"}
    root = Path(repository_root).expanduser().resolve()
    if not expected_commit:
        return {"start_line": None, "end_line": None, "basis": "function_local", "note": "sample commit_id unavailable; repository line mapping not trusted"}
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"start_line": None, "end_line": None, "basis": "function_local", "note": f"cannot verify repository HEAD: {exc}"}
    actual_commit = result.stdout.strip().lower()
    if actual_commit != expected_commit.lower():
        return {"start_line": None, "end_line": None, "basis": "function_local", "note": f"repository HEAD mismatch: expected {expected_commit}, found {actual_commit}"}
    target = (root / file_name).resolve()
    if not target.is_relative_to(root):
        return {"start_line": None, "end_line": None, "basis": "function_local", "note": "sample file path escapes repository root"}
    if not target.is_file():
        # PrimeVul commonly stores only a basename (for example ``psd.c``),
        # while projects keep the file below a component directory. Resolve
        # only a unique tracked basename; ambiguity must remain untrusted.
        try:
            tracked = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z"],
                check=True, capture_output=True, timeout=30,
            ).stdout.split(b"\0")
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "start_line": None, "end_line": None, "basis": "function_local",
                "note": f"cannot resolve repository basename: {exc}",
            }
        candidates = [
            (root / raw.decode("utf-8", errors="replace")).resolve()
            for raw in tracked if raw and Path(raw.decode("utf-8", errors="replace")).name == Path(file_name).name
        ]
        candidates = [path for path in candidates if path.is_file() and path.is_relative_to(root)]
        if len(candidates) == 1:
            target = candidates[0]
        elif not candidates:
            return {
                "start_line": None, "end_line": None, "basis": "function_local",
                "note": f"repository file {file_name!r} was not found at the validated commit",
            }
        else:
            return {
                "start_line": None, "end_line": None, "basis": "function_local",
                "note": f"repository basename {file_name!r} is ambiguous at the validated commit",
            }
    try:
        repository_source = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"start_line": None, "end_line": None, "basis": "function_local", "note": f"cannot read target file: {exc}"}

    function_tokens = _tokens(function_source)
    source_tokens = _tokens(repository_source)
    if not function_tokens:
        return {"start_line": None, "end_line": None, "basis": "function_local", "note": "function contains no matchable source tokens"}
    starts = [index for index, token in enumerate(source_tokens) if token[0] == function_tokens[0][0]]
    matches = [
        index for index in starts
        if source_tokens[index:index + len(function_tokens)]
        and [token[0] for token in source_tokens[index:index + len(function_tokens)]] == [token[0] for token in function_tokens]
    ]
    if len(matches) != 1:
        reason = "function token sequence not found" if not matches else "function token sequence is ambiguous"
        return {"start_line": None, "end_line": None, "basis": "function_local", "note": reason}

    first = source_tokens[matches[0]][1]
    last = source_tokens[matches[0] + len(function_tokens) - 1][2] - 1
    return {
        "start_line": repository_source.count("\n", 0, first) + 1,
        "end_line": repository_source.count("\n", 0, last) + 1,
        "basis": "repository_absolute",
        "note": None,
    }


def _tokens(source: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(), match.start(), match.end())
        for match in _TOKEN_RE.finditer(source)
        if not _COMMENT_RE.match(match.group())
    ]


def normalized_function_hash(source: str) -> str:
    """Hash C/C++ tokens, ignoring whitespace and comments but preserving literals/operators."""
    material = "\x1f".join(token for token, _, _ in _tokens(source))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def function_matches_repository(
    function_source: str,
    file_name: str | None,
    repository_root: str | Path,
    *,
    expected_commit: str,
) -> tuple[bool, str | None]:
    """Check whether the exact normalized function token sequence exists in a pinned file."""
    if not file_name:
        return False, "sample file path is unavailable"
    root = Path(repository_root).expanduser().resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"cannot verify repository HEAD: {exc}"
    if result.stdout.strip().lower() != expected_commit.lower():
        return False, f"repository HEAD mismatch: expected {expected_commit}, found {result.stdout.strip()}"

    relative = Path(file_name)
    target = (root / relative).resolve()
    if target.is_relative_to(root) and target.is_file():
        files = [target]
    else:
        try:
            tracked = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z"],
                check=True, capture_output=True, timeout=30,
            ).stdout.split(b"\0")
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"cannot enumerate repository paths: {exc}"
        files = [
            (root / item.decode("utf-8", errors="replace")).resolve()
            for item in tracked if item and Path(item.decode("utf-8", errors="replace")).name == relative.name
        ]
        files = [path for path in files if path.is_relative_to(root) and path.is_file()]

    expected_tokens = [token for token, _, _ in _tokens(function_source)]
    if not expected_tokens:
        return False, "sample function contains no matchable tokens"
    for path in files:
        try:
            actual_tokens = [token for token, _, _ in _tokens(path.read_text(encoding="utf-8", errors="replace"))]
        except OSError:
            continue
        width = len(expected_tokens)
        if any(actual_tokens[index:index + width] == expected_tokens for index in range(len(actual_tokens) - width + 1)):
            return True, None
    return False, f"normalized function hash {normalized_function_hash(function_source)} not found in {file_name} at {expected_commit}"


def index_key_operations(function_source: str, *, start_line: int | None = None) -> list[dict[str, Any]]:
    """Index source lines containing pointer-sensitive operations or control conditions."""
    base_line = start_line or 1
    entries: list[dict[str, Any]] = []
    for local_line, code in enumerate(function_source.splitlines(), start=1):
        excerpt = code.strip()
        if not excerpt:
            continue
        categories = [name for name, pattern in _PATTERNS.items() if pattern.search(code)]
        if categories:
            entry = {"line": base_line + local_line - 1, "categories": categories, "code": excerpt[:300]}
            if "alignment_sensitive_cast" in categories or "typed_dereference" in categories:
                entry["access_type"] = "float_or_double"
                entry["alignment_requirement"] = "natural alignment for the target scalar type"
                entry["address_source"] = "pointer expression inside the cast"
            if "integer_conversion" in categories:
                cast = _PATTERNS["integer_conversion"].search(code)
                entry["conversion_target_type"] = cast.group(0).strip("() ") if cast else "integer type"
                entry["conversion_source_expression"] = code[cast.end():].strip()[:180] if cast else "expression after cast"
                entry["representation_requirement"] = (
                    "source value must be representable in the target integer type; inspect signedness, "
                    "width, nested casts, and subsequent shifts/masks"
                )
            entries.append(entry)
    if len(entries) > 120:
        entries = entries[:120] + [{"truncated": True, "omitted_entries": len(entries) - 120}]
    return entries


__all__ = [
    "function_matches_repository", "index_key_operations", "locate_function_in_repository",
    "normalized_function_hash",
]
