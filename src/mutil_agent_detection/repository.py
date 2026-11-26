"""Cached, exact-commit PrimeVul repository acquisition."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .config import RepositoryConfig
from .dataset import PrimeVulSample
from .source_evidence import function_matches_repository, normalized_function_hash


class RepositoryError(RuntimeError):
    """Repository could not be safely acquired at the requested commit."""


@dataclass(frozen=True)
class RepositorySnapshotResolution:
    root: Path | None
    declared_commit: str
    matched_commit: str | None
    status: str
    sample_function_hash: str
    repository_function_hash: str | None
    checked_commits: tuple[str, ...]
    note: str | None = None


class RepositoryManager:
    """Clone/fetch each project once and create immutable detached worktrees."""

    def __init__(self, config: RepositoryConfig | None = None):
        self.config = config or RepositoryConfig()

    def prepare(self, sample: PrimeVulSample) -> Path:
        if not sample.project_url:
            raise RepositoryError(f"sample {sample.sample_id} has no project_url")
        if not sample.commit_id or not re.fullmatch(r"[0-9a-fA-F]{7,40}", sample.commit_id):
            raise RepositoryError(f"sample {sample.sample_id} has no valid Git commit_id")
        url = self._validate_url(sample.project_url)
        project_key = hashlib.sha256(url.encode()).hexdigest()[:16]
        project_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample.project or "project")[:80]
        local_root = self._find_local_repository(project_slug, sample.commit_id.lower())
        if (
            local_root is not None
            and self._read_commit(local_root) == sample.commit_id.lower()
            and self._is_clean_repository(local_root)
        ):
            return local_root
        project_dir = self.config.cache_dir / f"{project_slug}-{project_key}"
        project_dir.mkdir(parents=True, exist_ok=True)
        lock_path = project_dir / ".repository.lock"

        with lock_path.open("a+") as lock:
            self._lock(lock)
            mirror = project_dir / "mirror.git"
            if not mirror.exists():
                self._clone_mirror(url, project_dir, mirror, local_root=local_root)
            else:
                self._validate_origin(mirror, url)

            commit = sample.commit_id.lower()
            if not self._has_commit(mirror, commit):
                self._fetch_commit(mirror, commit)
            if not self._has_commit(mirror, commit):
                raise RepositoryError(f"commit {commit} is unavailable in repository {sample.project_url}")

            worktree = project_dir / "worktrees" / commit
            if worktree.exists():
                actual = self._read_commit(worktree)
                if actual != commit:
                    raise RepositoryError(f"cached worktree mismatch at {worktree}: found {actual}")
                return worktree

            worktree.parent.mkdir(parents=True, exist_ok=True)
            self._git(["--git-dir", str(mirror), "worktree", "add", "--detach", str(worktree), commit])
            actual = self._read_commit(worktree)
            if actual != commit:
                raise RepositoryError(f"checkout verification failed: expected {commit}, found {actual}")
            return worktree

    def prepare_matching_snapshot(
        self,
        sample: PrimeVulSample,
        *,
        max_first_parent_commits: int | None = None,
    ) -> RepositorySnapshotResolution:
        """Resolve the nearest matching source snapshot on the declared commit's first-parent chain."""
        if not sample.commit_id:
            raise RepositoryError(f"sample {sample.sample_id} has no declared commit")
        limit = (
            self.config.max_first_parent_commits
            if max_first_parent_commits is None else max_first_parent_commits
        )
        if limit < 0:
            raise ValueError("max_first_parent_commits must be non-negative")

        declared = sample.commit_id.lower()
        function_hash = normalized_function_hash(sample.func)
        current_sample = sample
        current_commit = declared
        checked: list[str] = []
        try:
            initial_root = self.prepare(current_sample)
        except RepositoryError as exc:
            return RepositorySnapshotResolution(
                root=None, declared_commit=declared, matched_commit=None,
                status="repository_version_mismatch", sample_function_hash=function_hash,
                repository_function_hash=None, checked_commits=(),
                note=f"declared repository snapshot unavailable: {exc}",
            )
        current_root = initial_root

        for depth in range(limit + 1):
            checked.append(current_commit)
            matches, note = function_matches_repository(
                sample.func, sample.file_name, current_root, expected_commit=current_commit,
            )
            if matches:
                return RepositorySnapshotResolution(
                    root=current_root, declared_commit=declared, matched_commit=current_commit,
                    status="declared_commit" if depth == 0 else "matched_first_parent",
                    sample_function_hash=function_hash, repository_function_hash=function_hash,
                    checked_commits=tuple(checked),
                    note=None if depth == 0 else f"function matched first-parent ancestor at depth {depth}",
                )

            if depth >= limit:
                break
            parent_result = self._git(
                ["-C", str(current_root), "rev-parse", "--verify", "HEAD^1"],
                check=False, capture_output=True,
            )
            parent = parent_result.stdout.strip().lower() if parent_result.returncode == 0 else ""
            if not parent or not re.fullmatch(r"[0-9a-f]{40}", parent):
                break
            current_commit = parent
            current_sample = sample.model_copy(update={"commit_id": current_commit})
            try:
                current_root = self.prepare(current_sample)
            except RepositoryError as exc:
                return RepositorySnapshotResolution(
                    root=None, declared_commit=declared, matched_commit=None, status="repository_version_mismatch",
                    sample_function_hash=function_hash, repository_function_hash=None,
                    checked_commits=tuple(checked), note=f"first-parent snapshot unavailable: {exc}",
                )

        return RepositorySnapshotResolution(
            root=None, declared_commit=declared, matched_commit=None, status="repository_version_mismatch",
            sample_function_hash=function_hash, repository_function_hash=None,
            checked_commits=tuple(checked),
            note=(note or f"no matching function found within {len(checked)} checked first-parent commit(s)"),
        )

    def _validate_url(self, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme.lower() not in self.config.allowed_schemes:
            raise RepositoryError(f"repository URL scheme is not allowed: {parsed.scheme or '<missing>'}")
        if parsed.username or parsed.password:
            raise RepositoryError("repository URLs containing embedded credentials are not allowed")
        if parsed.scheme.lower() != "file" and not parsed.hostname:
            raise RepositoryError("repository URL has no hostname")
        if self.config.allowed_hosts and parsed.hostname not in self.config.allowed_hosts:
            raise RepositoryError(f"repository URL host is not allowed: {parsed.hostname}")
        return value.rstrip("/")

    def _clone_mirror(
        self,
        url: str,
        project_dir: Path,
        mirror: Path,
        *,
        local_root: Path | None = None,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="clone-", dir=project_dir) as temp_dir:
            staged_mirror = Path(temp_dir) / "mirror.git"
            if local_root is not None:
                self._git(["clone", "--bare", "--no-hardlinks", "--no-tags", str(local_root), str(staged_mirror)])
                self._git(["--git-dir", str(staged_mirror), "remote", "set-url", "origin", url])
            else:
                self._git(["clone", "--bare", "--filter=blob:none", "--no-tags", url, str(staged_mirror)])
            self._git(["--git-dir", str(staged_mirror), "rev-parse", "--is-bare-repository"])
            os.replace(staged_mirror, mirror)

    def _find_local_repository(self, project_slug: str, commit: str) -> Path | None:
        source_cache = self.config.source_cache_dir
        if source_cache is None:
            return None
        candidates = [source_cache / project_slug / commit, source_cache / project_slug]
        for candidate in candidates:
            if candidate.is_dir() and self._is_git_repository(candidate):
                if self._has_local_commit(candidate, commit):
                    return candidate.resolve()
        return None

    def _is_git_repository(self, root: Path) -> bool:
        result = self._git(["-C", str(root), "rev-parse", "--is-inside-work-tree"], check=False, capture_output=True)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _is_clean_repository(self, root: Path) -> bool:
        result = self._git(
            ["-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=False,
            capture_output=True,
        )
        return result.returncode == 0 and not result.stdout.strip()

    def _has_local_commit(self, root: Path, commit: str) -> bool:
        result = self._git(
            ["-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
            capture_output=True,
        )
        return result.returncode == 0

    def _validate_origin(self, mirror: Path, url: str) -> None:
        result = self._git(["--git-dir", str(mirror), "remote", "get-url", "origin"], capture_output=True)
        if result.stdout.strip().rstrip("/") != url:
            raise RepositoryError(f"cached repository origin mismatch at {mirror}")

    def _has_commit(self, mirror: Path, commit: str) -> bool:
        result = self._git(
            ["--git-dir", str(mirror), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
            capture_output=True,
        )
        return result.returncode == 0

    def _fetch_commit(self, mirror: Path, commit: str) -> None:
        result = self._git(
            ["--git-dir", str(mirror), "fetch", "--no-tags", "origin", commit],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            # Some servers disallow fetching an unadvertised SHA. Fetch refs,
            # then verify the requested commit before creating a worktree.
            self._git(
                [
                    "--git-dir",
                    str(mirror),
                    "fetch",
                    "--no-tags",
                    "--filter=blob:none",
                    "origin",
                    "+refs/heads/*:refs/remotes/origin/*",
                ]
            )

    def _read_commit(self, root: Path) -> str:
        result = self._git(["-C", str(root), "rev-parse", "HEAD"], capture_output=True)
        return result.stdout.strip().lower()

    def _lock(self, lock_file) -> None:
        deadline = time.monotonic() + self.config.lock_timeout_seconds
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RepositoryError("timed out waiting for project repository lock")
                time.sleep(0.1)

    def _git(self, args: list[str], *, check: bool = True, capture_output: bool = False):
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(
            [self.config.git_executable, *args],
            check=False,
            capture_output=capture_output,
            text=True,
            timeout=self.config.timeout_seconds,
            env=env,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout or "git command failed")[-3000:]
            raise RepositoryError(f"git {args[0]} failed with exit {result.returncode}: {detail}")
        return result


__all__ = ["RepositoryError", "RepositoryManager", "RepositorySnapshotResolution"]
