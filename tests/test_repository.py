import subprocess

import pytest

from mutil_agent_detection.config import RepositoryConfig
from mutil_agent_detection.dataset import normalize_record
from mutil_agent_detection.repository import RepositoryError, RepositoryManager
from mutil_agent_detection.source_evidence import index_key_operations, locate_function_in_repository


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)


def test_repository_manager_clones_and_checks_out_exact_commit_from_local_remote(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "--initial-branch=main")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    (source / "sample.c").write_text("int sample(void) { return 1; }", encoding="utf-8")
    git(source, "add", "sample.c")
    git(source, "commit", "-m", "initial")
    commit = git(source, "rev-parse", "HEAD").stdout.strip()

    record = {
        "idx": 1,
        "target": 1,
        "func": "int sample(void) { return 1; }",
        "project": "test-project",
        "project_url": source.as_uri(),
        "commit_id": commit,
    }
    sample = normalize_record(record, dataset_name="test", source_line=1)
    manager = RepositoryManager(
        RepositoryConfig(cache_dir=tmp_path / "cache", allowed_schemes=["https", "file"], allowed_hosts=None)
    )

    worktree = manager.prepare(sample)
    assert git(worktree, "rev-parse", "HEAD").stdout.strip() == commit
    assert (worktree / "sample.c").is_file()
    assert manager.prepare(sample) == worktree


def test_repository_manager_uses_existing_matching_local_repository(tmp_path):
    source = tmp_path / "sources" / "test-project"
    source.mkdir(parents=True)
    git(source, "init", "--initial-branch=main")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    (source / "sample.c").write_text("int sample(void) { return 0; }", encoding="utf-8")
    git(source, "add", "sample.c")
    git(source, "commit", "-m", "initial")
    commit = git(source, "rev-parse", "HEAD").stdout.strip()
    sample = normalize_record(
        {
            "idx": 2,
            "target": 0,
            "func": "int sample(void) { return 0; }",
            "project": "test-project",
            "project_url": "https://github.com/test/project",
            "commit_id": commit,
        },
        dataset_name="test",
        source_line=1,
    )
    manager = RepositoryManager(
        RepositoryConfig(cache_dir=tmp_path / "managed", source_cache_dir=tmp_path / "sources")
    )

    assert manager.prepare(sample) == source.resolve()


def test_existing_repository_with_different_head_uses_requested_commit_without_switching_source(tmp_path):
    source = tmp_path / "sources" / "test-project"
    source.mkdir(parents=True)
    git(source, "init", "--initial-branch=main")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    source_file = source / "sample.c"
    source_file.write_text("int sample(void) { return 0; }", encoding="utf-8")
    git(source, "add", "sample.c")
    git(source, "commit", "-m", "vulnerable")
    target_commit = git(source, "rev-parse", "HEAD").stdout.strip()
    source_file.write_text("int sample(void) { return 1; }", encoding="utf-8")
    git(source, "commit", "-am", "fixed")
    source_head = git(source, "rev-parse", "HEAD").stdout.strip()
    sample = normalize_record(
        {
            "idx": 3,
            "target": 1,
            "func": "int sample(void) { return 0; }",
            "project": "test-project",
            "project_url": "https://github.com/test/project",
            "commit_id": target_commit,
        },
        dataset_name="test",
        source_line=1,
    )
    manager = RepositoryManager(
        RepositoryConfig(cache_dir=tmp_path / "managed", source_cache_dir=tmp_path / "sources")
    )

    worktree = manager.prepare(sample)
    assert git(worktree, "rev-parse", "HEAD").stdout.strip() == target_commit
    assert git(source, "rev-parse", "HEAD").stdout.strip() == source_head
    assert "return 0" in (worktree / "sample.c").read_text(encoding="utf-8")


def test_repository_manager_matches_declared_commit_by_normalized_function_tokens(tmp_path):
    source = tmp_path / "sources" / "test-project"
    source.mkdir(parents=True)
    git(source, "init", "--initial-branch=main")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    function = "int sample(void) {\n  return 7;\n}"
    (source / "sample.c").write_text(function + "\n", encoding="utf-8")
    git(source, "add", "sample.c")
    git(source, "commit", "-m", "matching source")
    commit = git(source, "rev-parse", "HEAD").stdout.strip()
    sample = normalize_record({
        "idx": 7, "target": 1, "func": "int sample(void){ return 7; }",
        "project": "test-project", "project_url": "https://github.com/test/project",
        "commit_id": commit, "file_name": "sample.c",
    }, dataset_name="test", source_line=1)
    manager = RepositoryManager(RepositoryConfig(
        cache_dir=tmp_path / "managed", source_cache_dir=tmp_path / "sources",
    ))

    result = manager.prepare_matching_snapshot(sample)

    assert result.status == "declared_commit"
    assert result.matched_commit == commit
    assert result.root == source.resolve()
    assert result.sample_function_hash == result.repository_function_hash


def test_repository_manager_finds_matching_first_parent_snapshot(tmp_path):
    source = tmp_path / "sources" / "test-project"
    source.mkdir(parents=True)
    git(source, "init", "--initial-branch=main")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    vulnerable = "int sample(void) { return 0; }"
    file_path = source / "sample.c"
    file_path.write_text(vulnerable + "\n", encoding="utf-8")
    git(source, "add", "sample.c")
    git(source, "commit", "-m", "vulnerable")
    parent = git(source, "rev-parse", "HEAD").stdout.strip()
    file_path.write_text("int sample(void) { return 1; }\n", encoding="utf-8")
    git(source, "commit", "-am", "fixed")
    fixed = git(source, "rev-parse", "HEAD").stdout.strip()
    sample = normalize_record({
        "idx": 8, "target": 1, "func": vulnerable,
        "project": "test-project", "project_url": "https://github.com/test/project",
        "commit_id": fixed, "file_name": "sample.c",
    }, dataset_name="test", source_line=1)
    manager = RepositoryManager(RepositoryConfig(
        cache_dir=tmp_path / "managed", source_cache_dir=tmp_path / "sources",
    ))

    result = manager.prepare_matching_snapshot(sample)

    assert result.status == "matched_first_parent"
    assert result.declared_commit == fixed
    assert result.matched_commit == parent
    assert result.checked_commits == (fixed, parent)
    assert git(result.root, "rev-parse", "HEAD").stdout.strip() == parent
    assert git(source, "rev-parse", "HEAD").stdout.strip() == fixed


def test_repository_manager_reports_version_mismatch_when_no_checked_commit_matches(tmp_path):
    source = tmp_path / "sources" / "test-project"
    source.mkdir(parents=True)
    git(source, "init", "--initial-branch=main")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    (source / "sample.c").write_text("int sample(void) { return 1; }\n", encoding="utf-8")
    git(source, "add", "sample.c")
    git(source, "commit", "-m", "different source")
    commit = git(source, "rev-parse", "HEAD").stdout.strip()
    sample = normalize_record({
        "idx": 9, "target": 1, "func": "int sample(void) { return 0; }",
        "project": "test-project", "project_url": "https://github.com/test/project",
        "commit_id": commit, "file_name": "sample.c",
    }, dataset_name="test", source_line=1)
    manager = RepositoryManager(RepositoryConfig(
        cache_dir=tmp_path / "managed", source_cache_dir=tmp_path / "sources",
        max_first_parent_commits=0,
    ))

    result = manager.prepare_matching_snapshot(sample)

    assert result.root is None
    assert result.status == "repository_version_mismatch"
    assert result.matched_commit is None
    assert "normalized function hash" in result.note


def test_matching_but_dirty_source_repository_is_not_used_as_analysis_snapshot(tmp_path):
    source = tmp_path / "sources" / "test-project"
    source.mkdir(parents=True)
    git(source, "init", "--initial-branch=main")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    source_file = source / "sample.c"
    source_file.write_text("int sample(void) { return 0; }", encoding="utf-8")
    git(source, "add", "sample.c")
    git(source, "commit", "-m", "initial")
    commit = git(source, "rev-parse", "HEAD").stdout.strip()
    source_file.write_text("int sample(void) { return 999; }", encoding="utf-8")
    sample = normalize_record(
        {
            "idx": 4,
            "target": 0,
            "func": "int sample(void) { return 0; }",
            "project": "test-project",
            "project_url": "https://github.com/test/project",
            "commit_id": commit,
        },
        dataset_name="test",
        source_line=1,
    )
    manager = RepositoryManager(
        RepositoryConfig(cache_dir=tmp_path / "managed", source_cache_dir=tmp_path / "sources")
    )

    worktree = manager.prepare(sample)

    assert worktree != source.resolve()
    assert git(worktree, "rev-parse", "HEAD").stdout.strip() == commit
    assert "return 0" in (worktree / "sample.c").read_text(encoding="utf-8")
    assert "return 999" in source_file.read_text(encoding="utf-8")


def test_repository_manager_rejects_non_https_by_default(tmp_path):
    record = {
        "idx": 1,
        "target": 0,
        "func": "int sample(void) { return 0; }",
        "project_url": "file:///tmp/example",
        "commit_id": "a" * 40,
    }
    sample = normalize_record(record, dataset_name="test", source_line=1)
    manager = RepositoryManager(RepositoryConfig(cache_dir=tmp_path / "cache"))
    with pytest.raises(RepositoryError, match="scheme is not allowed"):
        manager.prepare(sample)


def test_analysis_input_uses_exact_repository_lines_and_indexes_sensitive_operations(tmp_path):
    source = """#include <stddef.h>
void inspect(char *p, int i) {
    char *q = (char *)p;
    if (i > 0) {
        q++;
        p[i] = *q;
    }
}
"""
    (tmp_path / "sample.c").write_text(source, encoding="utf-8")
    git(tmp_path, "init", "--initial-branch=main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "add", "sample.c")
    git(tmp_path, "commit", "-m", "source snapshot")
    commit = git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    function = """void inspect(char *p, int i) {
    char *q = (char *)p;
    if (i > 0) {
        q++;
        p[i] = *q;
    }
}"""
    sample = normalize_record(
        {
            "idx": 5,
            "target": 1,
            "func": function,
            "file_name": "sample.c",
            "commit_id": commit,
        },
        dataset_name="test",
        source_line=1,
    )

    request = sample.to_analysis_input(context_mode="repository", repository_root=str(tmp_path))

    assert request.metadata["source_line_basis"] == "repository_absolute"
    assert request.metadata["source_line_start"] == 2
    assert request.metadata["source_line_end"] == 8
    assert request.scope.locations[0].line == 2
    assert request.scope.locations[0].end_line == 8
    indexed = {item["line"]: item["categories"] for item in request.metadata["key_operation_index"]}
    assert "cast" in indexed[3]
    assert "condition" in indexed[4]
    assert "pointer_step" in indexed[5]
    assert "index" in indexed[6]
    assert "dereference" in indexed[6]


def test_repository_line_mapping_refuses_a_different_commit(tmp_path):
    git(tmp_path, "init", "--initial-branch=main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "sample.c").write_text("int sample(void) { return 0; }\n", encoding="utf-8")
    git(tmp_path, "add", "sample.c")
    git(tmp_path, "commit", "-m", "source snapshot")
    sample = normalize_record(
        {
            "idx": 6,
            "target": 0,
            "func": "int sample(void) { return 0; }",
            "file_name": "sample.c",
            "commit_id": "f" * 40,
        },
        dataset_name="test",
        source_line=1,
    )

    request = sample.to_analysis_input(context_mode="repository", repository_root=str(tmp_path))

    assert request.metadata["source_line_basis"] == "function_local"
    assert request.metadata["source_line_start"] is None
    assert "HEAD mismatch" in request.metadata["source_line_mapping_note"]
    assert any("repository line mapping unavailable" in item for item in request.scope.unresolved_dependencies)


def test_key_operation_index_exposes_narrow_integer_conversion():
    indexed = index_key_operations(
        "unsigned char f(double pixel) { return ((unsigned char) pixel) & 7; }"
    )
    row = next(item for item in indexed if "narrow_integer_conversion" in item["categories"])
    assert row["conversion_target_type"] == "unsigned char"
    assert "representable" in row["representation_requirement"]


def test_repository_line_mapping_resolves_unique_basename(tmp_path):
    nested = tmp_path / "coders"
    nested.mkdir()
    source = "int helper(void) { return 0; }\n\nint target(void) {\n  return 7;\n}\n"
    (nested / "sample.c").write_text(source, encoding="utf-8")
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "coders/sample.c"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "-m", "sample"], check=True, capture_output=True)
    commit = git(tmp_path, "rev-parse", "HEAD").stdout.strip()

    mapped = locate_function_in_repository(
        "int target(void) { return 7; }", "sample.c", tmp_path, expected_commit=commit,
    )

    assert mapped["basis"] == "repository_absolute"
    assert mapped["start_line"] == 3
    assert mapped["end_line"] == 5
