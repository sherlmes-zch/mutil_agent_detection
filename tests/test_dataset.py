import json

from mutil_agent_detection.dataset import (
    apply_pair_links,
    iter_primevul,
    link_adjacent_pairs,
    normalize_record,
    quality_report,
)


def raw(idx, target, *, project="demo", cve="CVE-1", file_name="demo.c"):
    return {
        "idx": idx,
        "project": project,
        "commit_id": "abc123",
        "project_url": "https://example/demo",
        "commit_url": "https://example/commit/abc123",
        "commit_message": "fix",
        "target": target,
        "func": "static int copy_items(int n) { return n; }",
        "func_hash": idx,
        "file_name": file_name,
        "file_hash": "filehash",
        "cwe": ["CWE-787"] if target else [],
        "cve": cve,
        "cve_desc": "description",
        "nvd_url": "https://nvd/1",
    }


def test_normalize_and_map_to_analysis_input():
    sample = normalize_record(raw(1, 1), dataset_name="primevul_test", source_line=3)
    request = sample.to_analysis_input()

    assert sample.function_name == "copy_items"
    assert request.expected_label == 1
    assert request.dataset_split == "test"
    assert request.metadata["context_mode"] == "function_only"
    assert "repository context not loaded" in request.scope.unresolved_dependencies


def test_pairing_requires_opposite_labels_and_shared_metadata():
    rows = [
        normalize_record(raw(1, 1), dataset_name="paired", source_line=1),
        normalize_record(raw(2, 0), dataset_name="paired", source_line=2),
        normalize_record(raw(3, 1, project="other"), dataset_name="paired", source_line=3),
        normalize_record(raw(4, 0, project="different"), dataset_name="paired", source_line=4),
    ]
    links, unpaired = link_adjacent_pairs(rows)
    annotated = apply_pair_links(rows, links)

    assert len(links) == 1
    assert len(unpaired) == 2
    assert {row.pair_role for row in annotated[:2]} == {"vulnerable", "non_vulnerable"}


def test_quality_report_and_stream_reader(tmp_path):
    path = tmp_path / "primevul_test.jsonl"
    path.write_text("\n".join(json.dumps(raw(i, i % 2)) for i in range(1, 4)), encoding="utf-8")
    rows = list(iter_primevul(path))
    report = quality_report(rows)

    assert len(rows) == 3
    assert report.records == 3
    assert report.labels == {"1": 2, "0": 1}
    assert report.function_chars_min > 0
