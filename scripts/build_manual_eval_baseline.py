#!/usr/bin/env python3
"""Build a frozen 20-sample PrimeVul paired pilot set and annotation sheet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/primevul_test_paired.jsonl"
SUBSET = ROOT / "datasets/primevul_manual_eval_20.jsonl"
ANNOTATIONS = ROOT / "doc/primevul_manual_eval_annotations.jsonl"

# Pair indices are fixed before examining model outputs. Each selected pair has
# one vulnerable and one non-vulnerable function from the supplied paired set.
PAIR_INDICES = [0, 2, 7, 11, 16, 19, 26, 27, 30, 58]

PAIR_NOTES = {
    0: {
        "root_cause": "EXIF bytes are cast to float* and double* and dereferenced without establishing the required alignment.",
        "vulnerable_evidence": ["EXIFMultipleValues(4,\"%f\",(double) *(float *) p1);", "EXIFMultipleValues(8,\"%f\",*(double *) p1);"],
        "trigger": "A crafted EXIF profile reaches the single/double format branch with p1 not aligned for the target floating-point type.",
        "protection_to_check": "Length/offset checks do not establish address alignment; do not count them as alignment protection.",
        "confidence": "high",
        "basis": "CVE description plus vulnerable/fixed paired function diff.",
    },
    2: {
        "root_cause": "Kernel creation can fail after assigning an allocated kernel to item->kernel; the error path overwrites the pointer with nullptr without first releasing it.",
        "vulnerable_evidence": ["Status s = params_.create_kernel(n->properties(), &item->kernel);", "item->kernel = nullptr;"],
        "trigger": "Kernel creation returns an error after partially creating/assigning a kernel object; confirm API ownership semantics against repository context.",
        "protection_to_check": "Error handling must release any partially-created kernel before clearing the owning pointer.",
        "confidence": "high",
        "basis": "CVE description plus paired fix adding params_.delete_kernel(item->kernel).",
    },
    7: {
        "root_cause": "The failed-credentials branch updates the existing session for the requested username before returning HTTP 401, allowing failed authentication to mutate authenticated session identity/state.",
        "vulnerable_evidence": ["user_session_update(config, session_uid, u_map_get_case(request->map_header, \"user-agent\"), issued_for, json_string_value(json_object_get(j_param, \"username\")), NULL, 1)"],
        "trigger": "A request has an existing session and submits invalid credentials for another username; exact session semantics should be confirmed from repository code.",
        "protection_to_check": "The 401 response does not undo a prior session update; verify authorization state and session persistence.",
        "confidence": "medium",
        "basis": "CVE description plus paired fix removing session update from failed-authentication branch.",
    },
    11: {
        "root_cause": "An attacker-controlled input index is used to access node_t.args(ix), while the only range assertion is DCHECK and is absent in production builds.",
        "vulnerable_evidence": ["DCHECK(ix < node_t.args_size())", "input_types.emplace_back(node_t.args(ix));"],
        "trigger": "Malformed graph/type information supplies ix >= node_t.args_size() in a production build.",
        "protection_to_check": "A debug-only DCHECK is not a production bounds guard; an explicit check must precede args(ix).",
        "confidence": "high",
        "basis": "CVE description plus paired fix replacing DCHECK with a runtime check and early return.",
    },
    16: {
        "root_cause": "BiasAndClamp performs a modulo by bias_size and uses bias_size as a loop step without rejecting zero, causing division/modulo-by-zero or invalid loop behavior.",
        "vulnerable_evidence": ["TFLITE_DCHECK_EQ((array_size % bias_size), 0);", "array_offset += bias_size"],
        "trigger": "A crafted TFLite model reaches BiasAndClamp with bias_size == 0; whether the check is compiled in is build-configuration dependent.",
        "protection_to_check": "A debug assertion must not be treated as an unconditional runtime validation.",
        "confidence": "high",
        "basis": "CVE description plus paired fix adding `if (bias_size == 0) return;`.",
    },
    19: {
        "root_cause": "The parsed pps_id is checked only against the upper bound before indexing avc->pps; the paired fix also rejects negative values.",
        "vulnerable_evidence": ["if (pps_id > 255)", "si->pps = &avc->pps[pps_id];"],
        "trigger": "Malformed bitstream produces a negative pps_id that passes the upper-bound-only check; validate parser return semantics in repository context.",
        "protection_to_check": "Upper-bound validation alone does not establish a valid non-negative index.",
        "confidence": "medium",
        "basis": "CVE description plus paired fix adding a negative-value check.",
    },
    26: {
        "root_cause": "The TLS client sends hostname indication but does not configure certificate hostname verification, so a trusted certificate for a different host may be accepted.",
        "vulnerable_evidence": ["SSL_set_tlsext_host_name(ssl, hostname.c_str());", "int rc = SSL_connect(ssl);"],
        "trigger": "SSL MITM client mode connects to a server presenting a trusted certificate whose subject/SAN does not match hostname.",
        "protection_to_check": "SNI is routing metadata, not hostname validation; chain validation alone is insufficient.",
        "confidence": "high",
        "basis": "CVE description plus paired fix adding X509_VERIFY_PARAM_set1_host for OpenSSL 1.1+.",
    },
    27: {
        "root_cause": "The shared /tmp session directory is created and chmodded with mode 0703, allowing other local users to access/modify session entries; ownership/mode are not revalidated.",
        "vulnerable_evidence": ["mkdir(TMATE_WORKDIR \"/sessions\", 0703)", "chmod(TMATE_WORKDIR \"/sessions\", 0703)"],
        "trigger": "A local attacker exploits world access to the session directory to inspect or replace session symlinks/identifiers.",
        "protection_to_check": "The 0700 permissions and ownership/mode checks in the paired fixed version are the relevant protection.",
        "confidence": "high",
        "basis": "CVE description plus paired fix changing directory modes and validating ownership/mode.",
    },
    30: {
        "root_cause": "SetUnknownShape forwards output_port to ctx->set_output without checking that it is within [0, ctx->num_outputs()).", 
        "vulnerable_evidence": ["ctx->set_output(output_port, shape);"],
        "trigger": "An invalid output_port reaches SetUnknownShape and is used as an output-array index.",
        "protection_to_check": "The non-null context check protects ctx existence but not output_port bounds.",
        "confidence": "high",
        "basis": "CVE description plus paired fix adding explicit lower/upper output_port bounds validation.",
    },
    58: {
        "root_cause": "SparseTensorSliceDataset validates some tensor ranks but omits consistency checks among indices, values, and dense_shape before constructing the SparseTensor; malformed dimensions can reach unsafe downstream handling.",
        "vulnerable_evidence": ["SparseTensor::Create(\n                 *indices, *values, TensorShape(dense_shape->vec<int64_t>()),"],
        "trigger": "A crafted sparse tensor supplies inconsistent ranks/dimensions or an empty shape that violates SparseTensor construction preconditions.",
        "protection_to_check": "Rank checks alone do not establish dimension-count and element-count consistency; the paired version adds these guards.",
        "confidence": "high",
        "basis": "CVE description plus paired fix adding shape and dimensionality precondition checks.",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    source_rows = read_jsonl(SOURCE)
    subset: list[dict] = []
    annotations: list[dict] = []
    for pair_index in PAIR_INDICES:
        vulnerable, non_vulnerable = source_rows[2 * pair_index : 2 * pair_index + 2]
        if (vulnerable.get("target"), non_vulnerable.get("target")) != (1, 0):
            raise ValueError(f"unexpected labels at paired row {pair_index}")
        if vulnerable.get("cve") != non_vulnerable.get("cve"):
            raise ValueError(f"pair metadata mismatch at paired row {pair_index}")
        pair_id = f"manual-pair-{pair_index:03d}"
        note = PAIR_NOTES[pair_index]
        subset.extend((vulnerable, non_vulnerable))
        annotations.append(
            {
                "pair_id": pair_id,
                "source_pair_index": pair_index,
                "cve": vulnerable.get("cve"),
                "cwe": vulnerable.get("cwe", []),
                "project": vulnerable.get("project"),
                "file_name": vulnerable.get("file_name"),
                "vulnerable_sample": {"idx": vulnerable["idx"], "target": 1},
                "paired_non_vulnerable_sample": {"idx": non_vulnerable["idx"], "target": 0},
                **note,
                "annotation_status": "provisional_manual_review",
                "agent_risk_root_cause_score": None,
                "agent_risk_location_score": None,
                "agent_risk_trigger_score": None,
                "agent_protection_match_score": None,
                "unsupported_claims": None,
                "reviewer_notes": None,
            }
        )

    SUBSET.parent.mkdir(parents=True, exist_ok=True)
    SUBSET.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset), encoding="utf-8")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for row in annotations:
        row["source_dataset"] = str(SOURCE.relative_to(ROOT))
        row["source_dataset_sha256"] = digest
    ANNOTATIONS.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in annotations), encoding="utf-8")
    print(f"wrote {len(subset)} samples to {SUBSET.relative_to(ROOT)}")
    print(f"wrote {len(annotations)} pair annotations to {ANNOTATIONS.relative_to(ROOT)}")
    print(f"source_sha256={digest}")


if __name__ == "__main__":
    main()
