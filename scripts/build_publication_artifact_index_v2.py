"""Fresh full publication index audit, extending immutable v1 coverage.

Reuse the original inclusion and semantic gates in an isolated module instance;
do not rewrite v1, the canonical index, submission packages or scientific data.
Hash shared dependencies once, then independently rehash the entire observed
ancestry before publication. Audit completion is not experiment/publication readiness.
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.export_public_evidence import sanitize_value, scan_unsafe

EXTRA_INCLUDE = (
    "common_mechanism_presentation*_v1_*/presentation.manifest.json",
    "age_probe_absolute_presentation*_v1_*/presentation.manifest.json",
    "common_pair_protocol*_v1_*/summary.manifest.json",
    "common_pair_protocol*_v2_*/summary.manifest.json",
)
CURRENT_REQUIRED = (
    "common_mechanism_presentation6_v1_20261007/presentation.manifest.json",
    "age_probe_absolute_presentation11_v1_20261007/presentation.manifest.json",
    "common_pair_protocol6_v1_20261007/summary.manifest.json",
)
HEX = re.compile(r"[0-9a-f]{64}\Z")


def isolated_legacy(root, output):
    path = Path(__file__).with_name("build_publication_artifact_index.py")
    spec = importlib.util.spec_from_file_location("_publication_index_v2_legacy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJECT_ROOT = root
    module.METRICS = root / "metrics"
    module.OUTPUT = output
    module.INCLUDE = tuple(dict.fromkeys((*module.INCLUDE, *EXTRA_INCLUDE, *CURRENT_REQUIRED)))
    return module


class IntegrityAudit:
    """Current physical snapshots, not acceptance of an earlier declared checksum."""

    def __init__(self, legacy):
        self.legacy = legacy
        self.snapshots = {}

    def snapshot(self, path):
        path = path.resolve()
        if path not in self.snapshots:
            self.snapshots[path] = file_record(path)
        return self.snapshots[path]

    def check(self, records):
        if not isinstance(records, list):
            raise ValueError("manifest record list required")
        checked, private_missing, private_mismatch = 0, 0, 0
        missing, mismatch = [], []
        for record in records:
            if (not isinstance(record, dict) or not isinstance(record.get("path"), str)
                    or not record["path"].strip() or type(record.get("bytes")) is not int
                    or record["bytes"] < 0 or not isinstance(record.get("sha256"), str)
                    or not HEX.fullmatch(record["sha256"])):
                raise ValueError("well-formed native file record required")
            path = self.legacy._resolve_record(record)
            private = self.legacy._private_record(record)
            if not path.is_file():
                if private:
                    private_missing += 1
                else:
                    missing.append(record["path"])
                continue
            actual = self.snapshot(path)
            checked += 1
            if actual["sha256"] != record["sha256"] or actual["bytes"] != record["bytes"]:
                if private:
                    private_mismatch += 1
                else:
                    mismatch.append(record["path"])
        return dict(checked=checked, missing=missing, checksum_mismatch=mismatch,
                    private_missing_count=private_missing,
                    private_checksum_mismatch_count=private_mismatch,
                    valid=not missing and not mismatch and not private_missing and not private_mismatch)

    def guard(self):
        # Real byte rehash, not reliance on unchanged size/mtime or the discovery cache.
        if any(file_record(path) != before for path, before in self.snapshots.items()):
            raise RuntimeError("observed index ancestry changed")


def manifest_paths(legacy):
    return {p.resolve() for pattern in legacy.INCLUDE for p in legacy.METRICS.glob(pattern)}


def hide_controlled_paths(value):
    if isinstance(value, dict):
        return {key: hide_controlled_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [hide_controlled_paths(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/").lower()
        if "/private/" in normalized or "data/interim/faces/" in normalized:
            return "<controlled-access-artifact>"
    return value


def run(root, out):
    root, out = root.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError("fresh index-audit output root required")
    output = out / "publication_artifact_index.json"
    legacy = isolated_legacy(root, out / "private/legacy_index.json")
    audit = IntegrityAudit(legacy)
    selected = manifest_paths(legacy)
    for path in sorted(selected):
        audit.snapshot(path)
    for path in (Path(__file__), Path(legacy.__file__),
                 PROJECT_ROOT / "scripts/export_public_evidence.py",
                 PROJECT_ROOT / "src/age_gap/common/manifest.py",
                 PROJECT_ROOT / "src/age_gap/common/io.py"):
        audit.snapshot(path)
    # v1 reads these non-manifest semantic gates as well. They are bound, not silently
    # dropped or replaced by directory/queue-ledger claims of native36-cell completion.
    gate_paths = set(legacy.METRICS.glob("lfw_bound_evaluation_*/lfw_bound_evaluation.json"))
    gate_paths.update(legacy.METRICS.glob("strong_backbone_fixed8_bn_frozen_*/summary.json"))
    gate_paths.update(legacy.METRICS / name / "summary.json" for name in (
        "strong_backbone_study", "matched_agegap_arms", "matched_agegap_fixed10_last"))
    for path in sorted(gate_paths):
        if path.is_file():
            audit.snapshot(path)
    legacy._record_integrity = audit.check  # Isolated instance only; canonical v1 untouched.
    out.mkdir(parents=True)
    legacy.OUTPUT.parent.mkdir()
    legacy.main()
    raw = json.loads(legacy.OUTPUT.read_text(encoding="utf-8"))
    if selected != manifest_paths(legacy):
        raise RuntimeError("manifest selection changed during index audit")
    clean, _ = sanitize_value(hide_controlled_paths(raw), root)
    clean.update(index_audit_version=2, publication_ready=False,
                 limitation="physical artifact integrity snapshot; original v1 semantic gates retained; "
                            "no ethics/identity/disclosure clearance, native matrix completion, "
                            "full experiment reproduction or portal proof")
    try:
        scan_unsafe(clean)
    except Exception:
        raise ValueError("unreviewed index content refused; no publication clearance") from None
    output.write_text(json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8")
    audit.guard()
    target = out / "summary.manifest.json"
    metrics = dict(execution_complete=True, publication_ready=False,
                   artifact_index_complete=clean["complete"], entry_count=clean["entry_count"],
                   missing_count=len(clean["missing_expected_manifests"]),
                   invalid_count=len(clean["invalid_input_or_output_manifests"]),
                   incomplete_count=len(clean["incomplete_experiments"]),
                   observed_physical_inputs=len(audit.snapshots),
                   limitation=clean["limitation"])
    try:
        write_experiment_manifest(target, experiment="publication-artifact-index-audit-v2",
                                  parameters=dict(coverage="v1 plus feature/age/pair evidence",
                                                  canonical_index_updated=False,
                                                  historical_semantic_gates="retained, not native matrix proof"),
                                  metrics=metrics, inputs=sorted(audit.snapshots), outputs=[output])
        native = json.loads(target.read_text(encoding="utf-8"))
        if native["inputs"] != [audit.snapshots[p] for p in sorted(audit.snapshots)]:
            raise RuntimeError("manifest input snapshot differs")
        audit.guard()
    except Exception:
        if target.exists():
            target.unlink()  # Only this invocation's untrusted completion marker.
        raise
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.out), indent=2))


if __name__ == "__main__":
    main()
