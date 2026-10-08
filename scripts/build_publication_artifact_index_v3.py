"""Fresh publication index audit for the current v3 evidence set.

This is a bounded, fail-closed wrapper around the immutable v1 builder and its
v2 ancestry. It reuses the v1 inclusion/semantic gates and the v2
``IntegrityAudit``/privacy sanitizer through an isolated module instance, so
canonical v1/v2 globals and copied native evidence are never mutated. It then
binds the real bytes of the producing sources (v3, v2, v1 and shared modules)
plus every selected manifest, rehashes them, and refuses to publish a completion
marker when anything changed.

Audit completion is a physical snapshot only: it is NOT coordinator acceptance
of age14, full native36 matrix completion, ethics/identity/disclosure clearance,
experiment reproduction or a real upload. ``publication_ready`` is always False.
"""

import argparse
import importlib.util
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT

# New v2/v3 coverage on top of the inherited v1/v2 globs.
EXTRA_INCLUDE = (
    "common_mechanism_presentation*_v1_*/presentation.manifest.json",
    "age_probe_absolute_presentation*_v1_*/presentation.manifest.json",
    "common_pair_protocol*_v1_*/summary.manifest.json",
    "common_pair_protocol*_v2_*/summary.manifest.json",
    "common_mechanism_presentation*_v2_*/presentation.manifest.json",
    "age_probe_absolute_presentation*_v3_*/presentation.manifest.json",
    "common_pair_protocol*_v3_*/summary.manifest.json",
    "external_layout*_v2_*/summary.manifest.json",
    "scaling_layout*_v2_*/summary.manifest.json",
)
# Exactly the current required evidence. Historical v2 presentation literals
# (presentation6_v1 / age11_v1 / pair6_v1) are discovery-only here, never
# promoted back to CURRENT required.
CURRENT_REQUIRED = (
    "common_mechanism_presentation9_v2_20261008/presentation.manifest.json",
    "age_probe_absolute_presentation14_v3_full_20261008/presentation.manifest.json",
    "strong_fixed8_roc_presentation9_v1_20261008/presentation.manifest.json",
    "common_pair_protocol7_v2_20261008/summary.manifest.json",
    "external_layout_v2_20261008/summary.manifest.json",
    "scaling_layout_v2_20261008/summary.manifest.json",
)


def isolated_v2():
    """Load the immutable v2 ancestry in its own module namespace."""
    path = PROJECT_ROOT / "scripts" / "build_publication_artifact_index_v2.py"
    spec = importlib.util.spec_from_file_location("_publication_index_v3_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXTRA_INCLUDE = EXTRA_INCLUDE
    module.CURRENT_REQUIRED = CURRENT_REQUIRED
    return module


def run(root, out):
    root, out = root.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError("fresh index-audit output root required")
    output = out / "publication_artifact_index.json"
    v2 = isolated_v2()
    legacy = v2.isolated_legacy(root, out / "private/legacy_index.json")
    audit = v2.IntegrityAudit(legacy)

    selected = v2.manifest_paths(legacy)
    for path in sorted(selected):
        audit.snapshot(path)
    for path in (Path(__file__), Path(v2.__file__), Path(legacy.__file__),
                 PROJECT_ROOT / "scripts/export_public_evidence.py",
                 PROJECT_ROOT / "src/age_gap/common/manifest.py",
                 PROJECT_ROOT / "src/age_gap/common/io.py"):
        audit.snapshot(path)
    # v1 reads these non-manifest semantic gates as well; bind them instead of
    # replacing them with directory/ledger claims of native36 completion.
    gate_paths = set(legacy.METRICS.glob("lfw_bound_evaluation_*/lfw_bound_evaluation.json"))
    gate_paths.update(legacy.METRICS.glob("strong_backbone_fixed8_bn_frozen_*/summary.json"))
    gate_paths.update(legacy.METRICS / name / "summary.json" for name in (
        "strong_backbone_study", "matched_agegap_arms", "matched_agegap_fixed10_last"))
    for path in sorted(gate_paths):
        if path.is_file():
            audit.snapshot(path)

    legacy._record_integrity = audit.check  # Isolated v1 instance only; canonical v1 untouched.
    out.mkdir(parents=True)
    legacy.OUTPUT.parent.mkdir()
    legacy.main()
    raw = json.loads(legacy.OUTPUT.read_text(encoding="utf-8"))
    if selected != v2.manifest_paths(legacy):
        raise RuntimeError("manifest selection changed during index audit")
    clean, _ = v2.sanitize_value(v2.hide_controlled_paths(raw), root)
    clean.update(
        index_audit_version=3,
        publication_ready=False,
        physical_snapshot_complete=clean["complete"],
        current_required_manifests=[f"metrics/{name}" for name in CURRENT_REQUIRED],
        limitation="physical artifact integrity snapshot only; original v1 semantic gates "
                   "retained; historical v2/v3 evidence bound but not accepted; no coordinator "
                   "acceptance of age14, native36 matrix completion, ethics/identity/disclosure "
                   "clearance, full experiment reproduction or portal proof",
    )
    try:
        v2.scan_unsafe(clean)
    except Exception:
        raise ValueError("unreviewed index content refused; no publication clearance") from None
    output.write_text(json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                      encoding="utf-8")
    audit.guard()

    target = out / "summary.manifest.json"
    metrics = dict(execution_complete=True, publication_ready=False,
                   artifact_index_complete=clean["complete"],
                   physical_snapshot_complete=clean["complete"],
                   entry_count=clean["entry_count"],
                   missing_count=len(clean["missing_expected_manifests"]),
                   invalid_count=len(clean["invalid_input_or_output_manifests"]),
                   incomplete_count=len(clean["incomplete_experiments"]),
                   observed_physical_inputs=len(audit.snapshots),
                   limitation=clean["limitation"])
    try:
        v2.write_experiment_manifest(
            target, experiment="publication-artifact-index-audit-v3",
            parameters=dict(coverage="v1 plus current feature/age/roc/pair/layout evidence",
                            canonical_index_updated=False,
                            historical_evidence="integrity-bound, not independently accepted",
                            current_required=list(CURRENT_REQUIRED)),
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
