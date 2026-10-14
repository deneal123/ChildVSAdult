"""Fresh publication index audit for the current v4 evidence set.

This is a bounded, fail-closed wrapper around the immutable v1 builder and its
v2/v3 ancestry. It reuses the v1 inclusion/semantic gates and the v2
``IntegrityAudit``/privacy sanitizer through an isolated module instance, so
canonical v1/v2/v3 globals and copied native evidence are never mutated. It then
binds the real bytes of the producing sources (v4, v3, v2, v1 and shared
modules) plus every selected manifest, rehashes them, and refuses to publish a
completion marker when anything changed.

Audit completion is a physical snapshot only: it is NOT coordinator acceptance
of the newest presentations, full native36 matrix completion,
ethics/identity/disclosure clearance, experiment reproduction or a real upload.
``publication_ready`` is always False. Real presentations are not accepted here;
this audit never updates the canonical index or authorizes an upload.
"""

import argparse
import importlib.util
import json
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT

# New v4 coverage on top of the inherited v1/v2/v3 globs. The three newest
# evidence literals live in ``CURRENT_REQUIRED`` and are merged into the isolated
# legacy INCLUDE by ``isolated_legacy`` (exactly as v3 does), so a missing newest
# gate is reported as a missing *literal* rather than silently unmatched. This
# tuple stays a strict superset of the v3 patterns.
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
# Exactly the current required v4 evidence. Historical v2/v3 presentation
# literals are discovery-only here, never promoted back to CURRENT required.
CURRENT_REQUIRED = (
    "common_mechanism_presentation10_v2_20261009/presentation.manifest.json",
    "age_probe_absolute_presentation15_v3_full_20261009/presentation.manifest.json",
    "strong_fixed8_roc_presentation10_v1_20261009/presentation.manifest.json",
    "common_pair_protocol7_v2_20261008/summary.manifest.json",
    "external_layout_v2_20261008/summary.manifest.json",
    "scaling_layout_v2_20261008/summary.manifest.json",
)


def isolated_v3():
    """Load the immutable v3 ancestry in its own module namespace.

    Canonical v3 globals are never mutated: only a fresh module instance's
    ``EXTRA_INCLUDE``/``CURRENT_REQUIRED`` are replaced with the v4 sets, so its
    nested ``isolated_v2``/``isolated_legacy`` helpers reuse the v4 coverage.
    """
    path = PROJECT_ROOT / "scripts" / "build_publication_artifact_index_v3.py"
    spec = importlib.util.spec_from_file_location("_publication_index_v4_v3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXTRA_INCLUDE = EXTRA_INCLUDE
    module.CURRENT_REQUIRED = CURRENT_REQUIRED
    return module


def producing_sources(v3, v2, legacy):
    """Every source whose real bytes produced this index, de-duplicated in order.

    ``Path(__file__)`` binds this canonical v4 producer; older source files remain immutable.
    """
    candidates = (
        Path(__file__),
        Path(v3.__file__),
        Path(v2.__file__),
        Path(legacy.__file__),
        PROJECT_ROOT / "scripts" / "export_public_evidence.py",
        PROJECT_ROOT / "src" / "age_gap" / "common" / "manifest.py",
        PROJECT_ROOT / "src" / "age_gap" / "common" / "io.py",
    )
    seen: set[Path] = set()
    ordered = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return tuple(ordered)


def semantic_gate_paths(legacy):
    """Bind selection and missing/present state, including newly empty gate dirs."""
    paths = {directory / name
             for directory in legacy.METRICS.glob("lfw_bound_evaluation_*")
             for name in ("lfw_bound_evaluation.json", "lfw_bound_evaluation.manifest.json")}
    paths.update(directory / "summary.json"
                 for directory in legacy.METRICS.glob("strong_backbone_fixed8_bn_frozen_*"))
    paths.update(legacy.METRICS / name / "summary.json" for name in (
        "strong_backbone_study", "matched_agegap_arms", "matched_agegap_fixed10_last"))
    return {path.resolve(): path.is_file() for path in paths}


def run(root, out):
    root, out = root.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError("fresh index-audit output root required")
    output = out / "publication_artifact_index.json"
    v3 = isolated_v3()
    v2 = v3.isolated_v2()
    legacy = v2.isolated_legacy(root, out / "private/legacy_index.json")
    audit = v2.IntegrityAudit(legacy)

    selected = v2.manifest_paths(legacy)
    for path in sorted(selected):
        audit.snapshot(path)
    for path in producing_sources(v3, v2, legacy):
        audit.snapshot(path)
    # v1 reads these non-manifest semantic gates as well; bind them instead of
    # replacing them with directory/ledger claims of native36 completion.
    gates = semantic_gate_paths(legacy)
    for path in sorted(gates):
        if gates[path]:
            audit.snapshot(path)

    legacy._record_integrity = audit.check  # Isolated v1 instance only; canonical v1 untouched.
    out.mkdir(parents=True)
    legacy.OUTPUT.parent.mkdir()
    legacy.main()
    raw = json.loads(legacy.OUTPUT.read_text(encoding="utf-8"))
    # Selection-change guard BEFORE final writing: no new/dropped glob matches may
    # slip into the sanitized payload that was not snapshotted above.
    if selected != v2.manifest_paths(legacy):
        raise RuntimeError("manifest selection changed during index audit")
    if gates != semantic_gate_paths(legacy):
        raise RuntimeError("semantic gate selection changed during index audit")

    clean, _ = v2.sanitize_value(v2.hide_controlled_paths(raw), root)
    clean.update(
        index_audit_version=4,
        publication_ready=False,
        physical_snapshot_complete=clean["complete"],
        current_required_manifests=[f"metrics/{name}" for name in CURRENT_REQUIRED],
        limitation="physical artifact integrity snapshot only; original v1 semantic gates "
                   "retained; historical v2/v3/v4 evidence bound but not accepted; no "
                   "coordinator acceptance of the newest presentations, native36 matrix "
                   "completion, ethics/identity/disclosure clearance, full experiment "
                   "reproduction or portal proof",
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
        # Reuse the isolated v2 writer (loaded inside this run) so the call site
        # stays patchable exactly like the v3 ancestry and never touches canonical
        # module globals.
        v2.write_experiment_manifest(
            target, experiment="publication-artifact-index-audit-v4",
            parameters=dict(coverage="v1 plus newest mechanism/age/roc, pair and layout evidence",
                            canonical_index_updated=False,
                            historical_evidence="integrity-bound, not independently accepted",
                            current_required=list(CURRENT_REQUIRED)),
            metrics=metrics, inputs=sorted(audit.snapshots), outputs=[output])
        native = json.loads(target.read_text(encoding="utf-8"))
        if native["inputs"] != [audit.snapshots[p] for p in sorted(audit.snapshots)]:
            raise RuntimeError("manifest input snapshot differs")
        if native["outputs"] != [v2.file_record(output)]:
            raise RuntimeError("written index output snapshot differs")
        # Selection-change guard AFTER final writing: the completed snapshot must
        # cover exactly the same manifest selection that was serialized.
        if selected != v2.manifest_paths(legacy):
            raise RuntimeError("manifest selection changed during index audit")
        if gates != semantic_gate_paths(legacy):
            raise RuntimeError("semantic gate selection changed during index audit")
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
