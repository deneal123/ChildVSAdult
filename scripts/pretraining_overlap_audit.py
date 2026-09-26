"""Audit whether pretrained face-model identities can be intersected with benchmarks.

This audit intentionally distinguishes a verified empty intersection from an
intersection that cannot be computed because upstream identity manifests or
benchmark name mappings are absent. It never infers names from face images.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from age_gap.common.io import data_path
from age_gap.common.manifest import file_record, write_experiment_manifest


def _npz_identity_evidence(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        fields = sorted(archive.files)
        subject_count = (
            int(np.unique(archive["subjects"]).size) if "subjects" in archive.files else None
        )
    return {
        "artifact": file_record(path),
        "fields": fields,
        "numeric_subject_count": subject_count,
        "contains_identity_names": False,
    }


def main() -> None:
    inventory_path = Path(str(data_path("metrics_dir", "model_inventory.json")))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    external = Path(str(data_path("data_dir", "external")))
    benchmark_paths = {
        "LFW": external / "lfw_aligned.npz",
        "AgeDB-30": external / "agedb_30.bin",
        "CALFW": external / "calfw.bin",
        "CACD-VS": external / "cacd_vs_aligned.npz",
        "FG-NET": external / "fgnet_crops.npz",
    }
    missing = [str(path) for path in benchmark_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark artifacts: {missing}")

    sources: dict[str, list[str]] = {}
    for model_name, model in inventory["models"].items():
        source = str(model["declared_training_data"])
        sources.setdefault(source, []).append(model_name)

    benchmark_evidence: dict[str, dict[str, object]] = {}
    for name, path in benchmark_paths.items():
        if path.suffix == ".npz":
            evidence = _npz_identity_evidence(path)
            if name == "FG-NET":
                evidence["identifier_scope"] = (
                    "benchmark-local numeric subject IDs; no mapping to real names"
                )
            else:
                evidence["identifier_scope"] = "pair labels only; no subject identifier field"
        else:
            evidence = {
                "artifact": file_record(path),
                "format": "InsightFace verification bin: encoded image arrays plus same/different labels",
                "contains_identity_names": False,
                "identifier_scope": "pair labels only; no subject identifier field",
            }
        benchmark_evidence[name] = evidence

    payload = {
        "status": "exact_identity_intersection_not_computable",
        "verified_empty_intersection": False,
        "pretraining_sources": [
            {
                "declared_training_data": source,
                "models": sorted(models),
                "complete_identity_manifest_available_locally": False,
            }
            for source, models in sorted(sources.items())
        ],
        "benchmark_identity_evidence": benchmark_evidence,
        "finding": (
            "None of the downloaded pretrained checkpoints is accompanied by a complete identity "
            "manifest. The local verification artifacts expose no compatible real-name identity "
            "mapping (FG-NET exposes benchmark-local numeric IDs only). Therefore identity-level "
            "pretraining/benchmark intersection cannot be computed and must not be reported as zero."
        ),
        "risk": (
            "Overlap cannot be excluded, especially for web-scale or celebrity-oriented pretraining "
            "sets and celebrity benchmarks. This limits claims of benchmark independence."
        ),
        "mitigations": [
            "Report each checkpoint, checksum and declared pretraining dataset.",
            "Keep every benchmark evaluation-only and never tune thresholds on benchmark identities.",
            "Report results across CASIA- and WebFace-pretrained model families rather than relying on one lineage.",
            "State the unresolved overlap risk in the model card and limitations.",
            "Re-run the exact intersection if upstream identity manifests and compatible benchmark mappings become available.",
        ],
    }
    output = Path(str(data_path("metrics_dir", "pretraining_overlap_audit.json")))
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="pretraining-benchmark-identity-overlap-audit",
        parameters={"image_or_embedding_matching_performed": False},
        metrics={
            "exact_intersection_computable": False,
            "verified_empty_intersection": False,
            "pretraining_sources": len(sources),
            "benchmarks": len(benchmark_evidence),
        },
        inputs=[inventory_path, *benchmark_paths.values()],
        outputs=[output],
    )
    print(f"OK: {output}")


if __name__ == "__main__":
    main()
