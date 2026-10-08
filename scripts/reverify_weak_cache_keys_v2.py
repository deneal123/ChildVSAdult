"""Re-derive native-bound FaceNet cache keys; not proof of training independence."""

import argparse
import json
from pathlib import Path

import numpy as np

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.evaluate_oriented_cuda_v1 import validate_written_inputs
from scripts.fgnet_retrieval_study import embedding_cache_key
from scripts.run_mechanism_diagnostics_v1 import aligned_vectors
from scripts.run_oriented_campaign import paths_from, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output required")
    native = verified(args.binding, "fgnet-endpoint-age-matched-error-breakdown")
    source = PROJECT_ROOT / "data/external/fgnet_crops.npz"
    source_record = file_record(source)
    if source_record not in native["inputs"]:
        raise ValueError("FG-NET source must be bound")
    weights = {}
    for key, role, filename in [("frozen", "frozen_casia_webface", "20180408-102900-casia-webface.pt"),
                                *[(f"tuned_seed{s}", f"tuned_facenet_seed{s}", f"bb_facenet_seed{s}.pt")
                                  for s in (42, 1, 2)]]:
        matches = [r for r in native["inputs"] if Path(r["path"]).name == filename]
        if len(matches) != 1:
            raise ValueError("unique native weight locator required")
        weights[key] = (role, matches[0]["sha256"])
    inputs = sorted(set([Path(__file__), args.binding,
                         *[PROJECT_ROOT / p for p in ("scripts/fgnet_retrieval_study.py",
                            "scripts/run_mechanism_diagnostics_v1.py", "scripts/run_oriented_campaign.py",
                            "scripts/run_restricted_matched_campaign.py", "scripts/evaluate_oriented_cuda_v1.py",
                            "src/age_gap/common/manifest.py")], *paths_from(native)]))
    before = [file_record(p) for p in inputs]
    with np.load(source, allow_pickle=False) as data:
        count = len(data["subjects"])
    receipts = {}
    for key, (role, weight_sha) in weights.items():
        path = args.binding.parent / f"private/private_embeddings_{key}.npz"
        if file_record(path) not in native["outputs"]:
            raise ValueError("embedding output must be bound")
        aligned_vectors(path, count)
        with np.load(path, allow_pickle=False) as data:
            expected = embedding_cache_key(role=role, weights_sha256=weight_sha,
                                           source_sha256=source_record["sha256"], indices=data["indices"])
            if data["key"].ndim != 0 or str(data["key"]) != expected:
                raise ValueError("embedding key mismatch")
        receipts[key] = dict(key_verified=True, n_images=count, dimensions=512)
    if before != [file_record(p) for p in inputs]:
        raise RuntimeError("inputs changed")
    args.out.mkdir(parents=True)
    summary = args.out / "summary.json"
    metrics = dict(execution_complete=True, publication_ready=False, models=receipts,
                   limitation="cache source/weights/preprocessing/index linkage only; training history and identity independence unverified")
    summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(target, experiment="weak-native-cache-key-reverification-v2",
                              parameters=dict(device="cpu", network_calls=False),
                              metrics=metrics, inputs=inputs, outputs=[summary])
    validate_written_inputs(target, before)
    print("verified four native-bound weak embedding keys", flush=True)


if __name__ == "__main__":
    main()
