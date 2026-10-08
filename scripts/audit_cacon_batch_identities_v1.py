"""Prospective uniform-batch recorded-identity collisions, not human false-negative truth."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from age_gap.common.io import PROJECT_ROOT
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.mtlface_epoch_v2 import load_bound_dataset


def analyse_batches(identities, batches):
    if not identities or any(type(identity) is not int or identity < 0 for identity in identities):
        raise ValueError("nonempty recorded integer identities required")
    flat = [index for batch in batches for index in batch]
    if any(type(index) is not int for index in flat) or sorted(flat) != list(range(len(identities))):
        raise ValueError("exact one-pass source coverage required")
    rows = []
    for batch in batches:
        if not batch:
            raise ValueError("empty batch refused")
        counts = Counter(identities[index] for index in batch)
        pairs = sum(count * (count - 1) // 2 for count in counts.values())
        size = len(batch)
        rows.append(dict(sources=size, distinct_recorded_people=len(counts),
                         same_recorded_person_source_pairs=pairs,
                         anchors_with_another_same_recorded_person=3 * sum(c for c in counts.values() if c > 1),
                         directed_other_source_view_negatives=9 * size * (size - 1),
                         directed_same_recorded_person_view_negatives=18 * pairs))
    numerator = sum(row["directed_same_recorded_person_view_negatives"] for row in rows)
    denominator = sum(row["directed_other_source_view_negatives"] for row in rows)
    return dict(sources=len(identities), batches=len(rows),
                batches_with_identity_collision=sum(row["same_recorded_person_source_pairs"] > 0 for row in rows),
                same_recorded_person_source_pairs=sum(row["same_recorded_person_source_pairs"] for row in rows),
                anchors_with_another_same_recorded_person=sum(row["anchors_with_another_same_recorded_person"] for row in rows),
                directed_same_recorded_person_view_negatives=numerator,
                directed_other_source_view_negatives=denominator,
                recorded_identity_collision_fraction=numerator / denominator if denominator else None,
                per_batch=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=PROJECT_ROOT / "metrics/sota_recognition_face_list_20261003/summary.manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh output directory required")
    torch.set_num_threads(1)
    dataset, native = load_bound_dataset(args.faces, lambda image: image)
    identities = [record.identity for record in dataset.records]
    dependencies = [args.faces.resolve(), Path(__file__).resolve(), PROJECT_ROOT / "scripts/mtlface_epoch_v2.py",
                    PROJECT_ROOT / "scripts/mtlface_training_v2.py", PROJECT_ROOT / "scripts/prepare_sota_face_list.py",
                    PROJECT_ROOT / "src/age_gap/common/io.py", PROJECT_ROOT / "src/age_gap/common/manifest.py"]
    for record in native["inputs"] + native["outputs"]:
        path = Path(record["path"])
        dependencies.append(path if path.is_absolute() else PROJECT_ROOT / path)
    inputs = list(dict.fromkeys(dependencies))
    before = [file_record(path) for path in inputs]
    results = {}
    for seed in (42, 1, 2):
        loader = DataLoader(range(len(identities)), batch_size=64, shuffle=True,
                            generator=torch.Generator(device="cpu").manual_seed(seed), num_workers=0, drop_last=False)
        batches = [batch.tolist() for batch in loader]
        results[str(seed)] = analyse_batches(identities, batches)
    if [file_record(path) for path in inputs] != before:
        raise RuntimeError("audit inputs changed during execution")
    args.out.mkdir(parents=True)
    summary = args.out / "summary.json"
    metrics = dict(metadata_audit_complete=True, training_complete=False,
                   human_identity_verified=False, scientific_evaluation_complete=False, seeds=results)
    summary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(args.out / "summary.manifest.json", experiment="cacon-prospective-batch-identity-collisions",
                              parameters=dict(batch_size=64, seeds=[42, 1, 2], source_passes_per_seed=1,
                                              sampling="torch DataLoader CPU generator uniform shuffle, workers0/drop_last=false",
                                              conditioning="three views per source, two own-source positives; other sources are negatives",
                                              scope="candidate sampler risk from recorded identities; not a trained CACon result"),
                              metrics=metrics, inputs=inputs, outputs=[summary])
    print(json.dumps({seed: {k: v for k, v in row.items() if k != "per_batch"} for seed, row in results.items()}))


if __name__ == "__main__":
    main()
