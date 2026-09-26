"""Repeat clustering and dedup sensitivity with an independent FaceNet embedding.

The production curation uses InsightFace ArcFace.  This audit extracts CASIA-
WebFace FaceNet embeddings from the same aligned crops, then repeats both the
post-group merge and within-person near-duplicate threshold sweeps without
mutating canonical groups, pairs or splits.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from age_gap.common.device import torch_device
from age_gap.common.io import data_path, read_jsonl
from age_gap.common.manifest import write_experiment_manifest
from age_gap.common.schemas import FaceCrop, IdentityGroup
from age_gap.datasets.dedup import find_redundant_faces
from age_gap.datasets.person_clusters import cluster_groups_sweep
from age_gap.models.facenet import FaceNetBackbone, preprocess_bgr


class _FaceDataset(Dataset):
    def __init__(self) -> None:
        self.items: list[tuple[str, Path]] = []
        for row in read_jsonl(data_path("data_dir", "interim", "faces.jsonl")):
            face = FaceCrop.from_dict(row)
            if face.is_usable and face.face_crop_path:
                path = Path(face.face_crop_path)
                if not path.is_absolute():
                    path = data_path("data_dir").parent / path
                if path.exists():
                    self.items.append((face.face_id, path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        face_id, path = self.items[index]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"cannot read {path}")
        return face_id, torch.from_numpy(preprocess_bgr(image))


def extract_embeddings(output: Path, batch_size: int) -> Path:
    device = torch_device()
    model = FaceNetBackbone(pretrained="casia-webface").to(device).eval()
    dataset = _FaceDataset()
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for face_ids, images in DataLoader(dataset, batch_size=batch_size):
            vectors.append(model(images.to(device)).cpu().numpy())
            ids.extend(face_ids)
    matrix = np.concatenate(vectors).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, face_ids=np.asarray(ids, dtype=object), embeddings=matrix)
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="independent-facenet-embeddings",
        parameters={"model": "InceptionResnetV1 CASIA-WebFace", "batch_size": batch_size},
        metrics={"faces": len(ids), "dimension": matrix.shape[1]},
        inputs=[data_path("data_dir", "interim", "faces.jsonl")],
        outputs=[output],
    )
    return output


def _load(path: Path) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    archive = np.load(path, allow_pickle=True)
    ids = [str(value) for value in archive["face_ids"]]
    matrix = np.asarray(archive["embeddings"], dtype=np.float32)
    return ids, matrix, dict(zip(ids, matrix, strict=True))


def audit(embedding_path: Path, output: Path) -> dict:
    ids, matrix, embeddings = _load(embedding_path)
    original_groups = [
        IdentityGroup.from_dict(row)
        for row in read_jsonl(
            data_path("data_dir", "processed", "identity_groups.jsonl.post_bak")
        )
    ]
    group_by_face = {
        face_id: group.identity_group_id for group in original_groups for face_id in group.faces
    }
    keep = [index for index, face_id in enumerate(ids) if face_id in group_by_face]
    cluster_ids = [ids[index] for index in keep]
    cluster_matrix = matrix[keep]
    clustering: dict[str, dict[str, int]] = {}
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    mappings = cluster_groups_sweep(cluster_ids, cluster_matrix, group_by_face, thresholds)
    for threshold in thresholds:
        mapping = mappings[threshold]
        clustering[f"{threshold:.2f}"] = {
            "groups": len(mapping),
            "persons": len(set(mapping.values())),
            "merged_groups": len(mapping) - len(set(mapping.values())),
        }

    pre_prune = [
        IdentityGroup.from_dict(row)
        for row in read_jsonl(
            data_path("data_dir", "processed", "identity_groups.jsonl.pre_prune_bak")
        )
    ]

    def positives(redundant: dict[str, list[str]]) -> int:
        return sum(
            comb(
                max(
                    len([face for face in group.faces if face in embeddings])
                    - len(redundant.get(group.identity_group_id, [])),
                    0,
                ),
                2,
            )
            for group in pre_prune
        )

    raw = positives({})
    dedup: dict[str, dict[str, float | int]] = {}
    for threshold in [0.90, 0.93, 0.95, 0.97, 0.99]:
        redundant = find_redundant_faces(pre_prune, embeddings, threshold)
        remaining = positives(redundant)
        dedup[f"{threshold:.2f}"] = {
            "redundant_faces": sum(map(len, redundant.values())),
            "positives": remaining,
            "reduction_pct": 100 * (1 - remaining / raw),
        }
    payload = {
        "embedding_model": "FaceNet InceptionResnetV1 CASIA-WebFace",
        "clustering": clustering,
        "dedup_no_filter_positives": raw,
        "dedup": dedup,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_experiment_manifest(
        output.with_suffix(".manifest.json"),
        experiment="independent-embedding-clustering-dedup-sensitivity",
        parameters={
            "clustering_thresholds": [0.70, 0.75, 0.80, 0.85, 0.90],
            "dedup_thresholds": [0.90, 0.93, 0.95, 0.97, 0.99],
        },
        metrics=payload,
        inputs=[embedding_path],
        outputs=[output],
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=data_path("embeddings_cache_dir", "independent_facenet.npz"),
    )
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=data_path("metrics_dir", "independent_embedding_audit.json")
    )
    args = parser.parse_args()
    if not args.skip_embedding or not args.embeddings.exists():
        extract_embeddings(args.embeddings, args.batch_size)
    print(json.dumps(audit(args.embeddings, args.output), indent=2))


if __name__ == "__main__":
    main()
