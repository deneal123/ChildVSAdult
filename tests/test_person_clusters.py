"""Тест кластеризации групп в личности (union-find по высокому косинусу)."""

from __future__ import annotations

import numpy as np

from age_gap.datasets.person_clusters import cluster_groups, cluster_groups_sweep


def test_merges_groups_with_high_sim_face():
    # f0,f1 (группа A) и f2 (группа B) — почти одинаковые → A и B сливаются; f3 (C) ортогонален.
    face_ids = ["f0", "f1", "f2", "f3"]
    emb = np.array(
        [[1.0, 0.0], [0.99, 0.01], [0.995, 0.005], [0.0, 1.0]], dtype=np.float32
    )
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    gbf = {"f0": "A", "f1": "A", "f2": "B", "f3": "C"}
    g2p = cluster_groups(face_ids, emb, gbf, merge_threshold=0.85)
    assert g2p["A"] == g2p["B"]  # слиты в одну личность
    assert g2p["C"] != g2p["A"]  # отдельная личность
    assert len(set(g2p.values())) == 2


def test_no_merge_below_threshold():
    face_ids = ["f0", "f1"]
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # cos=0
    gbf = {"f0": "A", "f1": "B"}
    g2p = cluster_groups(face_ids, emb, gbf, merge_threshold=0.85)
    assert g2p["A"] != g2p["B"] and len(set(g2p.values())) == 2


def test_person_id_is_min_group_id():
    face_ids = ["f0", "f1"]
    emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    gbf = {"f0": "B", "f1": "A"}  # сливаются -> person_id = "A" (минимальный)
    g2p = cluster_groups(face_ids, emb, gbf, merge_threshold=0.85)
    assert g2p["A"] == "A" and g2p["B"] == "A"


def test_threshold_sweep_matches_independent_runs():
    rng = np.random.default_rng(7)
    emb = rng.normal(size=(12, 6)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    face_ids = [f"f{index}" for index in range(len(emb))]
    group_by_face = {face_id: f"g{index // 2}" for index, face_id in enumerate(face_ids)}
    thresholds = [0.2, 0.5, 0.8]

    swept = cluster_groups_sweep(face_ids, emb, group_by_face, thresholds, chunk=3)

    for threshold in thresholds:
        expected = cluster_groups(
            face_ids, emb, group_by_face, merge_threshold=threshold, chunk=3
        )
        assert swept[threshold] == expected
