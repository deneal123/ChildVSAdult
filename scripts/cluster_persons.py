"""CLI: кластеризация групп (постов) в личности по эмбеддингам (истинно leakage-safe сплит).

Пишет data/processed/person_clusters.jsonl (group_id -> person_id). Read-only по отношению к
pairs/splits. Требует посчитанных эмбеддингов (scripts/embed.py).

    uv run python scripts/cluster_persons.py --threshold 0.85
"""

from __future__ import annotations

import argparse

from age_gap.datasets.person_clusters import merge_to_person_groups, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster post-groups into persons by embeddings")
    parser.add_argument("--threshold", type=float, default=0.85, help="косинус слияния в личность")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="схлопнуть группы в person-level identity_groups (потом build_pairs + split)",
    )
    args = parser.parse_args()
    g2p = run(merge_threshold=args.threshold)
    print(f"Готово: групп={len(set(g2p))}, личностей={len(set(g2p.values()))}.")
    if args.apply:
        n = merge_to_person_groups(g2p)
        print(f"Person-level групп: {n} (запустите build_pairs.py + split.py).")


if __name__ == "__main__":
    main()
