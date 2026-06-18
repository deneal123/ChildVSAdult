"""CLI: посчитать baseline ArcFace-эмбеддинги для usable-лиц.

Пример:
    uv run python scripts/embed.py
"""

from __future__ import annotations

import argparse

from age_gap.models.embeddings import compute_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute baseline ArcFace embeddings")
    parser.add_argument("--faces", default=None, help="Путь к faces.jsonl")
    args = parser.parse_args()

    emb = compute_embeddings(faces_file=args.faces)
    print(f"Готово: эмбеддингов={len(emb)}.")


if __name__ == "__main__":
    main()
