"""CLI: gender в метаданные (Part 1) — genderage на все usable лица + агрегация в группы/посты.

Шаг 1 (резюмируемый) считает genderage по всем usable-лицам (CPU); шаг 2 сворачивает per-face
пол/возраст в IdentityGroup (apparent_gender/age/consistency) и пишет post-level сводку.

    uv run python scripts/enrich_gender.py
    uv run python scripts/enrich_gender.py --skip-compute   # только агрегация из готового sidecar
"""

from __future__ import annotations

import argparse

from age_gap.common.io import data_path, read_jsonl
from age_gap.datasets.gender_meta import enrich_groups, load_face_attributes, write_post_gender


def main() -> None:
    parser = argparse.ArgumentParser(description="Gender metadata: genderage + aggregate to groups")
    parser.add_argument(
        "--skip-compute", action="store_true", help="не считать genderage, только агрегация"
    )
    args = parser.parse_args()

    if not args.skip_compute:
        from age_gap.evaluation.fairness import compute_face_attributes

        ids = [
            r["face_id"]
            for r in read_jsonl(str(data_path("data_dir", "interim", "faces.jsonl")))
            if r.get("is_usable")
        ]
        compute_face_attributes(ids, device="cpu")  # резюмируемо

    attrs = load_face_attributes()
    groups = enrich_groups(attrs=attrs)
    write_post_gender(groups)


if __name__ == "__main__":
    main()
