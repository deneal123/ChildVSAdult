"""CLI: построение групп личностей (с извлечением/привязкой возраста).

Резюмируемо: при --age-extractor combined/llm длинный прогон можно перезапускать —
готовые группы пропускаются, LLM не вызывается повторно.

    uv run python scripts/build_groups.py --age-extractor combined
"""

from __future__ import annotations

import argparse

from age_gap.datasets.identity_groups import build_groups
from age_gap.datasets.llm_age_extractor import make_age_extractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Build identity groups (resumable)")
    parser.add_argument(
        "--age-extractor",
        choices=["regex", "llm", "combined"],
        default="regex",
        help="Источник возрастных якорей: regex (по умолч.) | llm (GigaChat) | combined",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Не продолжать, строить с нуля (перезаписи нет — append!)",
    )
    args = parser.parse_args()

    extractor = make_age_extractor(args.age_extractor)
    new = build_groups(extractor=extractor, resume=not args.no_resume)
    print(f"Готово: построено новых групп={new} (запускайте повторно для догона остатка).")


if __name__ == "__main__":
    main()
