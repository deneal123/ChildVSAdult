"""Локальный HTML-отчёт по «силе симпатии аудитории».

    uv run python scripts/engagement_report.py                    # без превью лиц
    uv run python scripts/engagement_report.py --with-thumbnails  # с кропами (ЛОКАЛЬНО!)

Отчёт пишется в reports/engagement/report.html (каталог в .gitignore).
Не публиковать и не коммитить: при --with-thumbnails внутри лежат кропы лиц.
"""

from __future__ import annotations

import argparse

from age_gap.engagement.report import build_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-thumbnails", action="store_true",
                    help="встроить кропы взрослых лиц (base64). Только локально!")
    ap.add_argument("--cards", type=int, default=16, help="сколько постов в топе и в низу")
    args = ap.parse_args()

    path = build_report(with_thumbnails=args.with_thumbnails, n_cards=args.cards)
    print(f"OK: {path}")


if __name__ == "__main__":
    main()
