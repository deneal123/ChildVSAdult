"""CLI: калибровка P(тот же человек | cosine, age_gap) (§5.3).

Сравнивает Platt(cos) и age-gap-aware(cos,gap) по ECE/Brier (в т.ч. по возрастному разрыву).
Требует эмбеддингов (scripts/embed.py) и пар с возрастом (build_pairs/split).

    uv run python scripts/calibrate.py
"""

from __future__ import annotations

import json

from age_gap.evaluation.calibration import evaluate_calibration


def main() -> None:
    res = evaluate_calibration()
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
