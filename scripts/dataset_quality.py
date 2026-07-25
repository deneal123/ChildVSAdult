"""Аудит качества датасета КАК БЕНЧМАРКА: чистота меток, покрытие, узкие места.

    uv run python scripts/dataset_quality.py

Зачем именно сейчас. Мы предъявляем набор как бенчмарк, на котором современные распознаватели
проваливаются (замороженный AdaFace IR-101: 0.956 на случайных негативах -> 0.705 на rank-10 и
0.474 на rank-1). Такое заявление живёт или умирает вместе с чистотой меток: одна систематическая
ошибка разметки объясняла бы весь эффект. Здесь всё, что можно проверить машинно.

Что считаем:
  1. ЧИСТОТА НЕГАТИВОВ. Негативы строились «кросс-группа», а личность — это КЛАСТЕР групп
     (слияние по косинусу >=0.85). Если два поста одного человека не слились, «негатив»
     оказывается позитивом. Проверяется по person_clusters.jsonl.
  2. ЧИСТОТА ПОЗИТИВОВ. Позитив = «один пост -> один человек». Риск обратный: в посте двое разных
     людей (мать/дочь) -> ложный позитив. Оцениваем через LLM-валидацию групп и через долю пар с
     аномально низким сходством внутри своего возрастного бакета.
  3. РАЗДЕЛЕНИЕ СПЛИТОВ по личностям (утечка идентичности между train и test).
  4. ПОКРЫТИЕ: возрастные разрывы, пол, кажущийся возраст, число фото на личность.
  5. ДУБЛИ: доля почти-дубликатов среди позитивов (косинус >=0.97).
  6. РАЗДЕЛИМОСТЬ по бакетам разрыва — то, ради чего бенчмарк и нужен.

Пишет metrics/dataset_quality.json.
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger
from age_gap.models.embeddings import load_embeddings

log = get_logger(__name__)


def _cos(emb, a, b):
    if a not in emb or b not in emb:
        return None
    x, y = emb[a], emb[b]
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dup-threshold", type=float, default=0.97)
    args = ap.parse_args()

    pairs = list(read_jsonl(data_path("data_dir", "processed", "pairs.jsonl")))
    emb = load_embeddings()
    g2p = {r["identity_group_id"]: r["person_id"]
           for r in read_jsonl(resolve_path("data", "processed", "person_clusters.jsonl"))}
    ga = {r["face_id"]: (float(r["age_est"]), int(r["gender"]))
          for r in read_jsonl(data_path("data_dir", "interim", "face_genderage.jsonl"))}
    out: dict[str, dict] = {}

    # ---------- 1. масштаб и разделение сплитов ----------
    by_split = collections.Counter(r.get("split") for r in pairs)
    persons_by_split = collections.defaultdict(set)
    for r in pairs:
        for g in (r["identity_group_a"], r["identity_group_b"]):
            persons_by_split[r.get("split")].add(g2p.get(g, g))
    leak = {}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                leak[f"{a}∩{b}"] = len(persons_by_split[a] & persons_by_split[b])
    out["1_масштаб"] = {"пар всего": len(pairs),
                        "по сплитам": dict(by_split),
                        "личностей по сплитам": {k: len(v) for k, v in persons_by_split.items()},
                        "ПЕРЕСЕЧЕНИЕ личностей между сплитами": leak}
    log.info("пар %d | утечка личностей между сплитами: %s", len(pairs), leak)

    # ---------- 2. чистота негативов ----------
    negs = [r for r in pairs if r["label"] == 0]
    same_person = sum(1 for r in negs
                      if g2p.get(r["identity_group_a"]) is not None
                      and g2p.get(r["identity_group_a"]) == g2p.get(r["identity_group_b"]))
    neg_sims = np.array([c for r in negs if (c := _cos(emb, r["face_a"], r["face_b"])) is not None])
    out["2_чистота_негативов"] = {
        "негативов": len(negs),
        "оказались ОДНОЙ личностью (ложный негатив)": same_person,
        "доля %": round(same_person / max(len(negs), 1) * 100, 3),
        "медианный косинус": round(float(np.median(neg_sims)), 3),
        "доля с косинусом >=0.85 (подозрение на пропущенное слияние) %":
            round(float((neg_sims >= 0.85).mean() * 100), 3),
    }

    # ---------- 3. чистота позитивов ----------
    poss = [r for r in pairs if r["label"] == 1]
    val = {r["post_id"]: r for r in read_jsonl(data_path("data_dir", "interim",
                                                         "group_validation_cache.jsonl"))}
    cat = collections.Counter(val[r["identity_group_a"]]["category"]
                              for r in poss if r["identity_group_a"] in val)
    conf = [float(val[r["identity_group_a"]]["confidence"])
            for r in poss if r["identity_group_a"] in val]
    pos_sims, pos_gaps = [], []
    for r in poss:
        c = _cos(emb, r["face_a"], r["face_b"])
        if c is not None:
            pos_sims.append(c)
            pos_gaps.append(r.get("age_gap") if r.get("age_gap") is not None else -1)
    pos_sims, pos_gaps = np.array(pos_sims), np.array(pos_gaps)
    out["3_чистота_позитивов"] = {
        "позитивов": len(poss),
        "LLM-категория группы": dict(cat),
        "доля 'single' среди проверенных %":
            round(cat.get("single", 0) / max(sum(cat.values()), 1) * 100, 1),
        "медианная уверенность LLM": round(float(np.median(conf)), 3) if conf else None,
        "непроверенных LLM": sum(1 for r in poss if r["identity_group_a"] not in val),
        f"почти-дубликаты (косинус >={args.dup_threshold:.2f}), %":
            round(float((pos_sims >= args.dup_threshold).mean() * 100), 2),
    }

    # ---------- 4. покрытие ----------
    gap_bins = [(0, 5), (5, 10), (10, 15), (15, 25), (25, 200)]
    cov = {}
    for lo, hi in gap_bins:
        m = (pos_gaps >= lo) & (pos_gaps < hi)
        cov[f"{lo}-{hi if hi < 200 else '∞'} лет"] = int(m.sum())
    cov["без метки возраста"] = int((pos_gaps < 0).sum())
    faces = {r["face_a"] for r in pairs} | {r["face_b"] for r in pairs}
    ages = np.array([ga[f][0] for f in faces if f in ga])
    gens = collections.Counter(ga[f][1] for f in faces if f in ga)
    out["4_покрытие"] = {
        "позитивов по разрыву": cov,
        "кажущийся возраст: медиана / p10 / p90":
            [round(float(np.median(ages)), 1), round(float(np.percentile(ages, 10)), 1),
             round(float(np.percentile(ages, 90)), 1)],
        "кажущийся пол (0=Ж,1=М)": {"Ж": gens.get(0, 0), "М": gens.get(1, 0),
                                     "доля Ж %": round(gens.get(0, 0) / max(sum(gens.values()), 1) * 100, 1)},
        "лиц всего": len(faces),
    }

    # ---------- 5. разделимость по бакетам (ради чего бенчмарк) ----------
    sep = {}
    for lo, hi in gap_bins:
        m = (pos_gaps >= lo) & (pos_gaps < hi)
        if m.sum() < 20:
            continue
        s = pos_sims[m]
        sep[f"{lo}-{hi if hi < 200 else '∞'} лет"] = {
            "n": int(m.sum()), "медиана косинуса": round(float(np.median(s)), 3),
            "ниже медианы негативов %": round(float((s < np.median(neg_sims)).mean() * 100), 1)}
    out["5_разделимость"] = {
        "позитивы по бакетам": sep,
        "негативы: медиана": round(float(np.median(neg_sims)), 3),
        "комментарий": "чем ближе медиана позитивов к медиане негативов, тем труднее бакет",
    }

    # ---------- 6. узкое место ----------
    n25 = cov.get("25-∞ лет", 0)
    out["6_узкие_места"] = {
        "позитивов 25+ в ТЕСТЕ (головной срез)": sum(
            1 for r in pairs if r.get("split") == "test" and r["label"] == 1
            and (r.get("age_gap") or -1) >= 25),
        "позитивов 25+ ВСЕГО": n25,
        "почему важно": "все заголовочные числа (large-gap AUC, кривая трудных негативов) "
                        "опираются на этот срез; при n~150 доверительные интервалы широки",
    }

    dst = data_path("metrics_dir", "dataset_quality.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    log.info("записано: %s", dst)


if __name__ == "__main__":
    main()
