"""Сборка post-level таблицы признаков для регрессии «силы симпатии аудитории».

Join-цепочка (ключи проверены на данных):
    posts.jsonl:  post_id -> photos[].photo_id
    faces.jsonl:  photo_id -> face_id (+ качество/bbox/детектор)
    face_genderage.jsonl: face_id -> (gender 0=F/1=M, age_est)
    identity_groups.jsonl: age_labels[].face_id -> age  (извлечённый из подписи возраст)
    person_clusters.jsonl: identity_group_id (== исходный post_id) -> person_id
    cache/embeddings/baseline_arcface.npz: face_id -> 512-d ArcFace

ЭТИЧЕСКИЕ GUARDRAILS (реализованы здесь, проверяются тестами):
  * ``ADULT_MIN_AGE = 18`` — признаки ВНЕШНОСТИ (эмбеддинги, качество, пол, возраст)
    считаются ТОЛЬКО по взрослым лицам. Детские лица никогда не попадают в эмбеддинги.
  * ``SUBJECT_MIN_AGE = 20`` — человек исключается целиком, если максимум кажущегося
    возраста по всем его лицам < 20 (буфер против занижения оценщика). Это убирает
    несовершеннолетних как СУБЪЕКТОВ, но сохраняет взрослых, выложивших своё детское фото.
  * Из детских лиц берётся ТОЛЬКО структурный факт наличия (``n_child_faces``) —
    это суть формата «тогда/сейчас», но не модель их внешности.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from age_gap.common.io import data_path, read_jsonl, resolve_path
from age_gap.common.logging import get_logger

log = get_logger(__name__)

ADULT_MIN_AGE = 18
SUBJECT_MIN_AGE = 20

_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002700-\U000027bf\U0001f1e6-\U0001f1ff☀-⛿]"
)
_HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)


@dataclass
class BuildStats:
    posts_total: int = 0
    posts_with_faces: int = 0
    persons_total: int = 0
    persons_dropped_minor: int = 0
    posts_dropped_minor_subject: int = 0
    posts_dropped_no_adult_face: int = 0
    posts_final: int = 0
    faces_total: int = 0
    faces_child_excluded_from_appearance: int = 0
    engagement_coverage: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------- text


def _text_features(caption: str) -> dict[str, Any]:
    text = caption or ""
    letters = [c for c in text if c.isalpha()]
    return {
        "caption_len": len(text),
        "word_count": len(text.split()),
        "emoji_count": len(_EMOJI_RE.findall(text)),
        "has_emoji": int(bool(_EMOJI_RE.search(text))),
        "n_hashtags": len(_HASHTAG_RE.findall(text)),
        "n_questions": text.count("?"),
        "n_exclam": text.count("!"),
        "upper_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        "has_digit": int(any(c.isdigit() for c in text)),
    }


# --------------------------------------------------------------------------- loaders


def _load_photo_to_post(posts_path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    photo2post: dict[str, str] = {}
    posts: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(posts_path):
        pid = row["post_id"]
        posts[pid] = row
        for ph in row.get("photos", []) or []:
            photo2post[ph["photo_id"]] = pid
    return photo2post, posts


def _load_face_ages(groups_path: Path) -> dict[str, float]:
    """face_id -> возраст, извлечённый из подписи (age_labels)."""
    out: dict[str, float] = {}
    for g in read_jsonl(groups_path):
        for al in g.get("age_labels", []) or []:
            fid, age = al.get("face_id"), al.get("age")
            if fid and age is not None:
                out[fid] = float(age)
    return out


def _load_person_map(clusters_path: Path) -> dict[str, str]:
    """post_id -> person_id (identity_group_id исходно равен post_id)."""
    return {r["identity_group_id"]: r["person_id"] for r in read_jsonl(clusters_path)}


def _bbox_area(bbox: list[float] | None) -> float:
    if not bbox or len(bbox) < 4:
        return float("nan")
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


# --------------------------------------------------------------- guardrails (чистые функции)


def minor_subject_ids(person_max_age: dict[str, float]) -> set[str]:
    """Субъекты-несовершеннолетние: максимум кажущегося возраста по всем их лицам < 20."""
    return {p for p, a in person_max_age.items() if a < SUBJECT_MIN_AGE}


def split_adult_child(
    faces: list[dict[str, Any]], genderage: dict[str, dict[str, Any]]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Разделить лица на взрослые (>=18) и детские. Лица без оценки возраста отбрасываются.

    Детские лица НИКОГДА не попадают в первый список, т.е. не участвуют в признаках внешности.
    """
    adult: list[tuple[dict[str, Any], dict[str, Any]]] = []
    child: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for f in faces:
        ga = genderage.get(f["face_id"])
        if ga is None:
            continue
        (adult if float(ga["age_est"]) >= ADULT_MIN_AGE else child).append((f, ga))
    return adult, child


# --------------------------------------------------------------------------- main


def build_table(
    platform: str = "vk",
    with_embeddings: bool = True,
    emb_prefix: str = "emb",
) -> tuple[pd.DataFrame, BuildStats]:
    """Собрать таблицу признаков. Возвращает (DataFrame, статистика guardrails)."""
    st = BuildStats()

    if platform == "vk":
        posts_path = data_path("data_dir", "raw", "posts.jsonl")
        faces_path = data_path("data_dir", "interim", "faces.jsonl")
        ga_path = data_path("data_dir", "interim", "face_genderage.jsonl")
        groups_path = data_path("data_dir", "processed", "identity_groups.jsonl")
        clusters_path = data_path("data_dir", "processed", "person_clusters.jsonl")
        eng_path = data_path("data_dir", "interim", "post_engagement.jsonl")
    else:
        base = resolve_path("data_reddit")
        posts_path = base / "raw" / "posts.jsonl"
        faces_path = base / "interim" / "faces.jsonl"
        ga_path = base / "interim" / "face_genderage.jsonl"
        groups_path = base / "processed" / "identity_groups.jsonl"
        clusters_path = base / "processed" / "person_clusters.jsonl"
        eng_path = base / "interim" / "post_engagement.jsonl"

    photo2post, posts = _load_photo_to_post(posts_path)
    st.posts_total = len(posts)

    engagement = {r["post_id"]: r for r in read_jsonl(eng_path)}
    if not engagement:
        raise RuntimeError(f"Нет engagement-backfill: {eng_path}. Запустите engagement_backfill.py")

    genderage = {r["face_id"]: r for r in read_jsonl(ga_path)}
    face_ages = _load_face_ages(groups_path)
    person_of_post = _load_person_map(clusters_path)

    # --- лица, сгруппированные по посту -------------------------------------------------
    faces_by_post: dict[str, list[dict[str, Any]]] = {}
    for f in read_jsonl(faces_path):
        st.faces_total += 1
        pid = photo2post.get(f.get("photo_id", ""))
        if pid is None:
            continue
        faces_by_post.setdefault(pid, []).append(f)
    st.posts_with_faces = len(faces_by_post)

    # --- возраст субъекта: максимум age_est по всем лицам его person_id -----------------
    person_max_age: dict[str, float] = {}
    for pid, flist in faces_by_post.items():
        person = person_of_post.get(pid, pid)
        for f in flist:
            ga = genderage.get(f["face_id"])
            if ga is None:
                continue
            person_max_age[person] = max(person_max_age.get(person, -1.0), float(ga["age_est"]))
    st.persons_total = len(person_max_age)
    minors = minor_subject_ids(person_max_age)
    st.persons_dropped_minor = len(minors)

    embeddings: dict[str, np.ndarray] = {}
    if with_embeddings:
        from age_gap.models.embeddings import load_embeddings

        embeddings = load_embeddings()
        log.info("Загружено эмбеддингов: %d", len(embeddings))

    rows: list[dict[str, Any]] = []
    emb_rows: list[np.ndarray] = []
    emb_dim = 0

    for pid, post in posts.items():
        eng = engagement.get(pid)
        if eng is None:
            continue  # пост исчез со стены / не покрыт backfill
        flist = faces_by_post.get(pid, [])
        if not flist:
            continue

        person = person_of_post.get(pid, pid)
        if person in minors:
            st.posts_dropped_minor_subject += 1
            continue

        adult, child = split_adult_child(flist, genderage)
        st.faces_child_excluded_from_appearance += len(child)

        adult_usable = [(f, g) for f, g in adult if f.get("is_usable")]
        if not adult_usable:
            st.posts_dropped_no_adult_face += 1
            continue

        det = np.array([float(f["det_score"]) for f, _ in adult_usable])
        qual = np.array([float(f["face_quality_score"]) for f, _ in adult_usable])
        areas = np.array([_bbox_area(f.get("bbox")) for f, _ in adult_usable])
        ages = np.array([float(g["age_est"]) for _, g in adult_usable])
        genders = np.array([int(g["gender"]) for _, g in adult_usable])  # 0=F, 1=M

        # возраст из подписи (then/now) — «насколько сильно изменился»
        label_ages = [face_ages[f["face_id"]] for f in flist if f["face_id"] in face_ages]
        age_gap_lbl = (max(label_ages) - min(label_ages)) if len(label_ages) >= 2 else np.nan

        ts = eng.get("ts_unix")
        dt = pd.to_datetime(ts, unit="s", utc=True) if ts else pd.NaT

        row: dict[str, Any] = {
            "post_id": pid,
            "person_id": person,
            "platform": platform,
            "owner_id": str(eng.get("owner_id", "")),
            # --- таргет-сырьё
            "likes": eng.get("likes", 0),
            "comments": eng.get("comments"),
            "reposts": eng.get("reposts", 0),
            "views": eng.get("views"),
            # --- reach / мета
            "ts_unix": ts,
            "hour": dt.hour if pd.notna(dt) else np.nan,
            "weekday": dt.weekday() if pd.notna(dt) else np.nan,
            "month": dt.month if pd.notna(dt) else np.nan,
            "year": dt.year if pd.notna(dt) else np.nan,
            "n_photos": len(post.get("photos", []) or []),
            "is_pinned": eng.get("is_pinned", 0),
            "marked_as_ads": eng.get("marked_as_ads", 0),
            "is_repost": eng.get("is_repost", 0),
            # --- лицо (только взрослые)
            "n_faces_total": len(flist),
            "n_adult_faces": len(adult),
            "n_adult_usable": len(adult_usable),
            "n_child_faces": len(child),  # структурный факт формата «тогда/сейчас»
            "has_child_photo": int(len(child) > 0),
            "frac_usable_adult": len(adult_usable) / max(1, len(adult)),
            "det_score_mean": float(det.mean()),
            "det_score_max": float(det.max()),
            "face_quality_mean": float(qual.mean()),
            "face_quality_max": float(qual.max()),
            "bbox_area_mean": float(np.nanmean(areas)),
            "bbox_area_max": float(np.nanmax(areas)),
            "share_female_adult": float((genders == 0).mean()),
            "age_est_median": float(np.median(ages)),
            "age_est_min": float(ages.min()),
            "age_est_max": float(ages.max()),
            # --- domain
            "age_gap_label": age_gap_lbl,
            "has_age_labels": int(len(label_ages) >= 2),
        }
        row.update(_text_features(post.get("caption", "")))
        rows.append(row)

        if with_embeddings:
            vecs = [embeddings[f["face_id"]] for f, _ in adult_usable if f["face_id"] in embeddings]
            if vecs:
                v = np.mean(np.stack(vecs), axis=0)
                emb_dim = v.shape[0]
            else:
                v = None
            emb_rows.append(v if v is not None else np.full(emb_dim or 512, np.nan, dtype=np.float32))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Пустая таблица признаков — проверьте backfill и артефакты лиц.")

    # возраст поста относительно самого свежего поста в выборке (детерминированно)
    ref = df["ts_unix"].max()
    df["post_age_days"] = (ref - df["ts_unix"]) / 86400.0

    if with_embeddings and emb_rows:
        emb = np.vstack(emb_rows).astype(np.float32)
        emb_cols = [f"{emb_prefix}{i:03d}" for i in range(emb.shape[1])]
        df = pd.concat([df, pd.DataFrame(emb, columns=emb_cols, index=df.index)], axis=1)
        st.notes.append(f"embeddings: {emb.shape[1]}-d, только взрослые usable-лица")

    st.posts_final = len(df)
    st.engagement_coverage = round(len(engagement) / max(1, st.posts_total), 4)
    log.info("Таблица признаков: %s", df.shape)
    return df, st


def feature_columns(df: pd.DataFrame, emb_prefix: str = "emb") -> dict[str, list[str]]:
    """Разделение колонок на блоки — нужно для reach/content-моделей и ablation-лестницы."""
    emb = sorted(c for c in df.columns if c.startswith(emb_prefix))
    reach = [
        "hour", "weekday", "month", "year", "post_age_days",
        "n_photos", "is_pinned", "marked_as_ads", "is_repost", "caption_len",
    ]
    meta_content = [
        "n_photos", "caption_len", "word_count", "emoji_count", "has_emoji",
        "n_hashtags", "n_questions", "n_exclam", "upper_ratio", "has_digit",
        "n_faces_total", "has_child_photo", "n_child_faces",
    ]
    face_attrs = [
        "n_adult_faces", "n_adult_usable", "frac_usable_adult",
        "det_score_mean", "det_score_max", "face_quality_mean", "face_quality_max",
        "bbox_area_mean", "bbox_area_max", "share_female_adult",
        "age_est_median", "age_est_min", "age_est_max",
    ]
    domain = ["age_gap_label", "has_age_labels"]
    keep = lambda cols: [c for c in cols if c in df.columns]  # noqa: E731
    return {
        "reach": keep(reach),
        "meta": keep(meta_content),
        "face": keep(face_attrs),
        "domain": keep(domain),
        "emb": emb,
    }


def save_table(df: pd.DataFrame, platform: str = "vk") -> Path:
    out = data_path("data_dir", "processed", f"engagement_features_{platform}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
    except Exception as exc:  # noqa: BLE001 — pyarrow может отсутствовать
        out = out.with_suffix(".csv.gz")
        df.to_csv(out, index=False, compression="gzip")
        log.warning("parquet недоступен (%s) -> сохранено в %s", exc, out)
    log.info("Сохранено: %s", out)
    return out
