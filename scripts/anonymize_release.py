"""CLI: экспорт ОБЕЗЛИЧЕННОГО релиза (docs/DATA_GOVERNANCE.md §4 / чек-лист).

Пишет в release/ только производные артефакты: обезличенные группы и пары (хеш-id, без подписей/VK-id),
manifest. Несовершеннолетние (apparent age < 18 по genderage) исключаются. Сырые лица/посты/подписи НЕ
экспортируются. Секретная соль — data/.anon_salt (gitignored, НЕ публикуется); без неё хеши необратимы.
Финальный safety-scan падает, если в выводе обнаружен сырой VK-идентификатор.

    uv run python scripts/anonymize_release.py
    uv run python scripts/anonymize_release.py --embeddings   # + эмбеддинги (только controlled access!)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from age_gap.common.io import data_path, read_jsonl, resolve_path, write_jsonl
from age_gap.common.logging import get_logger
from age_gap.common.schemas import IdentityGroup, Pair
from age_gap.datasets.anonymize import MINOR_AGE, anon_group, anon_pair, hash_id

log = get_logger(__name__)

# Сырой VK-id — длинное ОТРИЦАТЕЛЬНОЕ число (owner id, напр. -77072632); hex-хеши, типы пар, split,
# label и неотрицательный age_gap его не содержат → точный и обобщённый маркер для safety-scan.
_RAW_VK_ID = re.compile(r"-\d{5,}")


def _load_salt() -> bytes:
    """Секретная соль из data/.anon_salt (gitignored); генерируется при отсутствии. НЕ публикуется."""
    salt_path = Path(resolve_path(str(data_path("data_dir", ".anon_salt"))))
    if salt_path.exists():
        return bytes.fromhex(salt_path.read_text().strip())
    salt = os.urandom(32)
    salt_path.parent.mkdir(parents=True, exist_ok=True)
    salt_path.write_text(salt.hex())
    log.info("Сгенерирована новая соль -> %s (gitignored, НЕ публиковать)", salt_path)
    return salt


def _minor_faces() -> set[str]:
    """Лица с apparent age < 18 (genderage sidecar) — исключаются из релиза."""
    path = data_path("data_dir", "interim", "face_genderage.jsonl")
    return {r["face_id"] for r in read_jsonl(path) if int(r.get("age_est", 99)) < MINOR_AGE}


def _safety_scan(paths: list[Path]) -> None:
    """Гарантия: в релизных файлах нет сырых VK-идентификаторов (длинных отрицательных чисел). Иначе abort."""
    for p in paths:
        m = _RAW_VK_ID.search(p.read_text(encoding="utf-8"))
        if m:
            raise SystemExit(f"SAFETY-SCAN FAIL: возможный сырой VK-id {m.group()!r} в {p.name}")
    log.info("Safety-scan пройден: сырых VK-id в релизе нет (%d файлов)", len(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export de-identified research release")
    parser.add_argument(
        "--embeddings", action="store_true", help="включить эмбеддинги (controlled access!)"
    )
    args = parser.parse_args()

    salt = _load_salt()
    minors = _minor_faces()
    out_dir = Path(resolve_path("release"))
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = [
        IdentityGroup.from_dict(r)
        for r in read_jsonl(str(data_path("data_dir", "processed", "identity_groups.jsonl")))
    ]
    anon_groups = [a for g in groups if (a := anon_group(g, salt, minors)) is not None]
    write_jsonl(out_dir / "groups_anon.jsonl", anon_groups)

    pairs = [
        Pair.from_dict(r)
        for r in read_jsonl(str(data_path("data_dir", "processed", "pairs.jsonl")))
    ]
    anon_pairs = [a for p in pairs if (a := anon_pair(p, salt, minors)) is not None]
    write_jsonl(out_dir / "pairs_anon.jsonl", anon_pairs)

    written = [out_dir / "groups_anon.jsonl", out_dir / "pairs_anon.jsonl"]

    if args.embeddings:
        import numpy as np

        from age_gap.models.embeddings import load_embeddings

        emb = load_embeddings()
        ids = [f for f in emb if f not in minors]
        mat = np.stack([emb[f] for f in ids]) if ids else np.zeros((0, 512), np.float32)
        hashed = np.asarray([hash_id(f, salt) for f in ids], dtype=object)
        np.savez(out_dir / "embeddings_anon.npz", face_ids=hashed, embeddings=mat)
        log.info("Эмбеддинги (controlled access!): %d лиц -> embeddings_anon.npz", len(ids))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "salt_fingerprint": hashlib_fp(salt),
        "n_groups": len(anon_groups),
        "n_pairs": len(anon_pairs),
        "n_minor_faces_excluded": len(minors),
        "includes_embeddings": bool(args.embeddings),
        "contains_raw_images": False,
        "contains_captions_or_pii": False,
        "note": "De-identified research release. No raw faces/posts/IDs. Salt not published. "
        "Minors (apparent age<18) excluded. Research-only; governance and release policy are described in the accompanying paper.",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _safety_scan(written)
    log.info(
        "Релиз -> %s | групп=%d, пар=%d, исключено minor-лиц=%d",
        out_dir,
        len(anon_groups),
        len(anon_pairs),
        len(minors),
    )


def hashlib_fp(salt: bytes) -> str:
    """Отпечаток соли (для воспроизводимости релиза без раскрытия самой соли)."""
    import hashlib

    return hashlib.sha256(salt).hexdigest()[:12]


if __name__ == "__main__":
    main()
