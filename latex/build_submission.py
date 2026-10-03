#!/usr/bin/env python3
"""Собрать самодостаточный пакет для подачи журнальной статьи (T-BIOM / TNNLS).

    python latex/build_submission.py journal-1-tbiom
    python latex/build_submission.py journal-1-tnnls
    python latex/build_submission.py --all

Раньше пакет TNNLS собирался руками по инструкции в submission/SUBMISSION.md. Теперь это
воспроизводимо и одинаково для обоих журналов: en/ + shared/ -> submission/{manuscript,supplement}/.

Что делает:
  1. копирует en/main.tex, локализуя общие пути (\\graphicspath, \\bibliography) под папку пакета,
     чтобы архив был самодостаточным (порталы IEEE требуют именно этого);
  2. копирует refs.bib, используемые фигуры и IEEEtran.cls;
  3. ВЫРЕЗАЕТ комментарии LaTeX (submission/strip_comments.py). Это не косметика: рецензент,
     открывший исходник, иначе прочитал бы закомментированный camera-ready блок с именами
     авторов и деанонимизировал бы подачу;
  4. компилирует pdflatex -> bibtex -> pdflatex x2 и проверяет, что нет ошибок и битых ссылок.

Пакет лежит в .gitignore (он производный) — пересобирать перед каждой подачей.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

LATEX = Path(__file__).resolve().parent
PAPERS = ["journal-1-tbiom", "journal-1-tnnls"]


def _localize(tex: str) -> str:
    """Общие относительные пути -> локальные (пакет должен быть самодостаточным)."""
    tex = re.sub(r"\\graphicspath\{\{[^}]*\}\}", r"\\graphicspath{{./figures/}}", tex)
    return re.sub(r"\\bibliography\{[^}]*\}", r"\\bibliography{refs}", tex)


def _latex_reference_errors(log: str) -> int:
    """Include wrapped citation warnings, not just undefined cross-references."""
    flattened = re.sub(r"\s+", " ", log)
    return len(re.findall(
        r"(?:Citation|Reference) .*? undefined|multiply defined|There were undefined references",
        flattened,
    ))


def _figures(tex: str) -> list[str]:
    return sorted({m for m in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex)})


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace",
                          check=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_nonblank_pages(pdf: Path, pages: int) -> None:
    """Fail when the portal-facing PDF contains a textually empty page."""
    pdf = pdf.resolve()
    if not shutil.which("pdftotext"):
        print("    ! pdftotext не найден — проверка пустых страниц пропущена")
        return
    for page in range(1, pages + 1):
        result = _run(
            ["pdftotext", "-f", str(page), "-l", str(page), str(pdf), "-"],
            pdf.parent,
        )
        if result.returncode or not result.stdout.strip():
            raise RuntimeError(f"пустая или нечитаемая страница {page}: {pdf}")


def _check_source_privacy(paths: list[Path]) -> None:
    """Reject obvious row identifiers or secrets in portal-facing textual sources."""
    patterns = {
        "private corpus row/photo identifier": re.compile(
            r"(?:vk_|reddit_)?-?\d{5,}_\d{5,}", re.IGNORECASE
        ),
        "credential/token marker": re.compile(
            r"(?:vk1\.a\.|bearer\s+[a-z0-9._-]{20,}|client_secret\s*[=:])",
            re.IGNORECASE,
        ),
        "machine-specific absolute path": re.compile(
            r"(?<![a-z0-9])(?:[a-z]:[\\/]|/(?:home|users)/[^/\s]+/)", re.IGNORECASE
        ),
    }
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in patterns.items():
            if pattern.search(text):
                raise RuntimeError(f"{label} found in portal-facing source: {path}")


def _package_uploads(
    sub: Path, page_counts: dict[str, int], *, include_experiment_index: bool = False
) -> None:
    """Create portal artifacts without mixing PDFs, sources, or build logs."""
    evidence = None
    lfw_evidence = None
    curation_evidence = None
    roc_v2_evidence = None
    if include_experiment_index:
        root = LATEX.parent
        staging = root / ".work" / "submission-evidence" / uuid.uuid4().hex
        result = _run(
            [sys.executable, str(root / "scripts/export_public_evidence.py"),
             "--root", str(root), "--out", str(staging)],
            root,
        )
        if result.returncode:
            raise RuntimeError("public evidence export failed:\n" + result.stdout + result.stderr)
        evidence = staging / "public_evidence_bundle.zip"
        if not evidence.is_file():
            raise FileNotFoundError("public evidence builder produced no archive")
        lfw_staging = staging / "lfw"
        lfw_result = _run(
            [sys.executable, "-m", "scripts.export_lfw_evidence", "--root", str(root),
             "--out", str(lfw_staging)], root,
        )
        if lfw_result.returncode:
            raise RuntimeError("LFW evidence export failed:\n" + lfw_result.stdout + lfw_result.stderr)
        lfw_evidence = lfw_staging / "lfw_evidence_bundle.zip"
        if not lfw_evidence.is_file():
            raise FileNotFoundError("LFW evidence builder produced no archive")
        curation_staging = staging / "curation"
        curation_result = _run(
            [sys.executable, "-m", "scripts.export_curation_evidence", "--root", str(root),
             "--out", str(curation_staging)], root,
        )
        if curation_result.returncode:
            raise RuntimeError("curation evidence export failed:\n" + curation_result.stdout + curation_result.stderr)
        curation_evidence = curation_staging / "curation_evidence_bundle.zip"
        if not curation_evidence.is_file():
            raise FileNotFoundError("curation evidence builder produced no archive")
        roc_staging = staging / "roc-v2"
        roc_result = _run(
            [sys.executable, "-m", "scripts.export_roc_v2_evidence", "--root", str(root),
             "--out", str(roc_staging)], root,
        )
        if roc_result.returncode:
            raise RuntimeError("ROC-v2 evidence export failed:\n" + roc_result.stdout + roc_result.stderr)
        roc_v2_evidence = roc_staging / "roc_v2_evidence_bundle.zip"
        if not roc_v2_evidence.is_file():
            raise FileNotFoundError("ROC-v2 evidence builder produced no archive")
    man, supp = sub / "manuscript", sub / "supplement"
    upload = sub / "upload"
    upload.mkdir(exist_ok=True)
    managed = {
        "main.pdf",
        "supplement.pdf",
        "manuscript-source.zip",
        "artifact-manifest.json",
        "publication-evidence.zip",
        "lfw-evidence.zip",
        "curation-evidence.zip",
        "roc-v2-evidence.zip",
    }
    for name in managed:
        target = upload / name
        if target.exists():
            target.unlink()

    shutil.copy2(man / "main.pdf", upload / "main.pdf")
    if (supp / "supplement.pdf").exists():
        shutil.copy2(supp / "supplement.pdf", upload / "supplement.pdf")
    if evidence is not None:
        shutil.copy2(evidence, upload / "publication-evidence.zip")
    if lfw_evidence is not None:
        shutil.copy2(lfw_evidence, upload / "lfw-evidence.zip")
    if curation_evidence is not None:
        shutil.copy2(curation_evidence, upload / "curation-evidence.zip")
    if roc_v2_evidence is not None:
        shutil.copy2(roc_v2_evidence, upload / "roc-v2-evidence.zip")

    source_files = ["main.tex", "main.bbl", "refs.bib", "IEEEtran.cls"]
    figures = sorted((man / "figures").glob("*.pdf"))
    source_paths = [man / name for name in source_files]
    if (supp / "supplement.tex").exists():
        source_paths.append(supp / "supplement.tex")
    experiment_index = (
        LATEX.parent / "metrics" / "publication_artifact_index.json"
        if include_experiment_index
        else None
    )
    supplement_source = supp / "supplement.tex"
    if experiment_index is not None and supplement_source.is_file():
        supplement_text = supplement_source.read_text(encoding="utf-8")
        if "publication\\_artifact\\_index.json" in supplement_text and not experiment_index.is_file():
            raise FileNotFoundError(
                "supplement references metrics/publication_artifact_index.json, but it is missing"
            )
    if experiment_index is not None and experiment_index.is_file():
        source_paths.append(experiment_index)
    _check_source_privacy(source_paths)
    risky_figure_names = [
        figure.name
        for figure in [*figures, *(supp / "figures").glob("*.pdf")]
        if re.search(r"face|photo|crop|person|sample|example", figure.stem, re.IGNORECASE)
    ]
    if risky_figure_names:
        raise RuntimeError(f"potentially identifying figure names require manual review: {risky_figure_names}")
    with zipfile.ZipFile(upload / "manuscript-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in source_files:
            path = man / name
            if not path.exists():
                raise FileNotFoundError(f"нет обязательного файла submission: {path}")
            archive.write(path, name)
        for figure in figures:
            archive.write(figure, f"figures/{figure.name}")
        if experiment_index is not None and experiment_index.is_file():
            archive.write(experiment_index, "artifacts/publication_artifact_index.json")

    for stem, pages in page_counts.items():
        _check_nonblank_pages(upload / f"{stem}.pdf", pages)

    artifacts = []
    for path in sorted(upload.iterdir()):
        if path.name == "artifact-manifest.json" or not path.is_file():
            continue
        artifacts.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "pages": page_counts.get(path.stem),
            }
        )
    (upload / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "privacy_checks": {
                    "text_sources_scanned_for_row_ids_and_secret_markers": True,
                    "machine_specific_absolute_paths_rejected": True,
                    "source_archive_allowed_members": [
                        "tex",
                        "bbl",
                        "bib",
                        "cls",
                        "figures/*.pdf",
                        "artifacts/publication_artifact_index.json",
                    ],
                },
                "artifacts": artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"    upload: {upload} (PDF и source ZIP разделены, build-логи исключены)")


def build(paper: str, *, submission_dir: Path | None = None, compile_only: bool = False) -> bool:
    if compile_only and submission_dir is None:
        raise ValueError("compile-only requires an isolated submission_dir")
    src = LATEX / "papers" / paper
    en = src / "en"
    if not (en / "main.tex").exists():
        print(f"[пропуск] {paper}: нет en/main.tex")
        return False
    sub = submission_dir.resolve() if submission_dir is not None else src / "submission"
    man, supp = sub / "manuscript", sub / "supplement"
    for d in (man, man / "figures", supp):
        d.mkdir(parents=True, exist_ok=True)

    main_tex = _localize((en / "main.tex").read_text(encoding="utf-8"))
    (man / "main.tex").write_text(main_tex, encoding="utf-8", newline="\n")
    shutil.copy2(LATEX / "shared" / "refs.bib", man / "refs.bib")
    used = set(_figures(main_tex))
    for stale in (man / "figures").glob("*.pdf"):      # фигуры могли переехать в дополнение —
        if stale.name not in used:                     # старые копии удаляем, иначе поедут в архив
            stale.unlink()
    for fig in sorted(used):
        s = LATEX / "shared" / "figures" / fig
        if s.exists():
            shutil.copy2(s, man / "figures" / fig)
        else:
            print(f"    ! нет фигуры {fig}")
    cls = LATEX / "shared" / "vendor" / "ieee-tnnls" / "IEEEtran.cls"
    if cls.exists():
        shutil.copy2(cls, man / "IEEEtran.cls")

    if (en / "supplement.tex").exists():
        supp_tex = _localize((en / "supplement.tex").read_text(encoding="utf-8"))
        (supp / "supplement.tex").write_text(supp_tex, encoding="utf-8", newline="\n")
        shutil.copy2(LATEX / "shared" / "refs.bib", supp / "refs.bib")
        if cls.exists():
            shutil.copy2(cls, supp / "IEEEtran.cls")
        # дополнение тоже может содержать фигуры (в T-BIOM туда вынесены четыре кривые,
        # чтобы рукопись уложилась в 10 страниц) -> копируем их рядом с ним
        supp_figs = set(_figures(supp_tex))
        if supp_figs:
            (supp / "figures").mkdir(exist_ok=True)
            for stale in (supp / "figures").glob("*.pdf"):
                if stale.name not in supp_figs:
                    stale.unlink()
            for fig in sorted(supp_figs):
                s = LATEX / "shared" / "figures" / fig
                if s.exists():
                    shutil.copy2(s, supp / "figures" / fig)
                else:
                    print(f"    ! нет фигуры дополнения {fig}")

    # вырезаем комментарии (защита от деанонимизации через исходник)
    strip = src / "submission" / "strip_comments.py"
    if strip.exists():
        targets = [str(man / "main.tex")]
        if (supp / "supplement.tex").exists():
            targets.append(str(supp / "supplement.tex"))
        r = _run([sys.executable, str(strip), *targets], sub)
        if r.returncode:
            print(f"    ! strip_comments: {r.stderr.strip()[:200]}")
    else:
        print("    ! strip_comments.py не найден — исходник НЕ анонимизирован")

    ok = True
    page_counts: dict[str, int] = {}
    for d, stem in [(man, "main"), (supp, "supplement")]:
        if not (d / f"{stem}.tex").exists():
            continue
        _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem], d)
        has_bibliography = r"\bibliography{" in (d / f"{stem}.tex").read_text(encoding="utf-8")
        bibliography_result = _run(["bibtex", stem], d) if has_bibliography else None
        _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem], d)
        r = _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem], d)
        log = (d / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
        bad = _latex_reference_errors(log)
        pages = re.search(r"Output written on \S+ \((\d+) pages", log)
        bibliography_ok = bibliography_result is None or bibliography_result.returncode == 0
        status = "OK" if r.returncode == 0 and bad == 0 and bibliography_ok else "ПРОБЛЕМА"
        ok &= status == "OK"
        print(f"    {stem}.pdf: {status}, страниц {pages.group(1) if pages else '?'}"
              f"{f', битых ссылок {bad}' if bad else ''}")
        if pages:
            page_counts[stem] = int(pages.group(1))
    if ok and not compile_only:
        include_experiment_index = paper == "journal-1-tbiom"
        if include_experiment_index:
            index_builder = LATEX.parent / "scripts" / "build_publication_artifact_index.py"
            index_result = _run([sys.executable, str(index_builder)], LATEX.parent)
            if index_result.returncode:
                raise RuntimeError(
                    "не удалось обновить publication artifact index:\n"
                    + index_result.stdout
                    + index_result.stderr
                )
            print("    artifact index обновлён непосредственно перед упаковкой")
        _package_uploads(
            sub,
            page_counts,
            include_experiment_index=include_experiment_index,
        )
    if compile_only:
        print("    compile-only: artifact index and upload package were not updated")
    print(f"[{paper}] -> {sub}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper", nargs="?", choices=PAPERS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--submission-dir", type=Path,
                    help="isolated build destination; canonical sources remain unchanged")
    ap.add_argument("--compile-only", action="store_true",
                    help="compile isolated PDFs without refreshing the index or packaging uploads")
    args = ap.parse_args()
    if not args.paper and not args.all:
        ap.error("укажите статью или --all")
    if args.all and args.submission_dir is not None:
        ap.error("--submission-dir requires one paper, not --all")
    if args.compile_only and args.submission_dir is None:
        ap.error("--compile-only requires an isolated --submission-dir")
    ok = all(build(p, submission_dir=args.submission_dir, compile_only=args.compile_only)
             for p in (PAPERS if args.all else [args.paper]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
