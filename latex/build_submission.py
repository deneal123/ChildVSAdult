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
import re
import shutil
import subprocess
import sys
from pathlib import Path

LATEX = Path(__file__).resolve().parent
PAPERS = ["journal-1-tbiom", "journal-1-tnnls"]


def _localize(tex: str) -> str:
    """Общие относительные пути -> локальные (пакет должен быть самодостаточным)."""
    tex = re.sub(r"\\graphicspath\{\{[^}]*\}\}", r"\\graphicspath{{./figures/}}", tex)
    return re.sub(r"\\bibliography\{[^}]*\}", r"\\bibliography{refs}", tex)


def _figures(tex: str) -> list[str]:
    return sorted({m for m in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", tex)})


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace",
                          check=False)


def build(paper: str) -> bool:
    src = LATEX / "papers" / paper
    en = src / "en"
    if not (en / "main.tex").exists():
        print(f"[пропуск] {paper}: нет en/main.tex")
        return False
    sub = src / "submission"
    man, supp = sub / "manuscript", sub / "supplement"
    for d in (man, man / "figures", supp):
        d.mkdir(parents=True, exist_ok=True)

    main_tex = _localize((en / "main.tex").read_text(encoding="utf-8"))
    (man / "main.tex").write_text(main_tex, encoding="utf-8", newline="\n")
    shutil.copy2(LATEX / "shared" / "refs.bib", man / "refs.bib")
    for fig in _figures(main_tex):
        s = LATEX / "shared" / "figures" / fig
        if s.exists():
            shutil.copy2(s, man / "figures" / fig)
        else:
            print(f"    ! нет фигуры {fig}")
    cls = LATEX / "shared" / "vendor" / "ieee-tnnls" / "IEEEtran.cls"
    if cls.exists():
        shutil.copy2(cls, man / "IEEEtran.cls")

    if (en / "supplement.tex").exists():
        (supp / "supplement.tex").write_text(
            _localize((en / "supplement.tex").read_text(encoding="utf-8")),
            encoding="utf-8", newline="\n")

    # вырезаем комментарии (защита от деанонимизации через исходник)
    strip = sub / "strip_comments.py"
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
    for d, stem in [(man, "main"), (supp, "supplement")]:
        if not (d / f"{stem}.tex").exists():
            continue
        _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem], d)
        _run(["bibtex", stem], d)
        _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem], d)
        r = _run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem], d)
        log = (d / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
        bad = len(re.findall(r"Reference .* undefined|multiply defined", log))
        pages = re.search(r"Output written on \S+ \((\d+) pages", log)
        status = "OK" if r.returncode == 0 and bad == 0 else "ПРОБЛЕМА"
        ok &= status == "OK"
        print(f"    {stem}.pdf: {status}, страниц {pages.group(1) if pages else '?'}"
              f"{f', битых ссылок {bad}' if bad else ''}")
    print(f"[{paper}] -> {sub}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper", nargs="?", choices=PAPERS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not args.paper and not args.all:
        ap.error("укажите статью или --all")
    ok = all(build(p) for p in (PAPERS if args.all else [args.paper]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
