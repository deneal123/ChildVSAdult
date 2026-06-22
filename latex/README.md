# LaTeX sources — IEEE T-BIOM submission

Two mirrored versions of the paper "Age-Invariant Identity Matching from
Naturally-Anchored Social-Media Posts":

| File | Purpose |
| --- | --- |
| `main.tex` | **English** (the submission language for IEEE T-BIOM), 8 pages |
| `main_ru.tex` | **Russian** mirror — same structure, tables, figures — 9 pages |
| `refs.bib` | shared bibliography (compiled with `IEEEtran.bst`) |
| `figures/` | vector PDF figures from `../scripts/make_figures.py`: English (`fig_*.pdf`) and Russian (`fig_*_ru.pdf`) variants |
| `INFO.md` | IEEE T-BIOM author requirements (reference; git-ignored) |
| `FormatJournal/` | the official IEEEtran bundle + author HOWTOs (reference only; git-ignored) |
| `_attic/` | leftover template files from an unrelated project (git-ignored) |

## Build

```sh
pdflatex main    ; bibtex main    ; pdflatex main    ; pdflatex main
pdflatex main_ru ; bibtex main_ru ; pdflatex main_ru ; pdflatex main_ru
```

Compiles with MiKTeX, which auto-installs IEEEtran and the T2A Cyrillic fonts on
first run (`pdflatex --enable-installer`). Both papers compile cleanly (0 errors,
0 undefined references, 0 overfull boxes); EN is 8 pages and the RU mirror 9
(Cyrillic is wider), both within the 10-page regular-paper limit.

## Figures

`../scripts/make_figures.py` regenerates every figure from hardcoded curated-data
constants (no experiment re-run needed), in **two languages**: `fig_*.pdf`
(English, used by `main.tex`) and `fig_*_ru.pdf` (Russian, used by `main_ru.tex`).
Style: column-width, Times serif with embedded fonts, no in-figure titles,
Okabe-Ito colour-blind-safe palette with grayscale-safe markers/hatching.
Technical tokens (FG-NET, ROC-AUC, frozen, +pairs, backbone names) stay Latin in
both languages.

## Draft conventions

- **RED `[text]`** — real data the authors must still fill in before submission
  (currently only: the ethics-committee name, the protocol/exemption, and the
  approval date in the Ethics section).
- **GREEN "Review note" boxes** — the assistant's editorial notes; delete them
  before submission.

## Reproducibility / compute

The whole study runs on a single laptop: NVIDIA RTX 3060 Laptop GPU (6 GB), AMD
Ryzen 7 5800H (8C/16T), 16 GB RAM, Windows 11; Python 3.12, PyTorch 2.12 (CUDA
13.2, cuDNN 9.2), InsightFace 1.0 / ONNX Runtime 1.26. The full environment is
stated in each paper's Reproducibility section.

## Notes

- `FormatJournal/IEEEtran.cls` and `.bst` in the downloaded bundle are corrupt
  (they contain changelog text, not the class/style); the build relies on MiKTeX's
  genuine IEEEtran package.
- The de-identified `release/` (hashed IDs only, no raw faces/PII/salt, minors
  excluded) is tracked in this **private** repository as the controlled-access
  benchmark protocol. Revisit the publication decision before making the repo public.
