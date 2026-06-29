# Article 1 — IEEE TNNLS

"Cross-Age Face Verification from Mined Same-Person Social-Media Pairs."
Target: IEEE Transactions on Neural Networks and Learning Systems (TNNLS).

A **venue-retargeted copy** of `../journal-1-tbiom/` — same content (tables, figures, results,
reframed ethics), reformatted for TNNLS:

- `\documentclass[journal]{IEEEtran}` with the TNNLS running header (`\markboth`) and the
  **standard in-column abstract** (`\maketitle` → `\begin{abstract}` → `\begin{IEEEkeywords}`),
  instead of T-BIOM's `\IEEEtitleabstractindextext` full-width block.

| Path | Purpose |
| --- | --- |
| `en/main.tex` | English submission |
| `en/supplement.tex` | Supplementary material (standalone, 2 pp) |
| `ru/main_ru.tex` | Russian mirror (tempora + babel for Cyrillic bold) |

## Build

```sh
cd en && pdflatex main    ; bibtex main    ; pdflatex main    ; pdflatex main
cd ru && pdflatex main_ru ; bibtex main_ru ; pdflatex main_ru ; pdflatex main_ru
```

Shared assets via relative paths: `../../../shared/refs.bib`, `../../../shared/figures/`.
Builds on MiKTeX's IEEEtran (the TNNLS-provided `IEEEtran.cls` v1.8b in
`../../shared/vendor/ieee-tnnls/` is the same class; the `lettersize` option in the TNNLS
sample is ignored, so `[journal]` is used). Compiles clean: EN 11 pp / RU 12 pp / supplement
2 pp; 0 errors / 0 undefined / 0 overfull.

## Keep in sync

Content is identical to `../journal-1-tbiom/`; only the journal header and abstract layout
differ. When editing the science, update both (or re-copy and re-apply the four retargeting
edits: comment, `\markboth`, the two abstract-block edits). Ethics use the dataset-on-request
stance (no red `\fillin`). Author-owned before submit: real manual-validation counts.
