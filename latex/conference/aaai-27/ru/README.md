# AAAI-27 — Russian mirror (NOT for AAAI submission)

`main_ru.tex` is a **Russian-language mirror** of the conference paper (`../en/main.tex`),
for review/checking and the thesis/defense (ВКР). It is **not** an AAAI submission:

- The AAAI template (`aaai2027.sty`) **forbids Cyrillic in body text** (`babel` is banned; the
  AuthorKit requires non-Roman alphabets to be "restricted to bit-mapped figures"). So the
  official AAAI submission is English-only (`../en/`).
- This mirror is therefore built **outside** the AAAI template, on a Cyrillic-capable
  two-column `article` setup (`babel` russian + `tempora` for bold Cyrillic + `geometry`),
  the same font approach as `papers/journal-1-tbiom/ru/main_ru.tex`.
- Because it is not submitted under double-blind, it shows the **real author names**.

## Build

```sh
pdflatex main_ru ; bibtex main_ru ; pdflatex main_ru ; pdflatex main_ru
```

Uses shared assets via relative paths (`../../../shared/figures/` with the `_ru` figure
variants, `../../../shared/refs`). Current status: 7 pp, 0 errors / 0 undefined / 0 overfull,
no Type-3 fonts. Content mirrors `../en/main.tex` (same tables/figures/results), translated to
Russian; the supplementary appendix has not been mirrored (English `../en/supplement.tex`).
