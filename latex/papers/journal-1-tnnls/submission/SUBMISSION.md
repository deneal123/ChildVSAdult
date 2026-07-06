# TNNLS submission — file routing (journal-1-tnnls, English)

**This journal uses double-blind review.** The portal's *title page* (with author names) is
uploaded separately and is **not sent to reviewers**, so the manuscript itself is
**anonymized** — no author names, affiliation, e-mail, ORCID, or repository URL. The real
author info is kept in commented lines in `../en/main.tex` (search `CAMERA-READY`) and on the
separate title page. `../en/main.tex` and `../en/supplement.tex` are now the anonymized
(submission) versions.

**You are uploading the `.tex` source**, so the copies in `manuscript/` and `supplement/`
have had **all LaTeX comments removed** (via `strip_comments.py`). This is essential: a
reviewer who opens the source would otherwise read the commented-out camera-ready author
block and de-anonymize you. (`../en/*.tex` keep their comments — they are not uploaded.)

## What to upload where

| Portal slot | Upload | Source |
| --- | --- | --- |
| **Основная рукопись** (main manuscript — required) | a single ZIP of the **contents** of `manuscript/` (so `main.tex` sits at the zip root) | `submission/manuscript/` |
| **Титульная страница** (title page — required, MS Word) | `title_page_en.docx` | `submission/title_page_en.docx` |
| **Конфликт интересов** (required) | nothing — you already selected *"no conflict to disclose"* (it is also stated in the manuscript's Funding and Conflicts of Interest section) | — |
| **Дополнительные материалы для обзора** (optional) | `supplement.pdf` (and, if asked, `supplement.tex`) | `submission/supplement/` |
| **Дополнительный файл LaTeX** (optional) | `supplement.tex` — only if the system wants the supplement source separately | `submission/supplement/` |
| Изображение · Отслеживаемые изменения · Ранее опубликовано · Сопроводительное письмо | skip — optional / N/A for a first submission | — |

## The main-manuscript ZIP (`manuscript/`)

Self-contained, anonymized, already compiles to 11 pp (0 errors / 0 overfull):

```
main.tex   refs.bib   main.bbl   main.pdf   IEEEtran.cls   figures/ (6 PDFs)
```

- It **excludes the supplement** (the portal requires the main manuscript to contain no
  supplementary material — the supplement goes to the "Дополнительные материалы" slot).
- Zip the files **inside** `manuscript/` (not the folder itself), so `main.tex` is at the root.
- If the portal would rather have one review PDF than a source archive, just upload `main.pdf`.

## Rebuilding the bundle (it is git-ignored, regenerate any time)

From `latex/papers/journal-1-tnnls/`:

1. `manuscript/main.tex` = `en/main.tex` with `\graphicspath{{./figures/}}` and
   `\bibliography{refs}` (shared paths localized); copy `shared/refs.bib`, the six
   `shared/figures/fig_{headroom,scaling,shortcut,sota,fairness,external}.pdf`, and
   `shared/vendor/ieee-tnnls/IEEEtran.cls`.
2. **Strip all comments** (double-blind safety, since the `.tex` is uploaded):
   `python submission/strip_comments.py submission/manuscript/main.tex submission/supplement/supplement.tex`.
3. Compile: `pdflatex main; bibtex main; pdflatex main; pdflatex main` (bibtex uses MiKTeX's
   genuine `IEEEtran.bst` — do **not** copy the corrupt `ieee-tbiom/IEEEtran.bst`).
4. `supplement/` = `en/supplement.{tex,pdf}` (standalone + anonymized), then comment-stripped in step 2.

## Camera-ready (after acceptance)

In `en/main.tex` and `en/supplement.tex`, uncomment the real author block(s) (the
`CAMERA-READY` lines) and restore the `\markboth` running-title / supplement author line.
