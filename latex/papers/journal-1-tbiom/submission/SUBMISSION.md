# T-BIOM submission — file routing (journal-1-tbiom, English)

**Unlike the TNNLS bundle, this one is NOT anonymized.** IEEE T-BIOM reviews with the author
list visible, and `../en/main.tex` accordingly carries the real `\author{...}` block. Do not
copy the TNNLS "Anonymized for Double-Blind Review" block here.

Comments are still stripped from the uploaded `.tex` (see step 2 below). That is not an
anonymity measure here — it just keeps the uploaded source free of working notes.

## Rebuilding the bundle

The bundle is **derived** and git-ignored. Regenerate it before every submission:

```bash
python latex/build_submission.py journal-1-tbiom
```

That script does all of the following, and fails loudly if anything breaks:

1. `manuscript/main.tex` = `en/main.tex` with the shared paths localized
   (`\graphicspath{{./figures/}}`, `\bibliography{refs}`), plus `shared/refs.bib`, the six
   `fig_{headroom,scaling,shortcut,sota,fairness,external}.pdf`, and `IEEEtran.cls`.
2. Comments stripped via `submission/strip_comments.py`.
3. `pdflatex; bibtex; pdflatex; pdflatex`, then a check for compile errors and for undefined /
   multiply-defined references.
4. `supplement/` built the same way from `en/supplement.tex`.

Current state: `main.pdf` 11 pp, `supplement.pdf` 2 pp, 0 errors, 0 broken references.

## What to upload where

| Portal slot | Upload | Source |
| --- | --- | --- |
| Main manuscript | a ZIP of the **contents** of `manuscript/` (so `main.tex` sits at the zip root), or `main.pdf` if the portal prefers a single review PDF | `submission/manuscript/` |
| Supplementary material | `supplement.pdf` (and `supplement.tex` if the source is requested) | `submission/supplement/` |
| Conflict of interest | nothing to declare — also stated in the manuscript's Funding and Conflicts of Interest section | — |

The main manuscript **excludes** the supplement: IEEE requires supplementary material to be
separate files, not part of the article PDF.

## Open issue before submitting

**Page count.** T-BIOM allows 10 pages for a regular paper; the manuscript is **11 pp**, so it
would incur IEEE Mandatory Overlength Page Charges. This predates the July 2026 corrections
(the pre-correction source also compiled to 11 pp) — it is a layout decision to make, not a
regression. Options: trim a subsection, move a table to the supplement, or accept the MOPC.
