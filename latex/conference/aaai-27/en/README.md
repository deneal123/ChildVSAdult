# AAAI-27 paper (English) — anonymous submission

Self-contained AAAI-27 build. The folder carries its **own** copies of `refs.bib`,
`figures/`, and the class files (`aaai2027.sty`, `aaai2027.bst`) because AAAI requires a
self-contained source archive (single `.tex` + `.bib` + class + graphics).

| File | Purpose |
| --- | --- |
| `main.tex` | The paper (7-page body + references), `\usepackage[submission]{aaai2027}` |
| `supplement.tex` | Supplementary technical appendix (overflow tables/figures, claims-vs-evidence, model/data card, reproducibility) |
| `ReproducibilityChecklist.tex` | Filled AAAI reproducibility checklist |
| `refs.bib` | Bibliography (copy of `../../../shared/refs.bib`) |
| `figures/` | The figure PDFs used by `main.tex` (4) and `supplement.tex` (2) |
| `aaai2027.sty`, `aaai2027.bst` | AAAI-27 class/style (copies of the AuthorKit) |

## Build

```sh
pdflatex --enable-installer main         ; bibtex main
pdflatex main ; pdflatex main
pdflatex supplement ; bibtex supplement ; pdflatex supplement ; pdflatex supplement
pdflatex ReproducibilityChecklist
```

`--enable-installer` lets MiKTeX fetch the AAAI fonts (`newtx`, `helvetic`, `courier`,
`fontaxes`) non-interactively on first run. Current status: `main` 7 pp, `supplement` 5 pp,
checklist 2 pp; **0 errors / 0 undefined / 0 overfull**, no Type-3 fonts, fully anonymized
(`Anonymous submission`, no names/affiliations/repo).

## Switch to camera-ready (after acceptance)

1. In `main.tex` **and** `supplement.tex`: change `\usepackage[submission]{aaai2027}` →
   `\usepackage{aaai2027}` (restores the mandatory copyright footer; un-hides authors).
2. Replace `\author{Anonymous Submission}` / empty `\affiliations{}` with the real author(s)
   and affiliation(s) (see the AuthorKit for the multi-affiliation macro).
3. Uncomment / add a `\begin{links} \link{Code}{...} \end{links}` block (between abstract and
   body) with the public repository.
4. In the **Ethical Statement**, replace the anonymized IRB sentence with the real
   committee name, protocol number and approval date.
5. Replace the **illustrative** 100-post manual-validation numbers (supplement `tab:humanval`)
   with the real counts.
6. Sign and return the AAAI copyright form.

## Submitting

- AAAI wants a single self-contained archive ≤ 10 MB: `main.tex` + `refs.bib` + `main.bbl` +
  `figures/` (only those used) + the class files + the compiled PDF. Name the source by the
  first author's family name.
- Submit `supplement.tex`/PDF and the reproducibility checklist as **separate** supplementary
  material (the main paper's 7-page limit excludes references and the supplement).
- **Clear PDF metadata** before submitting (anonymity): e.g. `exiftool -all:all= main.pdf`
  or `qpdf --linearize` after stripping `/Info` (neither tool is installed here — run it in
  your environment). The `[submission]` option already suppresses author metadata in-PDF.

## Regenerating the self-contained copies from `shared/`

```sh
cp ../../../shared/refs.bib refs.bib
cp ../../../shared/figures/fig_{scaling,headroom,shortcut,external,sota,fairness}.pdf figures/
# class files come from ../../../shared/vendor/aaai2027/AuthorKit27/{aaai2027.sty,aaai2027.bst}
```

## Page limit

The AuthorKit defers the page limit to the track CFP. This build targets the historical AAAI
main-technical-track budget (~7 body pages; references and the reproducibility checklist do
not count). Confirm the exact limit in the AAAI-27 CFP and adjust the body/supplement split if
needed.

## Author-owned blockers

- IRB committee / protocol / approval date (anonymized for review; real values for camera-ready).
- Real manual-validation counts (the 100-post numbers are illustrative).
