# T-BIOM submission — what goes into which slot

**Blockers first — do not upload until both are resolved.** See "Before you submit" below.

## Slot-by-slot

| Portal slot | Upload | Where it comes from |
| --- | --- | --- |
| **Main Manuscript** *(required, 1 file)* | a ZIP of the **contents** of `manuscript/` — so `main.tex` sits at the ZIP root, together with `refs.bib`, `main.bbl`, `IEEEtran.cls` and `figures/` | `submission/manuscript/` |
| **Conflict of Interest** *(required, 1 file)* | `coi.pdf` | `submission/coi.tex` |
| **Cover letter / Comments** *(nominally optional — for us **required**)* | `cover_letter.pdf` | `submission/cover_letter.tex` |
| **Supplementary Material for Review** *(optional)* | `supplement.pdf` | `submission/supplement/` |
| **LaTeX Supplementary File** *(optional)* | `supplement.tex` — only if the portal asks for the supplement source separately | `submission/supplement/` |
| Main Document — Tracked Changes | — | first submission, nothing to track |
| Image | — | figures are embedded in the manuscript |
| Previously Published — Statement / Files | — | not an extension of a **published** conference paper (see below) |

Notes that follow from T-BIOM's Information for Authors:

- The main manuscript **must not contain the supplement**: "All supplemental material must be
  submitted as separate files and must not be included within the same PDF file as the main paper."
  `manuscript/` is built without it.
- The cover letter is not shown to reviewers. That is where the dataset-availability justification
  belongs — T-BIOM requires it "in the submission **and in the cover letter**".
- **Double-anonymous is optional at T-BIOM**, and we are *not* using it: `en/main.tex` carries the
  real author block. Do not copy the anonymized block from the TNNLS bundle.
- Conflict of interest is declared in **two** places, and they are worded identically so they cannot
  contradict: the manuscript's *Funding and Conflicts of Interest* section and `coi.pdf`.

## Before you submit

**1. Ethics review — unresolved.** The submission form asks whether the research involved a
human-subjects dataset collected during the research (for us: yes) and, if so, whether approval was
obtained from a review board. Only two answers are accepted: approval (with institution, board name
and date, which must also appear in the manuscript) or a documented exemption (with the reasoning,
likewise in the manuscript). The manuscript currently states the opposite — *"No formal
ethics-committee protocol has been filed for this study to date"* (Sec. Ethical Approval and Legal
Basis). **Obtain an approval or a written exemption from the HSE University ethics board first**,
then update both that subsection and the placeholder paragraph in `cover_letter.tex`. This cannot be
resolved by rewording.

**2. Page count — 11 pp against a 10 pp limit.** Regular papers are up to 10 pages; pages beyond that
are subject to IEEE Mandatory Overlength Page Charges. This predates the July 2026 corrections (the
pre-correction source also compiled to 11 pp). Options: trim a subsection, move a table to the
supplement, or accept the MOPC. Decide before uploading.

**3. Confirm nothing is under review elsewhere.** The AAAI-27 version in `latex/conference/aaai-27/`
shares content with this manuscript. T-BIOM requires the work not be under review elsewhere, so the
AAAI version must not be in review while this submission is active. An earlier version was submitted
to and **rejected** by another IEEE journal — a rejected manuscript was never published, so the
"Previously Published" slots do not apply; the cover letter discloses this anyway.

## Rebuilding the bundle

The bundle is derived and git-ignored. Regenerate before every submission:

```bash
python latex/build_submission.py journal-1-tbiom
cd latex/papers/journal-1-tbiom/submission && pdflatex coi && pdflatex cover_letter
```

`build_submission.py` copies `en/main.tex` with the shared paths localized
(`\graphicspath{{./figures/}}`, `\bibliography{refs}`), pulls in `refs.bib`, the six figures and
`IEEEtran.cls`, strips LaTeX comments, compiles `pdflatex; bibtex; pdflatex; pdflatex`, and fails
loudly on compile errors or undefined/multiply-defined references.

Current state: `main.pdf` 11 pp, `supplement.pdf` 2 pp, `coi.pdf` 1 p, `cover_letter.pdf` 2 pp —
0 errors, 0 broken references.
