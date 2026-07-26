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

**2. Page count — resolved, now 10 pp.** Regular papers are limited to 10 pages. The manuscript was
11 pp, so four figures that merely re-plotted data from adjacent tables (headroom curve, data-scaling
law, objective comparison, fairness audit) were moved to the supplement, which has no page limit.
Their references now read "Table~X; plotted in the supplement". **No number was removed** — every
value those plots showed is still tabulated in the main paper. No Mandatory Overlength Page Charges
apply.

The Russian mirror in `../ru/` deliberately keeps all six figures inline: it is not a submission,
has no page limit, and has no supplement of its own to hold them.

**3. T-BIOM and AAAI-27 cannot be in review at the same time.** This is not a formatting question —
the two versions report the same work, and T-BIOM's rules bar parallel review twice over:

> "Every manuscript submitted to TBIOM must … Not be previously published or under review elsewhere."

> "Extensions of conference papers may be submitted to TBIOM, though **not whilst the conference
> version is in review**."

So the extension route does not create a loophole either. AAAI has its own dual-submission policy;
check the AAAI-27 CFP text directly before relying on anything here. Two legal orders exist:

- **T-BIOM first** (recommended): submit here, hold the AAAI version. The TNNLS AE called the
  contribution "a valuable dataset and data curation contribution" while faulting the "advance in
  neural network learning methods" — an AI-methods conference is likely to repeat exactly that.
- **AAAI first, then T-BIOM as an extension:** submit to AAAI, *wait for the decision*, then submit
  here citing the conference paper and explaining the extension. The 30%-new-material rule is met
  with room to spare: the journal version has 7396 words and 20 tables/figures against the
  conference version's 4019 and 9, i.e. ~46% of it is material the conference paper does not carry.
  Cost: the journal submission idles for months.

Either way the ethics blocker (item 1) applies — AAAI reviews ethics too, and non-consensual
biometric data on identifiable people is precisely the category that gets flagged there.

An earlier version was submitted to and **rejected** by another IEEE journal. A rejected manuscript
was never published, so the "Previously Published" slots do not apply; the cover letter discloses it
anyway.

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

Current state: `main.pdf` 10 pp, `supplement.pdf` 3 pp, `coi.pdf` 1 p, `cover_letter.pdf` 2 pp —
0 errors, 0 broken references.
