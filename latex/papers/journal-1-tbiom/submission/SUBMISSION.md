# T-BIOM submission — what goes into which slot

**DO NOT SUBMIT YET.** The AAAI-27 version (paper #7133) is currently **under review**, which bars a
T-BIOM submission outright — see item 3. The bundle is ready and waits for the AAAI decision.

## Slot-by-slot

| Portal slot | Upload | Where it comes from |
| --- | --- | --- |
| **Main Manuscript** *(required, 1 file)* | a ZIP of the **contents** of `manuscript/` — so `main.tex` sits at the ZIP root, together with `refs.bib`, `main.bbl`, `IEEEtran.cls` and `figures/` | `submission/manuscript/` |
| **Conflict of Interest** *(required, 1 file)* | `coi.pdf` | `submission/coi.tex` |
| **Cover letter / Comments** *(nominally optional — for us **required**)* | `cover_letter.pdf` | `submission/cover_letter.tex` |
| — *(not a portal slot)* | `ethics_request_hse.md` — file with the HSE ethics committee **before** submitting | `submission/ethics_request_hse.md` |
| — *(not a portal slot)* | `portal_answers.md` — ready-to-paste answers for the form's ethics/COI questions | `submission/portal_answers.md` |
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

**1. Ethics review — submittable now, but a date must be filled in.** IEEE policy (PSPB 8.1.1.B/E)
accepts, in place of an approval reference, *"an explanation as to why such a review was not
conducted"* — and likewise *"or explain why consent was not obtained"*. The manuscript now gives
that explanation affirmatively (no interaction or intervention; subjects' own public posts; nobody
identified, contacted, profiled or ranked; nothing deployed; minors excluded; no raw faces, posts or
recoverable identifiers released), states the counter-consideration rather than resolving it in our
favour (the images are biometric and the people remain identifiable), and says a formal
determination has been requested from the university ethics committee.

The old wording — *"No formal ethics-committee protocol has been filed ... the authors will obtain
[one] should a venue or reviewer require one"* — was the actual problem: it read as an unfulfilled
promise and contradicted whichever answer you give on the submission form. It is gone.

**What you must still do:** have the exemption request in `ethics_request_hse.md` filed with the HSE
committee (it asks for an *exemption determination*, not a full protocol — much faster) **before**
you press submit, because the manuscript states the request has been made. No date appears anywhere
in the manuscript or the cover letter, on purpose: a date in a published paper cannot be corrected
if the request is actually filed on a different day, so both documents instead promise the
committee's decision **together with its filing date**. Ready-to-paste answers for the portal form
are in `portal_answers.md`.

**2. Page count — resolved, now 10 pp.** Regular papers are limited to 10 pages. The manuscript was
11 pp, so four figures that merely re-plotted data from adjacent tables (headroom curve, data-scaling
law, objective comparison, fairness audit) were moved to the supplement, which has no page limit.
Their references now read "Table~X; plotted in the supplement". **No number was removed** — every
value those plots showed is still tabulated in the main paper. No Mandatory Overlength Page Charges
apply.

The Russian mirror in `../ru/` deliberately keeps all six figures inline: it is not a submission,
has no page limit, and has no supplement of its own to hold them.

**3. BLOCKING: the AAAI-27 version is under review right now.** Paper #7133 is in review at AAAI-27
(confirmed by the program chairs' reciprocal-reviewing notice of 26 July 2026, which also states the
paper will not be desk-rejected on nomination grounds). T-BIOM's rules bar a submission twice over
while that is true:

> "Every manuscript submitted to TBIOM must … Not be previously published or under review elsewhere."

> "Extensions of conference papers may be submitted to TBIOM, though **not whilst the conference
> version is in review**."

So the extension route is not a loophole either. **The order is now decided by fact, not preference:
AAAI first.** What follows:

- **Wait for the AAAI-27 decision.** Do not submit here in the meantime, and do not describe the
  T-BIOM version as "not under review elsewhere" until that is true.
- **If AAAI accepts:** this becomes an *extension of a published conference paper*. Then the rules
  change — the manuscript must cite the AAAI paper, explain the extension, and carry ≥30% new
  material. We clear that comfortably: 7396 words and 20 tables/figures here against 4019 and 9
  there, i.e. ~46% of this version is material the conference paper does not carry (13 tables exist
  only here). The cover letter's originality paragraph and the "Previously Published" slots must
  both be rewritten at that point — they currently say the opposite.
- **If AAAI rejects:** submit here as-is; a rejected manuscript was never published, so the
  originality paragraph stands unchanged.

Either way the ethics item (1) still applies — AAAI reviews ethics too.

An earlier version was also submitted to and **rejected** by another IEEE journal (TNNLS). That one
is settled: rejected ≠ published, and the cover letter discloses it.

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

Current state: `main.pdf` 10 pp, `supplement.pdf` 4 pp, `coi.pdf` 1 p, `cover_letter.pdf` 2 pp —
0 errors, 0 broken references.
