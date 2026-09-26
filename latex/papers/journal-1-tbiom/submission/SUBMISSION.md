# T-BIOM resubmission bundle

This directory describes a future **new submission** after the rejection of
`TBIOM-2026-07-0222`. The working manuscript is not submission-ready yet.

## Portal files

Run from the repository root:

```powershell
uv run python latex/build_submission.py journal-1-tbiom
```

The command compiles and validates the paper, then creates an ignored `upload/`
directory. Upload artifacts are deliberately separated:

| Portal slot | Artifact |
| --- | --- |
| Main manuscript PDF, when accepted by the portal | `upload/main.pdf` |
| Main manuscript LaTeX source | `upload/manuscript-source.zip` |
| Supplementary material for review | `upload/supplement.pdf` |
| Conflict of interest | compile `coi.tex` to `coi.pdf` |
| Cover letter | compile `cover_letter.tex` to `cover_letter.pdf` after all results are frozen |

`manuscript-source.zip` contains only `main.tex`, `main.bbl`, `refs.bib`,
`IEEEtran.cls`, referenced figures, and the aggregate
`artifacts/publication_artifact_index.json`. The index exposes commands,
parameters and checksums, but no images, embeddings or row-level corpus data.
The archive never contains the compiled PDF,
`.aux`, `.log`, `.blg`, or the supplement. `artifact-manifest.json` records the
size and SHA-256 of each portal-facing artifact. The build fails on LaTeX errors,
undefined references, or a textually blank PDF page.

## Mandatory gates before upload

1. Obtain and record the ethics committee's actual decision, official name,
   date, and reference number. The request must cover retrospective biometric
   processing, lack of consent, minors, manual annotation, controlled access,
   and publication of examples. Do not select a portal ethics answer in advance.
2. Apply the decision consistently. If minors or the retrospective source are
   not covered, remove the affected material and regenerate every result.
3. Finish CACD-VS, common-protocol MTLFace and CACon baselines, the three-seed
   strong-backbone study, and the strengthened cross-source validation. Every
   reported number must be generated from a machine-readable manifest.
4. Complete the independent annotation audit and controlled-access procedure.
   Do not restore the removed illustrative manual-validation values.
5. Complete `reviewer_response_matrix.md` with one row per editor/reviewer
   comment: comment, change, evidence artifact, and manuscript location.
6. Resolve the status of any simultaneous conference submission. A manuscript
   under review elsewhere must not be submitted to T-BIOM. If a related paper
   has been accepted or published, rewrite the originality/extension statement.
7. Rebuild from a clean checkout, inspect `artifact-manifest.json`, upload the
   source ZIP and supplement separately, download the portal-generated proof,
   and visually inspect all pages.

The English manuscript is the master. Synchronize the Russian mirror only after
the scientific content and numbers are frozen.
