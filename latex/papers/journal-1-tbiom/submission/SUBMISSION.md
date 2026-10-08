# T-BIOM resubmission bundle

This directory describes a future **new submission** after the rejection of
`TBIOM-2026-07-0222`. The working manuscript is not submission-ready yet.

## Portal files

Run from the repository root:

```powershell
uv run python latex/build_submission.py journal-1-tbiom
```

To verify current sources without overwriting another live build destination:

```powershell
uv run python latex/build_submission.py journal-1-tbiom --submission-dir .work/submission-check
```

Use a fresh task-specific destination. This option is incompatible with `--all`;
it changes the output directory only, not the canonical source or data/experiment inputs.

After a completed isolated build, audit the local current-source/package snapshot:

```powershell
uv run python -m scripts.audit_submission_snapshot --submission-dir .work/submission-check --out metrics/submission-check-audit
```

The audit requires a fresh output directory and fails on source/ZIP/hash/page mismatches,
extra upload members or missing/timeout extractors. It re-derives the exact localized,
comment-stripped LF source bytes, checks canonical refs/class/literal PDF figures and
the archived index against the current local index. Evidence member certificates and
projected aggregate/original-manifest bindings are checked. It does not attest the
compiler execution, reverify all private experiment inputs, certify figure privacy,
perform a visual review or replace the portal proof. Repeat after any source change.

The command compiles and validates the paper, then creates an ignored `upload/`
directory. Upload artifacts are deliberately separated:

| Portal slot | Artifact |
| --- | --- |
| Main manuscript PDF, when accepted by the portal | `upload/main.pdf` |
| Main manuscript LaTeX source | `upload/manuscript-source.zip` |
| Supplementary material for review | `upload/supplement.pdf` |
| Supplementary data/code slot, if supported | `upload/publication-evidence.zip` |
| Additional supplementary data/code slot, if supported | `upload/lfw-evidence.zip` |
| Additional supplementary data/code slot, if supported | `upload/curation-evidence.zip` |
| Additional supplementary data/code slot, if supported | `upload/roc-v2-evidence.zip` |
| Additional supplementary data/code slot, if supported | `upload/cacd-evidence.zip` |
| Conflict of interest | compile `coi.tex` to `coi.pdf` |
| Cover letter | compile `cover_letter.tex` to `cover_letter.pdf` after all results are frozen |

`manuscript-source.zip` contains only `main.tex`, `main.bbl`, `refs.bib`,
`IEEEtran.cls`, referenced figures, and the aggregate
`artifacts/publication_artifact_index.json`. The index exposes commands,
parameters and checksums, but no images, embeddings or row-level corpus data.

`cacd-evidence.zip` is the separate seven-member source-bound serial CACD-VS aggregate
package (not part of the historical ROC-v2 ZIP). It includes the generated table and
main-summary TeX bytes, count-only native manifest projections and a certificate.
Its intervals resample pairs, not subjects or the training-seed population. Accuracy,
EER and TAR at FAR 1% worsen despite nearly unchanged AUC. Original alignment replay,
checkpoint training provenance and benchmark identity independence remain unverified.
The local snapshot audit checks this archive against its native sources, not only its
self-reported checksums. Availability of another portal slot and disclosure/access
approval remain required; exporting it locally is not permission to distribute it.
The archive never contains the compiled PDF,
`.aux`, `.log`, `.blg`, or the supplement. `artifact-manifest.json` records the
size and SHA-256 of each portal-facing artifact. The build fails on LaTeX errors,
undefined references, or a textually blank PDF page.

`publication-evidence.zip` contains eight explicitly allowlisted aggregate results
(corrected FG-NET endpoint, internal endpoint, comparators, retrieval and error
breakdowns), sanitized manifest projections, `README.txt` and `CERTIFICATE.json`.
The build verifies every declared direct original input/output locally and refuses
missing or mismatched artifacts. Private records and checkpoint/biometric input
digests are omitted from the projections; verification summaries contain counts
only. Original and exported checksums are distinguished, and every ZIP member is
verified against an exact membership/hash map. Building therefore requires local
controlled-access inputs even though the archive does not redistribute them.
This is partial aggregate evidence, not full experiment reproducibility or
publication clearance. Remaining manuscript artifacts and access procedures must
still be completed. Upload this ZIP separately, never inside the source/PDF item;
confirm the portal supports it and that reviewers can access it.

`lfw-evidence.zip` separately contains exactly six members: `results/lfw.json`,
`tables/lfw.tex`, two sanitized evaluation/presentation manifests, `README.txt`,
and `CERTIFICATE.json`. A strict scalar/schema allowlist preserves aggregate
official-fold metrics and conditional subject intervals, but omits checkpoint
names/digests, private record paths/digests and arbitrary original prose. All
declared direct inputs/outputs are checked locally before and after packaging;
the generated table must match the verified result after newline normalization.
The certificate distinguishes original and exported result/table checksums and
binds each member. The previous upload is preserved if any evidence exporter
fails preflight. These archives do not establish training independence, full
reproducibility or publication readiness. Confirm that the portal accepts the
separate archives and exposes them to reviewers; do not merge them into a PDF
or source upload item.

`curation-evidence.zip` is a separate eight-member local package candidate: three
schema-allowlisted aggregate results (lineage, global-ID shadow reconstruction,
constituent-caption sensitivity), three count-only sanitized manifests, README and
certificate. All declared direct inputs/outputs are verified locally, but no input
record paths/digests, commands, parameters, private candidates or captions are
redistributed. Exact recursive schemas reject unknown fields, vectors and changed
scope claims; member hashes and original-versus-exported aggregate hashes are
checked independently. The reconstruction does not recover historical provenance;
automatic categories are not human gold, and unbalanced count filters are not
retrained experimental arms. Marker/schema checks do not certify privacy: small-cell
disclosure and access review remain mandatory before release. Confirm the separate
portal slot and reviewer access only after the publication gates are cleared.

`roc-v2-evidence.zip` is a separate thirteen-member candidate: three corrected ROC
aggregates (global FG-NET, internal sensitivity, LFW compatibility), three generated
tables, five count-only sanitized manifests, README and certificate. Original and
exported hashes are distinct; parse/hash binding uses the same buffered bytes.
The snapshot audit compares projections and tables to current native results, not
only to hashes supplied inside the ZIP. Export failure preserves prior uploads.
Private scores, images, captions, embeddings and person IDs are not redistributed.
Fixed-checkpoint uncertainty is not a training-seed population interval; test ROC
thresholds are not deployment calibration. Both EER definitions remain labelled.
This is still partial evidence, not ethics/privacy clearance or full reproduction.
The experiment index includes the five corresponding ROC-v2 manifests but remains
incomplete until missing evidence and unfinished campaigns are resolved.

The index builder also discovers fixed8 native CUDA training/evaluation manifests,
including nested train/FG-NET outputs of the sequential queue, representation/age and
common-index diagnostics, recorded trajectories, cache-key checks, matched-denominator
constants, and per-checkpoint ROC presentations. These discovery rules are not a freshly
regenerated index or uploaded package. Queue ledgers are orchestration only and are not
indexed as scientific completion evidence; private checkpoint/cache/prediction records
remain withheld. Index integrity does not establish full-matrix mechanism, independent
human identity clearance, ethics approval, or publication readiness.

## Mandatory gates before upload

1. Obtain and record the ethics committee's actual decision, official name,
   date, and reference number. The request must cover retrospective biometric
   processing, lack of consent, minors, manual annotation, controlled access,
   and publication of examples. Do not select a portal ethics answer in advance.
2. Apply the decision consistently. If minors or the retrospective source are
   not covered, remove the affected material and regenerate every result.
3. Finish the three-seed strong-backbone study, two complete common-budget SOTA
   reproductions, and remaining matched-source, uncertainty and leakage gates.
   CACD-VS and three-seed cross-platform transfer are available with stated limits.
   Existing MTLFace/CACon-inspired common-protocol adaptations do not satisfy the
   full-method reproduction requirement. Finalize bounded interpretations and
   regenerate after confirmed exclusions. Every reported number must be generated
   from a machine-readable manifest.
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
