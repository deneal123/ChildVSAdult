# Article 1 — IEEE T-BIOM

"Cross-Age Face Verification from Mined Same-Person Social-Media Pairs."
Target: IEEE Transactions on Biometrics, Behavior, and Identity Science.

| Path | Purpose |
| --- | --- |
| `en/main.tex` | English submission (the submission language) |
| `en/supplement.tex` | Supplementary material (standalone) |
| `ru/main_ru.tex` | Russian mirror — same structure/tables/figures (tempora + babel for Cyrillic bold) |
| `INFO.md` | T-BIOM author requirements (reference) |

## Build

```sh
cd en && pdflatex main    ; bibtex main    ; pdflatex main    ; pdflatex main
cd ru && pdflatex main_ru ; bibtex main_ru ; pdflatex main_ru ; pdflatex main_ru
```

Shared assets via relative paths: `../../../shared/refs.bib`, `../../../shared/figures/`.
Compiles with MiKTeX (auto-installs IEEEtran + T2A Cyrillic fonts on first run). Validate the
English master after every scientific change; update the Russian mirror only after the English
content is frozen.

## Open / author-owned

- **Ethics gate**: obtain and record the committee name, decision, reference and date before
  resubmission. Experiments may proceed, but claims involving minors and social-media biometrics remain
  conditional on that decision.
- **Manual audit**: the earlier illustrative values have been removed from the English manuscript and
  supplement. The corrected blinded 5 x 400 pack reconstructs original caption/photo order and exposes
  only opaque copied-image names; GigaChat, regex and local-model predictions are frozen and the scorer
  exists. This does not restore a
  quantitative human-validation claim until both independent response files and adjudication are frozen.
- **Long experiments**: CACD-VS, three-seed MTLFace/CACon, the data funnel,
  independent-embedding sensitivity, the artifact-level pretraining-overlap audit and larger
  bidirectional cross-source runs are complete. The AdaFace IR-101 mechanism matrix remains in
  progress; `metrics/strong_backbone_study/summary.json` records expected, completed and missing runs.
- **Submission state**: the current text is a working revision, not a submittable package.

## Reproducibility / compute

Single laptop: NVIDIA RTX 3060 Laptop GPU (6 GB), AMD Ryzen 7 5800H (8C/16T), 16 GB RAM,
Windows 11; Python 3.12, PyTorch 2.12 (CUDA 13.2, cuDNN 9.2), InsightFace 1.0 / ONNX
Runtime 1.26. Stated in full in each paper's Reproducibility section.
