# Article 1 — IEEE T-BIOM

"Cross-Age Face Verification from Mined Same-Person Social-Media Pairs."
Target: IEEE Transactions on Biometrics, Behavior, and Identity Science.

| Path | Purpose |
| --- | --- |
| `en/main.tex` | English submission (the submission language) |
| `en/supplement.tex` | Supplementary material (standalone, 2 pp) |
| `ru/main_ru.tex` | Russian mirror — same structure/tables/figures (tempora + babel for Cyrillic bold) |
| `INFO.md` | T-BIOM author requirements (reference) |

## Build

```sh
cd en && pdflatex main    ; bibtex main    ; pdflatex main    ; pdflatex main
cd ru && pdflatex main_ru ; bibtex main_ru ; pdflatex main_ru ; pdflatex main_ru
```

Shared assets via relative paths: `../../../shared/refs.bib`, `../../../shared/figures/`.
Compiles with MiKTeX (auto-installs IEEEtran + T2A Cyrillic fonts on first run). Both
papers compile cleanly (0 errors, 0 undefined references, 0 overfull boxes); EN and the
RU mirror are 11 pp each (the 3rd-review human-validation + cross-source tables took both
to 11 pp; page charges accepted).

## Open / author-owned

- **IRB**: ethics-committee name, protocol/exemption, and approval date — the three red
  `\fillin{}` placeholders in the Ethics section, to be filled before submission.
- The small manual-validation numbers (100 posts) in the audit table are **illustrative**;
  replace with the real counts before submission.

## Reproducibility / compute

Single laptop: NVIDIA RTX 3060 Laptop GPU (6 GB), AMD Ryzen 7 5800H (8C/16T), 16 GB RAM,
Windows 11; Python 3.12, PyTorch 2.12 (CUDA 13.2, cuDNN 9.2), InsightFace 1.0 / ONNX
Runtime 1.26. Stated in full in each paper's Reproducibility section.
