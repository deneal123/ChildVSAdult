# AAAI-27 — Russian mirror (NOT for AAAI submission)

The AAAI-27 template (`aaai2027.sty`) **forbids Cyrillic in body text**: `babel` is on the
banned-package list, and the AuthorKit states non-Roman alphabets "must be restricted to
bit-mapped figures." A Russian-language paper therefore cannot be compiled in the AAAI
template, and the AAAI *submission* is English-only — see `../en/`.

This folder is reserved for a Russian rendering of the conference paper for the thesis /
defense (ВКР), built **outside** the AAAI template — e.g. with the same `tempora` + `babel`
setup as `papers/journal-1-tbiom/ru/main_ru.tex`. **Build deferred.**
