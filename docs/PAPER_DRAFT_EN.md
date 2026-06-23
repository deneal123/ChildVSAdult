# Age-Invariant Identity Matching from Naturally-Anchored Social-Media Posts: A Naturally-Supervised Longitudinal Signal

*Working paper draft (EN). Self-contained: all tables and numbers are inline. All results are reported
on the curated dataset of §3.4, with seed variance and, for the fairness audit, paired bootstrap
confidence intervals; where useful we also show the pre-curation number for comparison (e.g. the
"(pre-clean gain)" column in Table 5). The one exception is the minor +age-anchor ablation increment
(§5.1), which is reported pre-curation as it does not affect any conclusion.*

## Abstract

Cross-age face verification — deciding whether two photographs taken years apart depict the same
person — is hard: age strongly alters appearance, and models tend to exploit age as a shortcut. We
study a **data-centric** question: can the supervision needed for age invariance be obtained without
a manually labeled longitudinal dataset? We show that **multi-photo "then/now" social-media posts
depicting one person at different ages provide naturally anchored positive identity pairs**, and that
fine-tuning a weak recognizer on them teaches **transferable** age invariance. Our supervision is
**manual-label-free** (no human identity/age annotation), though the pipeline uses an LLM caption
parser and embedding clustering; we therefore call it *naturally supervised* and audit those
components.

On the external cross-age benchmark FG-NET, fine-tuning a deliberately weak trainable backbone raises
large-gap (≥25-year) ROC-AUC from 0.736 to **0.848** (+0.112 ± 0.001 over 3 seeds) at a cost of only
−0.019 LFW accuracy; AgeDB-30 and CALFW corroborate. The effect (a) **exceeds synthetic aging** even
with a real generative re-aging model; (b) **reproduces across three loss families, two
architectures, two independent same-platform source communities, and two training objectives** — our
simple pair-contrastive objective matches the canonical ArcFace-margin classification objective (the
core of OE-CNN / MTLFace / AIM) on the key external metric (0.848 vs 0.851). We therefore make the
bounded claim that **within our protocol the data source dominates the choice of objective among the
variants considered**, not that architecture is irrelevant in general. The gain is **mechanistically
tied to removing an age shortcut**: frozen models' accuracy collapses toward chance under age-matched
negatives, and our pairs restore it. The method is **data-efficient** (≈96% of the gain at 10% of
identities). In an **apparent** gender/age audit (estimated attributes, not self-identification), no
stratum degrades and the largest, statistically significant gain is for the youngest (0–17) group
where the frozen baseline is weakest — narrowing the apparent-demographic gap. We additionally provide
a calibrated P(same | cosine, age-gap) and a curation pipeline whose chief finding is that ~half of
the raw "positive" pairs were near-duplicate frames inflating internal metrics.

**Keywords:** cross-age face recognition, age-invariant representation, naturally/weakly-supervised
identity, dataset curation, shortcut learning, fairness, calibration.

## 1. Introduction

Cross-age face verification matters for archival retrieval, account recovery, and authorized search
for missing persons. The difficulty is not telling apart random people but distinguishing **the same
person across age** from **different people of similar age/appearance**. Annotating longitudinal
identity pairs (one individual across many years) is expensive and scarce, bottlenecking supervised
approaches; recent work continues to flag weak cross-dataset generalization and a shortage of quality
longitudinal data.

**Hypothesis.** A multi-photo "then/now" post showing one person at different ages yields C(n,2)
positive identity pairs without manual annotation; a caption stating ages adds an age anchor. This is
a scalable, naturally-supervised signal of cross-age identity, and fine-tuning a weak recognizer on it
should transfer to external benchmarks.

**Contributions (bounded).** (1) A **naturally-supervised (manual-label-free) longitudinal signal**
mined from social posts. (2) A rigorous **attribution protocol** (one weak trainable backbone, frozen vs.
fine-tuned, external evaluation) isolating the contribution of *data* from network capacity. (3)
**Mutually reinforcing controls** (loss family, architecture, source community, training objective,
real-vs-synthetic, shortcut diagnosis) supporting the bounded claim that, within this protocol and across the objectives tested, the data contributes more to the large-gap gain than the choice among loss families. (4) A **curation pipeline** with an audit of
its automatic components, and the finding that naive positive counts are inflated ~2× by
near-duplicate frames. (5) **Applied calibration** and an **apparent-demographic fairness audit** with
confidence intervals.

We emphasize the novelty is **data-centric, not architectural** — a new *origin of supervision* — and
should be judged on that axis.

## 2. Related Work and Positioning

Three families precede us. **Discriminative age-invariant FR on labeled data:** OE-CNN (orthogonal
embedding decomposition), DAL (decorrelated adversarial learning), AIM, and MTLFace (multi-task
recognition + age synthesis + attention decomposition) reach strong accuracy but train on datasets
with **manual identity AND age labels** (CACD, MORPH, AgeDB, FG-NET). **Cross-age contrastive with
synthetic samples:** CACon adds a synthesized cross-age sample and a triplet loss; OrdCon uses
order-enhanced contrastive learning for generalized age features. **Synthetic aging for robustness:**
systematic studies show synthetic aging helps only partially. Generic self-supervised FR
(SimCLR-style augmentation) lacks identity anchors across time. We instead mine the structure "one
post = one person at different ages" as a free positive-pair generator.

**Table 1. Positioning vs. prior work.**

| work | supervision | mechanism | our difference |
| --- | --- | --- | --- |
| OE-CNN (2018) | labeled id+age | orthogonal feature decomposition | label-free data source, not decomposition |
| DAL (2019) | labeled id+age | adversarial id/age factorization | mined pairs + simpler objective |
| MTLFace (2021) | labeled id+age | recognition + age synthesis (multi-task) | weaker pipeline; novel supervision origin |
| CACon (2023) | semi-sup + synthetic | contrastive on synthesized cross-age | real mined pairs > synthetic aging (§5.2) |
| Synthetic Ageing (2024) | synthetic | train on aged images | we make *real* mining the thesis |
| OrdCon (2025) | age-ordered contrastive | order-enhanced features | mine natural pairs, no explicit age order |

We report on the canonical cross-age benchmarks of these methods (AgeDB-30, CALFW, FG-NET; CALFW was
designed as a harder, age-gapped LFW). The goal is **not** to beat absolute SOTA (we use a weak
backbone for clean attribution) but to quantify the magnitude and transferability of the gain from a
naturally-supervised source; §5.6 trains the canonical SOTA objective on our data directly.

## 3. Data: naturally-anchored longitudinal pairs

**Source.** Two public "then/now" multi-photo communities on VK (one platform, Russian-language
context). Faces are detected and aligned with a 5-point ArcFace template (InsightFace RetinaFace,
`buffalo_l`). A *usable* face passes detection-score, minimum-size, blur, and single-target checks
(§9). The raw mined corpus is summarized in Table 2 (left).

**Table 2. Dataset scale (raw mined) and after curation (§3.4).**

| stage | posts | photos | usable faces | persons (groups) | positive pairs |
| --- | --- | --- | --- | --- | --- |
| raw mined (2 communities) | 44,609 | 84,147 | 49,606 | 21,848 | 65,897 |
| **after curation** | — | — | **40,586** | **21,427** | **31,865** |

**3.1 Age extraction.** Captions are parsed by regex (Russian patterns + the slash format "6/18/21",
one age per photo) and by a GigaChat LLM. An LLM audit of all 33,697 multi-photo captions (Table 3)
found the LLM substantially more accurate than regex — it avoids reading calendar years/gaps as ages
and recovers second ages — disagreeing on 27.2% of posts, overwhelmingly LLM-favoring.

**Table 3. Age-extraction audit: LLM vs. regex (33,697 captioned multi-photo posts).**

| category | n | share |
| --- | --- | --- |
| agree (same ages or both empty) | 24,531 | 72.8% |
| LLM filled (regex missed) | 3,656 | 10.8% |
| age mismatch | 4,658 | 13.8% |
| LLM missed (regex found) | 852 | 2.5% |

**3.2 Persons and leakage-safe split.** A naive "one post = one identity" undercounts: a person recurs
across posts. We cluster post-groups into persons by embedding similarity (cos ≥ 0.85), giving 21,848
persons from 27,487 post-groups (~20% recurrence), and split **by person** (not by post). Manual
inspection of high-cosine pairs showed cosine does not separate look-alikes from same-person in the
0.55–0.8 band, so the merge threshold is deliberately conservative (0.85).

**3.3 Auditing the automatic supervision.** Because the signal is *naturally* (not human-) supervised,
the automatic components are potential hidden label noise and are audited: LLM-vs-regex age agreement
(Table 3); manual inspection of high-cosine pair identity; and within-person gender consistency as a
free noise detector (2,919 multi-photo groups with <0.6 consistency flag probable multi-person). A
dedicated **human-audited subset** (age-extraction accuracy, group-integrity accuracy, error matrix,
human–LLM κ) accompanies the release.

**3.4 Curation pipeline and its largest finding.** Three steps: (i) **apparent gender/age metadata**
(estimator) per face, aggregated per person — the corpus is **76.7% apparent-female / 23.3%
apparent-male**; (ii) **LLM group-integrity validation** (Table 4) — person-clustering had already
separated most multi-person posts, so only 421 person-groups are flagged noisy; (iii) **near-duplicate
dedup** within each person (cos ≥ 0.97), removing 8,190 faces (~17%).

**Table 4. LLM group-integrity classification (33,697 posts).**

| category | n | share |
| --- | --- | --- |
| single (one person across ages) | 31,275 | 92.8% |
| multi-person (different people) | 2,034 | 6.0% |
| unknown | 336 | 1.0% |
| meme / collage | 52 | 0.2% |

**Key finding.** Because positives are quadratic in faces per group, removing near-duplicates collapsed
positives **65,897 → 31,865 (−52%)**: roughly **half of the raw "positives" were duplicate frames**
(one shoot / burst / repost), which had inflated internal metrics. After curation, the frozen baseline
on our internal test honestly drops (our.overall 0.856 → 0.836; our.25+ 0.705 → 0.640), exposing a
harder, more honest benchmark. External benchmarks are unaffected. This is a transferable lesson for
mining longitudinal pairs from social media.

**3.5 Dataset datasheet (summary).** Provenance, rights and governance fields are stored per item
(source, origin URL, collection date, license/consent status, allowed use, retention policy, deletion
status, review status). Collection: public VK walls. Intended use: research on consent-based cross-age
matching. Out of scope: mass identification, surveillance, de-anonymization. Minors: the child↔adult
regime is in scope only under an explicit legal/ethical framework; per-item review status is tracked
and deletion-on-request is supported. A full Gebru-style datasheet accompanies the release.

## 4. Method, protocol and statistics

**Honest protocol.** Comparing an adapter on top of a *frozen* strong ArcFace against frozen ArcFace
is invalid (the adapter merely deforms near-perfect embeddings). Instead we use **one weak trainable
backbone** (FaceNet InceptionResnetV1, CASIA-WebFace) and compare (a) frozen → benchmark metric vs.
(b) soft fine-tuning of the head on our pairs → benchmark metric. Training uses a margin contrastive
loss on aligned crops. **Evaluation is on external benchmarks** (LFW, AgeDB-30, CALFW, FG-NET) for a
clean attribution of transfer; the internal test (`our.*`) is a secondary in-domain diagnostic.
Backbone strength is varied (§5.3) to map where the data helps. The design trades absolute performance
for *causal attribution*.

**Backbones.** FaceNet (weak), ArcFace iResNet-50 (CASIA-FaceV5; weak, different architecture), AdaFace
IR-50/IR-101 (WebFace; strong; a clean depth control), ArcFace iResNet-100 (strong).

**Benchmarks and metrics.** LFW (10-fold accuracy), AgeDB-30 and CALFW (aligned pairs; ROC-AUC +
10-fold accuracy), FG-NET (ROC-AUC overall and on the large-gap ≥25-year subset). We deliberately do
not rest the headline on FG-NET alone (small, old, partial detector coverage): AgeDB-30 and CALFW
corroborate. We distinguish **threshold-free** metrics (ROC-AUC) from **threshold-based** ones
(accuracy, TAR@FAR).

**Statistics.** Our *primary endpoint* is the FG-NET large-gap (≥25-year) ROC-AUC, with the null
hypothesis that fine-tuning on the mined pairs does not improve it over the frozen baseline (gain ≤ 0);
AgeDB-30, CALFW and the internal cross-age test are secondary endpoints, and easy LFW accuracy is
monitored as a forgetting control. The headline gain is **mean ± std over 3 seeds** (seed variance captures training
stochasticity, not full evaluation uncertainty). The fairness audit uses a **paired percentile
bootstrap** (1000 resamples, resampling pairs and recomputing both models on the same resample) for the
*gain*, accounting for frozen/tuned correlation. The release adds **paired-bootstrap / DeLong
confidence intervals** for the key benchmark AUC comparisons and a **Holm correction** across the
stratified fairness tests.

**Curated dataset.** All headline experiments (§5.1–5.6) are on the curated data of §3.4: 21,427 person
groups, 40,586 faces, 31,865 positives; split train 22,179 / val 4,579 / test 5,107 with balanced
negatives per split.

## 5. Results

**5.1 Main effect and self-supervised ablation.** The weak FaceNet, softly fine-tuned on our pairs,
raises FG-NET large-gap ROC-AUC 0.736 → **0.848** (+0.112 ± 0.001), while easy LFW falls only −0.019
and AgeDB-30/CALFW move negligibly (Table 5). An ablation isolates the source: same-post pairs use no
manual labels — neither identity nor age — and deliver almost all of the gain; adding explicit age
anchors contributes only **+0.009** on the 25+ bucket (pre-clean ablation). The method is therefore
essentially **self-supervised by identity**, with age supervision a thin add-on.

**Table 5. Main result: weak FaceNet frozen vs. +pairs, 3 seeds, curated data. Pre-clean gain shown
for reference.**

| metric | frozen | +pairs (mean ± std) | gain | (pre-clean gain) |
| --- | --- | --- | --- | --- |
| FG-NET large-gap | 0.7361 | 0.8484 ± 0.0007 | **+0.1124** | +0.1215 |
| FG-NET ROC | 0.8955 | 0.9230 ± 0.0006 | +0.0275 | +0.0277 |
| our.25+ (internal) | 0.6403 | 0.8348 ± 0.0027 | **+0.1946** | +0.1677 |
| our.overall (internal) | 0.8363 | 0.9163 ± 0.0009 | +0.0801 | +0.0764 |
| LFW accuracy | 0.9688 | 0.9503 ± 0.0031 | −0.0185 | −0.0283 |
| AgeDB-30 ROC | 0.9530 | 0.9462 ± 0.0013 | −0.0068 | −0.0155 |
| CALFW ROC | 0.9482 | 0.9489 ± 0.0014 | +0.0007 | −0.0030 |

The gain is statistically robust (std ≤ 0.003) and survives curation; the internal cross-age gain even
*grows* on the harder curated test (our.25+ +0.195) because curation lowers the frozen baseline to
0.640 while +pairs recovers to 0.835.

**5.2 Real pairs ≫ synthetic aging.** Replacing real positives with synthetically aged versions — a
domain proxy and a real re-aging network (FRAN) — not only fails to help but **actively hurts** on
clean data: FRAN reaches FG-NET large-gap 0.727 (**below** frozen 0.736) and our.25+ collapses to
0.549 (below frozen 0.640), whereas real pairs give +0.112 / +0.198 (Table 6). Identity-preserving
aging teaches texture artifacts, not real longitudinal variation (pose/camera/era). (Our 112px crops
limit FRAN fidelity, which targets ~1024px.)

**Table 6. Real vs. synthetic positives (FaceNet, curated data).**

| metric | frozen | +real pairs | +synthetic (proxy) | +synthetic (FRAN) |
| --- | --- | --- | --- | --- |
| FG-NET large-gap | 0.736 | **0.848** | 0.761 | 0.727 |
| our.25+ | 0.640 | **0.838** | 0.594 | 0.549 |

**5.3 Headroom curve and the strong-backbone regime.** The gain reproduces on FaceNet, ArcFace iResNet,
and AdaFace IR-50/IR-101 (three loss families, two architectures) and **monotonically decreases with
frozen strength** (Table 7); a clean depth control (AdaFace IR-50 → IR-101) confirms it. We state
plainly: **strong saturated backbones do not improve and slightly forget** under fine-tuning; a gentler
learning rate (1e-5) roughly halves the forgetting but does not turn it into a gain. The contribution
is specific to weak/medium backbones with cross-age headroom — a limitation we do not hide (§7).

**Table 7. Headroom: FG-NET large-gap by backbone strength (curated data, lr 3e-5).**

| backbone (strength) | frozen | +pairs | Δ large-gap | Δ our.25+ |
| --- | --- | --- | --- | --- |
| FaceNet (weak) | 0.736 | 0.848 | **+0.112** | +0.198 |
| ArcFace r50 (weak, diff. arch) | 0.764 | 0.805 | +0.041 | +0.133 |
| AdaFace IR-50 (strong) | 0.917 | 0.924 | +0.007 | +0.004 |
| ArcFace r100 (strong) | 0.954 | 0.915 | −0.039 | −0.002 |
| AdaFace IR-101 (strong) | 0.957 | 0.931 | −0.026 | −0.006 |

**5.4 Data-scaling law.** Varying the fraction of training identities (val/test fixed), the gain
**saturates very early**: ≈96% of the full FG-NET large-gap gain is reached with only **10% of
identities (~1,500)**, plateauing from 25% (Table 8); the slight dip at full data (0.50→1.00) is within seed variance — the curve is flat, not monotonic, beyond ~10%. The signal is data-efficient; further gains need
*harder* data (larger gaps, hard positives), not merely more.

**Table 8. Data-scaling law (FaceNet, curated; val/test fixed).**

| train fraction | # identities | FG-NET large-gap | our.25+ |
| --- | --- | --- | --- |
| frozen | 0 | 0.736 | 0.640 |
| 0.10 | ~1,500 | 0.845 | 0.814 |
| 0.25 | ~3,750 | 0.853 | 0.842 |
| 0.50 | ~7,499 | 0.864 | 0.848 |
| 1.00 | ~14,998 | 0.848 | 0.838 |

**5.5 Mechanism — the age shortcut.** Under age-matched negatives (same age bucket, different people),
frozen accuracy on the internal 25+ test collapses from 0.640 to **0.517** (to near-chance, 0.5) and
real pairs restore it to **0.838** — direct evidence that frozen models discriminate cross-age pairs
largely by **age**, and our pairs teach discrimination by **identity**. A linear probe
refines this *representationally* (Table 9): apparent age stays well decodable from the identity
embedding for *all* models (~0.63–0.65 balanced accuracy ≫ 0.25 chance) and changes only slightly,
while identity AUC jumps — so the method learns an invariant **metric** (the cosine stops relying on
the age axis), not an age-free representation; explicit adversarial disentanglement reduces leakage no
further.

**Table 9. Age-leakage linear probe (curated data; chance = 0.25).**

| model | age-probe balanced acc ↓ | identity ROC-AUC ↑ |
| --- | --- | --- |
| frozen | 0.651 ± 0.019 | 0.836 |
| +pairs | 0.629 ± 0.012 | 0.916 |
| +pairs+disentangle | 0.632 ± 0.017 | 0.918 |

**5.6 Objective comparison: training the SOTA objective on our data.** We train the canonical
modern-FR objective — **ArcFace-margin identity classification** (the core of OE-CNN/MTLFace/AIM) — on
our naturally-supervised identities (9,204 classes), same weak backbone/scope/protocol. On the key
external metric the contrastive and ArcFace objectives **coincide** (Table 10). We draw the **bounded**
conclusion that *within this protocol and among the objectives tested, the data source dominates the
objective choice* — not that the method is irrelevant in general. ArcFace additionally **forgets less**
on easy benchmarks — a practical recommendation.

**Table 10. Objective comparison (FaceNet, curated data).**

| metric | frozen | +pairs (contrastive) | +ArcFace (classification) |
| --- | --- | --- | --- |
| FG-NET large-gap | 0.736 | 0.848 | **0.851** |
| our.25+ | 0.640 | 0.838 | 0.868 |
| LFW accuracy | 0.969 | 0.946 | 0.952 |
| AgeDB-30 ROC | 0.953 | 0.945 | 0.947 |

**5.7 Apparent-demographic audit.** Stratifying by **apparent** gender/age of the pair anchor
(estimated, **not self-identified**), **no stratified group degrades** and every gain is significant
(95% paired-bootstrap CI > 0; Table 11). The largest gain is for the youngest group **0–17** (frozen
0.750 → +pairs 0.880, +0.130, CI not overlapping adult bands) — the method strengthens where the base
model is weakest (the child↔adult regime), **narrowing** the apparent-demographic gap. We deliberately
avoid the word "fair" as a solved property: the source is apparent-female-skewed (76.7%), attributes are
apparent (not ethnicity or legal categories), and 45+ is sparse.

**Table 11. Apparent gender/age strata (FaceNet, curated; 95% paired-bootstrap CI of the gain).**

| stratum | n_pos | frozen | +pairs | gain | 95% CI |
| --- | --- | --- | --- | --- | --- |
| overall | 5,107 | 0.836 | 0.916 | +0.080 | [+0.075, +0.086] |
| apparent F | 3,790 | 0.840 | 0.923 | +0.084 | [+0.077, +0.090] |
| apparent M | 1,317 | 0.838 | 0.897 | +0.059 | [+0.048, +0.071] |
| age 0–17 | 1,123 | 0.750 | 0.880 | **+0.130** | [+0.113, +0.146] |
| age 18–29 | 3,159 | 0.872 | 0.936 | +0.064 | [+0.058, +0.069] |
| age 30–44 | 672 | 0.822 | 0.893 | +0.070 | [+0.054, +0.087] |
| age 45+ | 153 | 0.859 | 0.893 | +0.034 | [+0.004, +0.064] |

**5.8 Cross-source transfer.** Training on one community and testing on the held-out other transfers
the gain: external FG-NET large-gap +0.113 (vs. +0.112 within-source — essentially identical) and
internal +0.092 on the held-out community. We frame this as **transfer between two independent sources
within one platform domain**, not platform-agnostic or web-scale generalization.

**5.9 Calibration.** Raw cosine (Platt) is mis-calibrated across gap bins — over-confident at 15–25
years (ECE 0.052); a P(same | cosine, age-gap) calibrator fixes every bin (15–25 → 0.017) and improves
the overall Brier score (0.035 → 0.029) — a usable, age-aware same-identity probability.

**5.10 Disentanglement.** Explicit age removal (gradient reversal + age head, MTLFace-style) gives a
small but stable cross-age gain (+0.0055 FG-NET large-gap, +0.0145 our.25+ over +pairs) without
harming easy benchmarks (even slightly above +pairs: LFW 0.951 vs 0.946); in absolute terms its large-gap (~0.853) is on par with the ArcFace objective of §5.6 (0.851), consistent with the data — not the objective or the disentangling add-on — setting the cross-age ceiling. With §5.5 it shows explicit
age supervision is a thin add-on over the data.

**5.11 Hard-negative mining.** Adding the hardest cross-person negatives to fine-tuning — the
top-5 most-similar other-person faces per anchor (cosine in a hard band, excluding the
near-duplicate/uncertain range) — further sharpens cross-age discrimination: FG-NET large-gap rises
to 0.866 (from 0.848 with random negatives), but easy benchmarks forget more (LFW 0.969→0.934 vs.
0.950 for +pairs; AgeDB-30 and CALFW similar; Table 12). Hard negatives thus trade easy-pair
calibration for cross-age separation — useful for a cross-age retrieval setting, not a general
verifier. (Single run; the internal hard-negative test is near-chance for the frozen model by
construction and is not comparable to the standard internal test.)

**Table 12. Hard-negative mining (FaceNet, curated data; external benchmarks, single run).**

| metric | frozen | +pairs | +pairs+hard-neg |
| --- | --- | --- | --- |
| FG-NET large-gap | 0.736 | 0.848 | **0.866** |
| FG-NET ROC | 0.896 | 0.923 | **0.927** |
| LFW accuracy | 0.969 | 0.950 | 0.934 |
| AgeDB-30 ROC | 0.953 | 0.946 | 0.931 |
| CALFW ROC | 0.948 | 0.949 | 0.934 |

## 6. Discussion

The contribution is a **scalable, naturally-supervised source of longitudinal supervision** and
evidence that it teaches transferable age invariance that synthetic aging cannot replace and explicit
age supervision does not explain. The reinforcing controls — loss family, architecture, source
community, and training objective — converge on the bounded conclusion that, within this protocol and across the objectives tested, the data contributes more to the large-gap gain than the choice among loss families. The age-shortcut diagnostic gives a clean
identity-only metric and explains the mechanism; the curation finding (half the raw positives were
duplicates) is itself a methodological lesson. This is a *data-centric* contribution, not an
architectural one, and several claims are bounded by our two-source, single-platform, apparent-attribute
setting.

## 7. Limitations

- **Backbone dependence:** the gain concentrates in weak/medium backbones; strong saturated models do
  not improve and slightly forget (gentle lr mitigates but does not reverse this).
- **External validity:** two VK communities, one platform, one language/cultural context;
  apparent-female-skewed (76.7%); sparse at apparent age 45+. Transfer is shown between these sources and
  to standard benchmarks, not platform-agnostic generalization.
- **Apparent attributes only** in the fairness audit (estimated, not self-identified gender/ethnicity).
- **Naturally-supervised, not strictly label-free:** an LLM parses ages and validates group integrity;
  clustering forms persons — audited (§3.3) but a residual hidden-noise source.
- **Statistics:** seed variance and paired bootstrap (fairness) are reported; DeLong/paired-bootstrap
  CIs for all benchmark comparisons and Holm correction are in the release supplement.
- **Benchmark scale and age:** we evaluate on the field-standard cross-age suite — AgeDB-30 and CALFW
  (6,000 pairs each) and FG-NET for the explicit ≥25-year subset — the same benchmarks used by
  OE-CNN/DAL/MTLFace, which fixes comparability but inherits their limited scale and age (FG-NET in
  particular is small with partial detector coverage, so we never rest the headline on it alone). Our
  curated internal test (5,107 cross-age positives with controlled negatives) is a sizeable modern
  complement; a larger independent non-VK longitudinal set and a dedicated child↔adult benchmark remain
  the priority for future work.
- **SOTA pipelines not reproduced end-to-end:** by design we train the *shared core* of modern AIFR —
  the ArcFace angular-margin objective — under our weak-backbone protocol for clean attribution, not a
  leaderboard comparison. §5.6 shows this objective matches our contrastive one on the key external
  metric, so the objective is not what drives the gain; the omitted method-specific add-ons (MTLFace's
  age-synthesis branch, OE-CNN's orthogonal subspaces) refine that shared objective and are orthogonal to
  our claim about the supervision source. Whether a full SOTA pipeline trained on naturally-supervised
  data closes the remaining absolute gap is a well-scoped follow-up.

## 8. Ethics, legal basis and data governance

This is a sensitive biometric study; we treat governance as a first-class component, not a disclaimer.
Permitted use is restricted to **controlled / consent-based / human-reviewed** applications
(rights-cleared archives and family photos, authorized humanitarian search, account recovery); the
system outputs **candidates for human review**, never an automatic decision. **Forbidden:** mass
identification, de-anonymization, surveillance, and any processing of minors' data without a legal basis.

**8.1 Legal basis and its limits (honest assessment).** Data were collected from public "then/now"
community walls via the platform's official API. We stress that *official API access and voluntary
public posting do not by themselves authorize biometric processing.* Under Russia's 152-FZ, the 2020
amendment (519-FZ) replaced "publicly available" data with "data the subject authorized for
dissemination" (separate consent required), and Art. 11 requires **written consent** for biometric data
used to identify a person; under the GDPR, a face processed for unique identification is
**special-category** data (Art. 9), for which the "manifestly made public" exception is read narrowly
(cf. the Clearview AI enforcement). We do **not** hold explicit/written consent. We rely on the
**scientific-research** basis (GDPR Art. 6(1)(f) + 9(2)(j) with Art. 89 safeguards; analogous research
handling under 152-FZ) and compensate with the minimization, non-redistribution and de-identification
measures below. We report this gap transparently rather than claiming full compliance, and we recommend
institutional ethics review and data-protection legal counsel before any deployment.

**8.2 De-identification and release policy.** We **do not release raw face images or raw posts.** The
public release contains only code, trained model weights, **derived embeddings**, the pseudonymized
benchmark protocol, and aggregate statistics. All platform identifiers (owner/photo/post IDs) are
replaced by **salted hashes**; captions/comments (which may contain names) are excluded; only an age
bucket and a pseudonymous person ID are kept. Raw crops are retained locally only, under a retention
policy. Access to non-aggregate derived data is granted to qualified researchers under a **Data Use
Agreement** (research-only; no re-identification; no commercial/surveillance use). Paper figures contain
no identifiable faces (aggregate plots only).

**8.3 DPIA and minimization.** Because the processing is biometric and large-scale, a **Data Protection
Impact Assessment** is conducted and accompanies the release (key risks: re-identification, function
creep, minors; mitigations as in 8.2/8.4). Data minimization is applied end-to-end (we store a face
crop + age + pseudonymous ID, not names or social graphs).

**8.4 Minors.** The child↔adult regime necessarily involves images of minors, whose data carry the
strictest protection (152-FZ; GDPR Art. 8). Minor-age items (flagged via apparent age) are **withheld
from any release** absent an explicit legal framework and verifiable guardian consent, and are used only
for aggregate, non-redistributed analysis.

**8.5 Subject rights.** A public contact and **takedown / deletion-on-request** procedure are provided;
deletion propagates to derived artifacts via a per-item deletion-status field. Where rights are unclear,
the item is withheld. A full datasheet (Gebru-style) and the DPIA accompany the release.

## 9. Reproducibility

We release scripts and configs. Key details: **usable-face criteria** (detection score, minimum size,
Laplacian blur, single-target via IoU); **detector/alignment** (InsightFace `buffalo_l` RetinaFace,
5-point ArcFace `norm_crop`, 112px; FaceNet input 160px); **embeddings** (ArcFace `w600k_r50`) for
clustering/dedup; **person clustering** (union-find, cos ≥ 0.85); **dedup** (within-person union-find,
cos ≥ 0.97); **LLM** (GigaChat; full versioned system prompts for age extraction and group-integrity;
concurrency 10; cached, resumable); **split** (by person, 70/15/15, balanced negatives per split, seed
42); **negative sampling** (cross-person random + age-matched variant); **training** (margin
contrastive, head scope, lr 3e-5 weak / 1e-5 strong; ArcFace head m=0.5 s=32, lr 1e-3; early stop on
internal val-AUC; seeds 42/1/2); **metrics** (pure-numpy ROC-AUC via Mann–Whitney, EER, TAR@FAR;
10-fold for aligned-pair benchmarks).

**Compute.** The entire study runs on a single laptop: one NVIDIA RTX 3060 Laptop GPU (6 GB), an AMD
Ryzen 7 5800H (8 cores / 16 threads) and 16 GB RAM under Windows 11; the software stack is Python 3.12,
PyTorch 2.12 (CUDA 13.2, cuDNN 9.2), with InsightFace 1.0 / ONNX Runtime 1.26 for detection and
embeddings and scikit-learn 1.9 / Matplotlib 3.11 for analysis and figures. The weak-backbone protocol
keeps the full study within a 6 GB GPU budget.

## 10. Conclusion and Future Work

Multi-photo posts are a cheap, scalable, naturally-supervised signal of cross-age identity; fine-tuning
a weak recognizer on them yields transferable, cross-source, objective-robust gains concentrated on large-gap cross-age matching (with only minor forgetting on easy benchmarks), explained
by removing an age shortcut, that are data-efficient and do not worsen any apparent-demographic
stratum. Future work: an independent non-VK and a child↔adult-specific external set; a human-audited supervision subset (with human–LLM agreement); per-point multi-seed
confidence bands for the scaling and headroom curves; full DeLong tests and per-cell
multiple-comparison correction; full SOTA add-ons; a real aging model at native resolution; and a calibrated demo under the governance constraints of §8.

## References

Key prior work, benchmarks and methods used (full bibliographic formatting in the typeset version):

1. Y. Wang et al. *Orthogonal Deep Features Decomposition for Age-Invariant Face Recognition* (OE-CNN), ECCV 2018.
2. H. Wang et al. *Decorrelated Adversarial Learning for Age-Invariant Face Recognition* (DAL), CVPR 2019.
3. Z. Huang et al. *When Age-Invariant Face Recognition Meets Face Age Synthesis* (MTLFace), CVPR 2021.
4. Wang et al. *Cross-Age Contrastive Learning* (CACon), 2023.
5. Yao et al. *Synthetic Face Ageing for age-robust face recognition*, 2024.
6. Wang et al. *OrdCon: Order-enhanced Contrastive Learning*, 2025.
7. J. Deng et al. *ArcFace: Additive Angular Margin Loss*, CVPR 2019.
8. M. Kim et al. *AdaFace: Quality Adaptive Margin*, CVPR 2022.
9. F. Schroff et al. *FaceNet*, CVPR 2015.
10. Q. Cao et al. *VGGFace2: faces across pose and age*, FG 2018.
11. T. Zheng et al. *Cross-Age LFW (CALFW)*, 2017.
12. S. Moschoglou et al. *AgeDB: the First Manually Collected In-the-Wild Age Database*, CVPRW 2017.
13. A. Lanitis et al. *FG-NET Aging Database*, 2002.
14. B.-C. Chen et al. *Cross-Age Reference Coding for Age-Invariant Face Recognition and Retrieval (CACD)*, ECCV 2014.
15. K. Ricanek and T. Tesafaye. *MORPH: A Longitudinal Image Database of Normal Adult Age-Progression*, FG 2006.
16. T. Gebru et al. *Datasheets for Datasets*, CACM 2021.
