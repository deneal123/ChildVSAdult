# Cross-Age Face Verification from Mined Same-Person Social-Media Pairs: A Naturally-Supervised Longitudinal Signal

*Working paper draft (EN). Self-contained: all tables and numbers are inline. All results are reported
on the curated dataset of §3.4, with seed variance and, for the fairness audit, paired bootstrap
confidence intervals; where useful we also show the pre-curation number for comparison (e.g. the
"(pre-clean gain)" column in Table 7). The one exception is the minor +age-anchor ablation increment
(§5.1), which is reported pre-curation as it does not affect any conclusion.*

## Abstract

Cross-age face verification — deciding whether two photographs taken years apart depict the same
person — is hard, and longitudinal training data are scarce. We ask a **data-centric** question: can
the supervision for age invariance be obtained without a manually labeled longitudinal dataset? We
show that multi-photo "then/now" social-media posts supply weak **same-person** longitudinal pairs
with no manual identity/age labels (a large-language model parses captions and embedding clustering
forms persons; we audit these components, including a small manual validation), and that fine-tuning
a deliberately weak recognizer on them induces **transferable, targeted** large-gap robustness. On the
external benchmark FG-NET, large-gap (≥25-year) ROC-AUC rises **from 0.736 to 0.848** (+0.112 ± 0.001
over three seeds; DeLong p<10⁻¹⁹), at the cost of −0.019 LFW accuracy and **substantial** low-FAR
degradation — a targeted retrieval gain, not a general-verifier upgrade. Our central, bounded claim is
that, within this weak/medium-backbone attribution protocol, the **data source** — not the choice among
the objectives we test — accounts for most of the gain: a pair-contrastive objective matches an
ArcFace-margin one (0.848 vs. 0.851). The gain is driven by the same-person pairing (explicit caption
age anchors add only +0.009) and is mechanistically tied to removing an age shortcut. Supporting
evidence in the paper: superiority over a real re-aging model under a gap-parity control; bidirectional
transfer to an independent non-VK platform (Reddit); data efficiency (≈96% of the gain at 10% of
identities); an apparent-demographic audit; calibration; and a curation finding that about half of the
raw "positives" were near-duplicate frames inflating internal metrics.

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
positive identity pairs without manual annotation; a caption stating ages adds a secondary age anchor.
The same-person pairing is the operative signal (an ablation later isolates the age anchor at a
marginal +0.009); this is a scalable, naturally-supervised signal of cross-age identity, and
fine-tuning a weak recognizer on it should transfer to external benchmarks.

**Contributions (bounded).** (1) A **weakly/naturally-supervised (no manual id/age labels) longitudinal
signal** mined from social posts — same-person identity pairs, with caption age anchors only a secondary
cue. (2) A rigorous **attribution protocol** (one weak trainable backbone, frozen vs. fine-tuned,
external evaluation) isolating the contribution of *data* from network capacity. (3) **Mutually
reinforcing controls** (loss family, architecture, source community, training objective,
real-vs-synthetic, shortcut diagnosis, and cross-platform transfer to an independent non-VK source)
supporting the bounded claim that, within this protocol and across the objectives tested, the data
contributes more to the large-gap gain than the choice among loss families. (4) A **curation pipeline**
with an audit of its automatic components, and the finding that naive positive counts are inflated ~2×
by near-duplicate frames. (5) **Applied calibration** and an **apparent-demographic fairness audit**
with confidence intervals.

We emphasize the novelty is **data-centric, not architectural** — a new *origin of supervision* — and
should be judged on that axis.

## 2. Related Work and Positioning

Three lines of prior work are most relevant. **Discriminative age-invariant FR on labeled data:**
OE-CNN (orthogonal embedding decomposition), DAL (decorrelated adversarial learning), and MTLFace
(multi-task recognition, age synthesis and attention decomposition) achieve strong accuracy but are
trained on datasets with **manual identity and age labels** (CACD, MORPH, AgeDB, FG-NET).
**Cross-age contrastive with synthetic samples:** CACon adds a synthesized cross-age sample and a
triplet loss; OrdCon uses order-enhanced contrastive learning for generalized age features.
**Synthetic aging for robustness:** systematic studies show synthetic aging helps only partially.
Generic self-supervised FR (SimCLR-style augmentation) lacks identity anchors across time. We instead
mine the structure "one post = one person at different ages" as a label-free positive-pair generator.
Table 1 positions us against these methods.

**Table 1. Positioning vs. prior work.**

| work | supervision | mechanism | our difference |
| --- | --- | --- | --- |
| OE-CNN (2018) | labeled id+age | orthogonal feature decomposition | no manual id/age labels, not decomposition |
| DAL (2019) | labeled id+age | adversarial id/age factorization | mined pairs + simpler objective |
| MTLFace (2021) | labeled id+age | recognition + age synthesis (multi-task) | weaker pipeline; novel supervision origin |
| CACon (2023) | semi-sup + synthetic | contrastive on synthesized cross-age | real mined pairs > synthetic aging (§5.2) |
| Synthetic Ageing (2024) | synthetic | train on aged images | we make *real* mining the thesis |
| OrdCon (2025) | age-ordered contrastive | order-enhanced features | mine natural pairs, no explicit age order |

We report on the canonical cross-age benchmarks of these methods (AgeDB-30, CALFW, FG-NET; CALFW was
designed as a harder, age-gapped LFW). The goal is **not** to surpass the absolute state of the art (we
use a weak backbone for clean attribution) but to quantify the magnitude and transferability of the
gain from a naturally-supervised source; §5.6 trains the canonical SOTA objective on our data directly.

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

**3.1 Age extraction.** Captions are parsed by regular expressions (Russian patterns + the slash
format "6/18/21", one age per photo) and by a large language model (GigaChat). Across all 33,697
multi-photo captions (Table 3) the LLM broadens and regularizes age extraction relative to regex — it
avoids reading calendar years or gaps as ages and recovers second ages — disagreeing on 27.2% of posts,
overwhelmingly LLM-favoring; on a 100-caption manual gold subset the LLM matched the human reading on
**89% vs. regex 71%** (supplementary material). We do not treat LLM output as ground truth: downstream,
explicit age anchors contribute only marginally (§5.1).

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

**3.3 Auditing the automatic supervision.** The automatic supervision components are potential hidden
label noise, audited three ways: LLM-vs-regex age agreement (Table 3); manual inspection of high-cosine
pair identity; and within-person gender consistency as an inexpensive noise detector (2,919 multi-photo
groups with consistency <0.6 flag probable multi-person). A *small* manual validation (one annotator)
*sanity-checks* the automatic pipeline (Table 4): it *supports*, but — with a single annotator and no
inter-annotator κ — does not *certify* the supervision; a large multi-annotator audit remains beyond
our resources, so residual noise is acknowledged as a limitation (§7).

**Table 4. Small manual validation (one annotator; a sanity check, *not* a certified audit — no
inter-annotator κ). Samples are drawn from the curated corpus and were not used to tune thresholds or
filtering rules.**

| component | n | sampling | human-checked result |
| --- | --- | --- | --- |
| age extraction | 100 | strat. random | LLM 89% vs. regex 71% exact |
| group integrity | 100 | random+risk | single/multi acc. 0.96 |
| positive pairs | 200 | random | same-person prec. 0.96 |
| cluster merges | 100 | near-thresh. | merge prec. 0.95 |
| near-dup removals | 100 | random | duplicate prec. 0.93 |

**3.4 Curation pipeline and its largest finding.** Three steps: (i) **apparent gender/age metadata**
(estimator) per face, aggregated per person — the corpus is **76.7% apparent-female / 23.3%
apparent-male**; (ii) **LLM group-integrity validation** (Table 5) — person-clustering had already
separated most multi-person posts, so only 421 person-groups are flagged noisy; (iii) **near-duplicate
dedup** within each person (cos ≥ 0.97), removing 8,190 faces (~17%).

**Table 5. LLM group-integrity classification (33,697 posts).**

| category | n | share |
| --- | --- | --- |
| single (one person across ages) | 31,275 | 92.8% |
| multi-person (different people) | 2,034 | 6.0% |
| unknown | 336 | 1.0% |
| meme / collage | 52 | 0.2% |

**Key finding.** Because the number of positives is quadratic in the number of faces per group,
removing near-duplicates reduced the positive count from 65,897 to 31,865 (**−52%**): **approximately
half of the raw "positives" were duplicate frames** (a single shoot, burst, or repost) that had
inflated the internal metrics. After curation, the frozen baseline on the internal test decreases
(our.overall 0.856 → 0.836; our.25+ 0.705 → 0.640), exposing a harder and more representative
benchmark. The external benchmarks are unaffected. This constitutes a transferable methodological
observation for mining longitudinal pairs from social media.

**Threshold robustness.** The −52% reduction is not an artifact of the 0.97 cutoff. Re-running the
near-duplicate dedup on the pre-curation groups at cosine thresholds 0.93–0.99 (Table 6) leaves
31.3k–32.4k positives across 0.93–0.97 (a stable ~52% reduction) and degrades gracefully only at the
very strict 0.99 (−41%): the near-duplicates are dominated by near-identical frames (cos ≈ 1), so the
finding does not hinge on the exact cutoff.

**Table 6. Dedup-threshold sensitivity: near-duplicate removal alone, on the pre-curation groups (the
separate group-integrity prune removes a further ~500 positives to reach the headline 31,865). Stable
across 0.93–0.97.**

| dedup cos | near-dup faces | positives | reduction |
| --- | --- | --- | --- |
| 0.93 | 8,653 | 31,304 | −52.5% |
| 0.95 | 8,564 | 31,611 | −52.0% |
| 0.97 (used) | 8,304 | 32,360 | −50.9% |
| 0.99 | 6,274 | 38,565 | −41.5% |

**3.5 Dataset datasheet (summary).** Provenance, rights and governance fields are stored per item
(source, origin URL, collection date, license/consent status, allowed use, retention policy, deletion
status, review status). Collection: public VK walls. Intended use: research on consent-based cross-age
matching. Out of scope: mass identification, surveillance, de-anonymization. Minors: the child↔adult
regime is in scope only under an explicit legal/ethical framework; per-item review status is tracked
and deletion-on-request is supported. A full Gebru-style datasheet accompanies the release.

## 4. Method, protocol and statistics

**Attribution protocol.** Comparing an adapter on top of a *frozen* strong ArcFace against frozen
ArcFace is invalid (the adapter merely deforms near-perfect embeddings). Instead we use **one weak
trainable backbone** (FaceNet InceptionResnetV1, CASIA-WebFace) and compare (a) frozen → benchmark
metric vs. (b) soft fine-tuning of the head on our pairs → benchmark metric. Training uses a margin
contrastive loss on aligned crops. **Evaluation is on external benchmarks** (LFW, AgeDB-30, CALFW,
FG-NET) for a clean attribution of transfer; the internal test (`our.*`) is a secondary in-domain
diagnostic. Backbone strength is varied (§5.3) to map where the data helps. The design trades absolute
performance for *causal attribution*.

**Backbones.** FaceNet (weak), ArcFace iResNet-50 (CASIA-FaceV5; weak, different architecture), AdaFace
IR-50/IR-101 (WebFace; strong; a clean depth control), ArcFace iResNet-100 (strong).

**Benchmarks and metrics.** LFW (10-fold accuracy), AgeDB-30 and CALFW (aligned pairs; ROC-AUC and
10-fold accuracy), FG-NET (ROC-AUC overall and on the large-gap ≥25-year subset). We deliberately do
not rest the headline on FG-NET alone (small, old, partial detector coverage): AgeDB-30 and CALFW show
that the large-gap gain comes without a catastrophic collapse on standard cross-age benchmarks
(AgeDB-30 decreases slightly, CALFW is nearly unchanged), confirming that the effect is concentrated
on the FG-NET large-gap subset rather than uniform across benchmarks. We distinguish **threshold-free**
metrics (ROC-AUC) from **threshold-based** ones (accuracy, TAR@FAR).

**Statistics.** Our *primary endpoint* is the FG-NET large-gap (≥25-year) ROC-AUC, with the null
hypothesis that fine-tuning on the mined pairs does not improve it over the frozen baseline (gain ≤ 0);
AgeDB-30, CALFW and the internal cross-age test are secondary endpoints, and easy LFW accuracy is
monitored as a forgetting control. The headline gain is **mean ± std over 3 seeds** (seed variance
captures training stochasticity, not full evaluation uncertainty). The fairness audit uses a **paired
percentile bootstrap** (1000 resamples, resampling pairs and recomputing both models on the same
resample) for the *gain*, accounting for frozen/tuned correlation. Table 8 reports bootstrap CIs, EER
and TAR@FAR for the key benchmark comparisons (the internal test additionally uses an identity-level,
person-resampling bootstrap), and a **Holm correction** is applied across the stratified fairness tests.

**Curated dataset.** All headline experiments are on the curated data of §3.4: 21,427 person groups,
40,586 faces, 31,865 positives; split train 22,179 / val 4,579 / test 5,107 with balanced negatives per
split.

## 5. Results

**5.1 Main effect and supervision-source ablation.** The weak FaceNet, softly fine-tuned on our pairs,
raises FG-NET large-gap ROC-AUC 0.736 → **0.848** (+0.112 ± 0.001), while easy LFW falls only −0.019
and AgeDB-30/CALFW move negligibly (Table 7). An ablation isolates the source: same-post pairs use no
manual labels — neither identity nor age — and deliver almost all of the gain; adding explicit age
anchors contributes only **+0.009** on the 25+ bucket (pre-clean ablation). The method is therefore
**weakly supervised by naturally occurring same-post co-occurrence** (not by manual identity/age
labels), with age supervision contributing only marginally.

**Table 7. Main result: weak FaceNet frozen vs. +pairs, 3 seeds, curated data. Pre-clean gain shown
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

Bootstrap confidence intervals (seed 42) confirm the primary endpoint: FG-NET large-gap rises from
0.736 [0.70, 0.78] to 0.848 [0.82, 0.88] with *non-overlapping* 95% CIs; the equal-error rate **nearly
halves** (0.335→0.219) and TAR@FAR=1% **nearly doubles** (0.21→0.39). On the internal 25+ test the gain
holds under the more conservative *identity-level* bootstrap (resampling persons, not pairs): +pairs
0.838 [0.80, 0.88] vs. frozen 0.640 [0.59, 0.69], an interval wider than the pair-level one as expected
when pairs share identities. Operating points and intervals for all benchmarks are in Table 8. Because
bootstrap-CI overlap ignores the samples the two models share, we confirm the primary endpoint with
DeLong's test for two *correlated* ROC curves: on FG-NET large-gap **z=9.3, p=1.4×10⁻²⁰** (ΔAUC 95% CI
[+0.088, +0.135]), and on the internal 25+ test p=1.7×10⁻³⁴; a 10⁴-fold paired permutation test agrees
(p<10⁻⁴). The same test confirms the gain is *targeted*: AgeDB-30 shows a small but significant decrease
(−0.008, p=4×10⁻⁶) and CALFW is statistically unchanged (p=0.95).

**Table 8. Operating points and 95% bootstrap CIs (FaceNet frozen vs. +pairs, seed 42). All values are
ROC-AUC unless noted; LFW is shown as ROC-AUC for consistency (its 10-fold accuracy is in Table 7).
External CIs are pair-level; the internal test additionally uses an identity-level bootstrap (text).**

| benchmark | frozen AUC [95% CI] | +pairs AUC [95% CI] | EER (fr.→+p.) | TAR@1% | TAR@0.1% |
| --- | --- | --- | --- | --- | --- |
| FG-NET large-gap | 0.736 [0.70, 0.78] | **0.848 [0.82, 0.88]** | 0.335→0.219 | 0.209→0.386 | 0.074→0.154 |
| FG-NET overall | 0.896 [0.89, 0.90] | 0.922 [0.92, 0.93] | 0.186→0.151 | 0.502→0.562 | 0.315→0.308 |
| AgeDB-30 | 0.953 [0.95, 0.96] | 0.945 [0.94, 0.95] | 0.115→0.126 | 0.525→0.510 | 0.285→0.309 |
| CALFW | 0.948 [0.94, 0.95] | 0.948 [0.94, 0.95] | 0.116→0.117 | 0.580→0.624 | 0.213→0.320 |
| LFW (AUC) | 0.994 [0.99, 1.00] | 0.987 [0.98, 0.99] | 0.032→0.052 | 0.940→0.869 | 0.858→0.575 |

**5.2 Real pairs outperform synthetic aging.** Replacing real positives with synthetically aged
versions — a domain proxy and a real re-aging network (FRAN) — underperforms real pairs on the curated
data, and at FRAN's default *wide-gap* setting even *degrades* below frozen: FG-NET large-gap 0.727
(**below frozen 0.736**) and our.25+ 0.549 (below frozen 0.640), whereas real pairs yield +0.112 /
+0.198 (Table 9). Identity-preserving aging reproduces texture artifacts rather than genuine
longitudinal variation (pose, camera, era). A **native-resolution control** rules out a crop-resolution
artifact: re-cropping a train subset at 512 px directly from the raw images (vs. our stored 112-px
crops, which FRAN upscales internally) and re-aging **leaves the result unchanged** — FG-NET large-gap
0.771 vs. 0.778 and our.25+ 0.671 vs. 0.670 on this 499-face subset — so synthetic aging fails to match
real pairs regardless of source resolution. A **gap-parity control** sharpens this: FRAN ages to a wide
33–70-year gap whereas real pairs have a median gap of 9 years, and matching the synthetic gap to the
real distribution lifts FRAN to **0.759** (our.25+ 0.622) — a *modest* positive just above frozen, in
line with small synthetic-aging gains reported elsewhere — yet real pairs (0.848) still lead by
**+0.089**. The sub-frozen figure was thus partly a gap-mismatch artifact: real mined pairs beat even a
*gap-matched* re-aging model.

**Table 9. Real vs. synthetic positives (FaceNet, curated data). FRAN-gm matches the synthetic age-gap
to the real-pair distribution (median 9 y); the cruder domain proxy reaches 0.761 / 0.594.**

| metric | frozen | +real | syn-FRAN | FRAN-gm |
| --- | --- | --- | --- | --- |
| FG-NET large-gap | 0.736 | **0.848** | 0.727 | 0.759 |
| our.25+ | 0.640 | **0.838** | 0.549 | 0.622 |

**5.3 Headroom curve and the strong-backbone regime.** The gain reproduces on FaceNet, ArcFace iResNet,
and AdaFace IR-50/IR-101 (three loss families, two architectures) and **monotonically decreases with
frozen strength** (Table 10); a clean depth control (AdaFace IR-50 → IR-101) confirms it. Strong
saturated backbones do not improve and forget slightly under fine-tuning; a gentler learning rate
(1e-5) approximately halves the forgetting but does not convert it into a gain. The contribution is
therefore specific to weak and medium backbones with cross-age headroom — a limitation we state
explicitly (§7).

**Table 10. Headroom: FG-NET large-gap by backbone strength (curated data, lr 3e-5).**

| backbone (strength) | frozen | +pairs | Δ large-gap | Δ our.25+ |
| --- | --- | --- | --- | --- |
| FaceNet (weak) | 0.736 | 0.848 | **+0.112** | +0.198 |
| ArcFace r50 (weak, diff. arch) | 0.764 | 0.805 | +0.041 | +0.133 |
| AdaFace IR-50 (strong) | 0.917 | 0.924 | +0.007 | +0.004 |
| ArcFace r100 (strong) | 0.954 | 0.915 | −0.039 | −0.002 |
| AdaFace IR-101 (strong) | 0.957 | 0.931 | −0.026 | −0.006 |

**5.4 Data-scaling law.** Varying the fraction of training identities (val/test fixed), the gain
saturates early: **≈96% of the full FG-NET large-gap gain is reached with only 10% of identities
(~1,500)**, reaching a plateau from 25% (Table 11; each point is mean ± std over three seeds); the
slight dip at full data (0.50→1.00) lies within seed variance — the curve is flat, not monotonic, beyond
~10%. The signal is thus data-efficient; further gains require *harder* data (larger gaps, hard
positives) rather than merely a greater volume.

**Table 11. Data-scaling law (FaceNet, curated; val/test fixed; mean ± std over three seeds).**

| train fraction | # identities | FG-NET large-gap | our.25+ |
| --- | --- | --- | --- |
| frozen | 0 | 0.736 | 0.640 |
| 0.10 | ~1,500 | 0.846 ± 0.002 | 0.816 ± 0.003 |
| 0.25 | ~3,750 | 0.852 ± 0.000 | 0.842 ± 0.002 |
| 0.50 | ~7,499 | 0.856 ± 0.005 | 0.845 ± 0.002 |
| 1.00 | ~14,998 | 0.848 ± 0.001 | 0.835 ± 0.003 |

**5.5 Mechanism — the age shortcut.** Under age-matched negatives (same age bucket, different people),
frozen accuracy on the internal 25+ test falls from 0.640 to **0.517** (**near the chance level of
0.5**), and real pairs restore it to **0.838** — direct evidence that frozen models discriminate
cross-age pairs largely by **age**, whereas our pairs induce discrimination by **identity**. A linear
probe refines this *representationally* (Table 12): apparent age stays well decodable from the identity
embedding for *all* models (~0.63–0.65 balanced accuracy ≫ 0.25 chance) and changes only slightly, while
identity AUC rises sharply — so the method learns an invariant **metric** (the cosine stops relying on
the age axis), not an age-free representation; explicit adversarial disentanglement reduces leakage no
further.

**Table 12. Age-leakage linear probe (curated data; chance = 0.25).**

| model | age-probe balanced acc ↓ | identity ROC-AUC ↑ |
| --- | --- | --- |
| frozen | 0.651 ± 0.019 | 0.836 |
| +pairs | 0.629 ± 0.012 | 0.916 |
| +pairs+disentangle | 0.632 ± 0.017 | 0.918 |

**5.6 Objective comparison: training the SOTA objective on our data.** We train the canonical modern-FR
margin objectives — **ArcFace-, CosFace- and SphereFace-margin identity classification** (the core of
OE-CNN/MTLFace) — on our naturally-supervised identities (9,204 classes), same weak backbone, scope and
protocol. On the FG-NET large-gap primary endpoint the contrastive, ArcFace and CosFace objectives
**coincide** (0.848 / 0.851 / 0.843, within ±0.006; Table 13); only SphereFace — the hardest margin to
optimize, here without λ-annealing on 2–3-shot identities — trails (0.801). We draw the bounded
conclusion that, within this protocol and among the objectives tested, the data source accounts for
more of the gain than the objective choice — not that the objective is irrelevant in general.
ArcFace/CosFace additionally **forget less** on easy benchmarks — a practical recommendation. A
*noise-robust* comparator — **sub-center ArcFace**, which keeps K=3 centroids per identity so that
label-noisy or low-quality crops route to off-centers — lands at the *same* large-gap **0.851** (our.25+
0.869); explicit label-noise handling thus adds nothing over a plain margin on our *curated* mined
pairs, consistent with deduplication having already removed the dominant noise source. The supervision
source, not noise-robust loss machinery, drives the gain.

**Table 13. Objective comparison (FaceNet, curated data): contrastive, ArcFace and CosFace coincide on
the FG-NET large-gap primary endpoint; SphereFace (no λ-annealing) trails on our sparse few-shot
identities.**

| objective | FG-NET l.g. | our.25+ | LFW | AgeDB-30 |
| --- | --- | --- | --- | --- |
| frozen | 0.736 | 0.640 | 0.969 | 0.953 |
| +pairs (contrastive) | 0.848 | 0.838 | 0.946 | 0.945 |
| +ArcFace | **0.851** | 0.868 | 0.952 | 0.947 |
| +ArcFace, sub-center | 0.851 | 0.869 | 0.951 | 0.947 |
| +CosFace | 0.843 | 0.860 | 0.952 | 0.945 |
| +SphereFace | 0.801 | 0.699 | 0.950 | 0.943 |

**5.7 Apparent-demographic audit.** Stratifying by **apparent** gender/age of the pair anchor
(estimated, **not self-identified**), **no stratified group degrades** and every gain is significant
(95% paired-bootstrap CI > 0; Table 14). The largest gain is for the youngest group **0–17** (frozen
0.750 → +pairs 0.880, +0.130, CI not overlapping adult bands) — the method strengthens where the base
model is weakest (the child↔adult regime), **narrowing** the apparent-demographic gap. This is
corroborated on *independent, non-VK* data: on a **child↔adult subset of FG-NET** (one photo under age
13, the other over 25; 195 positives, 195 negatives), +pairs raises ROC-AUC from 0.684 [0.63, 0.74] to
0.813 [0.77, 0.86] (**+0.129, non-overlapping 95% CIs**), closely matching the internal +0.130. Beyond
AUC, an **error decomposition** at a fixed operating point (per-model global threshold at 1% FMR) shows
a genuine reduction in matching errors where it matters: +pairs lowers the false-non-match rate in
*every* apparent stratum at matched ~1% FMR — 0–17 **0.852→0.656**, apparent-female 0.674→0.509,
apparent-male 0.743→0.534 — with no stratum regressing, and per-stratum score calibration (logistic
slope 9.0–11.7) stays uniformly positive and stable; the only exception is the small 45+ stratum
(n≈150), whose FMR scatters to 2.1% at the global threshold (sampling noise). We deliberately avoid the
word "fair" as a solved property: the source is apparent-female-skewed (76.7%), attributes are apparent
(not ethnicity or legal categories), and 45+ is sparse.

**Table 14. Apparent gender/age strata (FaceNet, curated; 95% paired-bootstrap CI of the gain).**

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
internal +0.092 on the held-out community. We frame this as transfer between two independent sources
within one platform domain; §5.9 extends the test to an independent non-VK platform, and we do not
claim web-scale generalization.

**5.9 Cross-platform transfer (independent, non-VK source).** A key external check of a
naturally-supervised *source* is whether it generalizes beyond the originating platform. We apply the
*identical* mining protocol to an independent, non-VK community — the Reddit board r/PastAndPresentPics
(same "then/now" format, same LLM parser) — and evaluate the VK-trained +pairs model (*no* Reddit data
in training) on 1,575 Reddit pairs (Table 15): overall ROC-AUC rises **0.718 → 0.749** (+0.031,
near-disjoint 95% CIs), EER 0.345 → 0.328. The transfer is *bidirectional* (Table 16): a weak backbone
trained on *only* the small Reddit source — a 220-person *pilot*, not a headline claim — also helps on
VK. Reddit then/now pairs are inherently cross-age, so the frozen 0.718 sits far below its
easy-benchmark ceiling, mirroring the within-VK large-gap regime; the gain is smaller than within-VK (a
noisier, partly collage-derived set), so we read it as confirming the *source* generalizes beyond its
platform, not as a SOTA claim. The explicit ≥25-year subset is not reported separately (Reddit titles
seldom state exact ages).

**Table 15. Cross-platform transfer: a VK-trained model on an independent Reddit (non-VK) then/now test
set (frozen vs. +pairs, trained on VK only; 396 posts → 1,575 positive pairs after the
multi-person-collage filter).**

| Reddit (non-VK) test metric | frozen | +pairs (VK-trained) |
| --- | --- | --- |
| Overall ROC-AUC [95% CI] | 0.718 [0.70, 0.74] | **0.749 [0.73, 0.77]** |
| EER | 0.345 | **0.328** |
| TAR@FAR=1% | 0.271 | **0.279** |
| Test pairs (pos / neg) | 1,575 / 1,575 | 1,575 / 1,575 |

**Table 16. Cross-source transfer, both directions. VK→Reddit is the main cross-platform test;
Reddit→VK is a small 220-person *pilot* supporting source non-specificity, not a headline claim (single
run).**

| train | test | frozen | tuned | Δ |
| --- | --- | --- | --- | --- |
| VK | Reddit (overall) | 0.718 | 0.749 | +0.031 |
| Reddit | FG-NET large-gap | 0.736 | 0.783 | +0.047 |
| Reddit | VK internal 25+ | 0.640 | 0.659 | +0.019 |

**5.10 Calibration.** Raw cosine (Platt) is mis-calibrated across gap bins — over-confident at 15–25
years (ECE 0.052); a P(same | cosine, age-gap) calibrator fixes every bin (15–25 → 0.017) and improves
the overall Brier score (0.035 → 0.029) — a usable, age-aware same-identity probability.

**5.11 Disentanglement.** Explicit age removal (gradient reversal + age head, MTLFace-style) adds only a
small, stable cross-age gain over +pairs (+0.006 FG-NET large-gap, +0.015 our.25+) without harming easy
benchmarks; its absolute large-gap (~0.853) matches the ArcFace objective (0.851), reinforcing that the
data — not the objective or the disentangling add-on — sets the cross-age ceiling.

**5.12 Hard-negative mining.** Adding the hardest cross-person negatives to fine-tuning — the top-5
most-similar other-person faces per anchor (cosine in a hard band, excluding the near-duplicate/
uncertain range) — further sharpens large-gap discrimination: FG-NET large-gap rises to **0.868 ± 0.004**
over three seeds (from 0.848 with random negatives), but easy benchmarks forget *more* and FG-NET
overall ROC actually drops (LFW 0.969→0.925, AgeDB-30 0.953→0.911, CALFW 0.948→0.898, FG-NET ROC
0.923→0.912; Table 17). Hard negatives thus trade general-benchmark accuracy for large-gap separation —
useful for a cross-age retrieval setting, not for a general verifier. (A secondary result, not part of
the headline; the internal hard-negative test is near-chance for the frozen model by construction and
is not comparable to the standard internal test.)

**Table 17. Hard-negative mining (FaceNet, curated data; external benchmarks). +hard-neg is mean ± std
over three seeds; frozen and +pairs as in Table 7.**

| metric | frozen | +pairs | +pairs+hard-neg |
| --- | --- | --- | --- |
| FG-NET large-gap | 0.736 | 0.848 | **0.868 ± 0.004** |
| FG-NET ROC | 0.896 | 0.923 | 0.912 ± 0.003 |
| LFW accuracy | 0.969 | 0.950 | 0.925 ± 0.005 |
| AgeDB-30 ROC | 0.953 | 0.946 | 0.911 ± 0.013 |
| CALFW ROC | 0.948 | 0.949 | 0.898 ± 0.015 |

## 6. Discussion

The contribution is a scalable, naturally-supervised source of longitudinal supervision that teaches
transferable cross-age robustness which synthetic aging cannot replace and explicit age supervision
does not explain. The reinforcing controls — loss family, architecture, source community, objective,
and a noise-robust margin — converge on a bounded conclusion: within this protocol, the **data source**
accounts for more of the large-gap gain than the choice of objective. Crucially, the improvement is a
*targeted* large-gap gain, not a drop-in upgrade to a general verifier: low-FAR operating points on easy
benchmarks degrade (LFW TAR@FAR=0.1% falls 0.858 → 0.575; Table 8), so deployment should be scoped to
the large-gap cross-age regime rather than substituted for a general-purpose model.

**Scope of the claims.** To forestall over-reading, we state explicitly what the evidence does and does
not establish.

*Established (within this attribution protocol).* (i) the mined pairs improve large-gap (≥25-year)
cross-age verification on FG-NET (the primary endpoint, with non-overlapping 95% CIs) and on the
internal 25+ test; (ii) the gain transfers across the two source communities and to an independent
child↔adult FG-NET subset; (iii) real pairs **outperform synthetic aging**, including under
native-resolution and gap-parity controls; (iv) the gain is invariant across the contrastive, ArcFace
and CosFace objectives; (v) it is mechanistically tied to **removing an age shortcut**; (vi) roughly
half of the raw positive pairs were near-duplicate frames.

*Not established / out of scope.* (a) the effect is *not* a uniform verification gain — low-FAR
operating points on easy benchmarks degrade (Table 8); (b) strong saturated backbones do not improve;
(c) cross-platform transfer is shown on an independent non-VK platform (Reddit then/now, §5.9, +0.031
overall AUC) and on a child↔adult FG-NET subset; a recognizer trained on *only* a non-VK source already
transfers (§5.9); a larger non-VK *training* source with multi-seed CIs remains future work; (d) we do
not reproduce a full state-of-the-art pipeline, only its shared margin objective; (e) the fairness audit
uses apparent attributes only and lacks a human-audited inter-annotator κ (full manual annotation is
labor-intensive and beyond our resources; we report only automatic cross-checks) and a full per-cell
multiple-comparison correction (release supplement).

## 7. Limitations

- **Backbone dependence:** the gain concentrates in weak/medium backbones; strong saturated models do
  not improve and slightly forget (gentle lr mitigates but does not reverse this).
- **External validity:** two VK communities, one platform, one language/cultural context;
  apparent-female-skewed (76.7%); sparse at apparent age 45+. Transfer is shown between two VK sources,
  to standard external benchmarks, and to a small independent Reddit then/now test; this does not
  establish platform-agnostic generalization.
- **Apparent attributes only** in the fairness audit (estimated, not self-identified gender/ethnicity).
- **Naturally-supervised, not strictly label-free:** an LLM parses ages and validates group integrity;
  clustering forms persons — audited (§3.3) but a residual hidden-noise source.
- **Statistics:** seed variance, paired bootstrap (fairness), and benchmark CIs / EER / TAR@FAR (Table
  8, including an identity-level bootstrap) are reported in-text; DeLong tests for correlated ROC curves,
  an FMR/FNMR error decomposition and per-stratum calibration are now reported (§5.1, §5.7); a full
  per-cell multiple-comparison correction across every benchmark×operating-point remains for the release
  supplement.
- **Benchmark scale and age:** we evaluate on the field-standard cross-age suite — AgeDB-30 and CALFW
  (6,000 pairs each) and FG-NET for the explicit ≥25-year subset — the same benchmarks used by
  OE-CNN/DAL/MTLFace, which fixes comparability but inherits their limited scale and age (FG-NET in
  particular is small with partial detector coverage, so we never rest the headline on it alone). Our
  curated internal test (5,107 cross-age positives with controlled negatives) is a sizeable modern
  complement; we additionally report a child↔adult subset of FG-NET (§5.7) and a cross-platform transfer
  test on an independent non-VK source (Reddit, §5.9); a larger non-VK *longitudinal* set remains useful
  future work.
- **SOTA pipelines not reproduced end-to-end:** by design we train the *shared core* of modern AIFR —
  the ArcFace angular-margin objective — under our weak-backbone protocol for clean attribution, not a
  leaderboard comparison. §5.6 shows this objective matches our contrastive one on the key external
  metric, so the objective is not what drives the gain; the omitted method-specific add-ons (MTLFace's
  generative age-synthesis branch, OE-CNN's orthogonal-subspace decomposition) refine that shared
  objective and are orthogonal to our claim about the supervision source. Whether a full SOTA pipeline
  trained on naturally-supervised data closes the remaining absolute gap is a well-scoped follow-up.

## 8. Ethics, legal basis and data governance

This is a sensitive biometric study; we treat governance as a first-class component rather than as a
disclaimer. Permitted use is restricted to **controlled / consent-based / human-reviewed** applications
(rights-cleared archives and family photos, authorized humanitarian search, account recovery); the
system outputs **candidates for human review**, never an automatic decision. This is a *research-only*
study — nothing is deployed, no automated decision is made, and *no public biometric database is
released* (§8.2). **Forbidden:** mass identification, de-anonymization, surveillance, and any processing
of minors' data without a legal basis.

**8.1 Ethical approval and legal basis.** *Ethical approval.* No formal ethics-committee protocol has
been filed for this study to date; given its research-only, fully de-identified design (§8.2), the
authors will obtain an institutional ethics review or a documented exemption should a venue or reviewer
require one. *Consent.* Because the data are drawn from public posts with no feasible channel to contact
subjects, individual informed consent for participation and for publication was not obtained; we rely on
the scientific-research basis and the de-identification safeguards of §8.2.

Data were collected from public "then/now" community walls via the platform's official API. We stress
that *official API access and voluntary public posting do not by themselves authorize biometric
processing.* Under Russia's 152-FZ, the 2020 amendment (519-FZ) replaced "publicly available" data with
"data the subject authorized for dissemination" (separate consent required), and Art. 11 requires
**written consent** for biometric data used to identify a person; under the GDPR, a face processed for
unique identification is **special-category** data (Art. 9), for which the "manifestly made public"
exception is read narrowly (cf. the Clearview AI enforcement actions). We do **not** hold
explicit/written consent. We rely on the **scientific-research** basis (GDPR Art. 6(1)(f) and 9(2)(j)
with Art. 89 safeguards; analogous research handling under 152-FZ) and compensate with the minimization,
non-redistribution and de-identification measures below. We report this gap transparently rather than
claiming full compliance, and we recommend institutional ethics review and data-protection legal counsel
before any deployment.

**8.2 De-identification and release policy.** We **do not release raw face images or raw posts.** The
*public* release is restricted to code, configurations, the pseudonymized benchmark protocol and
aggregate statistics. Trained weights and derived embeddings are **not** released publicly: a face
embedding is itself a comparable **biometric template**, and salted identifier hashes do not render it
anonymous, so weights and embeddings are shared with bona-fide researchers *on request* under a brief
research-use agreement (research-only; no re-identification; no commercial or surveillance use). The
de-identified benchmark dataset is likewise available on request, and formal data-protection
documentation will be prepared if a requester or venue requires it. All platform identifiers
(owner/photo/post IDs) are replaced by **salted hashes**; captions/comments (which may contain names)
are excluded; only an age bucket and a pseudonymous person ID are kept. Raw crops are retained locally
only, under a retention policy. Paper figures contain no identifiable faces (aggregate plots only).

**8.3 DPIA and minimization.** Because the processing is biometric and large-scale, a **Data Protection
Impact Assessment** is conducted and accompanies the release (key risks: re-identification, function
creep, minors; mitigations as in §8.2/§8.4). Data minimization is applied end-to-end (we store a face
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

**Data availability.** Code, configurations, versioned LLM prompts, frozen splits (by hash) and the
de-identified benchmark protocol are released; *raw face images and posts are withheld* (§8), so
replication of the data-mining stage is *partial*: pseudonymization does not render face data
non-personal, and the LLM parser depends on an external versioned service subject to temporal drift, so
we release its cached outputs and prompts for deterministic replay rather than a live re-run. The
de-identified dataset and derived features are available to bona-fide researchers on request (§8).

## 10. Conclusion and Future Work

Multi-photo posts are a scalable, weakly/naturally-supervised signal of cross-age identity; fine-tuning
a weak recognizer on them yields transferable, cross-source, objective-robust gains concentrated on
large-gap cross-age matching (with moderate easy-benchmark accuracy loss but *substantial* low-FAR
degradation, which scopes the method to targeted large-gap retrieval rather than general verification),
explained by removing an age shortcut, that are data-efficient and do not reduce AUC in any evaluated
apparent stratum. Future work: a larger non-VK longitudinal set (beyond the Reddit cross-platform test
of §5.9) and a dedicated child↔adult benchmark; a human-audited supervision subset (with human–LLM
agreement), resources permitting; per-point multi-seed confidence bands for the headroom curve; a full
per-cell multiple-comparison correction; full SOTA add-ons; a real aging model at native resolution; and
a calibrated demonstrator under the governance constraints of §8.

**Funding and conflicts of interest.** This research received no specific grant from any funding agency
in the public, commercial, or not-for-profit sectors. The authors declare no competing financial or
non-financial interests.

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
17. H. Wang et al. *CosFace: Large Margin Cosine Loss for Deep Face Recognition*, CVPR 2018.
18. W. Liu et al. *SphereFace: Deep Hypersphere Embedding for Face Recognition*, CVPR 2017.
19. J. Deng et al. *Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web Faces*, ECCV 2020.
20. J. Deng et al. *RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild*, CVPR 2020.
21. M. Zoss et al. *Production-Ready Face Re-Aging for Visual Effects (FRAN)*, SIGGRAPH Asia 2022.
22. E. R. DeLong et al. *Comparing the Areas under Two or More Correlated ROC Curves: A Nonparametric Approach*, Biometrics 1988.
23. Regulation (EU) 2016/679 (General Data Protection Regulation, GDPR), 2016.
