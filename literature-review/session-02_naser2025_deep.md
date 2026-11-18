# Session 02 — Close Reading: Gaps & Tensions

## Session Metadata

| Field | Value |
|-------|-------|
| **Date** | 2026-11-16 |
| **Duration** | Several days |
| **Sections read** | Full paper (all sections) |
| **Goal** | Identify structural gaps and internal tensions relevant to our research focus |

---

## Papers Read

> **Naser, M. Z., et al.** "A review of machine learning with small and limited data."
> *Journal of Big Data*, 2025.
> — Full close reading following the initial survey pass in Session 01.

---

## Key Learnings

### 1. Intervention-Point Taxonomy (Refined)

The organising principle of the paper — **"actionable intervention points" in the ML pipeline** — proves more useful on close reading than it first appeared. The central recurring principle is:

> *Small-data learning is fundamentally about extracting maximum information from each sample while constraining model complexity to control the bias–variance trade-off and overfitting.*

The five intervention points, with their technical sub-themes:

| Intervention Point | Technical Sub-themes |
|--------------------|----------------------|
| **Theoretical foundations** | Bayesian/nonparametric methods (Dirichlet processes), variational inference, information geometry for parameter identifiability |
| **Classical methods** | SVMs (support vectors as structural fit for few samples), Gaussian processes (uncertainty quantification), tree ensembles, metric/manifold learning |
| **Algorithmic architectures** | Transfer learning, meta-learning (reuse from related tasks) |
| **Synthetic data generation** | Augmentation, GANs, VAEs, conditional GANs, diffusion models |
| **Emerging directions** | Domain knowledge + symbolic reasoning; causal inference; lifelong/online learning |

### 2. The Bias–Variance Lens

The paper is most coherent when read through the bias–variance lens: every intervention either **reduces variance** (regularisation, priors, ensembles) or **adds useful bias** (domain knowledge, structural constraints, pretrained representations). Naming this lens explicitly helps when comparing the emerging foundation-model approaches.

### 3. Evaluation as a Research Problem

The paper's most practically useful contribution for our focus: it treats small-sample evaluation as a *first-class research problem*, not a footnote. At very small *n*, standard cross-validation variance can exceed the signal from the model comparison itself. The survey mentions resampling strategies, information criteria (AIC/BIC/LOO-CV), and bootstrap confidence intervals as appropriate tools.

---

## Gaps Identified

### Gap 1 — No Operational Definition of "Extremely Small"

> **Description:** The survey uses "small," "limited," and "scarce" data as a broad, interchangeable umbrella. No principled, task-relative threshold is offered for the *extremely small* regime (e.g., tens of samples). Many cited techniques implicitly assume enough data to estimate distributions or fine-tune networks — that is, they are "small" only relative to internet-scale datasets, not absolutely small.

> **Significance for us:** Our research focus is precisely the absolute *n* ≈ 20–100 regime. Without a clear definition of this boundary, it is unclear which of the survey's cited methods genuinely apply vs. which only appear to apply because "small" is ill-defined.

### Gap 2 — Pretraining Dependence Unexamined

> **Description:** Transfer learning and meta-learning are presented as core solutions, but their dependence on large *source* datasets is not analysed critically. The paper does not distinguish between methods that **genuinely reduce sample complexity** and those that **relocate the data requirement to a pretraining phase** — a distinction that matters acutely when no suitable source domain exists.

> **Significance for us:** This is not merely academic. If a practitioner in a niche scientific domain has no suitable pretrained model and no large dataset to build one from, the "transfer learning" recommendation is not actionable.

### Gap 3 — No Controlled Head-to-Head Benchmarking

> **Description:** The survey synthesises and reports from many individual papers, each with different datasets, preprocessing pipelines, hyperparameter tuning budgets, and evaluation protocols. There is no controlled comparison under identical conditions. Decision criteria for choosing among methods are mostly qualitative.

> **Significance for us:** We cannot use the survey to answer "which method should I try first at *n* = 50 with 10 features?" — the question most practitioners need to answer.

### Gap 4 — Uncertainty Quantification Is Named but Not Operationalised

> **Description:** Reliability and uncertainty quantification are flagged as pressing concerns, especially at small *n*. Yet the practical protocol for producing *trustworthy* uncertainty estimates at very small *n* — not just any uncertainty estimate — is thin. The survey mentions Gaussian processes and Bayesian models but does not engage with practical uncertainty evaluation (calibration curves, coverage guarantees, conformal prediction).

> **Significance for us:** Given that overconfident predictions in small-sample regimes can cause real harm (especially in scientific and medical applications), uncertainty quality is not a secondary metric for us — it is a primary one.

---

## Contradictions / Tensions

### Tension 1 — Synthetic Data as Both Solution and Risk

> **The contradiction:** Generative augmentation (GANs/VAEs/diffusion models) is promoted to enlarge scarce datasets. Simultaneously, the paper cautions that these generators must be trained on data, and that over-resampling or synthesis can amplify noise when the original set is tiny.

> **The tension:** The proposed remedy partly presupposes the resource it is meant to supply. A GAN trained on 30 samples is not reliably generating the true data distribution — it is memorising and interpolating a tiny manifold, potentially with high variance.

> **Our working resolution:** Synthetic data generation from an *external prior* (e.g., structural causal model priors used by TabPFN) sidesteps this, because the generative process is not conditioned on the scarce target data. This needs to be tested.

### Tension 2 — Deep Learning for Small Data

> **The contradiction:** The survey states explicitly that advanced deep learning requires massive datasets, yet devotes a substantial section to deep-learning adaptations for small data (data augmentation, transfer learning, few-shot methods).

> **The tension:** These two positions are reconcilable in principle (DL works for small data *if* pre-trained or heavily regularised), but the paper does not clearly state the **boundary conditions** under which DL is advisable versus when classical/Bayesian methods dominate.

> **Our working resolution:** The boundary is likely a function of (a) availability of a suitable pretrained model, (b) problem structure (is there a natural augmentation symmetry?), and (c) computational budget. We will treat this as an empirical question in our experiments.

### Tension 3 — Causal Discovery with Few Samples

> **The contradiction:** Causal inference is offered as a way to reduce data complexity by finding relevant causal pathways. However, reliably discovering causal structure from very few observations is itself notoriously data-hungry (causal discovery algorithms have poor statistical power at small *n*).

> **The tension:** The paper does not acknowledge this conflict between the promise of causal inference and the data requirements of causal discovery.

> **Our working resolution:** At extremely small *n*, causal ideas are most useful as an **injected prior or constraint** derived from domain knowledge, not as learned structure. Causal discovery from the data itself is probably not viable in our regime without very strong regularisation.

---

## Implications for Experimental Plan

1. **Define the "extremely small" regime precisely** — we will adopt *n* ∈ {20, 50, 100} as our canonical sub-sample sizes.
2. **Stress-test transfer / foundation models** — test TabPFN on out-of-distribution datasets to probe where the amortised prior fails.
3. **Use the benchmark literature** (see Session 03) as a starting point for dataset selection rather than constructing our own ad-hoc benchmark.
4. **Treat uncertainty evaluation as primary** — calibration, conformal coverage, and reliability diagrams will be first-class outputs.

---

*← [Session 01](./session-01_naser2025.md) · [Literature Review Index](./README.md) · Next: [Session 03 →](./session-03_targeted.md)*
