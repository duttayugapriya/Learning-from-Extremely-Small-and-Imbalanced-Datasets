# Session 03 — Targeted Gap-Closing Reading

## Session Metadata

| Field | Value |
|-------|-------|
| **Date** | 2026-11-17 |
| **Duration** | ~1 day |
| **Goal** | Address the four gaps flagged in Session 02: (1) operational definition of "extremely small," (2) pretraining dependence of transfer methods, (3) absence of head-to-head benchmarks, (4) un-operationalised uncertainty quantification |

---

## Papers Read

> **Hollmann, N., et al.** "Accurate predictions on small data with a tabular foundation model."
> *Nature*, 2025.
> *TabPFN — the most consequential recent result for our specific regime.*

> **Erickson, N., et al.** "TabArena: A Living Benchmark for Machine Learning on Tabular Data."
> *NeurIPS*, 2025.
> *The controlled, continuously-updated benchmark the survey lacked.*

> **Larson, M. G.** (Bayesian methods tutorial), 2023.
> (Also: **van de Schoot, R., et al.**, 2015 — Bayesian inference tutorial.)
> *Establishing that Bayesian estimation does not require large-sample asymptotics.*

> **Shen, X., et al.** (2020). Empirical study of causal discovery algorithms at reduced sample sizes.
> *Demonstrating that small n severely degrades the power of causal-structure algorithms.*

---

## Key Learnings

### 1. TabPFN — In-Context Learning as an Amortised Prior (Hollmann et al., 2025)

TabPFN is the most consequential recent result for our research focus. Its design is worth understanding precisely:

**Architecture:** A transformer pre-trained once on approximately 100 million *synthetic* tabular datasets. The synthetic datasets are generated from **structural causal models** (SCMs) rather than from real data. At inference time, TabPFN takes the entire small training set as context in a single forward pass and outputs predictions — there is no per-dataset training, fine-tuning, or gradient computation.

**Empirical performance:**

| Comparison | Result |
|------------|--------|
| vs. tuned XGBoost (datasets up to ~10 k samples, 500 features) | TabPFN dominant |
| vs. tuned CatBoost | TabPFN matches or exceeds; achieves equivalent accuracy with **~half the training samples** |
| Speed | Thousands of times faster than tuned gradient-boosted trees |

**Two findings directly relevant to our project:**

1. **Bayesian interpretation:** TabPFN can be interpreted as approximating **Bayesian posterior-predictive inference** under the SCM-generated synthetic prior. This unifies the "Bayesian methods" and "synthetic data generation" threads from the survey into a single mechanism — the distinction between the two is largely one of implementation.

2. **Native uncertainty quantification:** TabPFN natively outputs a full predictive distribution over classes, providing calibrated uncertainty estimates "for free" — without additional calibration post-processing. This directly addresses Gap 4 (un-operationalised uncertainty).

**Limitation / caution:** TabPFN's capability is a **relocation** of the data requirement: it is "free" for practitioners because someone else paid the pretraining cost (synthetic data generation from SCMs). When the synthetic prior does not match the real data distribution (OOD conditions), the method may degrade. This is a concrete hypothesis to test.

---

### 2. TabArena — The Controlled Benchmark We Needed (Erickson et al., 2025)

TabArena addresses Gap 3 (absence of head-to-head benchmarks) directly. Key design choices:

- **51 rigorously curated datasets** (13 regression, 38 classification), selected for size, diversity, and lack of data leakage
- Standardised preprocessing and consistent hyperparameter tuning budgets across all methods
- **Elo-style rating system** for ranking, which accounts for relative method performance across datasets rather than averaging raw scores

**Current leaderboard highlights:**

| Rank | Method | Family |
|------|--------|--------|
| 1 | TabPFN v2.6 | Tabular foundation model |
| 2 | RealTabPFN | Tabular foundation model |
| 3 | TabICLv2 | Tabular foundation model |
| ~7 | Best boosted-tree method | XGBoost / CatBoost |

The dominance of tabular foundation models over tuned gradient-boosted trees in this controlled setting provides a **defensible baseline stack** for our experimental plan rather than relying on qualitative guesswork.

**Caveat for our work:** TabArena does not specifically isolate the *n* ≈ 20–100 regime. Our experiments should subset the TabArena datasets to these sizes to generate our own controlled comparison within the extremely-small-data regime.

---

### 3. Bayesian Methods at Small n — The Prior Does the Work (Larson 2023; van de Schoot et al., 2015)

These tutorial papers establish a foundational point for our theoretical framing:

> **Bayesian estimation does not rely on large-sample asymptotics.** Posterior inference is valid at any *n*, including *n* = 1, provided the prior and likelihood are correctly specified. At small *n*, the prior dominates the posterior — which is not a failure mode, it is by design.

**Reformulation:** The "small data" problem in a Bayesian frame is not "insufficient data to fit parameters" but **"how much trustworthy prior knowledge can we inject?"** The prior does the work that data cannot.

**Practical implication:** At *n* ≈ 20–100, the choice between an informative and a weakly-informative prior is not a philosophical preference — it is the dominant source of variation in model outputs. Our experiments should explicitly compare these.

---

### 4. Causal Discovery at Small n — The Tension Is Real (Shen et al., 2020)

Shen et al. provide a clean empirical characterisation of the tension I flagged in Session 02:

- Reducing sample size **sharply lowers the statistical power** of causal-discovery algorithms (FCI, FGES tested), inflating spurious "noise edges" in the recovered graph.
- The recommended fix is **more prior/background knowledge and longitudinal structure** — not more algorithmic sophistication.

**Conclusion:** This confirms Session 02's working resolution. Causal inference at extremely small *n* is viable only when heavily constrained by domain knowledge. Causal *discovery* from the data alone is not tractable in our regime.

---

## How the Targeted Reading Resolves Session 02's Tensions

| Session 02 Tension | Resolution |
|--------------------|------------|
| **Synthetic data as both solution and risk** | Real for GAN/VAE augmentation trained on the tiny set itself. TabPFN sidesteps it by generating synthetic data from an *SCM prior* rather than from the target data. The synthetic data trains the learner *offline*, not the target predictor. |
| **Transfer learning relocates, not eliminates, the data requirement** | Confirmed and sharpened. TabPFN's capability is a one-time synthetic pretraining cost that is acceptable when the prior matches, and a risk when it does not. Becoming a **concrete experimental hypothesis**. |
| **Causal discovery is data-hungry** | Confirmed by Shen et al. Causal ideas are most useful in our regime as an *injected prior/constraint*, not as something to discover from the data. |

---

## Sharpened Problem Framing

The unifying insight across all three sessions:

> **The extremely-small-data problem is best framed as maximising the amount of trustworthy prior information injected per sample** — whether through a Bayesian prior, a physics/domain constraint, or an amortised prior baked into a pretrained model like TabPFN.

This gives a **single comparative axis** for the four method families in our experimental plan:

| Method Family | How Prior Is Injected | Prior Trustworthiness Risk |
|---------------|-----------------------|---------------------------|
| Classical baselines (SVM, trees) | Implicit structural prior (inductive bias) | Low — but limited expressiveness |
| Bayesian / GP | Explicit parametric prior | Scales with domain knowledge quality |
| TabPFN (foundation model) | Amortised SCM prior from pretraining | Degrades under distribution shift |
| Domain-knowledge-guided | Hard constraints / physics equations | Scales with domain knowledge quality |

---

## Implications for Experimental Plan

### Datasets
- Use **TabArena** as the source dataset pool (standardised, vetted).
- Subsample each dataset to *n* ∈ {20, 50, 100} with stratified splits to preserve class ratios.
- Select ~5–8 datasets spanning a range of feature counts and class imbalance ratios.

### Method Stack
- **(i) Classical baselines:** Regularised logistic regression (L2), SVM (RBF kernel), random forest — performance floor.
- **(ii) Bayesian / GP:** Gaussian process classifier with RBF kernel; Bayesian logistic regression with informative vs. weakly-informative priors — quantify the prior's contribution.
- **(iii) Foundation model:** TabPFN v2 — amortised-prior approach.
- **(iv) Domain-knowledge-guided:** One simple constraint-encoded model (domain-specific, to be determined by dataset choice).

### Evaluation Metrics
| Metric | What It Measures |
|--------|-----------------|
| Accuracy / AUROC | Point prediction quality |
| Expected Calibration Error (ECE) | Reliability of confidence scores |
| Conformal prediction coverage | Distribution-free coverage guarantee |
| Brier score | Proper scoring rule combining accuracy + calibration |
| Posterior interval width (Bayesian) | Uncertainty quantification quality |

**Conformal prediction** is particularly well-suited to our regime: it provides distribution-free coverage guarantees without large-sample assumptions, and can be layered on top of any point predictor.

### Hypotheses to Test
1. TabPFN will outperform classical baselines at *n* ≤ 50 on standard datasets.
2. The advantage of TabPFN will shrink as datasets move further from tabular/structured data (OOD hypothesis).
3. Informative Bayesian priors will close a substantial fraction of the gap between weak baselines and TabPFN.
4. Calibration (ECE, conformal coverage) will show larger relative differences between methods than point accuracy.

---

*← [Session 02](./session-02_naser2025_deep.md) · [Literature Review Index](./README.md)*
