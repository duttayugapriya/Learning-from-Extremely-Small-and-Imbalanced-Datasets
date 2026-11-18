# Literature Review — Navigation Guide

This folder contains a chronological log of research sessions conducted as part of the *Extremely Small Imbalanced Learning* project. Each session is a self-contained document following a consistent schema: citation, key learnings, identified gaps, contradictions / tensions, and implications for the experimental plan.

---

## Reading Log

| # | Session File | Date | Papers Covered | Focus |
|---|-------------|------|----------------|-------|
| 1 | [session-01_naser2025.md](./session-01_naser2025.md) | 2026-11-15 | Naser et al. (2025) | First pass — abstract, intro, lit-review sections |
| 2 | [session-02_naser2025_deep.md](./session-02_naser2025_deep.md) | 2026-11-16 | Naser et al. (2025) | Close reading — full paper; gap & tension analysis |
| 3 | [session-03_targeted.md](./session-03_targeted.md) | 2026-11-17 | Hollmann et al. (2025); Erickson et al. (2025); Larson (2023); Shen et al. (2020) | Targeted gap-closing reading |

---

## Synthesis: Running Themes

The following themes recur across sessions and constitute the conceptual backbone of the project.

### 1. Prior Information as the Fundamental Currency

The unifying insight that emerged over the three sessions is that the extremely-small-data problem is best framed as **maximising the amount of trustworthy prior information injected per sample** — whether through a Bayesian prior, a physics/domain constraint, or an amortised prior baked into a pretrained model (e.g. TabPFN). This gives a single comparative axis for otherwise disparate methods.

### 2. Evaluation Is a First-Class Problem

Standard statistical metrics overfit to in-sample variance when *n* is very small, producing overconfident estimates of model quality. Reliable small-sample evaluation (resampling, information criteria, conformal prediction) is as important as the modelling choice itself.

### 3. The Synthetic-Data Paradox

Generative augmentation (GANs/VAEs/diffusion) is promoted as a remedy for data scarcity, yet these generators must themselves be trained on data. When the original dataset is tiny, over-resampling or synthesis can amplify noise rather than signal. TabPFN's approach — generating synthetic data from a structural causal model *prior* rather than from the target data — sidesteps this paradox.

### 4. Transfer Learning Relocates, Not Eliminates, the Data Requirement

Transfer and meta-learning methods that perform well at small *n* typically rely on large source datasets. This is a legitimate engineering trade when a suitable pretrained model exists, but it is not a true reduction in sample complexity — an important distinction when no suitable source domain is available.

### 5. Causal Discovery Is Data-Hungry

Causal inference is proposed as a way to reduce data requirements by identifying relevant pathways, but recovering causal structure from very few observations is itself statistically demanding. At extremely small *n*, causal ideas are most useful as *injected constraints* from domain knowledge, not as learned structure.

---

## Document Schema

Each session file follows this structure:

```
## Session Metadata
## Papers Read
## Key Learnings
## Gaps Identified
## Contradictions / Tensions
## Implications for Experimental Plan
```

---

*Navigate back to [repository root](../README.md)*
