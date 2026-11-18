# Session 01 — Initial Survey Read

## Session Metadata

| Field | Value |
|-------|-------|
| **Date** | 2026-11-15 |
| **Duration** | ~1 day |
| **Sections read** | Abstract, Introduction, Literature Review |
| **Goal** | Build a high-level map of the small-data ML field |

---

## Papers Read

> **Naser, M. Z., et al.** "A review of machine learning with small and limited data."
> *Journal of Big Data*, 2025.
> — Comprehensive survey of the state of the art in machine learning under data-scarcity constraints.

---

## Key Learnings

### 1. Central Framing: Maximum Information per Sample

The authors frame small-data learning as a problem where the goal **shifts from pure statistical fitting to extracting the maximum information from each data point** while constraining model complexity to avoid overfitting. This reframing is useful because it moves the conversation away from "get more data" toward "make each datum count more."

### 2. Taxonomy: Actionable Intervention Points

The survey organises the field not by algorithm family but by **where in the ML pipeline** one can intervene. The taxonomy consists of five axes:

| Intervention Point | Description |
|--------------------|-------------|
| Theoretical foundations | Bayesian/nonparametric frameworks, information geometry, identifiability |
| Data-related frameworks | Augmentation, synthetic generation, active learning |
| Algorithmic architectures | Transfer learning, meta-learning, model compression |
| Evaluation methods | Small-sample-appropriate validation, resampling, information criteria |
| Mitigation strategies | Regularisation, early stopping, ensemble methods |

This "intervention-point" framing proved more actionable than a flat algorithm listing.

### 3. Methods Landscape

**Classical methods that remain competitive under data scarcity:**

- **SVMs** — the support-vector mechanism naturally emphasises a small number of boundary-defining examples, making it a good structural fit for few samples.
- **Gaussian processes** — full nonparametric Bayesian treatment; yields uncertainty estimates without large-sample asymptotics.
- **Tree ensembles** — robust to noise; resist overfitting through aggregation.
- **Nearest-neighbour classifiers** — non-parametric; no distributional assumptions.
- **Bayesian / nonparametric methods** — Dirichlet processes, variational inference, kernel methods.

**Deep-learning adaptations for small data:**

- Transfer learning (fine-tune a network pretrained on large data)
- Data augmentation (domain-specific transformations)
- Synthetic data generation: **GANs**, **VAEs**, **conditional GANs**, **diffusion models**

**Emerging directions flagged as especially relevant:**

- Embedding **domain knowledge and symbolic reasoning** into model structure
- **Causal inference** to remove spurious correlations and reduce effective sample complexity
- **Lifelong / online learning** for sequential adaptation with few labelled instances

### 4. Evaluation as a First-Class Concern

A recurring theme: **standard statistical metrics can be misleading when sample sizes are very small**, leading to overconfidence in brittle models. The authors treat small-sample-appropriate evaluation as a research problem in its own right within their taxonomy — not merely a downstream concern.

---

## Gaps Identified

*To be elaborated in Session 02 after completing the full paper.*

Preliminary flags from this session:
- No precise, operational definition of "extremely small" vs. "small" vs. "limited" data.
- Transfer learning and meta-learning are presented positively; the dependence on large *source* datasets is not scrutinised.
- No controlled head-to-head empirical comparisons — the survey synthesises the literature rather than benchmarking it.

---

## Contradictions / Tensions

*To be elaborated in Session 02.*

Preliminary observation:
- Synthetic data generation is presented as a solution to data scarcity, yet generative models are themselves data-hungry — a circularity worth examining.

---

## Implications for Experimental Plan

- The intervention-point taxonomy gives a natural axis for organising experiments.
- The field-level emphasis on evaluation under small *n* confirms that **calibration and uncertainty quality** must be primary metrics, not secondary ones.
- The methods landscape suggests the following natural groupings for a comparative study: (a) strong classical baselines, (b) Bayesian / GP methods, (c) deep-learning adaptations, (d) domain-knowledge-guided approaches.

---

*← [Literature Review Index](./README.md) · Next: [Session 02 →](./session-02_naser2025_deep.md)*
