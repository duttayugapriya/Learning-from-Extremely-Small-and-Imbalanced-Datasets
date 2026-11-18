# Extremely Small Imbalanced Learning

> A structured research repository documenting the investigation into machine learning methods for **extremely small and class-imbalanced datasets** — typically in the tens-of-samples regime where standard statistical assumptions break down.

---

## 📌 Research Focus

This project investigates the intersection of two compounding challenges in real-world machine learning:

1. **Extreme data scarcity** — training sets in the range of *n* ≈ 20–100 samples
2. **Class imbalance** — skewed label distributions that further reduce the effective sample size for minority classes

The central research question is:

> *How much trustworthy prior information — Bayesian, domain-driven, or amortised via pretraining — must be injected per sample to achieve reliable prediction and well-calibrated uncertainty in the extremely-small-data regime?*

---

## 🗂️ Repository Structure

```
extremely-small-imbalanced-learning/
│
├── literature-review/              # Annotated reading log & synthesis
│   ├── README.md                   # Overview & navigation guide
│   ├── session-01_naser2025.md     # Day 1–2: Initial survey read (Naser et al., 2025)
│   ├── session-02_naser2025_deep.md # Day 3: Deep read + gap & tension analysis
│   └── session-03_targeted.md      # Day 4: Targeted gap-closing papers
│
├── experiments/                    # Reproducible experiment notebooks & scripts
│   └── .gitkeep
│
├── notes/                          # Raw working notes, drafts, diagrams
│   └── .gitkeep
│
└── README.md                       # This file
```

---

## 📚 Literature Review

The [`literature-review/`](./literature-review/) folder contains a chronological reading log structured as *research sessions*. Each session document captures:

- **Papers read** with full citations
- **Key learnings** and conceptual contributions
- **Gaps** relative to the research focus
- **Contradictions / tensions** in the literature
- **Implications** for the experimental plan

| Session | Focus | Key Sources |
|---------|-------|-------------|
| [Session 01](./literature-review/session-01_naser2025.md) | Survey overview — taxonomy & methods landscape | Naser et al. (2025) |
| [Session 02](./literature-review/session-02_naser2025_deep.md) | Deep read — gaps & tensions | Naser et al. (2025) |
| [Session 03](./literature-review/session-03_targeted.md) | Targeted reading — gap closing | Hollmann et al. (2025), Erickson et al. (2025), Larson (2023), Shen et al. (2020) |

---

## 🧪 Experimental Plan

A controlled comparison anchored to the **extremely-small-data regime** (*n* ≈ 20–100), using the smallest datasets from TabArena / OpenML subsampled to this range.

**Four method families under evaluation:**

| Family | Representative Methods | Role |
|--------|------------------------|------|
| Classical baselines | Regularised linear, SVM, random forests | Performance floor |
| Bayesian / GP | Gaussian processes, Bayesian regression (informative vs. weak priors) | Quantify the prior's value |
| Foundation models | TabPFN v2 | Amortised-prior approach |
| Domain-knowledge-guided | Constraint-encoded models | Explicit prior injection |

**Evaluation metrics:** Point accuracy, calibration (ECE), conformal prediction coverage, Brier score.

---

## 📖 Citation

If you reference this work, please cite the primary survey that anchored the literature review:

```bibtex
@article{naser2025review,
  title   = {A review of machine learning with small and limited data},
  author  = {Naser, M. Z. and others},
  journal = {Journal of Big Data},
  year    = {2025}
}
```

---

## 🔖 Status

| Milestone | Status |
|-----------|--------|
| Literature review — initial survey | ✅ Complete |
| Literature review — gap analysis | ✅ Complete |
| Literature review — targeted reading | ✅ Complete |
| Experimental design | 🔄 In progress |
| Experiments | ⬜ Planned |

---

*Last updated: November 2026*
