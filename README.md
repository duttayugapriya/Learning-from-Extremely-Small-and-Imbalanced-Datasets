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
├── experiments/                    # Reproducible experiment logs & benchmark reports
│   ├── exp01_raw_vs_smote_sonar.md # Phase 1: Raw data vs SMOTE on imbalanced Sonar
│   └── exp02_cross_conformal_sonar.md # Phase 2: Cross-conformal prediction & marginal coverage illusion
│
├── src/                            # Experiment runner scripts & pipelines
│   ├── augmented_sonar.py          # Phase 1 runner: Lesser vs. Extended SMOTE
│   └── conformal.py                # Phase 2 runner: Fold-safe cross-conformal prediction
│
├── results/                        # Curated experimental benchmark metrics & CSVs
│   ├── phase1/                     # Phase 1 results (PR-AUC, ECE comparisons)
│   └── phase2/                     # Phase 2 results (Cross-conformal coverage & set sizes)
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

**Evaluation metrics:** Point accuracy, calibration (ECE), conformal prediction coverage, Brier score, PR-AUC.

---

## 🔬 Experiment Log & Progress

Structured records of experimental phases, empirical findings, and validation studies:

| Phase | Experiment | Date | Target Benchmark | Status | Key Focus |
|---|---|---|---|---|---|
| Phase 1 | [Exp 01: Raw Data vs. SMOTE](./experiments/exp01_raw_vs_smote_sonar.md) | 2026-11-19 | Imbalanced Sonar (*n*=20, 50, 81) | ✅ Completed | Minority PR-AUC & ECE calibration under synthetic augmentation |
| Phase 2 | [Exp 02: Cross-Conformal Prediction](./experiments/exp02_cross_conformal_sonar.md) | 2026-11-21 | Imbalanced Sonar (*n*=20, 50, 81) | ✅ Completed | Marginal coverage illusion, SMOTE geometric floor & set efficiency |

### Headline Findings: Experiment 01 (Imbalanced Sonar Benchmark)
- **SVM Regime Reversal:** At extreme data scarcity ($n=20$, only 2 minority points), raw RBF-SVM collapses (0.2594 PR-AUC) and fold-safe SMOTE rescues it (+0.2786 to 0.5380); however, once sufficient real minority points exist ($n=100 / 81$ actual), SMOTE severely harms RBF-SVM performance (-0.1602 PR-AUC).
- **Tree Ensembles vs. Bayesian Prior Distortion:** Random Forest is the only model that consistently benefits from SMOTE across all sample sizes (up to +0.2009 PR-AUC), whereas TabPFN achieves superior discrimination on raw imbalanced data at $n=20$ and $n=81$, indicating synthetic SMOTE interpolation distorts its pre-trained amortized Bayesian prior.
- **The Calibration Penalty:** Synthetic oversampling degrades probability calibration across well-calibrated classifiers—pushing predicted minority probabilities upward and sharply worsening ECE (Random Forest ECE rises by +0.1011 to 0.1442 at $n=81$; TabPFN degrades from 0.0575 to 0.0775).

### Headline Findings: Experiment 02 (Cross-Conformal Prediction & Marginal Coverage Illusion)
- **The Marginal Coverage Illusion:** While overall empirical coverage clears the nominal 90% target across all models ($0.9091$–$0.9773$ at $n=81$), minority-class coverage collapses to an average of $0.440$ ($0.20$ for Random Forest; $0.40$ for SVM, LogReg, and TabPFN). Because Class 0 makes up 88.8% of the test set, pooled marginal nonconformity thresholds satisfy aggregate validity almost entirely by covering the majority class while sacrificing rare minority instances as prediction sets sharpen toward size 1.0.
- **SMOTE Fails to Rescue Marginal Minority Coverage:** Even with fold-safe SMOTE oversampling, minority coverage at $n=81$ averages only $0.408$ (worse than raw $0.440$), degrading Gaussian Process ($0.80 \to 0.60$), TabPFN ($0.40 \to 0.34$), and Random Forest ($0.20 \to 0.14$).
- **The $n=20$ Geometric Floor & Need for Mondrian Calibration:** At $n=20$ (2 minority points total), 2-fold inner cross-conformal splitting leaves only 1 minority sample per fold, causing SMOTE to fail interpolation and mathematically fall back to raw training. To guarantee reliable minority-class coverage under extreme imbalance, Mondrian (class-conditional) conformal calibration ($\hat{q}_0, \hat{q}_1$) is strictly required, demanding at least $n_1 \ge \lceil 1/\alpha - 1 \rceil = 9$ minority points for 90% coverage.

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
| Experimental design | ✅ Complete |
| Phase 1: Raw vs. SMOTE Baseline (Exp 01) | ✅ Complete |
| Phase 2: Uncertainty Quantification (Exp 02) | ✅ Complete |
| Phase 3: Multi-Dataset Scaling (Exp 03) | ⬜ Planned |

---

*Last updated: 21 November 2026*
