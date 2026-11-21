# Experiment 02: Fold-Safe Cross-Conformal Prediction & The Marginal Coverage Illusion

- **Phase:** Phase 2 (Distribution-Free Uncertainty Quantification on Imbalanced Sonar)
- **Date:** 21 November 2026
- **Status:** Completed

## Objective
To test whether fold-safe marginal Cross-Conformal Prediction (target coverage $1 - \alpha = 0.90$) can guarantee reliable minority-class coverage on extremely small, imbalanced data despite high Expected Calibration Error (ECE), and to examine how SMOTE affects conformal coverage and prediction set efficiency.

## Dataset
- **Benchmark:** Imbalanced Sonar (OpenML ID: `40`, 60 numeric features, binary classification)
- **Split & Imbalance (Same Controlled Protocol as Experiment 01):**
  - 125 total samples (111 majority Class 0, 14 minority Class 1 — 11.2% minority, `RandomState(42)`)
  - Stratified 65% / 35% split (`random_state=0`): **81-sample training pool** (72 majority, 9 minority) and **44-sample frozen test set** (39 majority, 5 minority).

## Experimental Setup
- **Requested vs. Actual Training Sample Sizes:**
  - `n = 20` → **20 actual samples** (18 majority, 2 minority)
  - `n = 50` → **50 actual samples** (44 majority, 6 minority)
  - `n = 100` → **81 actual samples** (72 majority, 9 minority — full pool)
- **Models Evaluated (5 families):** L2 Logistic Regression (`C=0.1`), RBF-SVM (`C=1.0`, Platt scaling), Random Forest (`200` trees, `max_depth=4`), Gaussian Process Classifier, and TabPFN (`v2.0.6`, `n_estimators=4`, CPU).
- **Conformal Calibration Protocol:**
  - Nested fold-safe **Marginal Cross-Conformal Prediction** with inner stratified folds $K = \min(5, \max(2, n_1))$.
  - Nonconformity score: $s(x, y) = 1 - \hat{p}(y \mid x)$, using finite-sample corrected quantile $\hat{q}_{\text{marg}} = s_{(\lceil (n+1)(1-\alpha) \rceil)}$ at $\alpha = 0.10$ (90% target coverage).
  - All preprocessing (`StandardScaler`) and optional `SMOTE` (`k_neighbors = min(5, n_min_fold - 1)`) strictly fit inside each inner training fold.
- **Repetitions:** 20 repeated stratified draws per sample size.
- **Evaluation Metrics:** Overall Marginal Coverage (`Cov`), Minority-Class Empirical Coverage (`Min. Cov`), and Average Prediction Set Size (`Set Size` $\in [0, 2]$).

## Results: Marginal Cross-Conformal Prediction (Target $1 - \alpha = 0.90$, 20 Repeats)

| Method | Requested n (Actual) | Raw Overall Cov. | SMOTE Overall Cov. | Raw Minority Cov. | SMOTE Minority Cov. | $\Delta$ Min. Cov. | Raw Set Size | SMOTE Set Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LogReg | 20 (20) | 0.9886 | 0.9886 | 0.90 | 0.90 | 0.00 | 1.7182 | 1.7182 |
| TabPFN | 20 (20) | 0.9818 | 0.9818 | 0.84 | 0.84 | 0.00 | 1.6364 | 1.6364 |
| RF | 20 (20) | 0.9727 | 0.9727 | 0.76 | 0.76 | 0.00 | 1.5636 | 1.5636 |
| GP | 20 (20) | 0.9591 | 0.9591 | 0.64 | 0.64 | 0.00 | 1.5045 | 1.5045 |
| SVM | 20 (20) | 0.9227 | 0.9273 | 0.32 | 0.36 | +0.04 | 1.4841 | 1.5023 |
| GP | 50 (50) | 0.9864 | 0.9636 | 0.88 | 0.84 | -0.04 | 1.4205 | 1.3341 |
| LogReg | 50 (50) | 0.9432 | 0.9545 | 0.50 | 0.62 | +0.12 | 1.1045 | 1.2273 |
| SVM | 50 (50) | 0.9636 | 0.9545 | 0.68 | 0.60 | -0.08 | 1.3727 | 1.1341 |
| TabPFN | 50 (50) | 0.9523 | 0.9500 | 0.58 | 0.56 | -0.02 | 1.1568 | 1.1773 |
| RF | 50 (50) | 0.9227 | 0.9432 | 0.32 | 0.50 | +0.18 | 1.0909 | 1.1432 |
| GP | 100 (81) | 0.9773 | 0.9295 | 0.80 | 0.60 | -0.20 | 1.3409 | 1.0000 |
| SVM | 100 (81) | 0.9318 | 0.9477 | 0.40 | 0.54 | +0.14 | 0.9955 | 1.0114 |
| LogReg | 100 (81) | 0.9318 | 0.8955 | 0.40 | 0.42 | +0.02 | 1.0000 | 0.9909 |
| TabPFN | 100 (81) | 0.9318 | 0.9250 | 0.40 | 0.34 | -0.06 | 1.0000 | 1.0205 |
| RF | 100 (81) | 0.9091 | 0.8977 | 0.20 | 0.14 | -0.06 | 1.0227 | 0.9705 |

### Mean Trend Across Models by Sample Size

| Requested n (Actual) | Mean Raw Overall Cov. | Mean Raw Minority Cov. | Mean Raw Set Size | Mean SMOTE Overall Cov. | Mean SMOTE Minority Cov. | Mean SMOTE Set Size |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 20 (20) | 0.9650 | 0.692 | 1.5814 | 0.9659 | 0.700 | 1.5850 |
| 50 (50) | 0.9536 | 0.592 | 1.2291 | 0.9532 | 0.624 | 1.2032 |
| 100 (81) | 0.9364 | 0.440 | 1.0718 | 0.9191 | 0.408 | 0.9987 |

## Key Observations
1. **The Marginal Coverage Illusion:** As sample size increases from `n=20` to `n=100 (81)`, average prediction sets sharpen from `1.5814` down to `1.0718` (raw), and every raw model clears the `90%` overall coverage guarantee (`0.9091–0.9773` at `n=100`). Yet **minority-class coverage collapses** from a mean of `0.692` at `n=20` down to **`0.440` at `n=100`** (`0.20` for RF; `0.40` for SVM, LogReg, and TabPFN). Because Class 0 makes up 88.8% of the test set, the pooled marginal threshold $\hat{q}_{\text{marg}}$ satisfies aggregate 90% validity almost entirely by covering the majority class while sacrificing rare events.
2. **SMOTE Fails to Rescue Marginal Minority Coverage:** Even with fold-safe SMOTE oversampling, minority coverage at `n=100 (81)` averages only **`0.408`** (worse than raw `0.440`), dropping for GP (`0.80 → 0.60`), TabPFN (`0.40 → 0.34`), and RF (`0.20 → 0.14`), while helping only SVM (`0.40 → 0.54`) and marginally LogReg (`0.40 → 0.42`).
3. **The `n=20` Geometric Floor for In-Fold SMOTE:** At `n=20` (only 2 minority samples total), 2-fold inner cross-conformal splitting leaves only `1` minority point per inner training fold ($n_{1,\text{fold}} < 2$). Because SMOTE requires at least 2 minority neighbors to interpolate, it mathematically falls back to raw training—producing identical conformal metrics for LogReg, TabPFN, RF, and GP (with SVM differing by `0.0046` solely due to internal Platt-scaling CV randomness).
4. **Why GP Appears Higher in Raw Minority Coverage:** GP's high minority coverage at `n=100` (`0.80`) is an artifact of its severe miscalibration (`ECE > 0.40`) forcing wider prediction sets (`1.3409` vs. `~1.00` for other models), rather than stronger discrimination.

## Limitations
- **Marginal-Only Calibration:** Pooling nonconformity scores across both classes cannot provide class-conditional guarantees under class imbalance.
- **Small Test Minority Count:** The frozen Sonar test set contains only 5 minority instances, causing minority coverage estimates to step in increments of `0.20` per repeat.
- **Single Dataset:** Multi-dataset validation is required to test whether the marginal coverage illusion holds across naturally imbalanced domains.

## Next Steps (Phase 3 / Experiment 03)
- **Implement Mondrian (Class-Conditional) Cross-Conformal Prediction:** Calibrate separate thresholds $\hat{q}_0, \hat{q}_1$ per class and empirically test the exact finite-sample feasibility threshold $n_1 \ge \lceil 1/\alpha - 1 \rceil = 9$ for $\alpha = 0.10$.
- **Scale to 10 Additional OpenML Benchmarks:** Evaluate both a 5-dataset severe-skew learning-curve suite (`n = 20, 50, 100, 200`) and a 5-dataset natively tiny validation suite (`N = 100–306`), backed by paired Wilcoxon signed-rank and Friedman–Nemenyi statistical tests.

---

## Artifacts & Reproducibility
- **Evaluation Script:** [`src/conformal.py`](../src/conformal.py)
- **Conformal Prediction Results Table:** [`results/phase2/phase4_conformal_results.csv`](../results/phase2/phase4_conformal_results.csv)

---

*Back to [Repository Root](../README.md)*
