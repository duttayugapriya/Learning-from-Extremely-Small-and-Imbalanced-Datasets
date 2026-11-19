# Experiment 01: Raw Data vs. SMOTE on Extremely Small & Imbalanced Data

- **Phase:** Phase 1 (Imbalanced Baseline & Synthetic Augmentation A/B Study)
- **Date:** 18–19 November 2026
- **Status:** Completed

## Objective
To investigate how classical, Bayesian, and tabular foundation models behave under severe class imbalance in the tens-of-samples regime, and to test whether fold-safe SMOTE augmentation improves minority-class discrimination (PR-AUC) or hurts probability calibration (ECE) compared to training on raw ("lesser") imbalanced data.

## Dataset
- **Benchmark:** Sonar (OpenML ID: `40`, 60 numeric features, binary classification)
- **Artificial Imbalance Injection:** Retained 100% of Class 0 (111 samples) and 15% of Class 1 (14 samples), seeded with `RandomState(42)` → **125 total samples** (88.8% majority, 11.2% minority).
- **Train/Test Split:** Stratified 65% / 35% split (`random_state=0`):
  - **Training Pool:** 81 samples (72 majority, 9 minority)
  - **Frozen Test Set:** 44 samples (39 majority, 5 minority)

## Experimental Setup
- **Requested vs. Actual Training Sample Sizes (Verified):**
  - `n = 20` → **20 actual samples** (18 majority, 2 minority)
  - `n = 50` → **50 actual samples** (44 majority, 6 minority)
  - `n = 100` → **81 actual samples** (72 majority, 9 minority — capped by the 81-sample imbalanced training pool)
- **Models Evaluated (5 families):**
  - L2 Logistic Regression (`C=0.1`, `max_iter=2000`)
  - RBF-SVM (`C=1.0`, `probability=True`)
  - Random Forest (`n_estimators=200`, `max_depth=4`, `random_state=0`)
  - Gaussian Process Classifier (`random_state=0`)
  - TabPFN (`v2.0.6`, `device="cpu"`, `n_estimators=4`, `random_state=0`)
- **Leakage Prevention:** `StandardScaler` and `SMOTE` (`k_neighbors = min(5, n_minority - 1)`) were strictly fit inside each training draw; the 44-sample test set remained untouched.
- **Repetitions:** 20 repeated stratified draws per sample size.
- **Evaluation Metrics:**
  - **PR-AUC** (`average_precision_score`) for imbalanced class discrimination
  - **ECE** (10-bin Expected Calibration Error) for probability reliability
- **Comparison:** Raw imbalanced training data ("Lesser") vs. SMOTE-augmented training data ("Extended").

## Reference Context: Balanced Small-Data Baseline (ROC-AUC)
*(Prior baseline on balanced Sonar before imbalance injection, showing model-ranking shifts across `n`)*

| Method | n = 20 ROC-AUC (± std) | n = 50 ROC-AUC (± std) | n = 100 ROC-AUC (± std) |
| :--- | :---: | :---: | :---: |
| Random Forest | 0.828 ± 0.034 | 0.865 ± 0.026 | 0.907 ± 0.018 |
| RBF-SVM | 0.672 ± 0.246 | 0.853 ± 0.036 | 0.928 ± 0.013 |
| Logistic Reg. (L2) | 0.807 ± 0.042 | 0.836 ± 0.022 | 0.856 ± 0.015 |
| Gaussian Process | 0.742 ± 0.043 | 0.791 ± 0.054 | 0.879 ± 0.026 |

## Results: Imbalanced Sonar — Raw vs. SMOTE (20 Repeats)

| Method | Requested n (Actual) | Raw PR-AUC | SMOTE PR-AUC | Δ PR-AUC | Raw ECE | SMOTE ECE | Δ ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LogReg | 20 (20) | 0.5512 | 0.5517 | +0.0005 | 0.0665 | 0.1019 | +0.0354 |
| SVM | 20 (20) | 0.2594 | 0.5380 | +0.2786 | 0.0965 | 0.0686 | -0.0279 |
| GP | 20 (20) | 0.3826 | 0.4736 | +0.0911 | 0.4000 | 0.4034 | +0.0034 |
| RF | 20 (20) | 0.3863 | 0.4372 | +0.0509 | 0.0553 | 0.0897 | +0.0344 |
| TabPFN | 20 (20) | 0.4274 | 0.4091 | -0.0183 | 0.0653 | 0.0882 | +0.0229 |
| SVM | 50 (50) | 0.7078 | 0.6910 | -0.0168 | 0.0709 | 0.0661 | -0.0048 |
| GP | 50 (50) | 0.5944 | 0.6770 | +0.0826 | 0.4340 | 0.4226 | -0.0114 |
| LogReg | 50 (50) | 0.6186 | 0.6030 | -0.0155 | 0.0608 | 0.1029 | +0.0421 |
| RF | 50 (50) | 0.4454 | 0.5968 | +0.1513 | 0.0646 | 0.1178 | +0.0532 |
| TabPFN | 50 (50) | 0.5589 | 0.5955 | +0.0366 | 0.0687 | 0.0895 | +0.0208 |
| SVM | 100 (81) | 0.9250 | 0.7648 | -0.1602 | 0.0662 | 0.0634 | -0.0028 |
| TabPFN | 100 (81) | 0.7738 | 0.7383 | -0.0355 | 0.0575 | 0.0775 | +0.0200 |
| GP | 100 (81) | 0.7033 | 0.7031 | -0.0002 | 0.4543 | 0.4305 | -0.0238 |
| RF | 100 (81) | 0.4679 | 0.6688 | +0.2009 | 0.0431 | 0.1442 | +0.1011 |
| LogReg | 100 (81) | 0.6283 | 0.6028 | -0.0255 | 0.0441 | 0.0613 | +0.0172 |

*(Note: In the SMOTE-augmented baseline runs, PR-AUC standard deviations across the 20 repeats narrowed sharply as sample size grew — e.g., at `n=20` vs `n=100 (81)`: LogReg `±0.1206 → ±0.0173`, SVM `±0.1654 → ±0.0127`, GP `±0.1500 → ±0.0241`, RF `±0.1589 → ±0.0759`, TabPFN `±0.1529 → ±0.0474`.)*

## Key Observations
1. **SVM Regime Reversal:** At `n=20` (only 2 minority points), raw RBF-SVM collapses (`0.2594` PR-AUC), and SMOTE rescues it by `+0.2786` (to `0.5380`), supplying spatial anchors for the RBF boundary. However, once enough real minority points exist at `n=100 (81)`, SMOTE severely degrades SVM by `-0.1602` (`0.9250 → 0.7648`).
2. **Tree Ensembles Consistently Benefit from Interpolation:** Random Forest is the only model that gains from SMOTE across every sample size (`+0.0509` at `n=20`, `+0.1513` at `n=50`, `+0.2009` at `n=100 (81)`), though at the cost of a steep calibration penalty.
3. **Prior Distortion in TabPFN:** TabPFN performs better on raw imbalanced data at both `n=20` (`0.4274` vs `0.4091`, `-0.0183`) and `n=100 (81)` (`0.7738` vs `0.7383`, `-0.0355`), helping only moderately at `n=50` (`+0.0366`). Collinear synthetic SMOTE points appear to distort its pre-trained amortized Bayesian prior.
4. **Regularization Stability:** Strongly regularized L2 Logistic Regression leads all models at `n=20` (`0.5512` raw / `0.5517` SMOTE) and is nearly invariant to augmentation at extreme scarcity (`+0.0005`), confirming that tight complexity constraints anchor the bias-variance trade-off.
5. **The Calibration Penalty of SMOTE:** Balancing the training fold pushes predicted minority probabilities upward, consistently worsening ECE for well-calibrated models: at `n=100 (81)`, RF ECE degrades from `0.0431 → 0.1442` (`+0.1011`), TabPFN from `0.0575 → 0.0775` (`+0.0200`), and LogReg from `0.0441 → 0.0613` (`+0.0172`). Gaussian Process remains severely miscalibrated across all runs (`ECE 0.4000–0.4543`).

## Limitations
- **Training Pool Cap at `n = 100`:** Because retaining 15% of Class 1 leaves 125 total rows and an 81-row training pool, the `n = 100` setting actually draws all 81 training samples (72 majority, 9 minority), reducing variance across repeats at `n = 100`.
- **Fixed Small Test Set:** All 20 repeats evaluate against the same frozen 44-sample test set (only 5 minority instances), making PR-AUC and ECE sensitive to single-sample flips.
- **Single Dataset & Artificial Imbalance:** Findings are currently limited to one high-dimensional dataset (`p = 60`) with synthetic imbalance rather than naturally skewed multi-domain benchmarks.

## Next Steps
- **Uncertainty Quantification (Experiment 02):** Implement fold-safe Cross-Conformal Prediction to test whether distribution-free prediction sets can protect minority-class coverage despite high ECE.
- **Multi-Dataset Scaling (Experiment 03):** Scale the pipeline across 5 naturally imbalanced OpenML benchmarks (`blood-transfusion`, `kc1`, `climate-crashes`, `pc1`, `ozone-8hr`) with larger frozen test sets and compare Marginal vs. Mondrian (class-conditional) conformal calibration.

---

## Artifacts & Reproducibility
- **Evaluation Script:** [`src/augmented_sonar.py`](../src/augmented_sonar.py)
- **Raw vs. SMOTE Comparison Table:** [`results/phase1/phase3_lesser_vs_extended.csv`](../results/phase1/phase3_lesser_vs_extended.csv)
- **Baseline Metric Statistics:** [`results/phase1/week1_results_sonar_imbalanced.csv`](../results/phase1/week1_results_sonar_imbalanced.csv)

---

*Back to [Repository Root](../README.md)*
