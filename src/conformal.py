import warnings
warnings.filterwarnings("ignore")

import numpy as np
import openml
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from imblearn.over_sampling import SMOTE
from tabpfn import TabPFNClassifier

# 1. LOAD DATA & INJECT IMBALANCE
print("Loading Sonar dataset...")
ds = openml.datasets.get_dataset(40)
X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
X = X.select_dtypes(include=[np.number]).to_numpy()
classes = y.unique()
y = (y == classes[int("0")]).astype(int).to_numpy()

c0 = np.where(y == int("0"))[int("0")]
c1 = np.where(y == int("1"))[int("0")]
rng_imb = np.random.RandomState(int("42"))
keep_c1 = rng_imb.choice(c1, size=int(len(c1) * float("0.15")), replace=False)
imb_idx = np.concatenate([c0, keep_c1])
rng_imb.shuffle(imb_idx)
X, y = X[imb_idx], y[imb_idx]

# 2. FREEZE TEST SET
X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=float("0.35"), stratify=y, random_state=int("0")
)

# 3. STRATIFIED SUBSAMPLE
def strat_subsample(Xp, yp, n, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(yp):
        pool = np.where(yp == c)[int("0")]
        k = max(int("1"), round(n * len(pool) / len(yp)))
        idx += list(rng.choice(pool, min(k, len(pool)), replace=False))
    return np.array(idx)

# 4. METHOD FACTORY
def make_methods():
    return {
        "LogReg": lambda: LogisticRegression(C=float("0.1"), max_iter=int("2000")),
        "SVM":    lambda: SVC(C=float("1.0"), probability=True),
        "RF":     lambda: RandomForestClassifier(n_estimators=int("200"), max_depth=int("4"), random_state=int("0")),
        "GP":     lambda: GaussianProcessClassifier(random_state=int("0")),
        "TabPFN": lambda: TabPFNClassifier(device="cpu", n_estimators=int("4"), random_state=int("0")),
    }

# 5. CROSS-CONFORMAL PREDICTION HELPER (Target Coverage = 1 - alpha)
def evaluate_conformal(make_fn, Xtr, ytr, Xte, yte, use_smote=False, alpha=float("0.10"), seed=int("0")):
    n_min = int(np.sum(ytr == int("1")))
    n_splits = min(int("5"), max(int("2"), n_min))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    oof_proba = np.zeros((len(ytr), int("2")))
    test_probas = []
    
    for tr_f, cal_f in skf.split(Xtr, ytr):
        X_f, y_f = Xtr[tr_f], ytr[tr_f]
        if use_smote:
            n_min_f = int(np.sum(y_f == int("1")))
            if n_min_f > int("1"):
                k_n = min(int("5"), n_min_f - int("1"))
                X_f, y_f = SMOTE(k_neighbors=k_n, random_state=seed).fit_resample(X_f, y_f)
        m = make_fn().fit(X_f, y_f)
        oof_proba[cal_f] = m.predict_proba(Xtr[cal_f])
        test_probas.append(m.predict_proba(Xte))
        
    mean_test_proba = np.mean(test_probas, axis=int("0"))
    cal_scores = float("1.0") - oof_proba[np.arange(len(ytr)), ytr]
    n_cal = len(cal_scores)
    q_level = min(float("1.0"), np.ceil((n_cal + int("1")) * (float("1.0") - alpha)) / n_cal)
    q_hat = np.quantile(cal_scores, q_level, method="higher")
    
    pred_sets = (float("1.0") - mean_test_proba) <= q_hat
    overall_cov = pred_sets[np.arange(len(yte)), yte].mean()
    min_mask = (yte == int("1"))
    min_cov = pred_sets[min_mask, int("1")].mean()
    set_size = pred_sets.sum(axis=int("1")).mean()
    return overall_cov, min_cov, set_size

# 6. RUN CONFORMAL EXPERIMENT
results = defaultdict(list)
ns = [int("20"), int("50"), int("100")]
repeats = int("10")

for n in ns:
    print(f"⏳ Running Conformal Prediction at n={n}...")
    for seed in range(repeats):
        tr_idx = strat_subsample(X_pool, y_pool, n, seed)
        if int(np.sum(y_pool[tr_idx] == int("1"))) < int("2"):
            continue
        scaler = StandardScaler().fit(X_pool[tr_idx])
        Xtr = scaler.transform(X_pool[tr_idx])
        ytr = y_pool[tr_idx]
        Xte = scaler.transform(X_test)
        
        for name, make in make_methods().items():
            try:
                cov_r, mcov_r, sz_r = evaluate_conformal(make, Xtr, ytr, Xte, y_test, use_smote=False, seed=seed)
                cov_s, mcov_s, sz_s = evaluate_conformal(make, Xtr, ytr, Xte, y_test, use_smote=True, seed=seed)
                results[(name, n)].append({
                    "raw_cov": cov_r, "smote_cov": cov_s,
                    "raw_min_cov": mcov_r, "smote_min_cov": mcov_s,
                    "raw_size": sz_r, "smote_size": sz_s
                })
            except Exception as e:
                print(f"⚠️ Skip {name} n={n} seed={seed}: {e}")

# 7. SUMMARIZE PHASE 4 TABLE
rows = []
for (name, n), lst in results.items():
    rows.append({
        "method": name,
        "n": n,
        "raw_cov": round(np.mean([r["raw_cov"] for r in lst]), int("4")),
        "smote_cov": round(np.mean([r["smote_cov"] for r in lst]), int("4")),
        "raw_min_cov": round(np.mean([r["raw_min_cov"] for r in lst]), int("4")),
        "smote_min_cov": round(np.mean([r["smote_min_cov"] for r in lst]), int("4")),
        "raw_set_size": round(np.mean([r["raw_size"] for r in lst]), int("4")),
        "smote_set_size": round(np.mean([r["smote_size"] for r in lst]), int("4")),
    })

df = pd.DataFrame(rows).sort_values(["n", "smote_min_cov"], ascending=[True, False])
df.to_csv("phase4_conformal_results.csv", index=False)

print("\n" + "=" * int("80"))
print("✅ PHASE 4: CONFORMAL COVERAGE & PREDICTION SET SIZE")
print("=" * int("80"))
print(df.to_string(index=False))