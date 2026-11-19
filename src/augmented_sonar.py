import warnings
warnings.filterwarnings("ignore")

import numpy as np
import openml
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from tabpfn import TabPFNClassifier

# 1. LOAD DATA (Sonar, ID 40)
print("Loading Sonar dataset...")
ds = openml.datasets.get_dataset(40)
X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
X = X.select_dtypes(include=[np.number]).to_numpy()
classes = y.unique()
y = (y == classes[0]).astype(int).to_numpy()

# 1.5 ARTIFICIAL IMBALANCE INJECTION (15% minority retained)
class_0_idx = np.where(y == 0)[0]
class_1_idx = np.where(y == 1)[0]
rng_imb = np.random.RandomState(42)
keep_class_1_idx = rng_imb.choice(
    class_1_idx, size=int(len(class_1_idx) * 0.15), replace=False
)
imbalanced_idx = np.concatenate([class_0_idx, keep_class_1_idx])
rng_imb.shuffle(imbalanced_idx)
X, y = X[imbalanced_idx], y[imbalanced_idx]

# 2. FREEZE TEST SET
X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=0.35, stratify=y, random_state=0
)

# 3. ECE FUNCTION
def ece(y_true, proba, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total_ece, N = 0.0, len(y_true)
    for i in range(bins):
        mask = (proba > edges[i]) & (proba <= edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = proba[mask].mean()
        total_ece += (mask.sum() / N) * abs(acc - conf)
    return total_ece

# 4. STRATIFIED SUBSAMPLE
def strat_subsample(Xp, yp, n, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(yp):
        pool = np.where(yp == c)[0]
        k = max(1, round(n * len(pool) / len(yp)))
        idx += list(rng.choice(pool, min(k, len(pool)), replace=False))
    return np.array(idx)

# 5. METHOD FACTORY
def make_methods():
    return {
        "LogReg": lambda: LogisticRegression(C=0.1, max_iter=2000),
        "SVM":    lambda: SVC(C=1.0, probability=True),
        "RF":     lambda: RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0),
        "GP":     lambda: GaussianProcessClassifier(random_state=0),
        "TabPFN": lambda: TabPFNClassifier(device="cpu", n_estimators=4, random_state=0),
    }

# 6. RUN SIDE-BY-SIDE EXPERIMENT (LESSER vs. EXTENDED)
results = defaultdict(list)
ns = [20, 50, 100]
repeats = 20

for n in ns:
    print(f"⏳ Evaluating Lesser (Raw) vs. Extended (SMOTE) at n={n}...")
    for seed in range(repeats):
        tr_idx = strat_subsample(X_pool, y_pool, n, seed)
        if len(np.unique(y_pool[tr_idx])) < 2:
            continue
            
        scaler = StandardScaler().fit(X_pool[tr_idx])
        Xtr_raw = scaler.transform(X_pool[tr_idx])
        ytr_raw = y_pool[tr_idx]
        Xte = scaler.transform(X_test)
        
        n_minority = sum(ytr_raw == 1)
        if n_minority > 1:
            k_neigh = min(5, n_minority - 1)
            smote = SMOTE(k_neighbors=k_neigh, random_state=seed)
            Xtr_aug, ytr_aug = smote.fit_resample(Xtr_raw, ytr_raw)
        else:
            Xtr_aug, ytr_aug = Xtr_raw, ytr_raw
            
        for name, make in make_methods().items():
            try:
                # 1. Lesser Data (Raw Imbalanced)
                m_raw = make().fit(Xtr_raw, ytr_raw)
                p_raw = m_raw.predict_proba(Xte)[:, 1]
                
                # 2. Extended Data (SMOTE Augmented)
                m_aug = make().fit(Xtr_aug, ytr_aug)
                p_aug = m_aug.predict_proba(Xte)[:, 1]
                
                results[(name, n)].append({
                    "raw_auc": average_precision_score(y_test, p_raw),
                    "aug_auc": average_precision_score(y_test, p_aug),
                    "raw_ece": ece(y_test, p_raw),
                    "aug_ece": ece(y_test, p_aug),
                })
            except Exception as e:
                print(f"⚠️ Skip {name} n={n} seed={seed}: {e}")

# 7. SUMMARIZE DELTA TABLE
rows = []
for (name, n), lst in results.items():
    raw_a = np.mean([r["raw_auc"] for r in lst])
    aug_a = np.mean([r["aug_auc"] for r in lst])
    raw_e = np.mean([r["raw_ece"] for r in lst])
    aug_e = np.mean([r["aug_ece"] for r in lst])
    rows.append({
        "method": name,
        "n": n,
        "raw_pr_auc": round(raw_a, 4),
        "smote_pr_auc": round(aug_a, 4),
        "delta_pr_auc": round(aug_a - raw_a, 4),
        "raw_ece": round(raw_e, 4),
        "smote_ece": round(aug_e, 4),
    })

df = pd.DataFrame(rows).sort_values(["n", "smote_pr_auc"], ascending=[True, False])
df.to_csv("phase3_lesser_vs_extended.csv", index=False)

print("\n" + "="*75)
print("✅ PHASE 3: LESSER (RAW) vs. EXTENDED (SMOTE) COMPARISON")
print("="*75)
print(df.to_string(index=False))