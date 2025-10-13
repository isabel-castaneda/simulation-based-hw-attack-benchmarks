import os, re
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

os.makedirs("metrics", exist_ok=True)

# --- Load data ---
df = pd.read_csv("filtered_dataset_clean.csv")
df["label"] = df["label"].map({"benign": 0, "malicious": 1})

X = df.drop(columns=["id", "label"])
y = df["label"].values
ids = df["id"].astype(str)

# Program group: everything before the last "_" (e.g., "loop_sum_100" -> "loop_sum")
def program_key(s):
    parts = s.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else s

groups = ids.apply(program_key).values

# --- Models ---
models = {
    "Random Forest": ("raw", RandomForestClassifier(n_estimators=100, random_state=42)),
    "SVM (scaled)": ("scaled", SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42)),
    "MLP (scaled)": ("scaled", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
}

logo = LeaveOneGroupOut()

summary_rows = []
# store concatenated predictions for ROC/PR later
stacked_scores = {name: [] for name in models}
stacked_truths = {name: [] for name in models}
stacked_program = {name: [] for name in models}

for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    prog = np.unique(groups[test_idx])[0]  # left-out program (one at a time)

    for name, (mode, clf) in models.items():
        if mode == "scaled":
            scaler = StandardScaler()
            X_tr_use = scaler.fit_transform(X_tr)
            X_te_use = scaler.transform(X_te)
        else:
            X_tr_use, X_te_use = X_tr, X_te

        clf.fit(X_tr_use, y_tr)
        y_pred = clf.predict(X_te_use)

        # Metrics
        acc = accuracy_score(y_te, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="weighted", zero_division=0)

        summary_rows.append({
            "program": prog, "model": name, "n_test": len(y_te),
            "accuracy": acc, "precision_w": prec, "recall_w": rec, "f1_w": f1
        })

        # Scores for ROC/PR (probability of class 1)
        if hasattr(clf, "predict_proba"):
            score = clf.predict_proba(X_te_use)[:, 1]
        else:
            # SVM with probability=True covers this, but just in case:
            score = clf.decision_function(X_te_use)
            # rescale to [0,1] in a monotonic way if needed
            score = (score - score.min()) / (score.max() - score.min() + 1e-9)

        stacked_scores[name].append(score)
        stacked_truths[name].append(y_te)
        stacked_program[name].append(np.array([prog]*len(y_te)))

# Save per-program metrics
summary_df = pd.DataFrame(summary_rows).sort_values(["model","program"])
summary_df.to_csv("metrics/lopo_summary.csv", index=False)

# Save overall averages
avg = summary_df.groupby("model")[["accuracy","precision_w","recall_w","f1_w"]].mean().reset_index()
avg.to_csv("metrics/lopo_avg.csv", index=False)

# Save concatenated predictions (for ROC/PR aggregation)
for name in models:
    y_all = np.concatenate(stacked_truths[name])
    s_all = np.concatenate(stacked_scores[name])
    p_all = np.concatenate(stacked_program[name])
    out = pd.DataFrame({"program": p_all, "y_true": y_all, "y_score": s_all})
    out.to_csv(f"metrics/lopo_scores_{name.replace(' ','_').replace('(','').replace(')','')}.csv", index=False)

print("Saved: metrics/lopo_summary.csv, metrics/lopo_avg.csv, metrics/lopo_scores_*.csv")
