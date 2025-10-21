# make_figures_ablation.py — Ablation per groups (PDF-first, memory-safe, CSVs)

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

SAVE_PNG = False
PNG_DPI  = 120

os.makedirs("figs", exist_ok=True)
os.makedirs("csv",  exist_ok=True)

DF_PATH = "filtered_dataset_clean.csv"
df = pd.read_csv(DF_PATH)

# label a {0,1}
if df["label"].dtype == object:
    df["label"] = df["label"].map({"benign": 0, "malicious": 1})
y = df["label"].astype(int)
X = df.drop(columns=["id", "label"])

def assign_group(col: str) -> str:
    c = col.lower()
    if "l1d-cache-0" in col:
        return "L1D"
    if "l1i-cache-0" in col:
        return "L1I"
    if "l2-cache-0" in col:
        return "L2"
    if c.startswith(("sim", "host", "finaltick", "numcycles", "committedinsts", "committedops", "simops", "simticks")):
        return "CPU Core"
    return "Other"

groups = {}
for col in X.columns:
    g = assign_group(col)
    groups.setdefault(g, []).append(col)

group_names = [g for g in groups.keys() if len(groups[g]) > 0]

# ---------- evaluation CV ----------
def eval_cv(Xsub: pd.DataFrame, y: pd.Series, k: int = 5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)

    accs, precs, recs, f1s = [], [], [], []
    for train_idx, test_idx in skf.split(Xsub, y):
        Xtr, Xte = Xsub.iloc[train_idx], Xsub.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        # defensa extra (no debería ocurrir con StratifiedKFold)
        if ytr.nunique() < 2 or yte.nunique() < 2:
            accs.append(np.nan); precs.append(np.nan); recs.append(np.nan); f1s.append(np.nan)
            continue

        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)

        rep = classification_report(yte, ypred, output_dict=True)
        accs.append(accuracy_score(yte, ypred))
        precs.append(rep["weighted avg"]["precision"])
        recs.append(rep["weighted avg"]["recall"])
        f1s.append(rep["weighted avg"]["f1-score"])

    return {
        "accuracy": np.nanmean(accs),
        "precision_weighted": np.nanmean(precs),
        "recall_weighted": np.nanmean(recs),
        "f1_weighted": np.nanmean(f1s),
    }

baseline = eval_cv(X, y, k=5)

rows = []
for g in group_names:
    cols = groups[g]
    if len(cols) == 0:
        continue
    metrics = eval_cv(X[cols], y, k=5)
    rows.append({
        "group": g,
        "n_features": len(cols),
        "accuracy": metrics["accuracy"],
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
    })

ablation_df = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)

ablation_df["f1_all_features"] = baseline["f1_weighted"]
ablation_df["delta_f1_vs_all"] = ablation_df["f1_weighted"] - baseline["f1_weighted"]

ablation_df.to_csv("csv/ablation_metrics.csv", index=False)

# ---------- figure: Delta F1 vs ALL ----------
labels = ablation_df["group"].tolist()
deltas = ablation_df["delta_f1_vs_all"].values

fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=110, constrained_layout=True)
x = np.arange(len(labels))
ax.bar(x, deltas)
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=8)
ax.set_ylabel("Δ F1 (group only − all features)")
ax.set_title("Ablation by hardware unit (Random Forest, 5-fold CV)")

fig.savefig("figs/ablation_delta_f1.pdf")
if SAVE_PNG:
    try:
        fig.savefig("figs/ablation_delta_f1.png", dpi=PNG_DPI)
    except Exception as e:
        print(f"[warn] PNG failed for ablation: {e}")
plt.close(fig)

print("\n=== BASELINE (all features, 5-fold CV) ===")
for k, v in baseline.items():
    print(f"{k}: {v:.4f}")

print("\nSaved: csv/ablation_metrics.csv, figs/ablation_delta_f1.pdf")
