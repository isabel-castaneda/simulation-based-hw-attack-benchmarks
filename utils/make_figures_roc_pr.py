import os, gc


os.makedirs("figs", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.environ["MPLCONFIGDIR"] = os.path.abspath("./.mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

PLOTS = os.environ.get("PLOTS", "1") == "1"

df = pd.read_csv("filtered_dataset_clean.csv")
df["label"] = df["label"].map({"benign": 0, "malicious": 1})
X = df.drop(columns=["id", "label"])
y = df["label"].values

# --- Models (SVM without probability to safe RAM) ---
models = {
    "Random Forest": ("raw",  RandomForestClassifier(n_estimators=100, random_state=42)),
    "SVM (scaled)":  ("scaled", SVC(kernel="rbf", C=1, gamma="scale", probability=False, random_state=42)),
    "MLP (scaled)":  ("scaled", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
}

# ---------- A) HOLDOUT ----------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

roc_aucs = []
pr_aps  = []
roc_series = []
pr_series  = []

# Training once per model
for name, (mode, clf) in models.items():
    if mode == "scaled":
        scaler = StandardScaler()
        X_tr_use = scaler.fit_transform(X_tr)
        X_te_use = scaler.transform(X_te)
    else:
        X_tr_use, X_te_use = X_tr, X_te

    clf.fit(X_tr_use, y_tr)

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_te_use)[:, 1]
    else:
        scores = clf.decision_function(X_te_use)

    fpr, tpr, _ = roc_curve(y_te, scores)
    roc_auc = auc(fpr, tpr)
    roc_aucs.append((name, roc_auc))
    roc_series.append((name, fpr, tpr))

    prec, rec, _ = precision_recall_curve(y_te, scores)
    ap = average_precision_score(y_te, scores)
    pr_aps.append((name, ap))
    pr_series.append((name, rec, prec))

# Saving metrics
pd.DataFrame(roc_aucs, columns=["model", "AUC"]).to_csv("metrics/roc_holdout_auc.csv", index=False)
pd.DataFrame(pr_aps,  columns=["model", "AP"]).to_csv("metrics/pr_holdout_ap.csv", index=False)

if PLOTS:
    # ROC holdout
    plt.figure(figsize=(3.8, 2.9))
    for name, fpr, tpr in roc_series:
        auc_val = [x[1] for x in roc_aucs if x[0] == name][0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})", linewidth=1.4)
    plt.plot([0,1], [0,1], "--", linewidth=1.0)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — Holdout"); plt.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig("figs/roc_holdout.pdf"); plt.close(); gc.collect()

    # PR holdout
    plt.figure(figsize=(3.8, 2.9))
    for name, rec, prec in pr_series:
        ap_val = [x[1] for x in pr_aps if x[0] == name][0]
        plt.plot(rec, prec, label=f"{name} (AP={ap_val:.2f})", linewidth=1.4)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall — Holdout"); plt.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig("figs/pr_holdout.pdf"); plt.close(); gc.collect()

# ---------- B) LOPOCV (adding metrics saved in train_lopocv.py) ----------
def load_lopo_scores(stub):
    path = f"metrics/lopo_scores_{stub}.csv"
    return pd.read_csv(path) if os.path.exists(path) else None

name2stub = {
    "Random Forest": "Random_Forest",
    "SVM (scaled)":  "SVM_scaled",
    "MLP (scaled)":  "MLP_scaled",
}

roc_aucs_lopo = []
pr_aps_lopo   = []
roc_series_lopo = []
pr_series_lopo  = []

for name, stub in name2stub.items():
    dfp = load_lopo_scores(stub)
    if dfp is None or dfp.empty:
        continue
    y_true  = dfp["y_true"].values
    y_score = dfp["y_score"].values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    roc_aucs_lopo.append((name, roc_auc))
    roc_series_lopo.append((name, fpr, tpr))

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    pr_aps_lopo.append((name, ap))
    pr_series_lopo.append((name, rec, prec))

# Saving metrics
if roc_aucs_lopo:
    pd.DataFrame(roc_aucs_lopo, columns=["model", "AUC"]).to_csv("metrics/roc_lopo_auc.csv", index=False)
if pr_aps_lopo:
    pd.DataFrame(pr_aps_lopo,  columns=["model", "AP"]).to_csv("metrics/pr_lopo_ap.csv", index=False)

if PLOTS and roc_series_lopo:
    # ROC LOPO
    plt.figure(figsize=(3.8, 2.9))
    for name, fpr, tpr in roc_series_lopo:
        auc_val = [x[1] for x in roc_aucs_lopo if x[0] == name][0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})", linewidth=1.4)
    plt.plot([0,1], [0,1], "--", linewidth=1.0)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — LOPOCV"); plt.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig("figs/roc_lopo.pdf"); plt.close(); gc.collect()

if PLOTS and pr_series_lopo:
    # PR LOPO
    plt.figure(figsize=(3.8, 2.9))
    for name, rec, prec in pr_series_lopo:
        ap_val = [x[1] for x in pr_aps_lopo if x[0] == name][0]
        plt.plot(rec, prec, label=f"{name} (AP={ap_val:.2f})", linewidth=1.4)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall — LOPOCV"); plt.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig("figs/pr_lopo.pdf"); plt.close(); gc.collect()

print("Done. CSVs in metrics/, PDFs (if PLOTS=1) in figs/")
