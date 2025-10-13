import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin GUI, más estable en cluster
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

SAVE_PNG = False
PNG_DPI = 120

os.makedirs("figs", exist_ok=True)
os.makedirs("csv", exist_ok=True)

df = pd.read_csv("filtered_dataset_clean.csv")
df["label"] = df["label"].map({"benign": 0, "malicious": 1})

X = df.drop(columns=["id", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm.fit(X_train_s, y_train)
svm_pred = svm.predict(X_test_s)

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train_s, y_train)
mlp_pred = mlp.predict(X_test_s)

def report_row(name, y_true, y_pred):
    rep = classification_report(y_true, y_pred, output_dict=True)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": rep["weighted avg"]["precision"],
        "recall_weighted": rep["weighted avg"]["recall"],
        "f1_weighted": rep["weighted avg"]["f1-score"],
        "precision_0": rep["0"]["precision"],
        "recall_0": rep["0"]["recall"],
        "f1_0": rep["0"]["f1-score"],
        "precision_1": rep["1"]["precision"],
        "recall_1": rep["1"]["recall"],
        "f1_1": rep["1"]["f1-score"],
        "support_0": rep["0"]["support"],
        "support_1": rep["1"]["support"],
    }

rows = [
    report_row("Random Forest", y_test, rf_pred),
    report_row("SVM (scaled)", y_test, svm_pred),
    report_row("MLP (scaled)", y_test, mlp_pred),
]
pd.DataFrame(rows).to_csv("csv/holdout_metrics.csv", index=False)

def save_cm(title, y_true, y_pred, stem):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["benign", "malicious"])
    fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=100, constrained_layout=True)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=9)
    pdf_path = f"figs/{stem}.pdf"
    fig.savefig(pdf_path)
    if SAVE_PNG:
        png_path = f"figs/{stem}.png"
        try:
            fig.savefig(png_path, dpi=PNG_DPI)
        except Exception as e:
            print(f"[warn] PNG failed for {stem}: {e}. PDF saved instead.")
    plt.close(fig)

save_cm("RF (holdout)",  y_test, rf_pred,  "confusion_rf_holdout")
save_cm("SVM (holdout)", y_test, svm_pred, "confusion_svm_holdout")
save_cm("MLP (holdout)", y_test, mlp_pred, "confusion_mlp_holdout")

# ---------- RF feature importance (top-20) ----------
importances = rf.feature_importances_
feat_names = np.array(X.columns)
order = np.argsort(importances)[::-1]
TOPK = min(20, len(order))
sel = order[:TOPK]

fi_df = pd.DataFrame({
    "feature": feat_names[sel],
    "importance": importances[sel]
})
fi_df.to_csv("csv/rf_feature_importance_top20.csv", index=False)

fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=110, constrained_layout=True)
ax.barh(range(TOPK), fi_df["importance"][::-1])
ax.set_yticks(range(TOPK))
ax.set_yticklabels(fi_df["feature"][::-1], fontsize=7)
ax.set_xlabel("Importance (MDI)")
ax.set_title("Top-20 RF Feature Importances (holdout)")
fig.savefig("figs/rf_feature_importance.pdf")
if SAVE_PNG:
    try:
        fig.savefig("figs/rf_feature_importance.png", dpi=PNG_DPI)
    except Exception as e:
        print(f"[warn] PNG failed for feature importance: {e}")
plt.close(fig)

# ---------- PCA 2D (sanity check) ----------
X_all_s = StandardScaler().fit_transform(X)
pca = PCA(n_components=2, random_state=42)
pc = pca.fit_transform(X_all_s)
pc_df = pd.DataFrame({"PC1": pc[:,0], "PC2": pc[:,1], "label": y.map({0:"benign",1:"malicious"})})
pc_df.to_csv("csv/pca_2d.csv", index=False)

fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=110, constrained_layout=True)
for lab, c in [("benign", "tab:blue"), ("malicious", "tab:orange")]:
    sub = pc_df[pc_df["label"] == lab]
    ax.scatter(sub["PC1"], sub["PC2"], s=14, alpha=0.85, label=lab)
ax.set_title("PCA (2D) of filtered features")
ax.legend(frameon=False, loc="best", fontsize=8)
fig.savefig("figs/pca_2d.pdf")
if SAVE_PNG:
    try:
        fig.savefig("figs/pca_2d.png", dpi=PNG_DPI)
    except Exception as e:
        print(f"[warn] PNG failed for PCA: {e}")
plt.close(fig)

print("Done. Saved PDFs under figs/ and CSVs under csv/.")
