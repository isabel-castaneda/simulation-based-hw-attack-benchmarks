import os, gc
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

TRY_PNG = False
PNG_DPI = 120
os.makedirs("figs", exist_ok=True)
os.makedirs("figs/csv", exist_ok=True)

def safe_save(fig, basepath):
    fig.savefig(f"{basepath}.pdf", bbox_inches="tight")
    if TRY_PNG:
        try:
            fig.savefig(f"{basepath}.png", dpi=PNG_DPI, bbox_inches="tight")
        except MemoryError:
            print(f"[warn] Skipped PNG for {basepath} (MemoryError). PDF written.")

def fresh_fig(w=6.5, h=4.5):
    plt.close('all'); gc.collect()
    return plt.figure(figsize=(w, h))

df = pd.read_csv("filtered_dataset_clean.csv")
df["label"] = df["label"].map({"benign": 0, "malicious": 1})
X = df.drop(columns=["id", "label"])
y = df["label"]

# ---------- Split holdout ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ============== 1) Random Forest feature importance ==============
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X.columns)
imp_sorted = importances.sort_values(ascending=False)
imp_sorted.to_csv("figs/csv/rf_feature_importances.csv")

top10 = imp_sorted.head(10)
fig = fresh_fig(6.4, 4.2)
ax = fig.gca()
top10.iloc[::-1].plot(kind="barh", ax=ax)
ax.set_xlabel("Importance (MDI)")
ax.set_ylabel("")
fig.tight_layout()
safe_save(fig, "figs/rf_feature_importance")
plt.close(fig); gc.collect()

# ============== 2) PCA 2D scatter ==============
scaler_for_pca = StandardScaler()
X_scaled = scaler_for_pca.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame({"PC1": Z[:,0], "PC2": Z[:,1], "label": y.values})
pca_df.to_csv("figs/csv/pca_2d_points.csv", index=False)

fig = fresh_fig(6.2, 4.6)
for lab, name in [(0,"benign"), (1,"malicious")]:
    sel = pca_df["label"]==lab
    plt.scatter(pca_df.loc[sel,"PC1"], pca_df.loc[sel,"PC2"], s=18, alpha=0.7, label=name)
plt.legend()
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
safe_save(fig, "figs/pca_scatter")
plt.close(fig); gc.collect()

# ============== 3) Boxplots top-4 ==============
top4 = imp_sorted.head(4).index.tolist()
melt = df[["label"]+top4].copy()
melt["label"] = melt["label"].map({0:"benign",1:"malicious"})
melt.to_csv("figs/csv/boxplot_top4_data.csv", index=False)

fig = fresh_fig(7.6, 5.6)
for i,feat in enumerate(top4, start=1):
    ax = plt.subplot(2,2,i)
    b = [melt.loc[melt["label"]=="benign", feat].values,
         melt.loc[melt["label"]=="malicious", feat].values]
    ax.boxplot(b, labels=["benign","malicious"], showfliers=False)
    ax.set_title(feat, fontsize=9)
plt.tight_layout()
safe_save(fig, "figs/boxplots_top4")
plt.close(fig); gc.collect()

# ============== 4) Learning curves (RF y SVM) ==============
def make_train_sizes(n_samples, cv_splits):
    max_train = int(n_samples * (cv_splits - 1) / cv_splits)
    min_train = max(20, int(0.3 * max_train))
    sizes = np.linspace(min_train, max_train, 8, dtype=int)
    sizes = np.unique(sizes)
    sizes = sizes[(sizes > 1) & (sizes <= max_train)]
    return sizes

def plot_learning_curve(estimator, Xd, yd, title, outbase, scale=False):
    if scale:
        sc = StandardScaler()
        Xd = sc.fit_transform(Xd)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = make_train_sizes(Xd.shape[0], cv_splits=cv.n_splits)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, Xd, yd, cv=cv, scoring="accuracy",
        train_sizes=train_sizes, shuffle=True, random_state=42
    )
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

    lc = pd.DataFrame({
        "train_size": train_sizes,
        "train_mean": train_mean, "train_std": train_std,
        "test_mean": test_mean, "test_std": test_std
    })
    lc.to_csv(f"figs/csv/{outbase}_learning_curve.csv", index=False)

    fig = fresh_fig(6.4, 4.4)
    plt.plot(train_sizes, train_mean, marker="o", label="Train")
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
    plt.plot(train_sizes, test_mean, marker="o", label="CV")
    plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, alpha=0.2)
    plt.xlabel("Training samples"); plt.ylabel("Accuracy"); plt.legend()
    plt.title(title)
    plt.tight_layout()
    safe_save(fig, f"figs/{outbase}_learning_curves")
    plt.close(fig); gc.collect()

plot_learning_curve(RandomForestClassifier(n_estimators=300, random_state=42), X, y,
                    "Learning Curves (Random Forest)", "rf", scale=False)
plot_learning_curve(SVC(kernel="rbf", C=1, gamma="scale"), X, y,
                    "Learning Curves (SVM, RBF, scaled)", "svm", scale=True)

# ============== 5) Confusion matricx + reports (holdout) ==============
def eval_and_save(name, clf, Xtr, Xte, ytr, yte, scale=False):
    if scale:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)

    rep = classification_report(yte, y_pred, output_dict=True)
    pd.DataFrame(rep).to_csv(f"figs/csv/{name}_report.csv")

    cm = confusion_matrix(yte, y_pred, labels=[0,1])
    pd.DataFrame(cm, index=["true_benign","true_malicious"],
                 columns=["pred_benign","pred_malicious"]).to_csv(f"figs/csv/{name}_cm.csv")

    fig = fresh_fig(4.4, 4.1)
    disp = ConfusionMatrixDisplay(cm, display_labels=["benign","malicious"])
    ax = fig.gca()
    disp.plot(ax=ax)
    plt.tight_layout()
    safe_save(fig, f"figs/{name}_cm")
    plt.close(fig); gc.collect()

eval_and_save("rf_holdout", RandomForestClassifier(n_estimators=300, random_state=42),
              X_train, X_test, y_train, y_test, scale=False)
eval_and_save("svm_holdout", SVC(kernel="rbf", C=1, gamma="scale"),
              X_train, X_test, y_train, y_test, scale=True)
eval_and_save("mlp_holdout", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
              X_train, X_test, y_train, y_test, scale=True)

print("Saved: figs/*.pdf and figs/csv/*.csv  (PNG disabled by default)")
