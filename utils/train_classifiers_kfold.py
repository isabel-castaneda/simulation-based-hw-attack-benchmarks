import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("filtered_dataset_clean.csv")
df["label"] = df["label"].map({"benign": 0, "malicious": 1})

X = df.drop(columns=["id", "label"])
y = df["label"]

classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine (scaled)": SVC(kernel="rbf", C=1, gamma="scale"),
    "Multi-Layer Perceptron (scaled)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
}

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

results = {name: [] for name in classifiers}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, clf in classifiers.items():
        if "scaled" in name:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        report = classification_report(y_test, y_pred, output_dict=True)
        results[name].append(report)

        print(f"\n{name}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

print("\n=== AVERAGE METRICS OVER ALL FOLDS ===")
for name, reports in results.items():
    avg_precision = np.mean([r["weighted avg"]["precision"] for r in reports])
    avg_recall = np.mean([r["weighted avg"]["recall"] for r in reports])
    avg_f1 = np.mean([r["weighted avg"]["f1-score"] for r in reports])
    print(f"\n{name} - Avg Precision: {avg_precision:.2f}, Avg Recall: {avg_recall:.2f}, Avg F1: {avg_f1:.2f}")
