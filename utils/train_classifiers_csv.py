import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("filtered_dataset_clean.csv")
df["label"] = df["label"].map({"benign": 0, "malicious": 1})

X = df.drop(columns=["id", "label"])
y = df["label"]

# train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = []

# --- 1. Random Forest (no scaling needed) ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_report = classification_report(y_test, rf_pred, output_dict=True)["weighted avg"]
results.append(["Random Forest", rf_report["precision"], rf_report["recall"], rf_report["f1-score"]])

print("\n--- Random Forest ---")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# --- 2. SVM (scaled) ---
scaler_svm = StandardScaler()
X_train_scaled_svm = scaler_svm.fit_transform(X_train)
X_test_scaled_svm = scaler_svm.transform(X_test)

svm = SVC(kernel="rbf", C=1, gamma="scale")
svm.fit(X_train_scaled_svm, y_train)
svm_pred = svm.predict(X_test_scaled_svm)
svm_report = classification_report(y_test, svm_pred, output_dict=True)["weighted avg"]
results.append(["SVM (scaled)", svm_report["precision"], svm_report["recall"], svm_report["f1-score"]])

print("\n--- Support Vector Machine (scaled) ---")
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# --- 3. MLP (scaled) ---
scaler_mlp = StandardScaler()
X_train_scaled_mlp = scaler_mlp.fit_transform(X_train)
X_test_scaled_mlp = scaler_mlp.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train_scaled_mlp, y_train)
mlp_pred = mlp.predict(X_test_scaled_mlp)
mlp_report = classification_report(y_test, mlp_pred, output_dict=True)["weighted avg"]
results.append(["MLP (scaled)", mlp_report["precision"], mlp_report["recall"], mlp_report["f1-score"]])

print("\n--- Multi-Layer Perceptron (scaled) ---")
print(confusion_matrix(y_test, mlp_pred))
print(classification_report(y_test, mlp_pred))

df_results = pd.DataFrame(results, columns=["Model", "Precision", "Recall", "F1-Score"])
df_results.to_csv("holdout_results_summary.csv", index=False)
