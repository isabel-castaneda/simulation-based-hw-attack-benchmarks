import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("filtered_dataset.csv")

df["label"] = df["label"].map({"benign": 0, "malicious": 1})

X = df.drop(columns=["id", "label"])
y = df["label"]

# train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 1. Random Forest ----------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n--- Random Forest ---")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# ---------- 2. Support Vector Machine (SVM) ----------
svm = SVC(kernel="rbf", C=1, gamma="scale")
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("\n--- Support Vector Machine ---")
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# ---------- 3. Multi-Layer Perceptron (MLP) ----------
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)

print("\n--- Multi-Layer Perceptron (MLP) ---")
print(confusion_matrix(y_test, mlp_pred))
print(classification_report(y_test, mlp_pred))
