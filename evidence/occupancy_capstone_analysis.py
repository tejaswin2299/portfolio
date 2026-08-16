"""End-to-end analysis for Sai Tejaswi Nooka's AI/ML portfolio capstone."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             balanced_accuracy_score, classification_report,
                             f1_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Occupancy_Estimation.csv"
OUT = Path(__file__).resolve().parent / "occupancy_outputs"
OUT.mkdir(exist_ok=True)

# I load the original UCI file and keep the target separate from sensor inputs.
df = pd.read_csv(DATA)
target = "Room_Occupancy_Count"
sensor_cols = [c for c in df.columns if c not in {"Date", "Time", target}]
X, y = df[sensor_cols], df[target]

audit = {
    "rows": int(len(df)), "input_columns": int(len(sensor_cols)),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
}

# The split is stratified because occupancy levels are highly imbalanced.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
scaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
preprocessor = ColumnTransformer([("sensors", scaled, sensor_cols)])

models = {
    "Multinomial logistic regression": LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42
    ),
    "K-nearest neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "Support vector machine": SVC(C=3, gamma="scale", class_weight="balanced"),
    "Random forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
comparison = []
pipelines = {}
for name, model in models.items():
    pipe = Pipeline([("prepare", preprocessor), ("model", model)])
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro",
                 "balanced_accuracy": "balanced_accuracy"}, n_jobs=-1
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    pipelines[name] = (pipe, pred)
    comparison.append({
        "model": name,
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_macro_f1_mean": scores["test_macro_f1"].mean(),
        "cv_balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
        "test_accuracy": accuracy_score(y_test, pred),
        "test_macro_f1": f1_score(y_test, pred, average="macro"),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, pred),
    })

results = pd.DataFrame(comparison).sort_values("test_macro_f1", ascending=False)
best_name = results.iloc[0]["model"]
best_pipe, best_pred = pipelines[best_name]

# Save exact evidence so the webpage's results are reproducible.
results.to_csv(OUT / "model_comparison.csv", index=False)
pd.DataFrame(classification_report(y_test, best_pred, output_dict=True)).T.to_csv(
    OUT / "best_model_classification_report.csv"
)
metrics = {
    **audit,
    "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
    "best_model": best_name,
    "test_accuracy": float(accuracy_score(y_test, best_pred)),
    "test_macro_f1": float(f1_score(y_test, best_pred, average="macro")),
    "test_balanced_accuracy": float(balanced_accuracy_score(y_test, best_pred)),
}
(OUT / "capstone_metrics.json").write_text(json.dumps(metrics, indent=2))

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(8.6, 4.8))
counts = y.value_counts().sort_index()
ax.bar([f"{i} people" if i != 1 else "1 person" for i in counts.index], counts.values,
       color=["#1f6f8b", "#d9a441", "#4f86a6", "#102a43"])
ax.set(title="Occupancy Classes Are Strongly Imbalanced", ylabel="Observations")
for i, value in enumerate(counts.values): ax.text(i, value + 90, f"{value:,}", ha="center")
fig.tight_layout(); fig.savefig(OUT / "class_distribution.png", dpi=170); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.4))
chart = results.sort_values("test_macro_f1")
ax.barh(chart["model"], chart["test_macro_f1"], color="#1f6f8b")
ax.set(xlim=(0, 1), xlabel="Held-out macro F1", title="Model Comparison on Unseen Test Data")
for i, value in enumerate(chart["test_macro_f1"]): ax.text(value + .01, i, f"{value:.3f}", va="center")
fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=170); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5.6))
ConfusionMatrixDisplay.from_predictions(y_test, best_pred, display_labels=[0, 1, 2, 3],
                                        cmap="Blues", colorbar=False, ax=ax)
ax.set_title(f"Best Model Confusion Matrix\n{best_name}")
fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=170); plt.close(fig)

rf = pipelines["Random forest"][0].named_steps["model"]
importance = pd.Series(rf.feature_importances_, index=sensor_cols).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(8.5, 5.4))
ax.barh(importance.index, importance.values, color="#d9a441")
ax.set(title="Random-Forest Feature Importance", xlabel="Mean decrease in impurity")
fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=170); plt.close(fig)

print(results.to_string(index=False))
print(json.dumps(metrics, indent=2))
