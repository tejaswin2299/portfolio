"""Reproducible classification audit for the UCI Rice dataset."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Rice_Cammeo_Osmancik.arff"
OUTPUT = Path(__file__).resolve().parent / "rice_outputs"
OUTPUT.mkdir(exist_ok=True)

raw, _ = arff.loadarff(DATA)
df = pd.DataFrame(raw)
df["Class"] = df["Class"].str.decode("utf-8")
features = [column for column in df.columns if column != "Class"]

# Short quality audit before any model fitting.
quality = {
    "rows": int(len(df)),
    "features": len(features),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "class_counts": df["Class"].value_counts().to_dict(),
}

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["Class"], test_size=0.20, random_state=42,
    stratify=df["Class"]
)

# Scaling is fitted only on training data inside the pipeline to prevent leakage.
model = Pipeline([
    ("scale", ColumnTransformer([("numeric", StandardScaler(), features)])),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = {
    **quality,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
    "classification_report": classification_report(
        y_test, predictions, output_dict=True
    ),
}
(OUTPUT / "rice_metrics.json").write_text(json.dumps(metrics, indent=2))

fig, ax = plt.subplots(figsize=(6, 4))
df["Class"].value_counts().sort_index().plot.bar(
    ax=ax, color=["#2dd4bf", "#818cf8"]
)
ax.set(title="Rice variety class distribution", xlabel="Variety", ylabel="Grains")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(OUTPUT / "rice_class_distribution.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, cmap="Blues", colorbar=False, ax=ax
)
ax.set_title("Logistic regression confusion matrix")
fig.tight_layout()
fig.savefig(OUTPUT / "rice_confusion_matrix.png", dpi=180)
plt.close(fig)

print(json.dumps({key: metrics[key] for key in
                  ["rows", "missing_values", "duplicate_rows", "accuracy", "macro_f1"]},
                 indent=2))
