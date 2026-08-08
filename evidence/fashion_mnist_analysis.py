"""Fashion-MNIST data-quality and ANN evidence for Sai Tejaswi Nooka's portfolio.

The script expects the included compact CSV sample. It performs reproducible data checks,
normalizes pixels, creates a stratified split, compares a majority baseline with an MLP,
and saves metrics and plots used by the HTML portfolio.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fashion_mnist_kaggle_sample.csv"
OUTPUT_DIR = ROOT / "evidence" / "outputs"
LABEL_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
RANDOM_STATE = 42


def load_and_validate(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    label_col = "label" if "label" in frame.columns else "y" if "y" in frame.columns else frame.columns[-1]
    feature_cols = [col for col in frame.columns if col != label_col]
    if len(feature_cols) != 784:
        raise ValueError(f"Expected 784 pixel columns, found {len(feature_cols)}")
    if frame[label_col].isna().any():
        raise ValueError("The label column contains missing values")
    return frame, label_col, feature_cols


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, label_col, feature_cols = load_and_validate(DATA_PATH)

    missing_values = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())
    pixel_min = int(frame[feature_cols].min().min())
    pixel_max = int(frame[feature_cols].max().max())
    class_counts = frame[label_col].value_counts().sort_index()

    X = frame[feature_cols].astype("float32") / 255.0
    y = frame[label_col].astype("int64")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(128,), activation="relu", solver="adam",
        batch_size=64, max_iter=80, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=8,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "dataset_rows": int(len(frame)),
        "pixel_features": int(len(feature_cols)),
        "classes": int(y.nunique()),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "pixel_min": pixel_min,
        "pixel_max": pixel_max,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "baseline_accuracy": round(float(accuracy_score(y_test, baseline_predictions)), 4),
        "ann_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "ann_macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "training_iterations": int(model.n_iter_),
        "hidden_layer": [128],
        "activation": "relu",
        "random_state": RANDOM_STATE,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pd.DataFrame({
        "check": ["rows", "pixel_features", "classes", "missing_values", "duplicate_rows", "pixel_min", "pixel_max"],
        "value": [len(frame), len(feature_cols), y.nunique(), missing_values, duplicate_rows, pixel_min, pixel_max],
    }).to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar([LABEL_NAMES[i] for i in class_counts.index], class_counts.values)
    ax.set_title("Fashion-MNIST Sample Class Distribution")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=LABEL_NAMES,
        xticks_rotation=45, colorbar=False, ax=ax,
    )
    ax.set_title("ANN Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    sample_indices = np.linspace(0, len(frame) - 1, 12, dtype=int)
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for ax, idx in zip(axes.flat, sample_indices):
        image = frame.loc[idx, feature_cols].to_numpy(dtype=float).reshape(28, 28)
        label = int(frame.loc[idx, label_col])
        ax.imshow(image, cmap="gray")
        ax.set_title(LABEL_NAMES[label], fontsize=8)
        ax.axis("off")
    fig.suptitle("Fashion-MNIST Sample Images")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_images.png", dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
