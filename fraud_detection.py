"""
Credit Card Fraud Detection with a PyOD AutoEncoder
----------------------------------------------------
Trains an unsupervised AutoEncoder on anonymized credit card transactions
and flags transactions with unusually high reconstruction error as fraud.

The model never sees the fraud labels during training. Labels are only used
afterward to measure how well the anomaly scores line up with real fraud.

Dataset: https://www.kaggle.com/datasets/whenamancodes/fraud-detection
         (save the file as creditcard.csv next to this script)

Author: Mohith Tummala
Course: MSCS-633-A01 Advanced Artificial Intelligence
Assignment: Hands-On Assignment 4 - Use Unsupervised Deep Learning Algorithm
            to Detect Fraud with PyOD
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Render plots to files so the script works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
LABEL_COLUMN = "Class"  # 1 = fraud, 0 = legitimate


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the transactions file and fail early with a clear message if it is missing."""
    if not csv_path.exists():
        sys.exit(
            f"Could not find {csv_path}. Download creditcard.csv from the Kaggle "
            "link in this file's docstring and place it next to this script."
        )
    df = pd.read_csv(csv_path)
    if LABEL_COLUMN not in df.columns:
        sys.exit(f"Expected a '{LABEL_COLUMN}' column in {csv_path}.")
    return df


def prepare_features(df: pd.DataFrame):
    """Split into features and labels, then split into train and test sets."""
    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN].to_numpy()

    # Stratify so the test set keeps the same (very small) share of fraud cases.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # V1-V28 are already PCA outputs, but Time and Amount are on very different
    # scales. The scaler is fit on training data only so nothing leaks from the test set.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def build_model(contamination: float, epochs: int) -> AutoEncoder:
    """
    Create the AutoEncoder. The hidden layers squeeze the 30 inputs down to a small
    bottleneck and rebuild them. Normal transactions reconstruct well; rare unusual
    ones do not, and that gap is the anomaly score.

    PyOD's hidden_neuron_list describes only the encoder half, and the last entry is
    the bottleneck. The decoder is mirrored automatically, so [16, 8, 4] builds
    30 -> 16 -> 8 -> 4 -> 8 -> 16 -> 30.
    """
    return AutoEncoder(
        hidden_neuron_list=[16, 8, 4],
        epoch_num=epochs,
        batch_size=256,
        dropout_rate=0.1,
        contamination=contamination,
        preprocessing=False,  # Features are already scaled in prepare_features()
        random_state=RANDOM_STATE,
        verbose=1,
    )


def evaluate(model: AutoEncoder, X_test, y_test) -> dict:
    """Score the test set and compute metrics that make sense for imbalanced data."""
    scores = model.decision_function(X_test)  # higher = more anomalous
    predictions = model.predict(X_test)  # 1 = flagged as outlier

    auc = roc_auc_score(y_test, scores)
    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test, predictions, target_names=["Legitimate", "Fraud"], digits=4, zero_division=0
    )
    return {"scores": scores, "predictions": predictions, "auc": auc,
            "matrix": matrix, "report": report}


def save_plots(results: dict, y_test, output_dir: Path) -> None:
    """Save two charts: the score distributions and the precision-recall curve."""
    scores = results["scores"]

    # Reconstruction error by true class. Log scale because a few scores are huge.
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(max(scores.min(), 1e-6)), np.log10(scores.max()), 60)
    ax.hist(scores[y_test == 0], bins=bins, alpha=0.6, label="Legitimate", color="#2a7ab0")
    ax.hist(scores[y_test == 1], bins=bins, alpha=0.8, label="Fraud", color="#d9534f")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reconstruction error (anomaly score)")
    ax.set_ylabel("Number of transactions")
    ax.set_title("AutoEncoder anomaly scores by true class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close(fig)

    # Precision-recall curve is more honest than ROC when fraud is this rare.
    precision, recall, _ = precision_recall_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#2a7ab0")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyOD AutoEncoder fraud detection")
    parser.add_argument("--data", type=Path, default=Path("creditcard.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    df = load_data(args.data)
    fraud_rate = df[LABEL_COLUMN].mean()
    print(f"Loaded {len(df):,} transactions, {int(df[LABEL_COLUMN].sum()):,} fraud "
          f"({fraud_rate:.4%})")

    X_train, X_test, y_train, y_test = prepare_features(df)

    # Contamination tells PyOD what share of data to flag. Using the real fraud rate
    # here is a simplification for the experiment; a production system would tune
    # the threshold against business cost instead.
    model = build_model(contamination=fraud_rate, epochs=args.epochs)
    print("\nTraining AutoEncoder (labels are not used for training)...")
    model.fit(X_train)

    results = evaluate(model, X_test, y_test)
    print(f"\nROC-AUC: {results['auc']:.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(results["matrix"])
    print("\nClassification report:")
    print(results["report"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_plots(results, y_test, args.output_dir)
    print(f"Saved plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
