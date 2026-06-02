"""
modelling.py (versi Workflow-CI)
=================================
Script pelatihan model yang mendukung MLflow Project entry point.
Dapat dipanggil via: python modelling.py --data_path ... --n_estimators ... dll
"""
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import dagshub

# DagsHub credentials dibaca dari env variable (di-set via GitHub Secrets)
DAGSHUB_OWNER = os.environ.get("DAGSHUB_OWNER", "nuris-isw")
DAGSHUB_REPO  = os.environ.get("DAGSHUB_REPO",  "Eksperimen_SML_Nuris")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "loan_status"
RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def plot_confusion_matrix(y_test, y_pred, path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected", "Approved"])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix - Random Forest", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_feature_importance(model, feature_names, path):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    importances.plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
    ax.set_title("Feature Importance - Random Forest", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="loan_approval_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    X, y = load_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Inisialisasi DagsHub tracking
    dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment("Loan_Approval_CI")

    with mlflow.start_run() as run:
        # Tulis run_id ke file agar dapat digunakan oleh GitHub Actions
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "run_id.txt"), "w") as f:
            f.write(run.info.run_id)
        # Manual logging
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", args.test_size)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        mlflow.log_metrics(metrics)

        # Artefak visual
        cm_path = "confusion_matrix.png"
        fi_path = "feature_importance.png"
        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_feature_importance(model, X_train.columns.tolist(), fi_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(fi_path, artifact_path="plots")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            pip_requirements=[
                "mlflow==2.19.0",
                "scikit-learn==1.5.2",
                "pandas==2.2.3",
                "numpy==1.26.4",
            ]
        )

        print("\n=== METRIK ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))


if __name__ == "__main__":
    main()
