"""Entrenamiento del modelo de churn con XGBoost.

Entrena un pipeline con preprocesamiento (one-hot de categóricas + passthrough numérico)
y clasificador XGBoost. Guarda el modelo serializado, métricas y columnas usadas.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


TARGET = "churn_30d"
ID_COLS = ["comercio_id", "fecha_corte"]

CATEGORICAL_COLS = [
    "mcc_segmento",
    "region",
    "tipo_persona",
    "recencia_bucket_0",
]


@dataclass
class TrainResult:
    auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    n_train: int
    n_test: int
    churn_rate_train: float
    churn_rate_test: float


def _split_features(mdt: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    feature_cols = [c for c in mdt.columns if c not in ID_COLS + [TARGET]]
    X = mdt[feature_cols].copy()
    y = mdt[TARGET].astype(int)

    categorical = [c for c in CATEGORICAL_COLS if c in X.columns]
    numeric = [c for c in feature_cols if c not in categorical]

    X[categorical] = X[categorical].astype(str)
    return X, y, numeric, categorical


def _build_pipeline(numeric: list[str], categorical: list[str], scale_pos_weight: float) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _pick_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Elige el umbral que maximiza F1."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    idx = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[idx]) if len(thr) else 0.5


def train(
    mdt_path: str | Path = "data/processed/mdt_churn.parquet",
    out_dir: str | Path = "outputs/model",
) -> TrainResult:
    mdt = pd.read_parquet(mdt_path)
    X, y, numeric, categorical = _split_features(mdt)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42,
    )

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    pipeline = _build_pipeline(numeric, categorical, pos_weight)
    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)
    threshold = _pick_threshold(y_test.values, proba_test)
    pred = (proba_test >= threshold).astype(int)

    result = TrainResult(
        auc=auc,
        pr_auc=pr_auc,
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        threshold=threshold,
        n_train=len(y_train),
        n_test=len(y_test),
        churn_rate_train=float(y_train.mean()),
        churn_rate_test=float(y_test.mean()),
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, out_dir / "churn_model.pkl")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(asdict(result), f, indent=2)
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump({"numeric": numeric, "categorical": categorical}, f, indent=2)

    print("\n=== Métricas del modelo ===")
    print(f"AUC-ROC         : {auc:.4f}")
    print(f"AUC-PR          : {pr_auc:.4f}")
    print(f"Umbral óptimo F1: {threshold:.4f}")
    print(f"Precision       : {result.precision:.4f}")
    print(f"Recall          : {result.recall:.4f}")
    print(f"F1              : {result.f1:.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n", classification_report(y_test, pred, digits=4))
    return result


if __name__ == "__main__":
    train()
