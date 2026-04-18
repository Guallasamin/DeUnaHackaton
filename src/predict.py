"""Scoring y segmentación de churn para el equipo comercial.

Replica la lógica de segmentación del flujo Databricks de referencia (Alerta Roja/
Amarilla/Baja/Muy Baja) sobre el percent_rank de la probabilidad de inactivación.
Genera outputs/predictions.csv consumible por el dashboard.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from train_model import ID_COLS, TARGET, _split_features


SEGMENTOS = [
    (0.95, "ALERTA_ROJA"),
    (0.89, "ALERTA_AMARILLA"),
    (0.82, "BAJA_PROBABILIDAD"),
    (0.00, "MUY_BAJA_PROBABILIDAD"),
]


def _segmentar(prob_rank: float) -> str:
    for cutoff, label in SEGMENTOS:
        if prob_rank >= cutoff:
            return label
    return "MUY_BAJA_PROBABILIDAD"


def score(
    mdt_path: str | Path = "data/processed/mdt_churn.parquet",
    model_path: str | Path = "outputs/model/churn_model.pkl",
    out_path: str | Path = "outputs/predictions.csv",
) -> pd.DataFrame:
    mdt = pd.read_parquet(mdt_path)
    pipeline = joblib.load(model_path)

    X, _, _, _ = _split_features(mdt)
    proba = pipeline.predict_proba(X)[:, 1]

    out = mdt[["comercio_id", "fecha_corte"]].copy()
    out["probabilidad_churn"] = proba
    out["prob_rank"] = out["probabilidad_churn"].rank(pct=True)
    out["segmento_churn"] = out["prob_rank"].apply(_segmentar)

    out = out.merge(
        mdt[["comercio_id", "mcc_segmento", "region", "tipo_persona", "tenure_meses",
             "volumen_sum_6m", "tx_sum_6m", "recencia_bucket_0"]],
        on="comercio_id",
        how="left",
    )

    out = out.sort_values("probabilidad_churn", ascending=False).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\nPredicciones guardadas en: {out_path}")
    print("\nDistribución por segmento:")
    print(out["segmento_churn"].value_counts().to_string())
    print(f"\nTop 10 comercios en riesgo:")
    print(
        out.head(10)[
            ["comercio_id", "probabilidad_churn", "segmento_churn",
             "mcc_segmento", "region", "volumen_sum_6m"]
        ].to_string(index=False)
    )
    return out


if __name__ == "__main__":
    score()
