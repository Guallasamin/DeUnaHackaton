"""Construcción de la MDT (Master Data Table) para el modelo de churn.

Sigue el patrón de ventanas _0, _1, _2, _3, _4 meses del flujo Databricks de referencia:
cada observación (comercio, fecha_corte) lleva los valores de los últimos 5 meses más
estadísticos agregados (3m, 6m, 12m), deltas y ratios.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ACTIVITY_METRICS = [
    "n_transacciones",
    "volumen_total",
    "ticket_promedio",
    "dias_desde_ult_tx",
    "n_chargebacks",
    "n_rechazos",
    "n_tickets_soporte",
    "dias_resolucion_soporte",
    "n_integraciones_activas",
]

FLAG_METRICS = ["flag_pos_fisico", "flag_ecommerce"]

LAGS = [0, 1, 2, 3, 4]


def _bucket_recencia(dias: float) -> str:
    if pd.isna(dias):
        return "sin_registro"
    if dias <= 5:
        return "menos_5"
    if dias <= 10:
        return "entre_5_y_10"
    if dias <= 20:
        return "entre_10_y_20"
    if dias <= 30:
        return "entre_20_y_30"
    return "mas_de_30"


def _lag_frame(activity: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Pivot: una fila por comercio con columnas metric_{lag}."""
    wide = None
    for lag in LAGS:
        mes = corte - pd.DateOffset(months=lag)
        snap = activity[activity["fecha_corte"] == mes].copy()
        snap = snap[["comercio_id", *ACTIVITY_METRICS, *FLAG_METRICS]]
        rename = {c: f"{c}_{lag}" for c in ACTIVITY_METRICS + FLAG_METRICS}
        snap = snap.rename(columns=rename)
        wide = snap if wide is None else wide.merge(snap, on="comercio_id", how="outer")
    return wide


def _rolling_aggregates(activity: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Agregados sobre ventanas de 3, 6 y 12 meses hacia atrás del corte."""
    out = None
    for window in (3, 6, 12):
        desde = corte - pd.DateOffset(months=window - 1)
        ventana = activity[(activity["fecha_corte"] >= desde) & (activity["fecha_corte"] <= corte)]
        agg = ventana.groupby("comercio_id").agg(
            **{
                f"tx_sum_{window}m":        ("n_transacciones", "sum"),
                f"tx_mean_{window}m":       ("n_transacciones", "mean"),
                f"tx_std_{window}m":        ("n_transacciones", "std"),
                f"volumen_sum_{window}m":   ("volumen_total", "sum"),
                f"volumen_mean_{window}m":  ("volumen_total", "mean"),
                f"volumen_std_{window}m":   ("volumen_total", "std"),
                f"chargebacks_sum_{window}m": ("n_chargebacks", "sum"),
                f"rechazos_sum_{window}m":  ("n_rechazos", "sum"),
                f"tickets_sum_{window}m":   ("n_tickets_soporte", "sum"),
                f"meses_activos_{window}m": ("n_transacciones", lambda s: int((s > 0).sum())),
            }
        )
        out = agg if out is None else out.join(agg, how="outer")
    return out.reset_index()


def _deltas(wide: pd.DataFrame) -> pd.DataFrame:
    """Deltas mensuales consecutivos sobre métricas clave (replica patrón DeltaSaldoX_Y)."""
    df = wide.copy()
    for metric in ["n_transacciones", "volumen_total", "ticket_promedio",
                   "dias_desde_ult_tx", "n_tickets_soporte"]:
        for newer, older in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            col_new = f"{metric}_{newer}"
            col_old = f"{metric}_{older}"
            if col_new in df.columns and col_old in df.columns:
                df[f"delta_{metric}_{newer}_{older}"] = df[col_new] - df[col_old]
    return df


def build_mdt(
    merchants: pd.DataFrame,
    activity: pd.DataFrame,
    fecha_corte: pd.Timestamp | None = None,
) -> pd.DataFrame:
    activity = activity.copy()
    activity["fecha_corte"] = pd.to_datetime(activity["fecha_corte"])

    if fecha_corte is None:
        fecha_corte = activity["fecha_corte"].max()
    else:
        fecha_corte = pd.Timestamp(fecha_corte)

    wide = _lag_frame(activity, fecha_corte)
    agg = _rolling_aggregates(activity, fecha_corte)
    mdt = wide.merge(agg, on="comercio_id", how="left")
    mdt = _deltas(mdt)

    mdt["fecha_corte"] = fecha_corte

    mdt = mdt.merge(
        merchants[["comercio_id", "mcc", "mcc_segmento", "region",
                   "tipo_persona", "tenure_meses"]],
        on="comercio_id",
        how="left",
    )

    mdt["recencia_bucket_0"] = mdt["dias_desde_ult_tx_0"].apply(_bucket_recencia)

    tx6 = mdt["tx_sum_6m"].replace(0, np.nan)
    mdt["chargeback_rate_6m"] = (mdt["chargebacks_sum_6m"] / tx6).fillna(0)
    mdt["decline_rate_6m"]    = (mdt["rechazos_sum_6m"] / tx6).fillna(0)
    mdt["tickets_per_tx_6m"]  = (mdt["tickets_sum_6m"] / tx6).fillna(0)

    mdt["volatilidad_volumen_6m"] = (
        mdt["volumen_std_6m"] / mdt["volumen_mean_6m"].replace(0, np.nan)
    ).fillna(0)
    mdt["volatilidad_tx_6m"] = (
        mdt["tx_std_6m"] / mdt["tx_mean_6m"].replace(0, np.nan)
    ).fillna(0)

    mdt["omnicanal_0"] = (
        mdt.get("flag_pos_fisico_0", 0).fillna(0).astype(int)
        + mdt.get("flag_ecommerce_0", 0).fillna(0).astype(int)
    )

    numeric_cols = mdt.select_dtypes(include=[np.number]).columns
    mdt[numeric_cols] = mdt[numeric_cols].fillna(0)

    return mdt


def main(
    raw_dir: str | Path = "data/raw",
    out_dir: str | Path = "data/processed",
) -> Path:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merchants = pd.read_csv(raw_dir / "merchants.csv")
    activity = pd.read_csv(raw_dir / "monthly_activity.csv", parse_dates=["fecha_corte"])
    labels = pd.read_csv(raw_dir / "churn_labels.csv", parse_dates=["fecha_corte"])

    mdt = build_mdt(merchants, activity)
    mdt = mdt.merge(labels[["comercio_id", "churn_30d"]], on="comercio_id", how="left")

    out_path = out_dir / "mdt_churn.parquet"
    mdt.to_parquet(out_path, index=False)
    print(f"MDT construida: {mdt.shape[0]} filas x {mdt.shape[1]} columnas → {out_path}")
    print(f"Tasa de churn: {mdt['churn_30d'].mean():.2%}")
    return out_path


if __name__ == "__main__":
    main()
