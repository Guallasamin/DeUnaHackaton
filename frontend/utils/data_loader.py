"""Capa de carga de datos para el frontend.

Lee los artefactos del directorio outputs/ (contrato con el equipo de modelo).
Usa st.cache_data para evitar recargas innecesarias.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# El frontend busca los datos en outputs/ relativo a la raíz del proyecto
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs_modelo"


@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame | None:
    """Carga predictions.csv del contrato de datos."""
    path = _OUTPUTS_DIR / "fact_churn_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_metrics() -> dict | None:
    """Carga metrics.json del contrato de datos."""
    path = _OUTPUTS_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_shap_values() -> pd.DataFrame | None:
    """Carga shap_values.parquet del contrato de datos."""
    path = _OUTPUTS_DIR / "shap_values.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_feature_columns() -> dict | None:
    """Carga feature_columns.json del contrato de datos."""
    path = _OUTPUTS_DIR / "feature_columns.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
