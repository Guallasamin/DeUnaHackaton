"""Configuración centralizada de paths y constantes del proyecto.

Ambos equipos (modelo y frontend) importan de aquí para evitar
paths hardcodeados y garantizar consistencia.

Uso:
    from config.settings import PATHS
    df = pd.read_csv(PATHS.PREDICTIONS)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ── Raíz del proyecto ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class _Paths:
    """Registro inmutable de todos los paths del proyecto."""

    # ── Raw inputs (generados por src/data/*) ───────────────────────────────
    RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    DIM_MERCHANTS:    Path = PROJECT_ROOT / "data" / "raw" / "dim_merchants.csv"
    FACT_PERFORMANCE: Path = PROJECT_ROOT / "data" / "raw" / "fact_performance_monthly.csv"
    FACT_TICKETS:     Path = PROJECT_ROOT / "data" / "raw" / "fact_support_tickets.csv"
    CHURN_LABELS:     Path = PROJECT_ROOT / "data" / "raw" / "churn_labels.csv"

    # ── Procesados ──────────────────────────────────────────────────────────
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    MDT:           Path = PROJECT_ROOT / "data" / "processed" / "mdt_churn.parquet"

    # ── Outputs del modelo (CONTRATO entre equipos) ─────────────────────────
    OUTPUT_DIR:        Path = PROJECT_ROOT / "outputs"
    MODEL_DIR:         Path = PROJECT_ROOT / "outputs" / "model"
    MODEL_PKL:         Path = PROJECT_ROOT / "outputs" / "model" / "churn_model.pkl"
    METRICS_JSON:      Path = PROJECT_ROOT / "outputs" / "model" / "metrics.json"
    FEATURE_COLS_JSON: Path = PROJECT_ROOT / "outputs" / "model" / "feature_columns.json"
    FIGURES_DIR:       Path = PROJECT_ROOT / "outputs" / "figures"
    SHAP_SUMMARY_PNG:  Path = PROJECT_ROOT / "outputs" / "figures" / "shap_summary.png"
    SHAP_BAR_PNG:      Path = PROJECT_ROOT / "outputs" / "figures" / "shap_bar.png"
    PREDICTIONS:       Path = PROJECT_ROOT / "outputs" / "predictions.csv"
    SHAP_VALUES:       Path = PROJECT_ROOT / "outputs" / "shap_values.parquet"


PATHS = _Paths()


# ── Constantes del modelo ────────────────────────────────────────────────────
TARGET = "abandono_30d"
ID_COLS = ["merchant_id", "fecha_corte"]

# Columnas categóricas que consume el modelo (one-hot).
# Se agrega `recencia_bucket_0` derivada en feature_engineering.
CATEGORICAL_COLS = [
    "segmento_comercial",
    "region",
    "tipo_negocio_desc",
    "recencia_bucket_0",
]

# Reproducibilidad
SEED = 42

# Split 60 / 20 / 20 (train / val / test). El val se usa para early stopping
# y selección de umbral; el test es hold-out para métricas finales no sesgadas.
TRAIN_SIZE = 0.60
VAL_SIZE = 0.20
TEST_SIZE = 0.20

# Fecha de corte del modelo.
#
# El dataset cubre 2025-04 a 2026-03. El decaimiento de los churners empieza
# entre noviembre 2025 y enero 2026 (3-5 meses de decay terminando en marzo 2026).
#
# Con corte en 2026-01: churners tienen 4x menos transacciones que sanos
# (ratio 4.08x) → el modelo lo aprende en 1 árbol → AUC 0.999 artificial,
# best_iteration=1, probabilidades comprimidas en 0.48-0.52, ranking inútil.
#
# Con corte en 2025-11: el ratio baja a 1.91x — señal presente pero no trivial.
# Solo ~33% de churners muestran 1 mes de decaimiento; el resto luce normal.
# El modelo necesita múltiples árboles y combinar lags + features estáticos,
# produciendo probabilidades con rango real y AUC realista (~0.80-0.88).
FECHA_CORTE = "2025-11-01"
