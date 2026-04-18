"""
====================================================================
 FEATURE ENGINEERING — Reto Deuna (Interact2Hack 2026)
====================================================================

 Construye features desde las 3 tablas fuente para alimentar
 el modelo de churn. Cada función es modular y puede llamarse
 independientemente.

 Granularidad de salida: una fila por merchant_id, con ~35 features.

 Uso típico:
     from feature_engineering import construir_dataset_features
     X, y, df_full = construir_dataset_features(
         path_merchants="data/raw/dim_merchants_con_abandono.csv",
         path_performance="data/raw/fact_performance_monthly.csv",
         path_tickets="data/raw/fact_support_tickets.csv",
         fecha_corte=datetime(2026, 3, 31)
     )
====================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. FEATURES ESTÁTICAS (desde dim_merchants)
# ============================================================

def features_estaticas(df_merchants: pd.DataFrame, fecha_corte: datetime) -> pd.DataFrame:
    """Features del perfil del comercio que no cambian temporalmente."""
    df = df_merchants.copy()

    # Parseo de fechas
    df["fecha_onboarding"] = pd.to_datetime(df["fecha_onboarding"])
    df["fecha_ultima_transaccion"] = pd.to_datetime(df["fecha_ultima_transaccion"])

    # --- Tenure en meses ---
    df["tenure_meses"] = ((fecha_corte - df["fecha_onboarding"]).dt.days / 30.44).round(1)
    df["es_onboarding_reciente"] = (df["tenure_meses"] < 3).astype(int)
    df["es_maduro"] = (df["tenure_meses"] > 24).astype(int)

    # --- Recencia directa ---
    df["dias_desde_ultima_trx_global"] = (fecha_corte - df["fecha_ultima_transaccion"]).dt.days.clip(lower=0)

    # --- Ejecutivo asignado ---
    df["tiene_ejecutivo"] = (df["ejecutivo_cuenta"] != "Auto-gestionado").astype(int)

    # --- Segmento como ordinal (Micro=0 ... Grande=3) ---
    mapa_segmento = {"Micro": 0, "Pequeña": 1, "Mediana": 2, "Grande": 3}
    df["segmento_ordinal"] = df["segmento_comercial"].map(mapa_segmento).fillna(0)

    # --- Región agrupada ---
    regiones_principales = ["Pichincha", "Guayas"]
    regiones_secundarias = ["Azuay", "Manabí", "Tungurahua", "El Oro"]
    regiones_amazonicas = ["Morona Santiago", "Napo", "Pastaza",
                           "Zamora Chinchipe", "Orellana", "Sucumbíos", "Galápagos"]
    def _grupo_region(r):
        if r in regiones_principales: return 3
        if r in regiones_secundarias: return 2
        if r in regiones_amazonicas: return 0
        return 1
    df["region_grupo"] = df["region"].apply(_grupo_region)

    # --- Tipo de negocio: flag de resilientes vs volátiles ---
    tipos_resilientes = ["Tiendas de abarrotes y víveres",
                         "Farmacias y artículos médicos",
                         "Panaderías y pastelerías",
                         "Cafeterías y panaderías"]
    tipos_volatiles = ["Boutiques y ropa",
                       "Bares y cantinas",
                       "Venta y repuestos de motos",
                       "Librerías y útiles escolares"]
    df["tipo_resiliente"] = df["tipo_negocio_desc"].isin(tipos_resilientes).astype(int)
    df["tipo_volatil"] = df["tipo_negocio_desc"].isin(tipos_volatiles).astype(int)

    cols_out = [
        "merchant_id", "tenure_meses", "es_onboarding_reciente", "es_maduro",
        "dias_desde_ultima_trx_global", "tiene_ejecutivo", "segmento_ordinal",
        "region_grupo", "tipo_resiliente", "tipo_volatil",
    ]
    return df[cols_out]


# ============================================================
# 2. FEATURES TEMPORALES (desde fact_performance_monthly)
# ============================================================

def features_performance(df_perf: pd.DataFrame, fecha_corte: datetime) -> pd.DataFrame:
    """
    Construye features sobre ventanas móviles desde la performance mensual.
    Usa datos ANTERIORES a fecha_corte.
    """
    df = df_perf.copy()
    df["mes_reporte"] = pd.to_datetime(df["mes_reporte"])
    df = df[df["mes_reporte"] <= fecha_corte].sort_values(["merchant_id", "mes_reporte"])

    features = []

    for mid, grp in df.groupby("merchant_id"):
        grp = grp.sort_values("mes_reporte").reset_index(drop=True)
        n_meses = len(grp)

        f = {"merchant_id": mid}

        # --- Agregados all-time ---
        f["tpv_total"] = grp["tpv_mensual"].sum()
        f["count_trx_total"] = grp["count_trx"].sum()
        f["n_meses_historia"] = n_meses
        f["meses_activos"] = (grp["count_trx"] > 0).sum()
        f["ratio_meses_activos"] = f["meses_activos"] / max(n_meses, 1)

        # --- Últimos 1, 3, 6 meses ---
        for n in [1, 3, 6]:
            tail = grp.tail(n)
            f[f"tpv_{n}m"] = tail["tpv_mensual"].sum()
            f[f"count_trx_{n}m"] = tail["count_trx"].sum()
            f[f"ticket_prom_{n}m"] = tail["ticket_promedio"].mean()
            f[f"tasa_rechazo_{n}m"] = tail["tasa_rechazo"].mean()
            f[f"tasa_rechazo_max_{n}m"] = tail["tasa_rechazo"].max()
            f[f"dias_sin_trx_max_{n}m"] = tail["dias_sin_transaccion_max"].max()

        # --- Tendencias (últimos 3 meses) ---
        if n_meses >= 3:
            ultimos3 = grp.tail(3)["count_trx"].values
            # Pendiente lineal normalizada por el promedio
            x = np.arange(len(ultimos3))
            y = ultimos3
            if y.mean() > 0:
                slope = np.polyfit(x, y, 1)[0]
                f["pendiente_count_trx_3m"] = slope / y.mean()
            else:
                f["pendiente_count_trx_3m"] = 0

            # TPV pendiente
            ultimos3_tpv = grp.tail(3)["tpv_mensual"].values
            if ultimos3_tpv.mean() > 0:
                slope_tpv = np.polyfit(x, ultimos3_tpv, 1)[0]
                f["pendiente_tpv_3m"] = slope_tpv / ultimos3_tpv.mean()
            else:
                f["pendiente_tpv_3m"] = 0
        else:
            f["pendiente_count_trx_3m"] = 0
            f["pendiente_tpv_3m"] = 0

        # --- Ratio recientes vs histórico ---
        tpv_ult3 = grp.tail(3)["tpv_mensual"].mean() if n_meses >= 3 else grp["tpv_mensual"].mean()
        tpv_hist = grp["tpv_mensual"].mean()
        f["ratio_tpv_3m_vs_total"] = tpv_ult3 / tpv_hist if tpv_hist > 0 else 1.0

        count_ult3 = grp.tail(3)["count_trx"].mean() if n_meses >= 3 else grp["count_trx"].mean()
        count_hist = grp["count_trx"].mean()
        f["ratio_count_3m_vs_total"] = count_ult3 / count_hist if count_hist > 0 else 1.0

        # --- Máxima caída mensual (mes a mes) ---
        if n_meses >= 2:
            tpv_mensual = grp["tpv_mensual"].values
            cambios = []
            for i in range(1, len(tpv_mensual)):
                if tpv_mensual[i-1] > 0:
                    cambios.append((tpv_mensual[i] - tpv_mensual[i-1]) / tpv_mensual[i-1])
            f["max_caida_mensual_tpv"] = min(cambios) if cambios else 0
            f["volatilidad_tpv"] = np.std(cambios) if cambios else 0
        else:
            f["max_caida_mensual_tpv"] = 0
            f["volatilidad_tpv"] = 0

        # --- Meses desde el último mes con trx > 0 ---
        meses_activos = grp[grp["count_trx"] > 0]
        if len(meses_activos) > 0:
            ultimo_activo = meses_activos["mes_reporte"].max()
            f["meses_desde_ultimo_activo"] = int(
                (fecha_corte - ultimo_activo).days / 30.44
            )
        else:
            f["meses_desde_ultimo_activo"] = n_meses

        features.append(f)

    return pd.DataFrame(features)


# ============================================================
# 3. FEATURES DE SOPORTE (desde fact_support_tickets)
# ============================================================

def features_soporte(df_tickets: pd.DataFrame, fecha_corte: datetime) -> pd.DataFrame:
    """Features derivadas del detalle de tickets de soporte."""
    df = df_tickets.copy()
    df["fecha_apertura"] = pd.to_datetime(df["fecha_apertura"])
    df = df[df["fecha_apertura"] <= fecha_corte]

    # --- Ventanas de tiempo ---
    limite_30 = fecha_corte - timedelta(days=30)
    limite_90 = fecha_corte - timedelta(days=90)
    limite_180 = fecha_corte - timedelta(days=180)

    df["en_30d"] = df["fecha_apertura"] >= limite_30
    df["en_90d"] = df["fecha_apertura"] >= limite_90
    df["en_180d"] = df["fecha_apertura"] >= limite_180

    features = []
    for mid, grp in df.groupby("merchant_id"):
        f = {"merchant_id": mid}

        # --- Conteos por ventana ---
        f["tickets_total"] = len(grp)
        f["tickets_30d"] = grp["en_30d"].sum()
        f["tickets_90d"] = grp["en_90d"].sum()
        f["tickets_180d"] = grp["en_180d"].sum()

        # --- Estados ---
        no_resueltos = grp[grp["estado"].isin(["abierto", "en_proceso", "escalado"])]
        f["tickets_no_resueltos_actuales"] = len(no_resueltos)
        f["tickets_escalados"] = (grp["estado"] == "escalado").sum()
        f["ratio_resueltos"] = (grp["estado"] == "resuelto").sum() / max(len(grp), 1)

        # --- Severidad ---
        f["severidad_max"] = grp["severidad"].max() if len(grp) > 0 else 0
        f["severidad_prom"] = grp["severidad"].mean() if len(grp) > 0 else 0
        if grp["en_90d"].sum() > 0:
            f["severidad_prom_90d"] = grp.loc[grp["en_90d"], "severidad"].mean()
            f["severidad_max_90d"] = grp.loc[grp["en_90d"], "severidad"].max()
        else:
            f["severidad_prom_90d"] = 0
            f["severidad_max_90d"] = 0

        # --- Tiempo de resolución ---
        resueltos = grp[grp["estado"] == "resuelto"]
        if len(resueltos) > 0:
            f["tiempo_resolucion_prom"] = resueltos["tiempo_resolucion_hrs"].mean()
            f["tiempo_resolucion_max"] = resueltos["tiempo_resolucion_hrs"].max()
        else:
            f["tiempo_resolucion_prom"] = 0
            f["tiempo_resolucion_max"] = 0

        # --- Satisfacción (si hay respuestas) ---
        sat = grp["satisfaccion_post_cierre"].dropna()
        f["satisfaccion_prom"] = sat.mean() if len(sat) > 0 else 3.0  # imputamos neutro

        # --- Categorías críticas (ratio) ---
        if len(grp) > 0:
            f["ratio_pago_rechazado"] = (grp["categoria"] == "pago_rechazado").mean()
            f["ratio_liquidacion_demora"] = (grp["categoria"] == "liquidacion_demora").mean()
            f["ratio_app_congelada"] = (grp["categoria"] == "app_congelada").mean()
        else:
            f["ratio_pago_rechazado"] = 0
            f["ratio_liquidacion_demora"] = 0
            f["ratio_app_congelada"] = 0

        features.append(f)

    return pd.DataFrame(features)


# ============================================================
# 4. ORQUESTADOR
# ============================================================

def construir_dataset_features(path_merchants: str,
                                path_performance: str,
                                path_tickets: str,
                                fecha_corte: datetime = None
                                ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Construye el dataset final para entrenar el modelo.

    Returns:
        X: DataFrame con features (solo numéricas)
        y: Series con la etiqueta abandono_30d
        df_full: DataFrame completo con merchant_id + features + y (para análisis)
    """
    df_m = pd.read_csv(path_merchants)
    df_p = pd.read_csv(path_performance)
    df_t = pd.read_csv(path_tickets)

    if fecha_corte is None:
        # Por defecto, fecha_corte = último día del dataset
        fecha_corte = pd.to_datetime(df_p["mes_reporte"]).max() + pd.offsets.MonthEnd(0)

    print(f"Construyendo features con fecha_corte = {fecha_corte.date()}")

    # 1. Calcular cada bloque de features
    fe_est = features_estaticas(df_m, fecha_corte)
    fe_perf = features_performance(df_p, fecha_corte)
    fe_sop = features_soporte(df_t, fecha_corte)

    # 2. Merge con el dataset maestro
    df = df_m[["merchant_id", "abandono_30d"]].copy()
    df = df.merge(fe_est, on="merchant_id", how="left")
    df = df.merge(fe_perf, on="merchant_id", how="left")
    df = df.merge(fe_sop, on="merchant_id", how="left")

    # 3. Imputar NaN (comercios sin tickets → 0, comercios sin performance → 0)
    df = df.fillna(0)

    # 4. Separar X / y
    y = df["abandono_30d"].copy()
    X = df.drop(columns=["merchant_id", "abandono_30d"])

    print(f"Dataset generado: {X.shape[0]} comercios × {X.shape[1]} features")
    print(f"Tasa de abandono: {y.mean():.3f}")
    print(f"Features disponibles:")
    for col in X.columns:
        print(f"  - {col}")

    return X, y, df


# ============================================================
# Utilidad para uso directo
# ============================================================
if __name__ == "__main__":
    X, y, df_full = construir_dataset_features(
        "data/raw/dim_merchants_con_abandono.csv",
        "data/raw/fact_performance_monthly.csv",
        "data/raw/fact_support_tickets.csv",
    )
    print("\nEjemplo de primeras 3 filas de X:")
    print(X.head(3))
    print(f"\nShape final: X={X.shape}, y={y.shape}")