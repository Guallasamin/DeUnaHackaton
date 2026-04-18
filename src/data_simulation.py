"""Generador de datos sintéticos de comerciantes Deuna para modelo de churn.

Produce tres tablas:
    - merchants.csv        : atributos estáticos del comercio
    - monthly_activity.csv : serie mensual de comportamiento (12 meses)
    - churn_labels.csv     : etiqueta de churn en los 30 días posteriores al corte
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


N_MERCHANTS = 2000
N_MONTHS = 12
SEED = 42

CORTE_MES_IDX = 10

MCC_CATALOG = {
    5411: ("Supermercados",          "retail",       0.45),
    5812: ("Restaurantes",            "retail",       0.75),
    5999: ("Retail general",          "retail",       0.85),
    5691: ("Tiendas de ropa",         "retail",       0.95),
    5732: ("Electrónica",             "retail",       1.05),
    7011: ("Hoteles",                 "servicios",    1.10),
    4722: ("Agencias de viaje",       "servicios",    1.55),
    8062: ("Servicios médicos",       "servicios",    0.60),
    7230: ("Peluquerías y estética",  "servicios",    1.20),
    5815: ("Bienes digitales",        "alto_riesgo",  1.80),
    5813: ("Bares y clubs",           "alto_riesgo",  1.70),
    7995: ("Apuestas",                "alto_riesgo",  2.20),
    4900: ("Servicios básicos",       "utility",      0.20),
    5541: ("Estaciones de servicio",  "utility",      0.35),
}

REGIONES = {
    "Pichincha":   0.28,
    "Guayas":      0.30,
    "Azuay":       0.09,
    "Manabí":      0.08,
    "Tungurahua":  0.05,
    "Loja":        0.04,
    "El Oro":      0.05,
    "Imbabura":    0.04,
    "Chimborazo":  0.04,
    "Esmeraldas":  0.03,
}

TIPO_PERSONA = {"PERSONA_NATURAL": 0.68, "PERSONA_JURIDICA": 0.32}


@dataclass
class SimConfig:
    n_merchants: int = N_MERCHANTS
    n_months: int = N_MONTHS
    seed: int = SEED
    start_date: str = "2025-05-01"
    churn_base_rate: float = 0.18


def _sample_categorical(rng: np.random.Generator, options: dict, size: int) -> np.ndarray:
    keys = list(options.keys())
    probs = np.array(list(options.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(keys, size=size, p=probs)


def generate_merchants(cfg: SimConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    mcc = _sample_categorical(
        rng,
        {code: 1.0 for code in MCC_CATALOG},
        cfg.n_merchants,
    )
    region = _sample_categorical(rng, REGIONES, cfg.n_merchants)
    tipo_persona = _sample_categorical(rng, TIPO_PERSONA, cfg.n_merchants)

    tenure_meses = rng.gamma(shape=2.2, scale=10.0, size=cfg.n_merchants).clip(1, 120).astype(int)

    fecha_corte = pd.Timestamp(cfg.start_date) + pd.DateOffset(months=cfg.n_months - 1)
    fecha_alta = [fecha_corte - pd.DateOffset(months=int(m)) for m in tenure_meses]

    merchants = pd.DataFrame({
        "comercio_id":    [f"M{str(i).zfill(6)}" for i in range(cfg.n_merchants)],
        "mcc":            mcc,
        "mcc_nombre":     [MCC_CATALOG[c][0] for c in mcc],
        "mcc_segmento":   [MCC_CATALOG[c][1] for c in mcc],
        "region":         region,
        "tipo_persona":   tipo_persona,
        "tenure_meses":   tenure_meses,
        "fecha_alta":     fecha_alta,
    })
    return merchants


def _churn_propensity(merchants: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Score latente de propensión a churnear (logit)."""
    mcc_risk = np.array([MCC_CATALOG[c][2] for c in merchants["mcc"]])
    tenure = merchants["tenure_meses"].values

    logit = (
        -1.6
        + 0.55 * (mcc_risk - 1.0)
        - 0.025 * np.log1p(tenure)
        + 0.25 * (merchants["tipo_persona"].values == "PERSONA_NATURAL").astype(float)
        + rng.normal(0, 0.45, size=len(merchants))
    )
    return 1 / (1 + np.exp(-logit))


def generate_monthly_activity(merchants: pd.DataFrame, cfg: SimConfig) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(cfg.seed + 1)
    prop = _churn_propensity(merchants, rng)

    will_churn = rng.uniform(size=len(merchants)) < prop

    churn_month_idx = np.where(
        will_churn,
        CORTE_MES_IDX + 1,
        cfg.n_months + 99,
    )

    base_tx = rng.gamma(shape=2.5, scale=60, size=len(merchants)).clip(5, 2500)
    base_ticket = rng.gamma(shape=3.0, scale=8, size=len(merchants)).clip(3, 400)

    start = pd.Timestamp(cfg.start_date)
    months = [start + pd.DateOffset(months=i) for i in range(cfg.n_months)]

    records = []
    for idx, row in merchants.iterrows():
        mcc_risk = MCC_CATALOG[row["mcc"]][2]
        churn_at = churn_month_idx[idx]

        for m, month_date in enumerate(months):
            months_to_churn = churn_at - m

            if months_to_churn <= 0:
                n_tx = 0
                volumen = 0.0
                ticket = 0.0
                dias_ult_tx = 30 + (m - churn_at + 1) * 30
                n_chargebacks = 0
                n_rechazos = 0
                n_tickets = 0
                dias_resolucion = 0.0
                flag_pos = 0
                flag_ecom = 0
                n_integraciones = 0
            else:
                decay = 1.0
                if 0 < months_to_churn <= 5:
                    decay = max(0.55, 1 - (6 - months_to_churn) * 0.08)

                shock_noise = rng.uniform() < 0.07
                shock_decay = rng.uniform(0.40, 0.80) if shock_noise else 1.0
                decay *= shock_decay

                seasonal = 1.0 + 0.10 * np.sin(2 * np.pi * m / 12)
                noise = rng.normal(1.0, 0.30)

                n_tx_float = base_tx[idx] * decay * seasonal * noise
                n_tx = max(0, int(n_tx_float))

                ticket_noise = rng.normal(1.0, 0.18)
                ticket = max(1.0, base_ticket[idx] * ticket_noise * (0.85 + 0.15 * decay))
                volumen = n_tx * ticket

                chargeback_rate = 0.003 * mcc_risk * (1.3 + 0.25 * (1 - decay))
                n_chargebacks = rng.binomial(n=max(n_tx, 1), p=min(chargeback_rate, 0.08))

                decline_rate = 0.05 + 0.02 * (1 - decay) + 0.02 * (mcc_risk - 0.5)
                n_rechazos = rng.binomial(n=max(n_tx, 1), p=min(max(decline_rate, 0.01), 0.30))

                dias_ult_tx = int(max(0, rng.exponential(scale=3 + (1 - decay) * 6)))

                tickets_lambda = 0.5 + 0.5 * (1 - decay) + 0.3 * mcc_risk
                n_tickets = rng.poisson(lam=max(0.1, tickets_lambda))
                dias_resolucion = 0.0 if n_tickets == 0 else rng.gamma(shape=2, scale=1.8 + (1 - decay) * 2)

                flag_pos = int(rng.uniform() < 0.65)
                flag_ecom = int(rng.uniform() < 0.45)
                if flag_pos + flag_ecom == 0:
                    flag_pos = 1
                n_integraciones = rng.poisson(lam=1.2 + 0.8 * (flag_ecom))

            records.append({
                "comercio_id": row["comercio_id"],
                "fecha_corte": month_date,
                "mes_idx": m,
                "n_transacciones": n_tx,
                "volumen_total": round(volumen, 2),
                "ticket_promedio": round(ticket, 2),
                "n_chargebacks": n_chargebacks,
                "n_rechazos": n_rechazos,
                "dias_desde_ult_tx": dias_ult_tx,
                "n_tickets_soporte": n_tickets,
                "dias_resolucion_soporte": round(dias_resolucion, 2),
                "flag_pos_fisico": flag_pos,
                "flag_ecommerce": flag_ecom,
                "n_integraciones_activas": n_integraciones,
            })

    activity = pd.DataFrame.from_records(records)
    activity = activity[activity["mes_idx"] <= CORTE_MES_IDX].copy()

    churn_label = pd.Series(will_churn.astype(int), index=merchants["comercio_id"], name="churn_30d")
    return activity, churn_label


def generate_churn_labels(
    merchants: pd.DataFrame,
    activity: pd.DataFrame,
    churn_series: pd.Series,
    cfg: SimConfig,
) -> pd.DataFrame:
    """Etiqueta construida sobre el último mes observado.

    churn_30d = 1 si el comercio no vuelve a transar en el siguiente período.
    """
    last_month = activity["fecha_corte"].max()
    corte = activity[activity["fecha_corte"] == last_month][["comercio_id"]].copy()
    corte["fecha_corte"] = last_month
    corte = corte.merge(
        churn_series.rename("churn_30d").reset_index(),
        on="comercio_id",
        how="left",
    )
    return corte


def main(out_dir: str | Path = "data/raw") -> dict[str, Path]:
    cfg = SimConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merchants = generate_merchants(cfg)
    activity, churn_series = generate_monthly_activity(merchants, cfg)
    labels = generate_churn_labels(merchants, activity, churn_series, cfg)

    paths = {
        "merchants":       out_dir / "merchants.csv",
        "monthly_activity": out_dir / "monthly_activity.csv",
        "churn_labels":    out_dir / "churn_labels.csv",
    }
    merchants.to_csv(paths["merchants"], index=False)
    activity.to_csv(paths["monthly_activity"], index=False)
    labels.to_csv(paths["churn_labels"], index=False)

    print(f"Merchants:        {len(merchants):>6} filas → {paths['merchants']}")
    print(f"Monthly activity: {len(activity):>6} filas → {paths['monthly_activity']}")
    print(f"Churn labels:     {len(labels):>6} filas → {paths['churn_labels']}")
    print(f"Tasa de churn:    {labels['churn_30d'].mean():.2%}")
    return paths


if __name__ == "__main__":
    main()
