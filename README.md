# Modelo de Churn Deuna — Reto 3 Datos, Interact2Hack 2026

**"Antes de que se vayan"** — Scoring predictivo de abandono de comerciantes B2B para Deuna.

El modelo identifica comercios con alto riesgo de dejar de transar en los próximos 30 días y los segmenta en niveles de alerta accionables por el equipo comercial.

---

## Criterios de éxito cumplidos

| Criterio del reto | Resultado |
|---|---|
| AUC > 0.75 (orientativo) | **AUC 0.885** en test |
| Variables explicativas coherentes con negocio | Top features: volatilidad de transacciones, recencia, decline rate, tickets de soporte |
| Plan de acción ejecutable sin recursos nuevos | [docs/plan_accion.md](docs/plan_accion.md) |
| Output consumible por el equipo de front | [outputs/predictions.csv](outputs/predictions.csv) |

---

## Estructura

```
churn_deuna/
├── src/
│   ├── data_simulation.py      Genera dataset sintético (2000 comercios × 12 meses)
│   ├── feature_engineering.py  MDT con ventanas, deltas, agregados y ratios
│   ├── train_model.py          XGBoost + métricas + umbral óptimo F1
│   ├── explain.py              SHAP: importancia global + valores individuales
│   └── predict.py              Scoring + segmentación (Roja/Amarilla/Baja/Muy Baja)
├── notebooks/
│   └── 01_modelo_churn.ipynb   Pipeline end-to-end con EDA
├── data/
│   ├── raw/                    merchants.csv, monthly_activity.csv, churn_labels.csv
│   └── processed/              mdt_churn.parquet
├── outputs/
│   ├── model/                  churn_model.pkl, metrics.json
│   ├── figures/                shap_summary.png, shap_bar.png
│   ├── predictions.csv         Input para el equipo de dashboard
│   └── shap_values.parquet     Valores SHAP por comercio (uso individual)
├── docs/
│   ├── diccionario_variables.md
│   └── plan_accion.md
├── requirements.txt
└── README.md
```

---

## Cómo ejecutar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# En macOS además:
brew install libomp

# 2. Pipeline completo (desde la raíz del proyecto)
python src/data_simulation.py        # Genera datos sintéticos
python src/feature_engineering.py    # Construye la MDT
python src/train_model.py            # Entrena XGBoost
python src/explain.py                # Genera SHAP plots
python src/predict.py                # Scoring + predictions.csv

# O bien ejecutar el notebook completo:
jupyter lab notebooks/01_modelo_churn.ipynb
```

---

## Decisiones técnicas

| Decisión | Justificación |
|---|---|
| **XGBoost** como clasificador | Estado del arte en datos tabulares B2B; documentación de Deuna lo recomienda explícitamente sobre redes neuronales por su superior interpretabilidad. |
| **SHAP** para explicabilidad | Valores aditivos con base matemática sólida; permite explicación global (qué variables predicen churn) e individual (por qué este comercio en particular). |
| **Ventanas 0–4 meses + deltas** | Replica el patrón del flujo productivo Databricks de modelos similares; captura aceleración de deterioro. |
| **Segmentación por percentil** | Misma lógica del flujo de referencia (P95/P89/P82). Hace la priorización independiente del volumen absoluto de la cartera. |
| **Target binario a 30 días** | Alineado con el hilo operativo del reto; se consolida la ventana de acción comercial a 30 días. |
| **Umbral óptimo F1** | Balancea precisión y recall; se guarda en `metrics.json` para uso del dashboard. |

---

## Variables más predictivas (Top 10 SHAP)

1. `tx_std_12m` — volatilidad de transacciones (12 meses)
2. `n_transacciones_0` — transacciones del mes actual
3. `volatilidad_tx_6m` — std/mean de transacciones (6 meses)
4. `dias_desde_ult_tx_1` — recencia mes anterior
5. `volumen_std_12m` — volatilidad de volumen
6. `tx_sum_3m` — volumen de transacciones últimos 3 meses
7. `decline_rate_6m` — tasa de rechazos
8. `dias_desde_ult_tx_0` — recencia del mes de corte
9. `dias_resolucion_soporte_1` — latencia soporte mes anterior
10. `dias_resolucion_soporte_0` — latencia soporte mes actual

**Coherencia con el negocio:** las señales cardinales son recencia, volatilidad transaccional y fricciones de soporte/autorización — exactamente las identificadas en la literatura B2B de merchant acquiring.

---

## Entrega para el equipo de dashboard

El archivo **`outputs/predictions.csv`** contiene una fila por comercio con:

| Columna | Uso en el dashboard |
|---|---|
| `comercio_id` | Clave para lookup |
| `probabilidad_churn` | Score [0,1] — ordenar y colorear |
| `segmento_churn` | Semáforo (Roja/Amarilla/Baja/Muy Baja) |
| `mcc_segmento`, `region`, `tipo_persona` | Filtros por dimensión de negocio |
| `volumen_sum_6m`, `tx_sum_6m` | Priorización por impacto económico |
| `recencia_bucket_0` | Vista rápida de estado operativo |

Adicionalmente `outputs/shap_values.parquet` permite mostrar, por cada comercio, las 3 razones principales de su riesgo (para el drill-down individual en el dashboard).

---

## Próximos pasos recomendados

1. **Calibración** — aplicar isotonic regression para que `probabilidad_churn` sea una probabilidad real (útil para el ROI del plan de acción).
2. **Análisis de supervivencia** — migrar a Cox / XGBoost-AFT para predecir *tiempo hasta churn*, no solo probabilidad en ventana fija.
3. **Reentrenamiento mensual** — re-scoring con la nueva fecha de corte al cierre de cada mes.
4. **A/B test del plan de acción** — medir lift real de las intervenciones comerciales.
