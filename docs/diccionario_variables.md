# Diccionario de Variables — Modelo de Churn Deuna

## Tablas fuente

### 1. `merchants.csv` — Atributos estáticos del comercio

| Variable | Tipo | Descripción |
|---|---|---|
| `comercio_id` | string | Identificador único del comercio |
| `mcc` | int | Merchant Category Code (ISO 18245) |
| `mcc_nombre` | string | Nombre legible del MCC |
| `mcc_segmento` | categoría | Agrupación de riesgo: `retail`, `servicios`, `alto_riesgo`, `utility` |
| `region` | categoría | Provincia del Ecuador |
| `tipo_persona` | categoría | `PERSONA_NATURAL` / `PERSONA_JURIDICA` |
| `tenure_meses` | int | Meses desde la afiliación del comercio |
| `fecha_alta` | date | Fecha de vinculación con Deuna |

### 2. `monthly_activity.csv` — Serie mensual de comportamiento

Una fila por `(comercio_id, fecha_corte)`. El corte corresponde al inicio de cada mes.

| Variable | Tipo | Descripción |
|---|---|---|
| `n_transacciones` | int | Transacciones liquidadas en el mes |
| `volumen_total` | float | Volumen procesado bruto (USD) |
| `ticket_promedio` | float | Ticket promedio del mes (AOV) |
| `n_chargebacks` | int | Contracargos registrados |
| `n_rechazos` | int | Transacciones rechazadas por el issuer/gateway |
| `dias_desde_ult_tx` | int | Días transcurridos desde la última transacción exitosa |
| `n_tickets_soporte` | int | Número de tickets de soporte abiertos |
| `dias_resolucion_soporte` | float | Días promedio de resolución de tickets |
| `flag_pos_fisico` | 0/1 | Uso activo de terminal POS en el mes |
| `flag_ecommerce` | 0/1 | Uso activo de canal ecommerce |
| `n_integraciones_activas` | int | API/conectores activos (ERP, CRM, etc.) |

### 3. `churn_labels.csv` — Etiqueta objetivo

| Variable | Tipo | Descripción |
|---|---|---|
| `comercio_id` | string | Identificador |
| `fecha_corte` | date | Fecha de corte del scoring |
| `churn_30d` | 0/1 | **Target.** 1 si el comercio dejó de transar en los 30 días siguientes al corte |

---

## Features derivadas (MDT — `mdt_churn.parquet`)

### Ventanas temporales `_{0,1,2,3,4}`
Para cada métrica de actividad se construyen 5 lags mensuales:
- `_0` = mes de corte
- `_1` = mes anterior al corte
- …
- `_4` = cuatro meses antes del corte

Métricas con lags: `n_transacciones`, `volumen_total`, `ticket_promedio`, `dias_desde_ult_tx`, `n_chargebacks`, `n_rechazos`, `n_tickets_soporte`, `dias_resolucion_soporte`, `n_integraciones_activas`, `flag_pos_fisico`, `flag_ecommerce`.

### Agregados de ventana rodante (3m / 6m / 12m)
- `tx_sum_{w}m`, `tx_mean_{w}m`, `tx_std_{w}m`
- `volumen_sum_{w}m`, `volumen_mean_{w}m`, `volumen_std_{w}m`
- `chargebacks_sum_{w}m`, `rechazos_sum_{w}m`, `tickets_sum_{w}m`
- `meses_activos_{w}m` — meses con transacciones > 0

### Deltas consecutivos
Siguiendo el patrón Databricks (`DeltaSaldoX_Y`):
- `delta_n_transacciones_{0_1, 1_2, 2_3, 3_4}`
- `delta_volumen_total_*`, `delta_ticket_promedio_*`, `delta_dias_desde_ult_tx_*`, `delta_n_tickets_soporte_*`

### Ratios y derivadas de negocio
| Variable | Fórmula | Interpretación |
|---|---|---|
| `chargeback_rate_6m` | `chargebacks_sum_6m / tx_sum_6m` | Tasa de disputa — riesgo operativo |
| `decline_rate_6m` | `rechazos_sum_6m / tx_sum_6m` | Fricción de autorización |
| `tickets_per_tx_6m` | `tickets_sum_6m / tx_sum_6m` | Fricción de soporte |
| `volatilidad_volumen_6m` | `volumen_std_6m / volumen_mean_6m` | Inestabilidad transaccional |
| `volatilidad_tx_6m` | `tx_std_6m / tx_mean_6m` | Inestabilidad de frecuencia |
| `omnicanal_0` | `flag_pos + flag_ecommerce` en mes 0 | Diversificación de canal |
| `recencia_bucket_0` | bucket categórico sobre `dias_desde_ult_tx_0` | Segmentación operativa |

### Bucket de recencia
Replica del flujo Databricks (`DiasUltimoIngreso`):
- `menos_5` — ≤ 5 días
- `entre_5_y_10`
- `entre_10_y_20`
- `entre_20_y_30`
- `mas_de_30`
- `sin_registro`

---

## Output del modelo — `outputs/predictions.csv`

| Variable | Descripción |
|---|---|
| `comercio_id` | Identificador |
| `fecha_corte` | Fecha del scoring |
| `probabilidad_churn` | Probabilidad calibrada de churn en 30 días [0, 1] |
| `prob_rank` | Percentil de la probabilidad sobre la cartera |
| `segmento_churn` | `ALERTA_ROJA` / `ALERTA_AMARILLA` / `BAJA_PROBABILIDAD` / `MUY_BAJA_PROBABILIDAD` |
| `mcc_segmento`, `region`, `tipo_persona`, `tenure_meses` | Atributos del comercio para filtrado |
| `volumen_sum_6m`, `tx_sum_6m`, `recencia_bucket_0` | Métricas clave para priorización comercial |

### Umbrales de segmentación
Basados en `prob_rank` (percentil sobre la cartera scoreada):
- `ALERTA_ROJA` — percentil ≥ 95 (top 5%)
- `ALERTA_AMARILLA` — percentil 89–95
- `BAJA_PROBABILIDAD` — percentil 82–89
- `MUY_BAJA_PROBABILIDAD` — percentil < 82
