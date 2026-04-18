# Plan de Acción por Nivel de Riesgo

Guía operativa para que el equipo comercial y de retención de Deuna accione sobre los comercios priorizados por el modelo. **Todas las acciones son ejecutables con recursos disponibles hoy** (no requieren integraciones externas, herramientas nuevas ni presupuesto adicional).

---

## 🔴 ALERTA ROJA — Top 5% (percentil ≥ 95)

**Perfil típico:** recencia > 20 días, caída de volumen > 40% vs. 3 meses atrás, tickets de soporte sin resolver.

**SLA:** contacto en las próximas **48 horas**.

| Acción | Responsable | Canal |
|---|---|---|
| Llamada del ejecutivo de cuenta con guion de recuperación | KAM / ejecutivo B2B | Teléfono |
| Auditoría técnica del POS y estado de la integración ecommerce | Soporte técnico Tier 2 | Remoto / presencial |
| Revisión de pricing y propuesta de descuento condicionado a volumen | Equipo comercial | Email / reunión |
| Escalamiento de tickets abiertos a resolución en < 24h | Customer Success | Interno |

**Señal de éxito:** comercio vuelve a transar en los siguientes 14 días.

---

## 🟡 ALERTA AMARILLA — Percentil 89–95

**Perfil típico:** señales mixtas — volatilidad creciente, decline rate elevado, pero aún con actividad regular.

**SLA:** contacto en la próxima **semana**.

| Acción | Responsable | Canal |
|---|---|---|
| Llamada de "check-in" proactivo sin guion de descuento | KAM | Teléfono |
| Envío de material educativo sobre nuevas funciones / mejores prácticas | Marketing | Email |
| Diagnóstico de fricciones transaccionales (declines recurrentes) | Soporte técnico | Email |
| Invitación a webinar o capacitación segmentada por MCC | Marketing | Email |

**Señal de éxito:** reducción de la `prob_rank` en el scoring del siguiente mes.

---

## 🟢 BAJA PROBABILIDAD — Percentil 82–89

**Perfil típico:** comercios estables con señales puntuales de deterioro que no han escalado.

**SLA:** touchpoint pasivo en el ciclo regular de **mensual**.

| Acción | Responsable | Canal |
|---|---|---|
| Email de valor con dashboard personalizado de su operación | CRM | Email automatizado |
| Encuesta NPS breve (1 pregunta) | CRM | Email |
| Oferta cruzada de productos complementarios (ecommerce si solo POS) | Comercial | Email |

**Señal de éxito:** respuesta a la encuesta y/o activación del canal sugerido.

---

## ⚪ MUY BAJA PROBABILIDAD — Percentil < 82

**Perfil típico:** comercios saludables. El foco es **crecimiento**, no retención.

**SLA:** comunicación estándar del ciclo comercial.

| Acción | Responsable | Canal |
|---|---|---|
| Newsletter mensual con novedades de producto | Marketing | Email |
| Programa de referidos y reconocimiento por volumen | Marketing | Email |
| Campañas de upsell (ej. herramientas premium, reporting avanzado) | Comercial | Email |

---

## Criterios de ejecución transversal

1. **Priorización diaria:** el equipo comercial recibe cada día el archivo `predictions.csv` ordenado por `probabilidad_churn`. Los 100 top (≈ top 5%) se asignan por KAM según la región y el volumen histórico.
2. **Trazabilidad:** cada contacto registrado en el CRM existente; el `comercio_id` sirve como llave de join.
3. **Feedback loop:** los resultados de las intervenciones (cierre / recuperación / sin respuesta) se vuelcan al dataset de entrenamiento en el siguiente ciclo mensual del modelo.
4. **Evitar sobre-contacto:** un comercio que ya fue contactado en la semana no se re-contacta, incluso si sigue en ALERTA ROJA.

## Métricas de seguimiento del programa

| Métrica | Fórmula | Meta inicial |
|---|---|---|
| Tasa de recuperación ALERTA ROJA | recuperados / contactados | ≥ 30% |
| Lift vs. línea base | churn observado en scoreados vs. población general | ≥ 2x |
| Cobertura operativa | comercios ALERTA ROJA contactados / total | ≥ 95% en 48h |
| Reducción de churn agregado | tasa churn trimestral pre/post modelo | −15% trimestre a trimestre |
