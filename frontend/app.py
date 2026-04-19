"""🏠 Churn Deuna — Dashboard Ejecutivo

Landing page del dashboard de predicción de churn de comerciantes B2B.
Muestra KPIs principales y navegación a las sub-páginas.
"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Asegurar imports del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontend.utils.data_loader import load_predictions, load_metrics

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Deuna — Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Cargar CSS custom
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="header-container">
            <h1 style="display: flex; align-items: center; gap: 12px; margin-bottom: 0;">
                <img src="https://res.cloudinary.com/doy9vd3pj/image/upload/q_auto/f_auto/v1776540402/unnamed_ooahqu.png" width="45" style="border-radius: 8px;">
                Churn Deuna
            </h1>
            <p class="subtitle">Sistema predictivo de abandono de comerciantes B2B</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Cargar datos ──────────────────────────────────────────────────────────
    predictions = load_predictions()
    metrics = load_metrics()

    if predictions is None:
        st.error(
            "⚠️ No se encontraron predicciones. Ejecuta primero el pipeline del modelo:\n\n"
            "```bash\nmake model\n```"
        )
        return

    # ── KPIs principales ─────────────────────────────────────────────────────
    total = len(predictions)
    alerta_roja = len(predictions[predictions["nivel_riesgo"].isin(["Crítico","Alto"])])
    alerta_amarilla = len(predictions[predictions["nivel_riesgo"] =="Medio" ])
    prob_media = predictions["probabilidad_churn"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{total:,}</div>
                <div class="kpi-label">Comercios Scoreados</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card kpi-danger">
                <div class="kpi-value">{alerta_roja}</div>
                <div class="kpi-label">🔴 Alerta Roja</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card kpi-warning">
                <div class="kpi-value">{alerta_amarilla}</div>
                <div class="kpi-label">🟡 Alerta Amarilla</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-value">{prob_media:.1%}</div>
                <div class="kpi-label">Prob. Media de Churn</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Métricas del modelo (si existen) ──────────────────────────────────────
    if metrics:
        # Usualmente en los dashboards mostramos el rendimiento de 'test' o 'val'
        st.markdown("### 📊 Rendimiento del Modelo (Test)")
        
        # 1. Extraemos el diccionario de métricas de test para no repetir código
        test_metrics = metrics.get("splits", {}).get("test", {})
        
        # 2. Creamos las 4 columnas
        m1, m2, m3, m4 = st.columns(4)
        
        # 3. Llenamos las 3 primeras sacando los datos de 'test_metrics'
        m1.metric("AUC-ROC", f"{test_metrics.get('auc_roc', 0):.3f}")
        m2.metric("AUC-PR", f"{test_metrics.get('auc_pr', 0):.3f}")
        m3.metric("F1 Score", f"{test_metrics.get('f1', 0):.3f}")
        
        # 4. Llenamos la 4ta sacando el dato directamente de 'metrics' (la raíz)
        m4.metric("Umbral F1", f"{metrics.get('threshold', 0):.3f}")

    st.markdown("---")

    # ── Navegación ────────────────────────────────────────────────────────────
    st.markdown("### 🧭 Navega el Dashboard")

    nav1, nav2, nav3 = st.columns(3)

    with nav1:
        st.markdown(
            """
            <div class="nav-card">
                <h3>📊 Dashboard</h3>
                <p>Distribución de segmentos, análisis por región y MCC, top comercios en riesgo.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav2:
        st.markdown(
            """
            <div class="nav-card">
                <h3>🔍 Explorador</h3>
                <p>Busca un comercio específico y analiza sus razones de riesgo con SHAP.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav3:
        st.markdown(
            """
            <div class="nav-card">
                <h3>📈 Modelo</h3>
                <p>Métricas detalladas, distribución de probabilidades y gráficos SHAP globales.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("Reto 3 — Interact2Hack 2026 | \"Antes de que se vayan\"")


if __name__ == "__main__":
    main()
