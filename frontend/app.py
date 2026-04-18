"""🏠 Churn Deuna — Dashboard Ejecutivo

Landing page del dashboard de predicción de churn de comerciantes B2B.
Muestra KPIs principales y navegación a las sub-páginas.
"""
import sys
from pathlib import Path

import streamlit as st

# Asegurar imports del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontend.utils.data_loader import load_predictions, load_metrics

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Deuna — Dashboard",
    page_icon="🔮",
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
    alerta_roja = len(predictions[predictions["segmento_churn"] == "ALERTA_ROJA"])
    alerta_amarilla = len(predictions[predictions["segmento_churn"] == "ALERTA_AMARILLA"])
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
        st.markdown("### 📊 Rendimiento del Modelo")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC-ROC", f"{metrics.get('auc', 0):.3f}")
        m2.metric("AUC-PR", f"{metrics.get('pr_auc', 0):.3f}")
        m3.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
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
