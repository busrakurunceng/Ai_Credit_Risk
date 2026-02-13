"""
AI Credit Risk — Streamlit Dashboard
=====================================
Kredi risk skoru hesaplama, senaryo simülasyonu ve açıklanabilirlik arayüzü.

Şu an mock data ile çalışır. Backend (model) hazır olduğunda
get_mock_*() fonksiyonları gerçek API çağrılarıyla değiştirilecek.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import time

# ─────────────────────────────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Credit Risk — Risk Analizi",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# ÖZEL CSS — Kurumsal finans teması
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Ana container padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Metrik kartları */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fc 0%, #eef1f8 100%);
        border: 1px solid #e2e6ef;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #5a6a85 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #1a2332 !important;
    }

    /* Risk bandı */
    .risk-banner {
        text-align: center;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        margin-top: 8px;
    }
    .risk-low    { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .risk-medium { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .risk-high   { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }

    /* Bölüm başlıkları */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a2332;
        border-bottom: 2px solid #4a6cf7;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Sidebar bilgi kutusu */
    .info-box {
        background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
        border-left: 4px solid #4a6cf7;
        border-radius: 0 8px 8px 0;
        padding: 14px 16px;
        font-size: 0.85rem;
        color: #3b4f74;
        line-height: 1.5;
    }

    /* Simülasyon sonuç kutusu */
    .sim-result {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .sim-result-negative {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fca5a5;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    /* Küçük footer */
    .footer-text {
        text-align: center;
        color: #9ca3af;
        font-size: 0.75rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# MOCK DATA — Backend hazır olduğunda bu bölüm değiştirilecek
# ═════════════════════════════════════════════════════════════════════

MOCK_CUSTOMERS = {
    "Müşteri #10234 — Düşük Risk": {
        "id": 10234,
        "risk_score": 18,
        "credit_amount": 450_000,
        "annual_income": 285_000,
        "current_debt": 32_000,
        "credit_score": 0.76,          # EXT_SOURCE_2
        "credit_term": 24,             # ay
        "age": 42,
        "employment_years": 8.5,
        "late_payment_ratio": 0.0,
        "previous_refusals": 0,
        "goods_credit_diff": 0,
        "education": "Yüksek Lisans",
        "payment_history": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "shap_values": {
            "Harici Kredi Skoru (EXT_SOURCE_2)":  -0.81,
            "Kredi-Mal Fiyat Farkı":              -0.20,
            "Geç Ödeme Oranı":                    -0.09,
            "Eğitim Seviyesi":                    -0.10,
            "Kredi Vadesi":                       -0.06,
            "Çalışma Süresi":                     -0.05,
            "Gelir / Kredi Oranı":                -0.04,
            "Önceki Başvuru Reddi":               -0.02,
        },
    },
    "Müşteri #20567 — Orta Risk": {
        "id": 20567,
        "risk_score": 47,
        "credit_amount": 675_000,
        "annual_income": 165_000,
        "current_debt": 128_000,
        "credit_score": 0.48,
        "credit_term": 48,
        "age": 33,
        "employment_years": 2.3,
        "late_payment_ratio": 0.15,
        "previous_refusals": 1,
        "goods_credit_diff": -85_000,
        "education": "Lisans",
        "payment_history": [0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 0],
        "shap_values": {
            "Harici Kredi Skoru (EXT_SOURCE_2)":  +0.12,
            "Kredi-Mal Fiyat Farkı":              +0.10,
            "Geç Ödeme Oranı":                    +0.08,
            "Kredi Vadesi":                       +0.09,
            "Çalışma Süresi":                     +0.06,
            "Eğitim Seviyesi":                    -0.03,
            "Gelir / Kredi Oranı":                +0.05,
            "Önceki Başvuru Reddi":               +0.04,
        },
    },
    "Müşteri #35891 — Yüksek Risk": {
        "id": 35891,
        "risk_score": 82,
        "credit_amount": 900_000,
        "annual_income": 108_000,
        "current_debt": 340_000,
        "credit_score": 0.11,
        "credit_term": 72,
        "age": 26,
        "employment_years": 0.8,
        "late_payment_ratio": 0.50,
        "previous_refusals": 4,
        "goods_credit_diff": -142_000,
        "education": "Lise",
        "payment_history": [0, 2, 3, 1, 2, 5, 3, 2, 4, 1, 3, 2],
        "shap_values": {
            "Harici Kredi Skoru (EXT_SOURCE_2)":  +0.64,
            "Kredi-Mal Fiyat Farkı":              +0.16,
            "Geç Ödeme Oranı":                    +0.18,
            "Kredi Vadesi":                       +0.18,
            "Çalışma Süresi":                     +0.13,
            "Eğitim Seviyesi":                    +0.05,
            "Gelir / Kredi Oranı":                +0.14,
            "Önceki Başvuru Reddi":               +0.13,
        },
    },
}


def get_mock_customer(name: str) -> dict:
    """
    Mock müşteri verisi döndürür.
    # TODO: Backend hazır olduğunda bu fonksiyon API çağrısına dönüşecek:
    #   response = requests.post(API_URL + "/predict", json=customer_data)
    #   return response.json()
    """
    return MOCK_CUSTOMERS[name]


def simulate_risk(customer: dict, new_income: float, new_credit: float) -> dict:
    """
    Değişen parametrelerle yeni risk skoru hesaplar (mock).
    # TODO: Backend hazır olduğunda:
    #   payload = {**customer, "annual_income": new_income, "credit_amount": new_credit}
    #   response = requests.post(API_URL + "/simulate", json=payload)
    #   return response.json()
    """
    original_score = customer["risk_score"]
    original_income = customer["annual_income"]
    original_credit = customer["credit_amount"]

    # Basit mock hesaplama: gelir artarsa risk düşer, kredi artarsa risk artar
    income_effect = (new_income - original_income) / original_income * -15
    credit_effect = (new_credit - original_credit) / original_credit * 20

    new_score = original_score + income_effect + credit_effect
    new_score = int(np.clip(new_score, 1, 99))

    return {
        "original_score": original_score,
        "new_score": new_score,
        "change": new_score - original_score,
    }


def parse_uploaded_json(uploaded_file) -> dict | None:
    """
    Yüklenen JSON dosyasını parse eder.
    # TODO: Backend hazır olduğunda validasyon eklenecek.
    """
    try:
        data = json.load(uploaded_file)
        return data
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════
# UI BİLEŞENLERİ (COMPONENTS)
# ═════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """Sol panel: müşteri seçimi, dosya yükleme, bilgi kutusu."""
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/bank-building.png",
            width=64,
        )
        st.markdown("## 🏦 AI Credit Risk")
        st.caption("Akıllı Kredi Risk Değerlendirme Sistemi")
        st.divider()

        # ── Dosya Yükleme ──
        st.markdown("##### 📂 Müşteri Verisi Yükle")
        uploaded = st.file_uploader(
            "JSON formatında müşteri verisi",
            type=["json"],
            help="Müşteri bilgilerini içeren .json dosyası yükleyin.",
        )

        if uploaded is not None:
            parsed = parse_uploaded_json(uploaded)
            if parsed:
                st.success("Dosya başarıyla yüklendi!")
                st.json(parsed)
            else:
                st.error("Geçersiz JSON formatı.")

        st.divider()

        # ── Örnek Müşteri Seçimi ──
        st.markdown("##### 👤 Örnek Müşteri Profili")
        selected = st.selectbox(
            "Profil seçin",
            options=list(MOCK_CUSTOMERS.keys()),
            index=0,
            help="Farklı risk seviyelerindeki örnek müşterileri inceleyin.",
        )

        st.divider()

        # ── Bilgi Kutusu ──
        st.markdown(
            """
            <div class="info-box">
                <strong>ℹ️ Proje Hakkında</strong><br>
                Bu sistem, Home Credit Default Risk veriseti üzerinde eğitilmiş
                bir <strong>LightGBM</strong> modeli kullanarak müşterilerin kredi
                temerrüt olasılığını tahmin eder.<br><br>
                <strong>Model:</strong> AUC 0.787 · PR-AUC 0.280<br>
                <strong>Kalibrasyon:</strong> Platt Scaling<br>
                <strong>Açıklanabilirlik:</strong> SHAP TreeExplainer
            </div>
            """,
            unsafe_allow_html=True,
        )

    return get_mock_customer(selected)


def render_kpi_panel(customer: dict):
    """Üst metrik paneli — 4 KPI kartı."""
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            label="Kredi Tutarı",
            value=f"₺{customer['credit_amount']:,.0f}",
            delta=f"Vade: {customer['credit_term']} ay",
        )
    with k2:
        st.metric(
            label="Yıllık Gelir",
            value=f"₺{customer['annual_income']:,.0f}",
            delta=f"Yaş: {customer['age']}",
        )
    with k3:
        st.metric(
            label="Mevcut Borç",
            value=f"₺{customer['current_debt']:,.0f}",
            delta=f"Geç Ödeme: %{customer['late_payment_ratio']*100:.0f}",
            delta_color="inverse",
        )
    with k4:
        score_pct = customer["credit_score"] * 100
        st.metric(
            label="Harici Kredi Puanı",
            value=f"{score_pct:.0f} / 100",
            delta=f"Eğitim: {customer['education']}",
        )


def render_risk_gauge(score: int):
    """Plotly gauge chart ile risk skoru gösterimi."""
    # Renk belirleme
    if score <= 30:
        bar_color = "#22c55e"       # yeşil
        level_text = "DÜŞÜK RİSK"
        css_class = "risk-low"
    elif score <= 60:
        bar_color = "#f59e0b"       # turuncu
        level_text = "ORTA RİSK"
        css_class = "risk-medium"
    else:
        bar_color = "#ef4444"       # kırmızı
        level_text = "YÜKSEK RİSK"
        css_class = "risk-high"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 56, "color": bar_color}, "suffix": ""},
        title={"text": "Risk Skoru", "font": {"size": 18, "color": "#64748b"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 2,
                "tickcolor": "#cbd5e1",
                "tickvals": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "tickfont": {"size": 11, "color": "#94a3b8"},
            },
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#f1f5f9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#dcfce7"},   # yeşil bölge
                {"range": [30, 60], "color": "#fef9c3"},   # sarı bölge
                {"range": [60, 100], "color": "#fee2e2"},   # kırmızı bölge
            ],
            "threshold": {
                "line": {"color": "#1e293b", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )

    st.plotly_chart(fig, width="stretch")

    # Risk seviyesi bandı
    st.markdown(
        f'<div class="risk-banner {css_class}">⚡ Risk Seviyesi: {level_text}</div>',
        unsafe_allow_html=True,
    )


def render_simulation_panel(customer: dict):
    """Senaryo analizi paneli — gelir ve kredi miktarı değiştirilebilir."""
    st.markdown('<div class="section-header">🔬 Senaryo Analizi</div>', unsafe_allow_html=True)

    st.caption(
        "Parametreleri değiştirerek risk skorunun nasıl etkilendiğini görün. "
        "\"Eğer bu müşteri daha az kredi isteseydi ne olurdu?\""
    )

    col_income, col_credit = st.columns(2)

    with col_income:
        new_income = st.number_input(
            "💰 Yıllık Gelir (₺)",
            min_value=30_000,
            max_value=2_000_000,
            value=customer["annual_income"],
            step=10_000,
            format="%d",
            help="Müşterinin yıllık gelirini değiştirin.",
        )

    with col_credit:
        new_credit = st.number_input(
            "🏷️ Kredi Miktarı (₺)",
            min_value=50_000,
            max_value=5_000_000,
            value=customer["credit_amount"],
            step=25_000,
            format="%d",
            help="Talep edilen kredi tutarını değiştirin.",
        )

    # Ek parametreler (genişletilmiş)
    with st.expander("⚙️ Gelişmiş Parametreler", expanded=False):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            new_term = st.number_input(
                "📅 Vade (ay)",
                min_value=6, max_value=120,
                value=customer["credit_term"],
                step=6,
            )
        with adv2:
            new_employment = st.number_input(
                "💼 Çalışma Süresi (yıl)",
                min_value=0.0, max_value=40.0,
                value=customer["employment_years"],
                step=0.5,
                format="%.1f",
            )
        with adv3:
            new_late_ratio = st.slider(
                "⏰ Geç Ödeme Oranı",
                min_value=0.0, max_value=1.0,
                value=customer["late_payment_ratio"],
                step=0.05,
                format="%.0f%%",
            )

    st.markdown("")  # spacing

    # Simülasyon butonu
    if st.button("🚀  Simülasyonu Çalıştır", use_container_width=True, type="primary"):
        with st.spinner("Risk skoru yeniden hesaplanıyor..."):
            time.sleep(1.2)  # Gerçekçi bekleme efekti
            # TODO: Backend hazır olduğunda:
            #   result = requests.post(API_URL + "/simulate", json=payload).json()
            result = simulate_risk(customer, new_income, new_credit)

        # Sonuç gösterimi
        change = result["change"]
        new_score = result["new_score"]

        if change < 0:
            st.markdown(
                f"""
                <div class="sim-result">
                    <span style="font-size:2rem;">📉</span><br>
                    <span style="font-size:0.9rem;color:#6b7280;">Yeni Risk Skoru</span><br>
                    <span style="font-size:2.5rem;font-weight:800;color:#16a34a;">{new_score}</span>
                    <span style="font-size:1.1rem;color:#16a34a;font-weight:600;">({change:+d} puan)</span><br>
                    <span style="font-size:0.85rem;color:#4b5563;margin-top:4px;display:inline-block;">
                        ✅ Risk skoru düştü! Bu senaryo müşteri için daha güvenli.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif change > 0:
            st.markdown(
                f"""
                <div class="sim-result-negative">
                    <span style="font-size:2rem;">📈</span><br>
                    <span style="font-size:0.9rem;color:#6b7280;">Yeni Risk Skoru</span><br>
                    <span style="font-size:2.5rem;font-weight:800;color:#dc2626;">{new_score}</span>
                    <span style="font-size:1.1rem;color:#dc2626;font-weight:600;">({change:+d} puan)</span><br>
                    <span style="font-size:0.85rem;color:#4b5563;margin-top:4px;display:inline-block;">
                        ⚠️ Risk skoru arttı! Bu senaryo daha riskli.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Risk skoru değişmedi. Farklı parametreler deneyin.")

        # Karşılaştırma çıkarımı
        if new_credit < customer["credit_amount"] and change < 0:
            diff = customer["credit_amount"] - new_credit
            st.markdown(
                f"""
                > 💡 **Çıkarım:** Eğer bu müşteri talep ettiği
                **₺{customer['credit_amount']:,.0f}** yerine **₺{new_credit:,.0f}**
                kredi isterse, risk skoru **{result['original_score']}**'dan
                **{new_score}**'a düşer.
                """
            )


def render_shap_chart(customer: dict):
    """SHAP waterfall — yatay çubuk grafik ile risk faktörleri."""
    st.markdown(
        '<div class="section-header">🔍 Risk Faktörleri — SHAP Analizi</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Her bir faktörün risk skoruna olan katkısı. "
        "Kırmızı = riski artırıyor, Yeşil = riski azaltıyor."
    )

    shap_data = customer["shap_values"]

    # Sırala: en büyük etkiden en küçüğe
    sorted_items = sorted(shap_data.items(), key=lambda x: abs(x[1]))
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
        textfont={"size": 12, "color": "#475569"},
    ))

    fig.update_layout(
        height=max(280, len(features) * 38),
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "title": {"text": "SHAP Katkısı (risk etkisi)", "font": {"size": 12, "color": "#64748b"}},
            "zeroline": True,
            "zerolinewidth": 2,
            "zerolinecolor": "#94a3b8",
            "gridcolor": "#f1f5f9",
        },
        yaxis={
            "tickfont": {"size": 12},
        },
        font={"family": "Inter, sans-serif"},
    )

    st.plotly_chart(fig, width="stretch")


def render_payment_history(customer: dict):
    """Son 12 aylık ödeme geçmişi çizgi grafiği."""
    st.markdown(
        '<div class="section-header">📊 Ödeme Geçmişi (Son 12 Ay)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "0 = Zamanında ödeme, 1+ = Gecikme gün sayısı seviyesi "
        "(DPD: Days Past Due)."
    )

    months = [f"Ay {i+1}" for i in range(12)]
    history = customer["payment_history"]

    # Renk: 0 ise yeşil, >0 ise kırmızı tonları
    colors = ["#22c55e" if v == 0 else "#ef4444" for v in history]

    fig = go.Figure()

    # Çizgi
    fig.add_trace(go.Scatter(
        x=months, y=history,
        mode="lines+markers",
        line={"color": "#4a6cf7", "width": 3, "shape": "spline"},
        marker={"size": 10, "color": colors, "line": {"width": 2, "color": "#fff"}},
        fill="tozeroy",
        fillcolor="rgba(74,108,247,0.08)",
        hovertemplate="<b>%{x}</b><br>DPD Seviyesi: %{y}<extra></extra>",
    ))

    # Güvenli bölge çizgisi
    fig.add_hline(
        y=0, line_dash="dot", line_color="#22c55e",
        annotation_text="Zamanında",
        annotation_position="bottom right",
        annotation_font_color="#22c55e",
    )

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"gridcolor": "#f1f5f9"},
        yaxis={
            "title": {"text": "DPD Seviyesi", "font": {"size": 12, "color": "#64748b"}},
            "gridcolor": "#f1f5f9",
            "dtick": 1,
        },
        font={"family": "Inter, sans-serif"},
    )

    st.plotly_chart(fig, width="stretch")


# ═════════════════════════════════════════════════════════════════════
# ANA UYGULAMA
# ═════════════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ──
    customer = render_sidebar()

    # ── Başlık ──
    st.markdown(
        f"### 📋 Müşteri #{customer['id']} — Kredi Risk Analizi"
    )
    st.markdown("")

    # ── KPI Paneli (üst) ──
    render_kpi_panel(customer)

    st.markdown("")
    st.divider()

    # ── Orta Bölüm: Simülasyon (sol) + Risk Gauge (sağ) ──
    col_sim, col_spacer, col_gauge = st.columns([5, 0.5, 4])

    with col_sim:
        render_simulation_panel(customer)

    with col_gauge:
        st.markdown(
            '<div class="section-header">🎯 Mevcut Risk Skoru</div>',
            unsafe_allow_html=True,
        )
        render_risk_gauge(customer["risk_score"])

        # Threshold bilgisi
        threshold_cost = 9    # 0.090 * 100
        threshold_f1 = 18     # 0.178 * 100
        score = customer["risk_score"]

        st.markdown(
            f"""
            <div style="background:#f8fafc;border-radius:8px;padding:12px 16px;
                        border:1px solid #e2e8f0;font-size:0.82rem;color:#475569;
                        margin-top:12px;">
                <strong>📌 Karar Eşikleri (Calibrated)</strong><br>
                • Cost-Optimal: <strong>9</strong> {"✅" if score < threshold_cost else "🔴"} &nbsp;
                • F1-Optimal: <strong>18</strong> {"✅" if score < threshold_f1 else "🔴"}<br>
                <span style="color:#9ca3af;font-size:0.75rem;">
                    Skor eşik değerinin üzerindeyse → Yüksek risk kararı
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Alt Bölüm: SHAP (sol) + Ödeme Geçmişi (sağ) ──
    col_shap, col_payment = st.columns([1, 1])

    with col_shap:
        render_shap_chart(customer)

    with col_payment:
        render_payment_history(customer)

    # ── Footer ──
    st.markdown(
        """
        <div class="footer-text">
            AI Credit Risk v1.0 · LightGBM + Platt Calibration + SHAP ·
            Model AUC: 0.787 · Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
