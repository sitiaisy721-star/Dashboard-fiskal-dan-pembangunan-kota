from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
import streamlit as st

# ======================================================================
# KONFIGURASI
# ======================================================================
CSV_NAME = "dataset_fiskal_97_hipoFull.csv"
ACCENT = "#EEC4D8"
SERIES = ["#B39DDB", "#B45309", "#1D4ED8", "#9D174D", "#4D7C0F",
          "#6D28D9", "#0E7490", "#A16207", "#BE123C", "#475569"]

LABELS = {
    "IPM": "IPM",
    "KEMISKINAN": "Kemiskinan (%)",
    "PENGANGGURAN": "Pengangguran (%)",
    "KETIMPANGAN": "Ketimpangan (Gini)",
    "PE": "Pertumbuhan ekonomi (%)",
}
# (kolom, apakah kenaikan = kabar buruk)
KPIS = [("IPM", False), ("KEMISKINAN", True), ("PENGANGGURAN", True),
        ("KETIMPANGAN", True), ("PE", False)]

st.set_page_config(page_title="Dashboard Fiskal dan Pembangunan",
                   page_icon="📊", layout="wide")


def label(col: str) -> str:
    return LABELS.get(col, col)


def angka(n: int) -> str:
    return f"{n:,}".replace(",", ".")


# ======================================================================
# TAMPILAN (CSS)
# ======================================================================
def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap');
    .stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea,
    .stApp button, .stApp [data-testid="stMetric"] *, .stApp [data-testid="stCaptionContainer"] *
        { font-family: 'IBM Plex Sans', sans-serif; }
    /* jaga font ikon bawaan Streamlit agar tidak tampil sebagai teks */
    span[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded
        { font-family: 'Material Symbols Rounded' !important; }
    h1, h2, h3 { font-family: 'IBM Plex Serif', Georgia, serif !important;
                 letter-spacing: -0.01em; }
    .block-container { padding-top: 5rem; max-width: 1300px; }

    .page-title { font-family: 'IBM Plex Serif', Georgia, serif; font-size: 2.3rem;
                  font-weight: 600; color: #14213D; line-height: 1.15;
                  letter-spacing: -0.02em; }
    .page-desc { color: #5B6475; font-size: 1.02rem; line-height: 1.55;
                 max-width: 46rem; margin: 0.7rem 0 1rem; }
    .chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem;
             padding-bottom: 1.3rem; border-bottom: 3px solid #B39DDB; }
    .chips span { background: rgba(15,118,110,0.10); color: #5B3E96; font-size: 0.85rem;
                  font-weight: 500; padding: 0.25rem 0.8rem; border-radius: 999px; }

    /* KPI: garis aksen di kiri, bukan kartu */
    div[data-testid="stMetric"] { background: none; border: none;
                                  border-left: 3px solid #B39DDB;
                                  padding: 2px 0 2px 14px; }
    div[data-testid="stMetricLabel"] p { color: #5B6475; font-size: 0.85rem; }
    div[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 600; }

    .stTabs [data-baseweb="tab"] { font-weight: 500; padding: 0 1.1rem; }
    section[data-testid="stSidebar"] h3 { font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)


def style_fig(fig, height=390):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=24, b=8),
                      legend_title_text="", font_family="IBM Plex Sans")
    return fig


# ======================================================================
# DATA
# ======================================================================
@st.cache_data
def load_csv(src) -> pd.DataFrame:
    return pd.read_csv(src)


def find_csv():
    base = Path(__file__).parent
    for p in [base / CSV_NAME, *base.glob(f"*/{CSV_NAME}")]:
        if p.exists():
            return p
    return None


def get_data() -> pd.DataFrame:
    path = find_csv()
    if path:
        return load_csv(path)
    st.warning(f"File {CSV_NAME} tidak ditemukan di folder aplikasi. "
               "Unggah file di bawah ini atau letakkan di folder yang sama dengan app.py.")
    up = st.file_uploader("Unggah dataset (CSV)", type="csv")
    if up is None:
        st.stop()
    return load_csv(up)


# ======================================================================
# SIDEBAR
# ======================================================================
def sidebar_filters(df: pd.DataFrame):
    st.sidebar.markdown("### Filter data")
    tmin, tmax = int(df["Tahun"].min()), int(df["Tahun"].max())
    tahun = st.sidebar.slider("Rentang tahun", tmin, tmax, (tmin, tmax))

    cakupan = st.sidebar.radio("Cakupan kota", ["Semua kota", "Pilih kota"])
    kota = []
    if cakupan == "Pilih kota":
        kota = st.sidebar.multiselect("Kota", sorted(df["Kota"].unique()))

    data = df[df["Tahun"].between(*tahun)]
    if kota:
        data = data[data["Kota"].isin(kota)]
    return tahun, data.copy()


# ======================================================================
# KOMPONEN
# ======================================================================
def header(data, tahun):
    st.markdown(
        '<div class="page-title">Dashboard fiskal dan pembangunan kota</div>'
        '<div class="page-desc">Pantau IPM, kemiskinan, pengangguran, dan ketimpangan, '
        'lalu uji pengaruh belanja daerah terhadap indikator tersebut lewat regresi data panel.</div>'
        '<div class="chips">'
        f'<span>Periode {tahun[0]}–{tahun[1]}</span>'
        f'<span>{data["Kota"].nunique()} kota</span>'
        f'<span>{angka(len(data))} observasi</span>'
        '</div>',
        unsafe_allow_html=True)


def kpi_strip(data):
    y0, y1 = int(data["Tahun"].min()), int(data["Tahun"].max())
    for col, (name, bad_up) in zip(st.columns(len(KPIS)), KPIS):
        last = data.loc[data["Tahun"] == y1, name].mean()
        first = data.loc[data["Tahun"] == y0, name].mean()
        col.metric(label(name), f"{last:.2f}",
                   f"{last - first:+.2f}" if y1 != y0 else None,
                   delta_color="inverse" if bad_up else "normal")
    st.caption(f"Rata-rata tahun {y1}. Selisih dihitung terhadap rata-rata tahun {y0}.")


# ======================================================================
# TAB 1: TREN DAN KORELASI
# ======================================================================
def tab_trend(data, num_cols):
    # ---- Bagian 1: tren indikator ----
    st.subheader("Tren indikator")
    metric = st.selectbox("Indikator", num_cols, format_func=label,
                          index=num_cols.index("IPM"), key="trend_metric")
    n_kota = data["Kota"].nunique()
    per_kota = False
    if 1 < n_kota <= 10:
        per_kota = st.toggle("Tampilkan per kota")

    if per_kota:
        fig = px.line(data, x="Tahun", y=metric, color="Kota", markers=True,
                      color_discrete_sequence=SERIES)
    else:
        avg = data.groupby("Tahun")[metric].mean().reset_index()
        fig = px.line(avg, x="Tahun", y=metric, markers=True,
                      color_discrete_sequence=[ACCENT])
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title=label(metric))
    st.plotly_chart(style_fig(fig, 380))
    if not per_kota:
        st.caption("Garis menunjukkan rata-rata seluruh kota yang terpilih.")

    st.divider()

    # ---- Bagian 2: korelasi antar variabel ----
    st.subheader("Korelasi antar variabel")
    default = [c for c in ["IPM", "KEMISKINAN", "PENGANGGURAN", "PAD", "bm_pendidikan"]
               if c in num_cols]
    picks = st.multiselect("Variabel", num_cols, default=default,
                           format_func=label, key="corr_vars")
    if len(picks) >= 2:
        corr = data[picks].corr().round(2)
        corr.index = [label(c) for c in corr.index]
        corr.columns = [label(c) for c in corr.columns]
        fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1,
                        color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(coloraxis_colorbar=dict(title="r", thickness=12))
        st.plotly_chart(style_fig(fig, 460))
        st.caption("Nilai mendekati 1 berarti hubungan searah, mendekati -1 berlawanan arah, "
                   "mendekati 0 berarti hampir tidak ada hubungan linear.")
    else:
        st.info("Pilih minimal dua variabel untuk melihat korelasi.")


# ======================================================================
# TAB 2: PERINGKAT KOTA
# ======================================================================
def tab_rank(data, num_cols):
    ctl, chart = st.columns([1, 3], gap="large")

    with ctl:
        st.subheader("Pengaturan")
        metric = st.selectbox("Indikator", num_cols, format_func=label,
                              index=num_cols.index("IPM"), key="rank_metric")
        year = st.selectbox("Tahun", sorted(data["Tahun"].unique(), reverse=True))
        sub = data[data["Tahun"] == year]
        max_n = min(20, len(sub))
        n = st.slider("Jumlah kota", 5, max_n, min(10, max_n)) if max_n > 5 else max_n
        order = st.radio("Urutan", ["Tertinggi", "Terendah"])

    with chart:
        pick = sub.nlargest(n, metric) if order == "Tertinggi" else sub.nsmallest(n, metric)
        st.subheader(f"{n} kota dengan {label(metric)} {order.lower()}, {year}")
        fig = px.bar(pick, x=metric, y="Kota", orientation="h", text=metric,
                     color_discrete_sequence=[ACCENT])
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                          cliponaxis=False)
        fig.update_yaxes(categoryorder="total ascending", title=None)
        fig.update_xaxes(title=label(metric))
        st.plotly_chart(style_fig(fig, 440))


# ======================================================================
# TAB 3: MODEL REGRESI PANEL
# ======================================================================
def tab_model(data, num_cols):
    st.subheader("Regresi data panel")

    c1, c2, c3 = st.columns([1.2, 2.5, 1.3], gap="large")
    with c1:
        y_var = st.selectbox("Variabel dependen (Y)", num_cols, format_func=label,
                             index=num_cols.index("KEMISKINAN"), key="y_var")
    x_opts = [c for c in num_cols if c != y_var]
    with c2:
        x_vars = st.multiselect(
            "Variabel independen (X)", x_opts, format_func=label, key="x_vars",
            default=[c for c in ["PAD", "bm_ekonomi", "bm_kesehatan"] if c in x_opts])
    with c3:
        fe = st.radio("Metode estimasi", ["Pooled OLS", "Fixed effect (LSDV)"]
                      ).startswith("Fixed")

    if not x_vars:
        st.info("Pilih minimal satu variabel X untuk menjalankan model.")
        return

    d = data[["Kota", y_var, *x_vars]].dropna()
    formula = f"{y_var} ~ " + " + ".join(x_vars) + (" + C(Kota)" if fe else "")
    n_params = len(x_vars) + 1 + (d["Kota"].nunique() - 1 if fe else 0)
    if len(d) <= n_params:
        st.error(f"Observasi ({len(d)}) tidak cukup untuk {n_params} parameter. "
                 "Kurangi variabel X, perluas rentang tahun, atau tambah kota.")
        return

    try:
        res = smf.ols(formula, data=d).fit()
    except Exception as err:
        st.error(f"Model gagal diestimasi: {err}")
        return

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R-squared", f"{res.rsquared:.3f}")
    m2.metric("Adj. R-squared", f"{res.rsquared_adj:.3f}")
    m3.metric("F-statistic", f"{res.fvalue:.2f}")
    m4.metric("Prob (F)", f"{res.f_pvalue:.4g}")
    st.caption(f"{len(d)} observasi, {d['Kota'].nunique()} kota. Rumus: {formula}")

    t_coef, t_diag, t_raw = st.tabs(["Koefisien", "Diagnostik", "Ringkasan lengkap"])

    with t_coef:
        ci = res.conf_int()
        coef = pd.DataFrame({
            "Koefisien": res.params, "Std. error": res.bse, "t": res.tvalues,
            "p-value": res.pvalues, "CI 2.5%": ci[0], "CI 97.5%": ci[1]})
        coef["Signifikan (5%)"] = np.where(coef["p-value"] < 0.05, "Ya", "Tidak")
        coef = coef.rename(index={"Intercept": "Konstanta"})
        if fe and not st.checkbox("Tampilkan dummy kota"):
            coef = coef[~coef.index.str.startswith("C(Kota)")]
        st.dataframe(coef.round(4))

        st.markdown("**Cara membaca hasil**")
        for v in x_vars:
            b, p = res.params[v], res.pvalues[v]
            arah = "naik" if b > 0 else "turun"
            sig = "signifikan" if p < 0.05 else "tidak signifikan"
            st.markdown(
                f"- Kenaikan 1 satuan **{label(v)}** membuat {label(y_var)} {arah} "
                f"**{abs(b):.4f}**, dengan variabel lain tetap (p = {p:.3f}, {sig}).")

    with t_diag:
        d = d.assign(Prediksi=res.fittedvalues, Residual=res.resid)
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("**Aktual vs prediksi**")
            fig = px.scatter(d, x=y_var, y="Prediksi", hover_data=["Kota"], opacity=0.75,
                             color_discrete_sequence=[ACCENT])
            lo, hi = float(d[y_var].min()), float(d[y_var].max())
            fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                          line=dict(dash="dash", color="#94A3B8"))
            fig.update_xaxes(title=f"{label(y_var)} aktual")
            st.plotly_chart(style_fig(fig, 340))
        with b:
            st.markdown("**Residual vs prediksi**")
            fig = px.scatter(d, x="Prediksi", y="Residual", hover_data=["Kota"],
                             opacity=0.75, color_discrete_sequence=[ACCENT])
            fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
            st.plotly_chart(style_fig(fig, 340))
        st.caption("Titik yang dekat garis putus-putus berarti prediksi mendekati nilai aktual. "
                   "Residual yang membentuk pola (melengkung atau menyebar) menandakan model kurang pas.")

    with t_raw:
        st.text(res.summary().as_text())


# ======================================================================
# TAB 4: DATA
# ======================================================================
def tab_data(data):
    st.subheader("Dataset terfilter")
    st.dataframe(data, hide_index=True)
    st.download_button("Unduh CSV", data.to_csv(index=False),
                       "data_fiskal_terfilter.csv", "text/csv")
    with st.expander("Statistik deskriptif"):
        st.dataframe(data.drop(columns=["Kota", "Tahun"]).describe().T.round(3))


# ======================================================================
# MAIN
# ======================================================================
inject_css()
df = get_data()
tahun, data = sidebar_filters(df)

if data.empty:
    st.warning("Tidak ada data untuk filter ini. Ubah rentang tahun atau pilihan kota.")
    st.stop()

num_cols = [c for c in df.columns if c not in ("Kota", "Tahun")]

header(data, tahun)
kpi_strip(data)
st.write("")

t1, t2, t3, t4 = st.tabs(["Tren dan korelasi", "Peringkat kota", "Model regresi", "Data"])
with t1:
    tab_trend(data, num_cols)
with t2:
    tab_rank(data, num_cols)
with t3:
    tab_model(data, num_cols)
with t4:
    tab_data(data)