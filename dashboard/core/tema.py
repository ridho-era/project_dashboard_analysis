"""Konstanta tampilan: warna, label, dan acuan. Diubah di sini saja agar konsisten di semua halaman."""

HIJAU_TUA = "#1B4332"
HIJAU_DAUN = "#2D6A4F"
LATAR = "#F4F8F2"

# Warna kategori ISPU PM2.5 (pilihan desain tim, dijelaskan di legenda; bukan warna resmi)
WARNA_ISPU = {
    "Baik": "#2E9E5B",
    "Sedang": "#3B82C4",
    "Tidak Sehat": "#F2B134",
    "Sangat Tidak Sehat": "#D64545",
    "Berbahaya": "#2B2B2B",
}
URUTAN_ISPU = list(WARNA_ISPU)

# Status kelengkapan data stasiun (definisi di halaman Metodologi)
WARNA_STATUS = {"Efektif": "#2D6A4F", "Terbatas": "#E0A526", "Buta": "#9AA39C"}

# Kelas persen hari melampaui baku mutu 24 jam (untuk peta)
KELAS_LEWAT = ["0%", "0–10%", "10–25%", "25–50%", "> 50%"]
WARNA_LEWAT = dict(zip(KELAS_LEWAT, ["#2E9E5B", "#C9D96F", "#F2B134", "#E06B3A", "#A50F15"]))

# Acuan (terverifikasi saat analisis)
BM_PM25_24J = 55.0        # PP 22/2021, rata-rata 24 jam
BATAS_STS = 150.4         # ISPU: di atas ini = Sangat Tidak Sehat atau lebih
TAHUN = [2024, 2025, 2026]
CATATAN_2026 = "2026 parsial: data berakhir 14 Okt 2026."

FONT = "Poppins, Segoe UI, Arial, sans-serif"


def gaya_grafik(fig, tinggi=420):
    """Gaya seragam untuk semua grafik Plotly."""
    fig.update_layout(
        height=tinggi,
        font=dict(family=FONT, size=13, color="#1B2A22"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_family=FONT),
        separators=",.",          # koma desimal, titik ribuan (format Indonesia)
    )
    fig.update_xaxes(gridcolor="#E6EEE3", zeroline=False, linecolor="#C9D6C5")
    fig.update_yaxes(gridcolor="#E6EEE3", zeroline=False, linecolor="#C9D6C5")
    return fig


def angka(x, desimal=1):
    """Format angka gaya Indonesia: titik ribuan, koma desimal."""
    s = f"{x:,.{desimal}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


# ======================================================================
# Lapisan gaya tampilan (dipanggil sekali di app.py)
# ======================================================================
import re as _re

import streamlit as _st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, p, li, label, input, textarea, h1, h2, h3, h4, .stMarkdown, [data-baseweb="select"] div {
    font-family: 'Poppins', 'Segoe UI', Arial, sans-serif;
}
h1, h2, h3, h4, .nn-hero, .nn-hero *, .nn-card, .nn-card *, .nn-insight, .nn-insight *, .nn-brand *,
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    font-family: 'Poppins', 'Segoe UI', Arial, sans-serif !important;
}
[data-testid="stMain"] .block-container { padding-top: 2.2rem; max-width: 1280px; }

/* ---------- Banner judul halaman ---------- */
.nn-hero {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, #1B4332 0%, #2D6A4F 62%, #40916C 100%);
    color: #F4F8F2; border-radius: 18px; padding: 26px 32px 24px 32px; margin-bottom: 26px;
    box-shadow: 0 10px 28px rgba(27, 67, 50, 0.18);
}
.nn-hero::after {
    content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
    border-radius: 50%; background: radial-gradient(circle, rgba(183, 228, 199, 0.28) 0%, rgba(183, 228, 199, 0) 70%);
}
.nn-eyebrow { font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; color: #B7E4C7; font-weight: 600; }
.nn-title { font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 6px 0 8px 0; color: #FFFFFF; }
.nn-sub { font-size: 1rem; color: #DCEFE2; max-width: 860px; margin: 0; }

/* ---------- Judul bagian ---------- */
[data-testid="stMain"] h3 {
    font-weight: 650; font-size: 1.45rem; color: #1B4332; padding-left: 14px; border-left: 5px solid #52B788;
    margin-top: 1.6rem;
}

/* ---------- Kartu KPI ---------- */
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #DDE7DA; border-radius: 14px; padding: 16px 18px 14px 18px;
    box-shadow: 0 2px 10px rgba(27, 67, 50, 0.06); border-top: 4px solid #2D6A4F;
}
[data-testid="stMetricValue"] { color: #1B4332; font-weight: 700; }
[data-testid="stMetricLabel"] p { font-weight: 500; color: #4A5D52; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}

/* ---------- Kotak temuan ---------- */
.nn-insight {
    background: #FFFFFF; border: 1px solid #DDE7DA; border-left: 5px solid #E0A526;
    border-radius: 12px; padding: 14px 18px; margin: 6px 0 14px 0; color: #1B2A22; line-height: 1.6;
}
.nn-insight .nn-tag {
    display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: #9A6B00; background: #FFF4D6; border-radius: 6px; padding: 2px 8px; margin-right: 8px;
}
.nn-card {
    background: #FFFFFF; border: 1px solid #DDE7DA; border-radius: 14px; padding: 18px 20px; height: 100%;
    box-shadow: 0 2px 10px rgba(27, 67, 50, 0.06);
}
.nn-card .nn-num { font-size: 0.8rem; font-weight: 700; color: #52B788; letter-spacing: 0.08em; }
.nn-card .nn-head { font-size: 1.05rem; font-weight: 650; color: #1B4332; margin: 4px 0 6px 0; }
.nn-card p { margin: 0; color: #33463B; line-height: 1.55; }

/* ---------- Grafik dan tabel sebagai kartu ---------- */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: #FFFFFF; border: 1px solid #DDE7DA; border-radius: 14px; padding: 10px 12px;
    box-shadow: 0 2px 10px rgba(27, 67, 50, 0.05);
}

/* ---------- Bilah samping ---------- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #E3EDE0 0%, #EEF4EB 100%); border-right: 1px solid #D4E0D0; }
.nn-brand { padding: 2px 4px 12px 4px; }
.nn-brand .nn-b1 { font-weight: 700; font-size: 1.15rem; color: #1B4332; }
.nn-brand .nn-b2 { font-size: 0.78rem; color: #4A5D52; }
</style>
"""


def pasang_gaya():
    """Menyisipkan CSS tampilan; dipanggil sekali di app.py."""
    _st.markdown(_CSS, unsafe_allow_html=True)


def _md(teks):
    """Ubah **tebal** sederhana menjadi HTML agar bisa dipakai di dalam kotak."""
    return _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", teks)


def judul_halaman(judul, sub, label):
    """Banner judul di bagian atas setiap halaman."""
    _st.markdown(f'<div class="nn-hero"><div class="nn-eyebrow">{label}</div>'
                 f'<div class="nn-title">{judul}</div><p class="nn-sub">{sub}</p></div>',
                 unsafe_allow_html=True)


def temuan(teks, tag="Temuan"):
    """Kotak penekanan untuk kalimat temuan yang dihitung dari data."""
    _st.markdown(f'<div class="nn-insight"><span class="nn-tag">{tag}</span>{_md(teks)}</div>',
                 unsafe_allow_html=True)


def kartu(nomor, judul, teks):
    """Kartu temuan bernomor (dipakai di Ringkasan)."""
    _st.markdown(f'<div class="nn-card"><div class="nn-num">{nomor}</div><div class="nn-head">{judul}</div>'
                 f'<p>{_md(teks)}</p></div>', unsafe_allow_html=True)
