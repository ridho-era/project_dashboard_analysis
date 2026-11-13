"""Titik masuk dashboard: konfigurasi halaman, logo, penyaring global, dan navigasi."""
from pathlib import Path

import streamlit as st

from core.data import muat
from core.tema import CATATAN_2026, TAHUN, pasang_gaya

st.set_page_config(page_title="Napas Nusantara | DA Gelar Rasa 2026", page_icon="🌿", layout="wide")
pasang_gaya()

LOGO = Path(__file__).parent / "assets" / "logo_gelar_rasa.png"
if LOGO.exists():
    st.logo(str(LOGO), size="large")

# ---------- Penyaring global (berlaku di semua halaman) ----------
S = muat("stasiun.csv")
with st.sidebar:
    
    st.markdown("### Penyaring")
    tahun = st.multiselect("Tahun", TAHUN, default=TAHUN, key="f_tahun", help=CATATAN_2026,
                           placeholder="Semua tahun")
    pulau = st.multiselect("Pulau", sorted(S["pulau"].unique()), key="f_pulau",
                           placeholder="Semua pulau")
    kelompok = st.multiselect("Tipe lokasi", sorted(S["kelompok"].unique()), key="f_kelompok",
                              placeholder="Semua tipe")
    hanya_efektif = st.toggle("Hanya stasiun dengan data lengkap", key="f_efektif",
                              help="Stasiun 'Efektif': minimal 75% hari punya rata-rata PM2.5 harian yang valid.")

    kandidat = S.copy()
    if pulau:
        kandidat = kandidat[kandidat["pulau"].isin(pulau)]
    if kelompok:
        kandidat = kandidat[kandidat["kelompok"].isin(kelompok)]
    if hanya_efektif:
        kandidat = kandidat[kandidat["kategori_data"] == "Efektif"]
    stasiun = st.multiselect("Stasiun", sorted(kandidat["stasiun"]), key="f_stasiun",
                             placeholder="Semua stasiun di atas")
    st.caption("Kosongkan pilihan untuk menampilkan semuanya.")

st.session_state["penyaring"] = {
    "tahun": tahun or TAHUN,
    "stasiun": stasiun or kandidat["stasiun"].tolist(),
}

# ---------- Navigasi ----------
halaman = st.navigation([
    st.Page("views/ringkasan.py", title="Ringkasan", icon="🏠", default=True),
    st.Page("views/ruang_waktu.py", title="Ruang dan waktu", icon="🗺️"),
    st.Page("views/episode_faktor.py", title="Episode dan faktor", icon="🔥"),
    st.Page("views/kesehatan.py", title="Dampak kesehatan", icon="🫁"),
    st.Page("views/jaringan.py", title="Jaringan pemantauan", icon="📡"),
    st.Page("views/metodologi.py", title="Metodologi", icon="📘"),
])



halaman.run()