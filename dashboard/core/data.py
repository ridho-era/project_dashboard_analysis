"""Pemuatan data dan penyaring global. Semua halaman membaca data lewat modul ini."""
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data"


@st.cache_data
def muat(nama: str) -> pd.DataFrame:
    """Membaca satu CSV dari dashboard/data (disimpan di cache agar tidak dibaca ulang)."""
    df = pd.read_csv(DATA / nama)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def penyaring() -> dict:
    """Pilihan penyaring global yang disimpan oleh app.py."""
    return st.session_state.get("penyaring", {})


def stasiun_terpilih() -> list:
    return penyaring().get("stasiun", muat("stasiun.csv")["stasiun"].tolist())


def tahun_terpilih() -> list:
    return penyaring().get("tahun", [2024, 2025, 2026])


def udara_harian_tersaring():
    """Data udara harian sesuai stasiun dan tahun yang dipilih."""
    d = muat("harian_udara.csv")
    return d[d["stasiun"].isin(stasiun_terpilih()) & d["tahun"].isin(tahun_terpilih())]
