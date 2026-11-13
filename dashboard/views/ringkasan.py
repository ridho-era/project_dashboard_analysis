"""Halaman 1: ringkasan kondisi, KPI, peta, dan tiga temuan utama."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core.data import muat, stasiun_terpilih, tahun_terpilih, udara_harian_tersaring
from core.tema import (judul_halaman, temuan, kartu, BATAS_STS, BM_PM25_24J, CATATAN_2026, KELAS_LEWAT, WARNA_LEWAT,
                       WARNA_STATUS, angka, gaya_grafik)

judul_halaman("Napas Nusantara", "Kualitas udara dan dampaknya pada kesehatan di 27 stasiun pemantau Indonesia, 1 Januari 2024 – 14 Oktober 2026.", "Ringkasan · 1 dari 6")

S = muat("stasiun.csv")
S_pilih = S[S["stasiun"].isin(stasiun_terpilih())]
D = udara_harian_tersaring()
Dv = D[D["pm25_mean"].notna()]

if Dv.empty:
    st.warning("Tidak ada hari dengan data PM2.5 valid untuk pilihan ini. Ubah penyaring di bilah samping.")
    st.stop()

# ---------- KPI ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Stasiun dengan data lengkap",
          f"{(S_pilih['kategori_data'] == 'Efektif').sum()} dari {len(S_pilih)}",
          help="Stasiun yang minimal 75% harinya punya rata-rata PM2.5 harian valid (≥18 dari 24 jam).")
k2.metric("Hari melampaui baku mutu",
          f"{angka((Dv['pm25_mean'] > BM_PM25_24J).mean() * 100)}%",
          help="Persen hari-stasiun dengan rata-rata PM2.5 24 jam di atas 55 µg/m³ (PP 22/2021).")
k3.metric("Hari Sangat Tidak Sehat atau lebih",
          angka(int((Dv['pm25_mean'] > BATAS_STS).sum()), 0),
          help="Jumlah hari-stasiun dengan rata-rata PM2.5 24 jam di atas 150,4 µg/m³ (kategori ISPU).")
k4.metric("Rata-rata PM2.5 harian",
          f"{angka(Dv['pm25_mean'].mean())} µg/m³",
          help="Rata-rata dari rata-rata harian yang valid, pada stasiun dan tahun terpilih.")
if 2026 in tahun_terpilih():
    st.caption(CATATAN_2026)

# ---------- Tiga temuan utama (seluruh data, tidak terpengaruh penyaring) ----------
st.subheader("Tiga temuan utama")
E = muat("efek_kategori.csv")
e = E[(E["paparan"] == "PM2.5 hari itu") & (E["kategori"] == "55,5-150,4") & (E["hasil"] == "ISPA_balita")]
nef = int((S["kategori_data"] == "Efektif").sum())

c1, c2, c3 = st.columns(3)
with c1:
    kartu("01", "Dua jenis masalah udara", "Jakarta tercemar sepanjang tahun, sedangkan Sumatera dan "
          "Kalimantan melonjak saat musim kabut asap Agustus–Oktober.")
with c2:
    if len(e):
        r = e.iloc[0]
        kartu("02", "Kesehatan ikut melonjak", f"Pada hari PM2.5 55,5–150,4 µg/m³, rawat inap ISPA balita "
              f"<b>{r['persen']:+.0f}%</b> dibanding hari ≤55,4 (selang 95%: {r['bawah']:+.0f}% sampai {r['atas']:+.0f}%).")
with c3:
    kartu("03", "Jaringan belum merata", f"Hanya <b>{nef} dari {len(S)}</b> stasiun punya data lengkap, "
          "dan beberapa di antaranya berada di wilayah rawan kabut asap.")

# ---------- Peta ----------
st.subheader("Di mana udara paling sering melampaui baku mutu?")
per_st = (Dv.assign(lewat=Dv["pm25_mean"] > BM_PM25_24J)
            .groupby("stasiun")
            .agg(persen_lewat=("lewat", "mean"), hari_valid=("pm25_mean", "size"), pm25=("pm25_mean", "mean"))
            .reset_index())
per_st["persen_lewat"] *= 100
P = S_pilih.merge(per_st, on="stasiun", how="left")

mode = st.radio("Warnai titik menurut", ["Persen hari > 55 µg/m³", "Kelengkapan data"], horizontal=True)
umum = dict(lat="latitude_dd", lon="longitude_dd", hover_name="stasiun", zoom=4.3,
            center={"lat": -2.3, "lon": 118.0}, map_style="carto-positron")
info = {"provinsi": True, "kelompok": True, "persen_valid": ":.1f",
        "latitude_dd": False, "longitude_dd": False}

if mode.startswith("Persen"):
    # Stasiun 'Buta' (<10% hari valid) disembunyikan: warnanya akan terbaca seolah udara bersih
    tampil = P[(P["kategori_data"] != "Buta") & P["persen_lewat"].notna()].copy()
    tampil["kelas"] = pd.cut(tampil["persen_lewat"], bins=[-0.01, 0, 10, 25, 50, 100],
                             labels=KELAS_LEWAT).astype(str)
    fig = px.scatter_map(tampil, color="kelas", color_discrete_map=WARNA_LEWAT,
                         category_orders={"kelas": KELAS_LEWAT},
                         hover_data={**info, "persen_lewat": ":.1f", "kelas": False},
                         labels={"kelas": "% hari > 55 µg/m³", "persen_lewat": "% hari > 55",
                                 "persen_valid": "% hari valid"}, **umum)
    disembunyikan = sorted(set(P["stasiun"]) - set(tampil["stasiun"]))
    if disembunyikan:
        st.caption("Tidak ditampilkan karena data tidak cukup (kurang dari 10% hari valid "
                   f"atau tanpa hari valid pada pilihan ini): {', '.join(disembunyikan)}. "
                   "Lihat mode Kelengkapan data.")
else:
    fig = px.scatter_map(P, color="kategori_data", color_discrete_map=WARNA_STATUS,
                         category_orders={"kategori_data": ["Efektif", "Terbatas", "Buta"]},
                         hover_data=info,
                         labels={"kategori_data": "Kelengkapan data", "persen_valid": "% hari valid"}, **umum)

fig.update_traces(marker=dict(size=16, opacity=0.9))
st.plotly_chart(gaya_grafik(fig, 520), width="stretch")
