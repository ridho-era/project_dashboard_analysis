"""Halaman 3: kalender episode, pola jam, hari kerja vs akhir pekan, dan faktor yang memengaruhi PM2.5."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.data import muat, stasiun_terpilih, tahun_terpilih, udara_harian_tersaring
from core.tema import judul_halaman, temuan, kartu, BATAS_STS, URUTAN_ISPU, WARNA_ISPU, angka, gaya_grafik

MIN_HARI_STASIUN = 500   # stasiun dengan < 500 hari valid tidak dipakai pada perbandingan antarstasiun

judul_halaman("Episode dan faktor", "Kapan episode pencemaran terjadi, pada jam berapa PM2.5 memuncak, dan faktor apa yang berkaitan dengannya?", "Waktu kritis & penyebab · 3 dari 6")

S = muat("stasiun.csv")
pilih = stasiun_terpilih()
D = udara_harian_tersaring()
Dv = D[D["pm25_mean"].notna()]

# ======================================================================
# A. Kalender episode
# ======================================================================
st.subheader("Kapan episode terjadi?")
hanya_episode = st.toggle("Tampilkan hanya stasiun yang pernah Tidak Sehat atau lebih", value=True, key="ef_episode")
K = D.copy()
K["kode"] = K["ispu_pm25"].map({k: i for i, k in enumerate(URUTAN_ISPU)})
if hanya_episode:
    pernah = Dv.loc[Dv["ispu_pm25"].isin(URUTAN_ISPU[2:]), "stasiun"].unique()
    K = K[K["stasiun"].isin(pernah)]
urut = (Dv[Dv["stasiun"].isin(K["stasiun"].unique())].assign(sts=lambda x: x["pm25_mean"] > BATAS_STS)
          .groupby("stasiun")["sts"].sum().sort_values(ascending=False).index.tolist())
if not urut:
    st.info("Tidak ada stasiun yang memenuhi pilihan ini.")
else:
    Z = K.pivot(index="stasiun", columns="date", values="kode").reindex(urut)
    n = len(URUTAN_ISPU)
    skala = []
    for i, k in enumerate(URUTAN_ISPU):   # skala warna diskrit: satu warna per kategori
        skala += [[i / n, WARNA_ISPU[k]], [(i + 1) / n, WARNA_ISPU[k]]]
    teks = np.vectorize(lambda v: "" if np.isnan(v) else URUTAN_ISPU[int(v)])(Z.values.astype(float))
    fig = go.Figure(go.Heatmap(z=Z.values, x=Z.columns, y=Z.index, zmin=-0.5, zmax=n - 0.5, colorscale=skala,
                               text=teks, hovertemplate="%{y}<br>%{x|%d %b %Y}<br>%{text}<extra></extra>",
                               colorbar=dict(tickvals=list(range(n)), ticktext=URUTAN_ISPU, title="ISPU PM2.5"),
                               hoverongaps=False))
    fig.update_yaxes(autorange="reversed")
    fig = gaya_grafik(fig, 140 + 24 * len(Z))
    fig.update_layout(plot_bgcolor="#B9C2BB")
    st.plotly_chart(fig, width="stretch")
    st.caption("Setiap sel = satu hari. Abu-abu = hari tanpa rata-rata 24 jam yang valid. Diurutkan dari stasiun "
               "dengan hari Sangat Tidak Sehat atau lebih terbanyak.")

    # Ringkasan episode per stasiun-tahun (hari > 150,4)
    ep = Dv[(Dv["pm25_mean"] > BATAS_STS) & Dv["stasiun"].isin(urut)]
    if len(ep):
        R = (ep.groupby(["stasiun", "tahun"])
               .agg(hari=("pm25_mean", "size"), mulai=("date", "min"), akhir=("date", "max"), puncak=("pm25_mean", "max"))
               .reset_index().sort_values(["tahun", "hari"], ascending=[True, False]))
        R["mulai"], R["akhir"] = R["mulai"].dt.strftime("%d %b"), R["akhir"].dt.strftime("%d %b")
        R["puncak"] = R["puncak"].map(lambda v: angka(v, 0))
        st.markdown("**Hari Sangat Tidak Sehat atau lebih (> 150,4 µg/m³) per stasiun dan tahun**")
        st.dataframe(R.rename(columns={"stasiun": "Stasiun", "tahun": "Tahun", "hari": "Jumlah hari",
                                       "mulai": "Hari pertama", "akhir": "Hari terakhir",
                                       "puncak": "Puncak harian (µg/m³)"}),
                     hide_index=True, width="stretch")
        per_th = ep.groupby("tahun")["stasiun"].nunique()
        temuan("Stasiun yang mengalami hari Sangat Tidak Sehat atau lebih: "
                    + ", ".join(f"**{th}: {v} stasiun**" for th, v in per_th.items())
                    + ". 2026 hanya sampai 14 Oktober, jadi episode tahun itu belum selesai.")

# ======================================================================
# B. Pola jam
# ======================================================================
st.subheader("Pada jam berapa PM2.5 memuncak?")
P = muat("profil_jam.csv")
P = P[P["stasiun"].isin(pilih)].merge(S[["stasiun", "kelompok"]], on="stasiun")
if P.empty:
    st.info("Tidak ada data untuk pilihan ini.")
else:
    P["relatif"] = P["pm25_rata2"] / P.groupby("stasiun")["pm25_rata2"].transform("mean")
    G = P.groupby(["kelompok", "jam"])["relatif"].mean().reset_index()
    nst = P.groupby("kelompok")["stasiun"].nunique()
    G["kelompok_n"] = G["kelompok"].map(lambda k: f"{k} ({nst[k]} stasiun)")
    fig = px.line(G, x="jam", y="relatif", color="kelompok_n", markers=True,
                  labels={"jam": "Jam (sesuai kolom waktu data)", "relatif": "PM2.5 relatif (1 = rata-rata harian stasiun)",
                          "kelompok_n": "Tipe lokasi"})
    fig.add_hline(y=1, line_color="#555", line_dash="dot")
    fig.update_xaxes(tickvals=list(range(0, 24, 3)))
    st.plotly_chart(gaya_grafik(fig, 420), width="stretch")
    puncak = P.loc[P.groupby("stasiun")["pm25_rata2"].idxmax()]
    pagi = int(puncak["jam"].between(6, 9).sum())
    temuan(f"Puncak harian jatuh pada pukul 06–09 di **{pagi} dari {len(puncak)} stasiun**, dengan puncak kedua yang "
                "lebih kecil sekitar pukul 18 dan titik terendah sekitar pukul 12–14.")
    st.caption("Nilai dinormalisasi per stasiun supaya stasiun dengan konsentrasi tinggi tidak mendominasi. Kolom waktu "
               "data tertulis WIB; stasiun WITA dan WIT tidak menunjukkan pergeseran jam puncak, sehingga jam dibaca apa adanya.")

# ======================================================================
# C. Hari kerja vs akhir pekan
# ======================================================================
st.subheader("Apakah akhir pekan lebih bersih?")
if len(Dv):
    W = Dv.assign(akhir_pekan=Dv["date"].dt.dayofweek >= 5).groupby(["stasiun", "akhir_pekan"])["pm25_mean"].mean().unstack()
    W = W.dropna()
    W = W.loc[W.index.isin(S.loc[S["kategori_data"] != "Buta", "stasiun"])]
    W["rasio"] = W[True] / W[False]
    W = W.sort_values("rasio").reset_index()
    fig = px.scatter(W, x="rasio", y="stasiun", labels={"rasio": "Rasio PM2.5 akhir pekan / hari kerja", "stasiun": ""},
                     hover_data={"rasio": ":.3f"})
    fig.update_traces(marker=dict(size=10, color="#2D6A4F"))
    fig.add_vrect(x0=0.99, x1=1.01, fillcolor="#9AA39C", opacity=0.25, line_width=0)
    fig.add_vline(x=1, line_color="#555")
    st.plotly_chart(gaya_grafik(fig, 120 + 18 * len(W)), width="stretch")
    dalam = int(W["rasio"].between(0.99, 1.01).sum())
    temuan(f"**{dalam} dari {len(W)} stasiun** berada dalam ±1% dari rasio 1,00 (pita abu-abu): tidak ada perbedaan "
                "bermakna antara hari kerja dan akhir pekan. Penyimpangan kecil muncul di stasiun yang rata-ratanya "
                "didominasi episode atau datanya tidak merata.")

# ======================================================================
# D. Faktor lokasi dan cuaca
# ======================================================================
st.subheader("Faktor apa yang berkaitan dengan PM2.5?")
hv = Dv.groupby("stasiun").size()
layak = hv[hv >= MIN_HARI_STASIUN].index
F = (Dv[Dv["stasiun"].isin(layak)].groupby("stasiun")["pm25_mean"].mean().rename("pm25").reset_index()
       .merge(S[["stasiun", "kelompok", "distance_to_road_m", "traffic_intensity_per_h"]], on="stasiun"))
if len(F) >= 5:
    k1, k2 = st.columns(2)
    for kol, x, judul in [(k1, "distance_to_road_m", "Jarak ke jalan raya (m)"),
                          (k2, "traffic_intensity_per_h", "Intensitas lalu lintas (kendaraan/jam)")]:
        rho = F["pm25"].corr(F[x], method="spearman")
        kol.markdown(f"**{judul.split(' (')[0]}**: Spearman ρ = {angka(rho, 2)} (n = {len(F)} stasiun)")
        fig = px.scatter(F, x=x, y="pm25", color="kelompok", hover_name="stasiun",
                         labels={x: judul, "pm25": "Rata-rata PM2.5 harian (µg/m³)", "kelompok": "Tipe lokasi"})
        fig.update_traces(marker=dict(size=11))
        kol.plotly_chart(gaya_grafik(fig, 400), width="stretch")
    st.caption(f"Hanya stasiun dengan ≥{MIN_HARI_STASIUN} hari valid pada pilihan ini. Korelasi antarstasiun dengan n kecil "
               "adalah petunjuk, bukan bukti sebab-akibat: stasiun di wilayah yang sama saling berkaitan.")

    # Cuaca harian setelah pola musiman bulanan dibuang
    rows = []
    for st_, g in Dv[Dv["stasiun"].isin(layak)].groupby("stasiun"):
        for v, nama in [("temp_mean", "Suhu"), ("rh_mean", "Kelembapan"), ("ws_mean", "Kecepatan angin")]:
            gv = g.dropna(subset=[v])
            if len(gv) < 100:
                continue
            a = gv["pm25_mean"] - gv.groupby("bulan")["pm25_mean"].transform("mean")
            b = gv[v] - gv.groupby("bulan")[v].transform("mean")
            rows.append({"stasiun": st_, "variabel": nama, "rho": a.corr(b, method="spearman")})
    C = pd.DataFrame(rows)
    if len(C):
        fig = px.strip(C, x="rho", y="variabel", hover_name="stasiun",
                       labels={"rho": "Korelasi Spearman harian dengan PM2.5 (pola musiman dibuang)", "variabel": ""})
        fig.update_traces(marker=dict(size=9, color="#2D6A4F", opacity=0.7))
        fig.add_vrect(x0=-0.1, x1=0.1, fillcolor="#9AA39C", opacity=0.25, line_width=0)
        fig.update_xaxes(range=[-1, 1])
        st.plotly_chart(gaya_grafik(fig, 300), width="stretch")
        kecil = int((C["rho"].abs() <= 0.1).sum())
        temuan(f"**{kecil} dari {len(C)} korelasi** cuaca harian berada di antara −0,1 dan 0,1: setelah pola musim "
                    "dibuang, suhu, kelembapan, dan kecepatan angin harian hampir tidak berkaitan dengan PM2.5. Yang paling "
                    "berkaitan adalah lokasi (jarak ke jalan, lalu lintas, tipe lokasi) dan musim episode.")
        st.caption("Setiap titik = satu stasiun. Arah angin tidak ditampilkan karena sebagian stasiun hanya mencatat 2–5 "
                   "sektor arah dan hubungannya belum dikendalikan terhadap musim.")
else:
    st.info("Terlalu sedikit stasiun dengan data cukup untuk perbandingan antarstasiun pada pilihan ini.")
