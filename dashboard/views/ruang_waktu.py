"""Halaman 2: kondisi per stasiun menurut waktu, komposisi ISPU, tren 2024-2026, dan IKU vs PM2.5."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.data import muat, stasiun_terpilih, tahun_terpilih, udara_harian_tersaring
from core.tema import judul_halaman, temuan, kartu, CATATAN_2026, URUTAN_ISPU, WARNA_ISPU, angka, gaya_grafik

judul_halaman("Ruang dan waktu", "Bagaimana kualitas udara berbeda antarstasiun, berubah sepanjang tahun, dan bergerak dari 2024 ke 2026?", "Kondisi udara · 2 dari 6")

S = muat("stasiun.csv")
pilih = stasiun_terpilih()
# Urutan stasiun konsisten di semua grafik: per pulau, lalu tipe lokasi, lalu kode stasiun
urut = S.sort_values(["pulau", "kelompok", "stasiun"])["stasiun"]
urut = [s for s in urut if s in pilih]
label = S.set_index("stasiun").apply(lambda r: f"{r.name} ({r['pulau']})", axis=1).to_dict()

# ======================================================================
# A. Heatmap stasiun x bulan
# ======================================================================
st.subheader("Kapan dan di mana PM2.5 tinggi?")
B = muat("bulanan_udara.csv")
B["tahun"] = B["ym"].str[:4].astype(int)
B = B[B["stasiun"].isin(pilih) & B["tahun"].isin(tahun_terpilih())].copy()
# Bulan dengan kurang dari 50% hari valid dikosongkan agar tidak tampak seolah bersih
B["cukup"] = B["hari_valid"] >= 0.5 * B["hari_total"]
B["persen_lewat"] = (B["hari_di_atas_55"] / B["hari_valid"].where(B["hari_valid"] > 0) * 100)

mode = st.radio("Tampilkan", ["% hari melampaui baku mutu (> 55 µg/m³)", "Rata-rata PM2.5 bulanan (µg/m³)"],
                horizontal=True, key="rw_mode")
kolom = "persen_lewat" if mode.startswith("%") else "pm25_rata2"
B["nilai"] = B[kolom].where(B["cukup"])
Z = B.pivot(index="stasiun", columns="ym", values="nilai").reindex(urut)

if Z.empty:
    st.warning("Tidak ada data untuk pilihan ini. Ubah penyaring di bilah samping.")
else:
    satuan = "% hari" if kolom == "persen_lewat" else "µg/m³"
    zmax = 100 if kolom == "persen_lewat" else max(150.0, float(pd.Series(Z.values.ravel()).quantile(0.98)))
    fig = go.Figure(go.Heatmap(
        z=Z.values, x=Z.columns, y=[label[s] for s in Z.index], colorscale="YlOrRd", zmin=0, zmax=zmax,
        colorbar=dict(title=satuan), hoverongaps=False,
        hovertemplate="%{y}<br>%{x}<br>%{z:.1f} " + satuan + "<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-45)
    fig = gaya_grafik(fig, 120 + 22 * len(Z))
    fig.update_layout(plot_bgcolor="#B9C2BB")   # abu-abu tegas = data tidak cukup
    st.plotly_chart(fig, width="stretch")
    st.caption("Sel abu-abu berarti kurang dari 50% hari pada bulan itu punya data valid. "
               + ("Warna adalah rata-rata bulanan dari rata-rata harian; kategori ISPU berlaku untuk rata-rata 24 jam, "
                  "bukan rata-rata bulanan." if kolom == "pm25_rata2" else "") + " " + CATATAN_2026)

# ======================================================================
# B. Komposisi kategori ISPU per stasiun
# ======================================================================
st.subheader("Seberapa sering tiap stasiun berada di kategori tidak sehat?")
D = udara_harian_tersaring()
Dv = D[D["ispu_pm25"].notna()]
# Stasiun 'Buta' (<10% hari valid) disembunyikan: 2 hari tidak bisa mewakili 3 tahun
buta = S.loc[S["kategori_data"] == "Buta", "stasiun"]
disembunyikan = sorted(set(Dv["stasiun"]) & set(buta))
Dv = Dv[~Dv["stasiun"].isin(buta)]
if disembunyikan:
    st.caption(f"Tidak ditampilkan karena data tidak cukup (kurang dari 10% hari valid): {', '.join(disembunyikan)}.")
if Dv.empty:
    st.info("Tidak ada hari valid untuk pilihan ini.")
else:
    K = (Dv.groupby(["stasiun", "ispu_pm25"]).size().rename("hari").reset_index())
    K["persen"] = K["hari"] / K.groupby("stasiun")["hari"].transform("sum") * 100
    buruk = (K[K["ispu_pm25"].isin(URUTAN_ISPU[2:])].groupby("stasiun")["persen"].sum()
               .reindex(K["stasiun"].unique()).fillna(0).sort_values())
    nhari = Dv.groupby("stasiun").size()
    K["label"] = K["stasiun"].map(lambda s: f"{s} (n={nhari[s]} hari)")
    fig = px.bar(K, x="persen", y="label", color="ispu_pm25", orientation="h",
                 color_discrete_map=WARNA_ISPU, category_orders={
                     "ispu_pm25": URUTAN_ISPU, "label": [f"{s} (n={nhari[s]} hari)" for s in buruk.index[::-1]]},
                 labels={"persen": "% hari valid", "label": "", "ispu_pm25": "Kategori ISPU PM2.5"},
                 hover_data={"hari": True, "persen": ":.1f"})
    fig.update_xaxes(range=[0, 100])
    st.plotly_chart(gaya_grafik(fig, 140 + 20 * K["stasiun"].nunique()), width="stretch")
    st.caption("Diurutkan dari persen hari Tidak Sehat atau lebih yang terbesar. Kategori dari rata-rata PM2.5 24 jam "
               "(Permen LHK P.14/2020); hari dihitung hanya jika ≥18 jam valid. n = jumlah hari valid.")

# ======================================================================
# C. Tren pada jendela yang sama (1 Jan - 14 Okt)
# ======================================================================
st.subheader("Membaik atau memburuk?")
T = muat("tren_jendela.csv")
T = T[T["stasiun"].isin(pilih) & T["memenuhi_kelengkapan"].astype(bool)]
W = T.pivot(index="stasiun", columns="tahun", values="pm25_rata2")
if {2024, 2026}.issubset(W.columns):
    W["selisih"] = W[2026] - W[2024]
    W["arah"] = pd.cut(W["selisih"], [-1e9, -5, 5, 1e9],
                       labels=["Membaik (turun > 5)", "Stabil (±5)", "Memburuk (naik > 5)"]).astype(str)
    W.loc[W["selisih"].isna(), "arah"] = "Tidak bisa dibandingkan"
    warna_arah = {"Memburuk (naik > 5)": "#A50F15", "Stabil (±5)": "#9AA39C",
                  "Membaik (turun > 5)": "#2E9E5B", "Tidak bisa dibandingkan": "#D9DED8"}
    L = T.merge(W[["arah", "selisih"]].reset_index(), on="stasiun")
    fig = px.line(L.sort_values("tahun"), x="tahun", y="pm25_rata2", color="arah", line_group="stasiun",
                  markers=True, color_discrete_map=warna_arah, hover_name="stasiun",
                  hover_data={"arah": False, "selisih": ":.1f", "tahun": True},
                  labels={"pm25_rata2": "Rata-rata PM2.5 (µg/m³)", "tahun": "", "arah": "Perubahan 2024→2026",
                          "selisih": "selisih 2026 − 2024"})
    fig.update_xaxes(tickvals=[2024, 2025, 2026], ticktext=["2024", "2025", "2026*"])
    st.plotly_chart(gaya_grafik(fig, 460), width="stretch")

    n = W["arah"].value_counts()
    naik = W[W["arah"].str.startswith("Memburuk")].sort_values("selisih", ascending=False)
    teks = (f"Dari {int(W['selisih'].notna().sum())} stasiun yang bisa dibandingkan, "
            f"{n.get('Memburuk (naik > 5)', 0)} memburuk, {n.get('Stabil (±5)', 0)} stabil, "
            f"dan {n.get('Membaik (turun > 5)', 0)} membaik.")
    if len(naik):
        teratas = ", ".join(f"{s} ({angka(v, 1)} µg/m³)" for s, v in naik["selisih"].head(3).items())
        teks += f" Kenaikan terbesar: {teratas}."
    temuan(teks)
    st.caption("Rata-rata 1 Jan–14 Okt tiap tahun agar 2026 (*parsial) sebanding; hanya stasiun-tahun dengan ≥80% "
               "hari valid. Grafik ini mengikuti penyaring stasiun, tidak mengikuti penyaring tahun.")
else:
    st.info("Pilih stasiun yang punya data 2024 dan 2026 untuk melihat tren.")

# ======================================================================
# D. IKU vs PM2.5 tahunan
# ======================================================================
st.subheader("Apakah IKU menangkap kabut asap?")
I = muat("iku_tahunan.csv")
I = I[I["stasiun"].isin(pilih) & I["memenuhi_kelengkapan"].astype(bool)].dropna(subset=["IKU", "PM25_tahunan"]).copy()
if I.empty:
    st.info("Tidak ada stasiun-tahun yang memenuhi kelengkapan 80% pada pilihan ini.")
else:
    I["tahun_label"] = I["tahun"].astype(str).replace({"2026": "2026 (parsial)"})
    fig = px.scatter(I, x="PM25_tahunan", y="IKU", color="tahun_label", hover_name="stasiun",
                     color_discrete_sequence=["#9FC5A8", "#2D6A4F", "#1B4332"],
                     hover_data={"SO2": ":.1f", "NO2": ":.1f", "tahun_label": False},
                     labels={"PM25_tahunan": "Rata-rata PM2.5 tahunan (µg/m³)", "IKU": "IKU (SO₂ dan NO₂)",
                             "tahun_label": "Tahun"})
    fig.update_traces(marker=dict(size=11, opacity=0.85))
    fig.add_vline(x=15, line_dash="dot", line_color="#555", annotation_text="15 µg/m³ (baku mutu tahunan)",
                  annotation_position="bottom right")
    st.plotly_chart(gaya_grafik(fig, 440), width="stretch")
    temuan("IKU dihitung hanya dari SO₂ dan NO₂ (Permen LHK 27/2021). Titik yang berjajar mendatar berarti IKU "
                "hampir sama walaupun PM2.5 sangat berbeda, sehingga IKU tidak menggambarkan beban kabut asap.")
    st.caption("IKU di sini adalah proksi dari satu stasiun AQMS per lokasi, bukan metode resmi (passive sampler di "
               "empat lokasi per kota). Stasiun BJB-01, MDN-01, dan PSW-01 memakai nilai hasil konversi ppm.")
