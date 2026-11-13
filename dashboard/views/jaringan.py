"""Halaman 5: kecukupan jaringan pemantauan dan wilayah prioritas."""
import pandas as pd
import plotly.express as px
import streamlit as st

from core.data import muat, stasiun_terpilih
from core.tema import judul_halaman, temuan, kartu, WARNA_STATUS, angka, gaya_grafik

# Prioritas usulan tim, dirumuskan dari hasil analisis jaringan (05e). Ubah di sini bila analisis berubah.
PRIORITAS = [
    ("Sumatera Selatan", "Pulihkan PLB-01", "Hari Sangat Tidak Sehat+ terbanyak kedua; hanya 1 stasiun dan 61,7% hari valid."),
    ("Kalimantan Selatan", "Pulihkan BJB-01", "Sensor polutan macet 7 Sep 2024 – 9 Jul 2025; musim kemarau 2024 tidak terukur."),
    ("Jambi", "Ganti JMB-02", "JMB-02 rusak (0% hari valid); koordinatnya hanya ~1 km dari JMB-01."),
    ("Kalimantan Timur", "Perkuat SM2-01", "Hari Sangat Tidak Sehat+ terbanyak, tetapi data musim kabut asap hanya 62–72%."),
    ("Papua Selatan", "Pulihkan MER-01", "Hanya 2 hari valid: paparan tidak diketahui, bukan berarti rendah."),
    ("Sumatera Utara", "Pulihkan MDN-01", "Hanya 36,4% hari valid."),
]

judul_halaman("Jaringan pemantauan", "Apakah stasiun yang ada cukup untuk memantau wilayah yang paling terpapar?", "Pemantauan · 5 dari 6")

S = muat("stasiun.csv")
pilih = S[S["stasiun"].isin(stasiun_terpilih())]

k = pilih["kategori_data"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stasiun terpasang", len(pilih))
c2.metric("Data lengkap (Efektif)", int(k.get("Efektif", 0)), help="≥75% hari punya rata-rata PM2.5 harian valid.")
c3.metric("Data terbatas", int(k.get("Terbatas", 0)), help="10–75% hari valid.")
c4.metric("Hampir tanpa data (Buta)", int(k.get("Buta", 0)), help="<10% hari valid.")

# ---------- Peta status data dan paparan ----------
st.subheader("Di mana data hilang justru saat paling dibutuhkan?")
P = pilih.copy()
# Ukuran = persen hari > 55 dari hari tercatat (bukan jumlah hari, yang otomatis kecil di stasiun berdata sedikit)
P["persen_di_atas_55"] = (P["hari_di_atas_55"] / P["hari_valid"].where(P["hari_valid"] > 0) * 100).fillna(0)
P["ukuran"] = P["persen_di_atas_55"] + 8
fig = px.scatter_map(P, lat="latitude_dd", lon="longitude_dd", color="kategori_data", size="ukuran", size_max=30,
                     color_discrete_map=WARNA_STATUS, category_orders={"kategori_data": ["Efektif", "Terbatas", "Buta"]},
                     hover_name="stasiun", zoom=4.3, center={"lat": -2.3, "lon": 118.0}, map_style="carto-positron",
                     hover_data={"provinsi": True, "persen_valid": ":.1f", "persen_di_atas_55": ":.1f", "hari_di_atas_55": True,
                                 "hari_sangat_tidak_sehat_plus": True, "ukuran": False,
                                 "latitude_dd": False, "longitude_dd": False},
                     labels={"kategori_data": "Kelengkapan data", "persen_valid": "% hari valid",
                             "persen_di_atas_55": "% hari > 55", "hari_di_atas_55": "hari > 55",
                             "hari_sangat_tidak_sehat_plus": "hari > 150,4"})
st.plotly_chart(gaya_grafik(fig, 500), width="stretch")
st.caption("Warna = kelengkapan data; ukuran titik = persen hari di atas 55 µg/m³ dari hari yang tercatat. Titik kuning "
           "atau abu-abu yang besar berarti wilayah terpapar dengan pemantauan yang tidak andal. JMB-02 bertumpuk di atas "
           "JMB-01 karena koordinatnya hampir sama.")

# ---------- Tabel prioritas ----------
st.subheader("Wilayah prioritas")
J = muat("jaringan_provinsi.csv")
pr = pd.DataFrame(PRIORITAS, columns=["provinsi", "Tindakan", "Dasar dari data"])
pr.insert(0, "Prioritas", range(1, len(pr) + 1))
pr = pr.merge(J[["provinsi", "stasiun", "efektif", "hari_sangat_tidak_sehat_plus"]], on="provinsi", how="left")
st.dataframe(pr.rename(columns={"provinsi": "Provinsi", "stasiun": "Stasiun", "efektif": "Efektif",
                                "hari_sangat_tidak_sehat_plus": "Hari > 150,4"}),
             hide_index=True, width="stretch")
satu = J[J["tingkat"].str.startswith("A")]["provinsi"].tolist()
if satu:
    temuan("**Redundansi:** provinsi terpapar yang hanya punya satu stasiun efektif, sehingga satu kerusakan alat "
                f"membuatnya buta: {', '.join(satu)}.")
st.caption("Terpapar = ≥20% hari di atas 55 µg/m³ atau ada hari > 150,4. Prioritas adalah usulan tim berdasarkan data, "
           "bukan peringkat resmi. Dataset hanya mencakup 21 provinsi yang punya stasiun; kebutuhan di provinsi tanpa "
           "stasiun tidak dapat dinilai dari data ini.")
with st.expander("Lihat semua provinsi"):
    st.dataframe(J, hide_index=True, width="stretch")

# ---------- Keterwakilan ----------
st.subheader("Seberapa jauh satu stasiun mewakili wilayah sekitarnya?")
T = muat("tetangga.csv").dropna(subset=["korelasi_tanpa_musim"])
T = T[T["stasiun"].isin(pilih["stasiun"])].copy()
if len(T):
    T["pasangan"] = T["stasiun"] + " ↔ " + T["tetangga_terdekat"]
    fig = px.scatter(T, x="jarak_km", y="korelasi_tanpa_musim", hover_name="pasangan",
                     labels={"jarak_km": "Jarak ke stasiun terdekat (km)",
                             "korelasi_tanpa_musim": "Kemiripan PM2.5 harian (korelasi, pola musim dibuang)"},
                     hover_data={"korelasi_mentah": ":.2f", "n_hari_bersama": True})
    fig.update_traces(marker=dict(size=11, color="#2D6A4F"))
    fig.update_yaxes(range=[-0.2, 1.05])
    st.plotly_chart(gaya_grafik(fig, 400), width="stretch")
    temuan("Stasiun di wilayah kabut asap tetap sangat mirip walaupun berjauhan (sampai sekitar 300 km), sedangkan "
                "stasiun perkotaan yang hanya berjarak 11–18 km tidak mirip sama sekali. Artinya, untuk kabut asap satu "
                "stasiun mewakili wilayah luas, tetapi pencemaran perkotaan butuh stasiun di tiap kawasan.")
    st.caption("Korelasi mentah antarstasiun selalu tinggi karena pola musim nasional yang sama, sehingga yang dipakai "
               "adalah korelasi setelah rata-rata bulanan dibuang. KTG-01 dan KTG-02 hampir identik (r = 1,00).")
