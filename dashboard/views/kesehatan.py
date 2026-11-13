"""Halaman 4: hubungan PM2.5 dengan kasus kesehatan, efek jeda, balita vs lansia, dan penjelajah per RS."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.data import muat, stasiun_terpilih
from core.tema import judul_halaman, temuan, kartu, HIJAU_DAUN, WARNA_ISPU, angka, gaya_grafik

NAMA = {"ISPA_balita": "ISPA balita", "ISPA_lansia": "ISPA lansia", "asma": "Asma darurat", "PPOK": "PPOK",
        "resp_umum": "Pernapasan umum", "kardiovask": "Kardiovaskular", "kematian": "Kematian"}
URUT = list(NAMA.values())
JEDA_TAMPIL = 4


def pr(x, d=1):
    """Persen bertanda dengan koma desimal, mis. +2,7."""
    return f"{x:+.{d}f}".replace(".", ",")
   # jeda 5-7 tidak ditampilkan: koefisien negatif di sana artefak pemisahan komponen (lihat Metodologi)

judul_halaman("Dampak kesehatan", "Apakah kasus di rumah sakit naik ketika PM2.5 tinggi, seberapa besar, dan berapa hari efeknya bertahan?", "Kesehatan · 4 dari 6")
temuan("Angka di halaman ini adalah **hubungan statistik** dari model regresi Poisson pada 24 rumah sakit "
        "(dikendalikan untuk rumah sakit, hari, bulan, tahun, suhu, dan kelembapan), bukan bukti sebab-akibat. "
        "Model dihitung dari seluruh data, sehingga tidak berubah oleh penyaring.", "Cara membaca")

# ======================================================================
# A. Kenaikan kasus per kategori ISPU
# ======================================================================
st.subheader("Berapa besar kenaikan kasus saat udara tidak sehat?")
E = muat("efek_kategori.csv")
N = muat("efek_kategori_n.csv")
paparan = st.radio("Ukuran paparan", ["PM2.5 hari itu", "rata-rata 7 hari"], horizontal=True, key="kes_paparan",
                   format_func=lambda x: "PM2.5 hari itu" if x == "PM2.5 hari itu" else "Rata-rata 7 hari terakhir")
e = E[E["paparan"] == paparan].copy()
e["hasil_n"] = e["hasil"].map(NAMA)
e["kat_label"] = e["kategori"].map({"55,5-150,4": "Tidak Sehat (55,5–150,4)", ">150,4": "Sangat Tidak Sehat+ (>150,4)"})
fig = px.scatter(e, x="persen", y="hasil_n", color="kat_label",
                 error_x=e["atas"] - e["persen"], error_x_minus=e["persen"] - e["bawah"],
                 color_discrete_map={"Tidak Sehat (55,5–150,4)": WARNA_ISPU["Tidak Sehat"],
                                     "Sangat Tidak Sehat+ (>150,4)": WARNA_ISPU["Sangat Tidak Sehat"]},
                 category_orders={"hasil_n": URUT},
                 hover_data={"bawah": ":.1f", "atas": ":.1f", "persen": ":.1f", "kat_label": False, "hasil_n": False},
                 labels={"persen": "Perubahan kasus dibanding hari ≤55,4 µg/m³ (%)", "hasil_n": "", "kat_label": "Kategori hari",
                         "bawah": "batas bawah 95%", "atas": "batas atas 95%"})
fig.update_traces(marker=dict(size=12))
fig.add_vline(x=0, line_color="#555")
st.plotly_chart(gaya_grafik(fig, 430), width="stretch")
n = N[N["paparan"] == paparan].set_index("kategori")["hari_RS"]
st.caption(f"Titik = estimasi; garis = selang kepercayaan 95%. Jumlah hari-rumah sakit: ≤55,4 = {angka(n.get('<=55,4', 0), 0)}, "
           f"55,5–150,4 = {angka(n.get('55,5-150,4', 0), 0)}, >150,4 = {angka(n.get('>150,4', 0), 0)}. "
           "Kategori >150,4 berdasar sedikit hari dan terkonsentrasi di beberapa rumah sakit, jadi selangnya lebar.")

ts = e[e["kategori"] == "55,5-150,4"].set_index("hasil")
if {"ISPA_balita", "ISPA_lansia"}.issubset(ts.index):
    temuan(f"Pada hari Tidak Sehat, rawat inap ISPA balita **{pr(ts.loc['ISPA_balita', 'persen'], 0)}%** dan ISPA lansia "
                f"**{pr(ts.loc['ISPA_lansia', 'persen'], 0)}%** dibanding hari Baik–Sedang. Kenaikan ini jauh lebih besar "
                "daripada yang diperkirakan dari hubungan garis lurus, sehingga ambang 55 µg/m³ adalah titik penting untuk peringatan dini.")

# ======================================================================
# B. Profil jeda
# ======================================================================
st.subheader("Berapa hari efeknya bertahan?")
C = muat("efek_komponen_cepat.csv")
pilihan = st.selectbox("Hasil kesehatan", URUT, key="kes_hasil")
kode = {v: k for k, v in NAMA.items()}[pilihan]
c = C[(C["hasil"] == kode) & (C["jeda_hari"] <= JEDA_TAMPIL)].sort_values("jeda_hari")
fig = go.Figure([
    go.Scatter(x=list(c["jeda_hari"]) + list(c["jeda_hari"][::-1]), y=list(c["atas"]) + list(c["bawah"][::-1]),
               fill="toself", fillcolor="rgba(45,106,79,0.18)", line=dict(width=0), mode="lines", hoverinfo="skip",
               name="Selang 95%"),
    go.Scatter(x=c["jeda_hari"], y=c["persen"], mode="lines+markers", line=dict(color=HIJAU_DAUN, width=3),
               marker=dict(size=9), name="Estimasi",
               hovertemplate="Jeda %{x} hari<br>%{y:+.2f}% per +10 µg/m³<extra></extra>"),
])
fig.add_hline(y=0, line_color="#555")
fig.update_xaxes(title="Jeda setelah lonjakan PM2.5 (hari)", tickvals=list(range(JEDA_TAMPIL + 1)))
fig.update_yaxes(title="Perubahan kasus per +10 µg/m³ (%)")
st.plotly_chart(gaya_grafik(fig, 380), width="stretch")
if len(c):
    puncak = c.loc[c["persen"].idxmax()]
    sig = c[c["bawah"] > 0]["jeda_hari"]
    teks = f"Untuk {pilihan}, efek terbesar pada jeda {int(puncak['jeda_hari'])} hari ({pr(puncak['persen'])}% per +10 µg/m³)"
    teks += (f" dan masih bermakna sampai jeda {int(sig.max())} hari." if len(sig) else ", tetapi tidak ada jeda yang bermakna.")
    temuan(teks, "Efek jeda")
L = muat("efek_komponen_lambat.csv").set_index("hasil")
if kode in L.index:
    r = L.loc[kode]
    st.caption(f"Grafik ini mengukur lonjakan harian (selisih dari rata-rata 29 hari). Paparan berkelanjutan (rata-rata 29 hari) "
               f"untuk {pilihan}: {pr(r['persen'])}% per +10 µg/m³ (selang 95%: {pr(r['bawah'])}% sampai {pr(r['atas'])}%). "
               f"Jeda 5–7 hari tidak ditampilkan karena estimasinya artefak metode, bukan efek pelindung.")

# ======================================================================
# C. Balita vs lansia
# ======================================================================
st.subheader("Siapa yang lebih terdampak: balita atau lansia?")
H = muat("kesehatan_harian.csv")
H = H[H["stasiun"].isin(stasiun_terpilih()) & ~H["hospital_id"].isin(["RS-XXX-23", "RS-XXX-27"])]
k1, k2 = st.columns(2)
with k1:
    rel = E[(E["paparan"] == "PM2.5 hari itu") & E["hasil"].isin(["ISPA_balita", "ISPA_lansia"])].copy()
    rel["kelompok"] = rel["hasil"].map({"ISPA_balita": "Balita", "ISPA_lansia": "Lansia"})
    rel["kat_label"] = rel["kategori"].map({"55,5-150,4": "Tidak Sehat", ">150,4": "Sangat Tidak Sehat+"})
    fig = px.bar(rel, x="kat_label", y="persen", color="kelompok", barmode="group",
                 error_y=rel["atas"] - rel["persen"], error_y_minus=rel["persen"] - rel["bawah"],
                 color_discrete_map={"Balita": "#3B82C4", "Lansia": "#8C5A9E"},
                 labels={"kat_label": "", "persen": "Kenaikan relatif kasus ISPA (%)", "kelompok": ""})
    st.plotly_chart(gaya_grafik(fig, 360), width="stretch")
    st.caption("Kenaikan relatif hampir sama dan selangnya tumpang-tindih: tidak ada bukti salah satu lebih rentan secara relatif.")
with k2:
    if len(H):
        ab = pd.DataFrame({"kelompok": ["Balita", "Lansia"],
                           "kasus": [H["resp_admissions_under5"].mean(), H["resp_admissions_over60"].mean()]})
        fig = px.bar(ab, x="kelompok", y="kasus", color="kelompok", text_auto=".1f",
                     color_discrete_map={"Balita": "#3B82C4", "Lansia": "#8C5A9E"},
                     labels={"kelompok": "", "kasus": "Rata-rata rawat inap ISPA per hari per RS"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(gaya_grafik(fig, 360), width="stretch")
        st.caption(f"Rata-rata deskriptif pada {H['hospital_id'].nunique()} rumah sakit terpilih (tanpa RS-23 dan RS-27 yang tidak punya data paparan andal). Dengan kenaikan relatif yang sama, "
                   "kelompok dengan kasus dasar lebih banyak menyumbang tambahan kasus lebih banyak.")

# ======================================================================
# D. Penjelajah per rumah sakit
# ======================================================================
st.subheader("Lihat langsung di satu rumah sakit")
S = muat("stasiun.csv")
opsi = S[S["stasiun"].isin(stasiun_terpilih())].sort_values("stasiun")
if opsi.empty:
    st.info("Pilih minimal satu stasiun di bilah samping.")
else:
    lbl = {r.stasiun: f"{r.hospital_id} – pasangan stasiun {r.stasiun} ({r.provinsi})" for r in opsi.itertuples()}
    awal = list(lbl).index("KTG-01") if "KTG-01" in lbl else 0   # stasiun dengan episode jelas
    st_pilih = st.selectbox("Rumah sakit", list(lbl), index=awal, format_func=lbl.get, key="kes_rs")
    h = muat("kesehatan_harian.csv")
    h = h[h["stasiun"] == st_pilih].sort_values("date").copy()
    h["ispa"] = h["resp_admissions_under5"] + h["resp_admissions_over60"]
    h["ispa_7h"] = h["ispa"].rolling(7, min_periods=4).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("PM2.5 harian (µg/m³)", "Rawat inap ISPA balita + lansia (rata-rata 7 hari)"))
    fig.add_trace(go.Scatter(x=h["date"], y=h["pm25_mean"], mode="lines", line=dict(color="#C0392B", width=1),
                             name="PM2.5", connectgaps=False), row=1, col=1)
    fig.add_hline(y=55, line_dash="dot", line_color="#555", row=1, col=1)
    fig.add_trace(go.Scatter(x=h["date"], y=h["ispa_7h"], mode="lines", line=dict(color=HIJAU_DAUN, width=2),
                             name="ISPA"), row=2, col=1)
    fig.update_layout(showlegend=False)
    st.plotly_chart(gaya_grafik(fig, 520), width="stretch")
    kosong = h["pm25_mean"].isna().mean() * 100
    st.caption(f"Garis titik-titik = baku mutu 24 jam 55 µg/m³. Celah pada garis PM2.5 = hari tanpa data valid "
               f"({angka(kosong)}% hari). " + ("Rumah sakit ini tidak dipakai dalam model karena paparannya tidak andal."
                                              if st_pilih in ("MER-01", "JMB-02", "MDN-01") else ""))
