from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

# =================== KONFIGURASI ===================
# Pengelompokan site_type -> kelompok. URUTAN = PRIORITAS (kata kunci pertama yang cocok menang).
KELOMPOK_URUT = [
    ("peat", "Gambut"), ("industrial", "Industri"),
    ("traffic", "Lalu lintas"), ("roadside", "Lalu lintas"), ("highway", "Lalu lintas"), ("artery", "Lalu lintas"),
    ("coastal", "Pesisir"), ("maritime", "Pesisir"),
    ("clean", "Latar bersih"), ("regional", "Latar bersih"), ("remote", "Latar bersih"),
    ("urban", "Perkotaan"),
]
MIN_HARI_STASIUN = 500            # stasiun dengan < 500 hari valid tidak dipakai pada perbandingan antarstasiun
SEKTOR = ["U", "TL", "T", "TG", "S", "BD", "B", "BL"]   # arah ASAL angin: Utara, Timur Laut, Timur, ...
# ===================================================

def kelompok(tipe):
    t = str(tipe).lower()
    for kunci, nama in KELOMPOK_URUT:
        if kunci in t:
            return nama
    return None

dim = pd.read_parquet(OUT / "dim_stasiun.parquet")
dim["kelompok"] = dim["site_type"].map(kelompok)
assert dim["kelompok"].notna().all(), f"site_type belum terpetakan: {dim.loc[dim['kelompok'].isna(), 'site_type'].tolist()}"
print("1. PENGELOMPOKAN site_type (periksa; ubah KELOMPOK_URUT bila tidak setuju)")
print(dim[["stasiun", "site_type", "kelompok"]].sort_values(["kelompok", "stasiun"]).to_string(index=False))
print("\nJumlah stasiun per kelompok:", dim["kelompok"].value_counts().to_dict())

H = pd.read_parquet(OUT / "aqms_hourly_clean.parquet",
                    columns=["stasiun", "ts_jam", "pm25_ug_m3", "wind_direction_deg"])
H["jam"] = H["ts_jam"].dt.hour
D = pd.read_parquet(OUT / "air_daily.parquet")
D["bulan"] = D["date"].dt.month
D["akhir_pekan"] = D["date"].dt.day_of_week >= 5
D = D.merge(dim[["stasiun", "kelompok", "pulau"]], on="stasiun", how="left")
Dv = D[D["pm25_mean"].notna()]
hari_valid = Dv.groupby("stasiun").size()
layak = hari_valid[hari_valid >= MIN_HARI_STASIUN].index

# ---------- 2. Pola jam (WIB sesuai data) ----------
print("\n" + "=" * 80)
print("2. POLA JAM-DALAM-HARI (jam sesuai kolom waktu data = WIB; baris VALID saja)")
prof = H.groupby(["stasiun", "jam"])["pm25_ug_m3"].mean().unstack("jam")
ring = pd.DataFrame({"jam_puncak": prof.idxmax(axis=1), "jam_terendah": prof.idxmin(axis=1),
                     "rasio_puncak_per_lembah": (prof.max(axis=1) / prof.min(axis=1)).round(2)})
print(ring.join(dim.set_index("stasiun")["kelompok"]).to_string())
norm = prof.div(prof.mean(axis=1), axis=0)
print("\nBentuk pola jam per kelompok (1,00 = rata-rata harian stasiun itu; rata-rata antarstasiun):")
print(norm.join(dim.set_index("stasiun")["kelompok"]).groupby("kelompok").mean().round(2).to_string())

# ---------- 3. Hari kerja vs akhir pekan ----------
print("\n" + "=" * 80)
print("3. AKHIR PEKAN vs HARI KERJA (rasio rata-rata PM2.5 harian; > 1 = akhir pekan lebih tinggi)")
wk = Dv.groupby(["stasiun", "akhir_pekan"])["pm25_mean"].mean().unstack("akhir_pekan")
if True in wk.columns and False in wk.columns:
    wk["rasio_akhir_pekan"] = (wk[True] / wk[False]).round(3)
    print(wk.join(dim.set_index("stasiun")["kelompok"]).round(2).to_string())
    print("\nMedian rasio per kelompok:")
    print(wk.join(dim.set_index("stasiun")["kelompok"]).groupby("kelompok")["rasio_akhir_pekan"].median().round(3).to_string())

# ---------- 4. Cuaca vs PM2.5 (Spearman, harian, dalam-stasiun) ----------
print("\n" + "=" * 80)
print("4. KORELASI SPEARMAN HARIAN PM2.5 vs CUACA, per stasiun (mentah dan tanpa pola musiman bulanan)")
rows = []
for st, g in Dv[Dv["stasiun"].isin(layak)].groupby("stasiun"):
    r = {"stasiun": st, "n_hari": len(g)}
    for v in ["temp_mean", "rh_mean", "ws_mean"]:
        gv = g.dropna(subset=[v])
        r[f"{v}_mentah"] = gv["pm25_mean"].corr(gv[v], method="spearman")
        a = gv["pm25_mean"] - gv.groupby("bulan")["pm25_mean"].transform("mean")
        b = gv[v] - gv.groupby("bulan")[v].transform("mean")
        r[f"{v}_tanpa_musim"] = a.corr(b, method="spearman")
    rows.append(r)
K = pd.DataFrame(rows).set_index("stasiun")
print(K.round(2).to_string())
print("\nMedian per kelompok:")
print(K.drop(columns="n_hari").join(dim.set_index("stasiun")["kelompok"]).groupby("kelompok").median().round(2).to_string())

# ---------- 5. Antarstasiun: lalu lintas, jarak jalan, baseline, tipe lokasi ----------
print("\n" + "=" * 80)
print(f"5. ANTARSTASIUN (hanya stasiun dengan >= {MIN_HARI_STASIUN} hari valid; n kecil, tafsirkan hati-hati)")
S = pd.DataFrame({"pm25_rata2": Dv.groupby("stasiun")["pm25_mean"].mean(),
                  "persen_hari_di_atas_55": Dv.groupby("stasiun")["pm25_di_atas_bm_nasional"].mean() * 100})
S = S.loc[layak].join(dim.set_index("stasiun")[["kelompok", "distance_to_road_m", "traffic_intensity_per_h", "baseline_pm25"]])
print(f"n stasiun: {len(S)}")
for kol in ["distance_to_road_m", "traffic_intensity_per_h", "baseline_pm25"]:
    print(f"Spearman PM2.5 rata-rata vs {kol}: {S['pm25_rata2'].corr(S[kol], method='spearman'):.2f}")
print("\nRingkasan per kelompok:")
print(S.groupby("kelompok").agg(stasiun=("pm25_rata2", "size"), pm25_rata2=("pm25_rata2", "mean"),
                                persen_hari_di_atas_55=("persen_hari_di_atas_55", "mean"),
                                jarak_jalan_m=("distance_to_road_m", "median"),
                                lalu_lintas_per_jam=("traffic_intensity_per_h", "median")).round(1).to_string())

# ---------- 6. Arah angin ----------
print("\n" + "=" * 80)
print("6. PM2.5 RATA-RATA PER SEKTOR ARAH ASAL ANGIN (8 sektor, per stasiun)")
W = H.dropna(subset=["pm25_ug_m3", "wind_direction_deg"]).copy()
W["sektor"] = (((W["wind_direction_deg"] + 22.5) // 45) % 8).astype(int)
m = W.groupby(["stasiun", "sektor"])["pm25_ug_m3"].mean().unstack("sektor")
m.columns = [SEKTOR[c] for c in m.columns]
sek = pd.DataFrame({"sektor_tertinggi": m.idxmax(axis=1), "sektor_terendah": m.idxmin(axis=1),
                    "rasio_tertinggi_per_terendah": (m.max(axis=1) / m.min(axis=1)).round(2)})
print(m.round(0).join(sek).to_string())