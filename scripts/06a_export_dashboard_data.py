from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
DASH = ROOT / "dashboard" / "data"
DASH.mkdir(parents=True, exist_ok=True)

# =================== KONFIGURASI ===================
HARI_TOTAL = 1018
BATAS_EFEKTIF = 75.0              # stasiun efektif bila >= 75% hari punya PM2.5 harian valid
BATAS_BM, BATAS_STS = 55.0, 150.4 # PP 22/2021 (24 jam) dan batas ISPU 'Sangat Tidak Sehat'
REF_SO2, REF_NO2 = 20.0, 40.0     # rumus IKU (Permen LHK 27/2021)
KELENGKAPAN_80 = 0.80
JENDELA_AKHIR = 1014              # bulan*100 + hari = 14 Okt
MIN_HARI_PASANGAN = 200
KELOMPOK_URUT = [
    ("peat", "Gambut"), ("industrial", "Industri"),
    ("traffic", "Lalu lintas"), ("roadside", "Lalu lintas"), ("highway", "Lalu lintas"), ("artery", "Lalu lintas"),
    ("coastal", "Pesisir"), ("maritime", "Pesisir"),
    ("clean", "Latar bersih"), ("regional", "Latar bersih"), ("remote", "Latar bersih"),
    ("urban", "Perkotaan"),
]
# ===================================================

def kelompok(tipe):
    t = str(tipe).lower()
    for kunci, nama in KELOMPOK_URUT:
        if kunci in t:
            return nama
    return None

def simpan(df, nama):
    df.to_csv(DASH / nama, index=False, encoding="utf-8")
    print(f"{nama:<28} {len(df):>7} baris | {(DASH / nama).stat().st_size / 1e6:6.2f} MB")

dim = pd.read_parquet(OUT / "dim_stasiun.parquet")
dim["kelompok"] = dim["site_type"].map(kelompok)
assert dim["kelompok"].notna().all()
D = pd.read_parquet(OUT / "air_daily.parquet")
D["date"] = pd.to_datetime(D["date"])
D["tahun"], D["bulan"] = D["date"].dt.year, D["date"].dt.month
A = pd.read_parquet(OUT / "analysis_daily.parquet")
H = pd.read_parquet(OUT / "aqms_hourly_clean.parquet", columns=["stasiun", "ts_jam", "pm25_ug_m3"])
Dv = D[D["pm25_mean"].notna()]

# ---- 1. stasiun.csv ----
kolom = ["stasiun", "hospital_id", "sensor_id", "provinsi", "kota_kabupaten", "kecamatan", "pulau", "kelompok",
         "site_type", "latitude_dd", "longitude_dd", "status_operasional", "kelas_faskes", "kapasitas_tempat_tidur",
         "distance_to_road_m", "traffic_intensity_per_h", "baseline_pm25"]
S = dim[[c for c in kolom if c in dim.columns]].copy().set_index("stasiun")
S["hari_valid"] = Dv.groupby("stasiun").size().reindex(S.index).fillna(0).astype(int)
S["persen_valid"] = (S["hari_valid"] / HARI_TOTAL * 100).round(1)
S["kategori_data"] = np.where(S["persen_valid"] >= BATAS_EFEKTIF, "Efektif",
                              np.where(S["persen_valid"] >= 10, "Terbatas", "Buta"))
S["hari_di_atas_55"] = (Dv["pm25_mean"] > BATAS_BM).groupby(Dv["stasiun"]).sum().reindex(S.index).fillna(0).astype(int)
S["hari_sangat_tidak_sehat_plus"] = (Dv["pm25_mean"] > BATAS_STS).groupby(Dv["stasiun"]).sum().reindex(S.index).fillna(0).astype(int)
S["pm25_rata2"] = Dv.groupby("stasiun")["pm25_mean"].mean().reindex(S.index).round(1)
S["resp_rata2_per_hari_RS"] = A.groupby("stasiun")["resp"].mean().reindex(S.index).round(1)
simpan(S.reset_index(), "stasiun.csv")

# ---- 2. harian_udara.csv ----
kol_d = ["stasiun", "date", "tahun", "bulan", "pm25_mean", "pm10_mean", "so2_mean", "no2_mean", "temp_mean", "rh_mean",
         "ws_mean", "n_jam_pm25", "pm25_max_jam", "ispu_pm25", "pm25_di_atas_bm_nasional", "pm25_di_atas_who", "dikonversi"]
Dd = D[[c for c in kol_d if c in D.columns]].copy()
Dd["date"] = Dd["date"].dt.strftime("%Y-%m-%d")
simpan(Dd.round(2), "harian_udara.csv")

# ---- 3. bulanan_udara.csv ----
D["ym"] = D["date"].dt.to_period("M").astype(str)
B = D.groupby(["stasiun", "ym"]).agg(hari_total=("date", "size"), hari_valid=("pm25_mean", "count"),
                                     pm25_rata2=("pm25_mean", "mean"), pm25_maks_harian=("pm25_mean", "max")).reset_index()
bm = D.assign(a=D["pm25_mean"] > BATAS_BM, s=D["pm25_mean"] > BATAS_STS).groupby(["stasiun", "ym"])[["a", "s"]].sum().reset_index()
B = B.merge(bm.rename(columns={"a": "hari_di_atas_55", "s": "hari_sangat_tidak_sehat_plus"}), on=["stasiun", "ym"])
simpan(B.round(2), "bulanan_udara.csv")

# ---- 4. profil_jam.csv ----
H["jam"] = H["ts_jam"].dt.hour
P = H.groupby(["stasiun", "jam"])["pm25_ug_m3"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "pm25_rata2", "count": "n_jam"})
simpan(P.round(2), "profil_jam.csv")

# ---- 5. kesehatan_harian.csv ----
kol_h = ["hospital_id", "stasiun", "date", "dow", "resp_admissions_under5", "resp_admissions_over60", "asthma_emergency_cases",
         "copd_admissions", "resp", "cvd", "death", "temp_c", "rhum_pct", "pm25_mean", "ispu_pm25", "resp_inkonsisten"]
Ah = A[[c for c in kol_h if c in A.columns]].copy()
Ah["date"] = pd.to_datetime(Ah["date"]).dt.strftime("%Y-%m-%d")
simpan(Ah.round(2), "kesehatan_harian.csv")

# ---- 6. iku_tahunan.csv ----
rows = []
for (st, th), g in D.groupby(["stasiun", "tahun"]):
    gg = g.dropna(subset=["so2_mean", "no2_mean"])
    perlu = int(np.ceil(KELENGKAPAN_80 * len(g)))
    ok = len(gg) >= perlu
    so2, no2 = gg["so2_mean"].mean(), gg["no2_mean"].mean()
    ieu = (so2 / REF_SO2 + no2 / REF_NO2) / 2
    pm = g["pm25_mean"].dropna()
    rows.append({"stasiun": st, "tahun": th, "hari_valid": len(gg), "hari_perlu": perlu, "memenuhi_kelengkapan": ok,
                 "SO2": so2 if ok else np.nan, "NO2": no2 if ok else np.nan,
                 "IKU": (100 - (50 / 0.9) * (ieu - 0.1)) if ok else np.nan,
                 "PM25_tahunan": pm.mean() if len(pm) >= perlu else np.nan, "parsial": th == 2026})
simpan(pd.DataFrame(rows).round(2), "iku_tahunan.csv")

# ---- 7. tren_jendela.csv (1 Jan - 14 Okt tiap tahun) ----
md = D["date"].dt.month * 100 + D["date"].dt.day
W = D[md <= JENDELA_AKHIR]
rows = []
for (st, th), g in W.groupby(["stasiun", "tahun"]):
    p = g["pm25_mean"].dropna()
    ok = len(p) >= np.ceil(0.8 * len(g))
    rows.append({"stasiun": st, "tahun": th, "hari_valid": len(p), "hari_jendela": len(g),
                 "pm25_rata2": p.mean() if ok else np.nan, "memenuhi_kelengkapan": ok})
simpan(pd.DataFrame(rows).round(2), "tren_jendela.csv")

# ---- 8. jaringan: tetangga terdekat dan ringkasan provinsi ----
S2 = S.copy()
S2["efektif"] = S2["persen_valid"] >= BATAS_EFEKTIF
lat, lon = np.radians(S2["latitude_dd"].to_numpy(float)), np.radians(S2["longitude_dd"].to_numpy(float))
dl, dn = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
a = np.sin(dl / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dn / 2) ** 2
dist = 2 * 6371 * np.arcsin(np.sqrt(a))
np.fill_diagonal(dist, np.inf)
Pw = D.pivot(index="date", columns="stasiun", values="pm25_mean")
Pm = Pw - Pw.groupby(Pw.index.month).transform("mean")
rows = []
for i, st in enumerate(S2.index):
    j = int(np.argmin(dist[i]))
    tj = S2.index[j]
    ok, okm = Pw[[st, tj]].dropna(), Pm[[st, tj]].dropna()
    rows.append({"stasiun": st, "tetangga_terdekat": tj, "jarak_km": round(dist[i, j]), "n_hari_bersama": len(ok),
                 "korelasi_mentah": ok[st].corr(ok[tj]) if len(ok) >= MIN_HARI_PASANGAN else np.nan,
                 "korelasi_tanpa_musim": okm[st].corr(okm[tj]) if len(okm) >= MIN_HARI_PASANGAN else np.nan})
simpan(pd.DataFrame(rows).round(2), "tetangga.csv")

prov = S2.groupby("provinsi").agg(pulau=("pulau", "first"), stasiun=("persen_valid", "size"), efektif=("efektif", "sum"),
                                  hari_valid=("hari_valid", "sum"), hari_sangat_tidak_sehat_plus=("hari_sangat_tidak_sehat_plus", "sum"),
                                  hari_di_atas_55=("hari_di_atas_55", "sum"))
prov["persen_hari_di_atas_55"] = (prov["hari_di_atas_55"] / prov["hari_valid"].replace(0, np.nan) * 100).round(1)
prov["resp_rata2_per_hari_RS"] = A.groupby("provinsi")["resp"].mean().round(1)
def tingkat(r):
    if r["hari_valid"] / (r["stasiun"] * HARI_TOTAL) < 0.10:
        return "BUTA: cakupan data < 10%"
    terpapar = (r["persen_hari_di_atas_55"] >= 20) or (r["hari_sangat_tidak_sehat_plus"] > 0)
    if r["efektif"] == 0:
        return "TERBATAS: tidak ada stasiun efektif" + (" + terpapar" if terpapar else "")
    if terpapar and r["efektif"] <= 1:
        return "A: terpapar, hanya <=1 stasiun efektif"
    if terpapar and r["efektif"] < r["stasiun"]:
        return "B: terpapar, ada stasiun terbatas"
    return "C: tidak ada tanda kekurangan"
prov["tingkat"] = prov.apply(tingkat, axis=1)
simpan(prov.reset_index(), "jaringan_provinsi.csv")
print("\nSelesai. Folder:", DASH)