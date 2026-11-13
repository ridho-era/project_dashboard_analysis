from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

# =================== KONFIGURASI ===================
MIN_JAM_HARIAN = 18        # >= 18 dari 24 jam (75%): Permen LHK P.14/2020 Pasal 7 ayat 3
MIN_HARI_TAHUNAN = 274     # >= 75% dari 365 hari: aturan kelengkapan KITA (analog), bukan kutipan peraturan
# Kategori ISPU PM2.5 24 jam (ug/m3): Baik <=15,5 | Sedang <=55,4 | Tidak Sehat <=150,4 | Sangat Tidak Sehat <=250,4 | Berbahaya >250,4
ISPU_BATAS = [-np.inf, 15.5, 55.4, 150.4, 250.4, np.inf]
ISPU_LABEL = ["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]
BM_NASIONAL_PM25_24J = 55      # PP 22/2021
BM_NASIONAL_PM10_24J = 75      # PP 22/2021
BM_NASIONAL_PM25_TAHUN = 15    # PP 22/2021
PEDOMAN_WHO_PM25_24J = 15      # WHO 2021 (24 jam)
PEDOMAN_WHO_PM10_24J = 45      # WHO 2021 (24 jam)
# ===================================================

C = pd.read_parquet(OUT / "aqms_hourly_clean.parquet")
C["date"] = C["ts_jam"].dt.normalize().astype("datetime64[ns]")
rad = np.deg2rad(C["wind_direction_deg"])
C["_s"], C["_c"] = np.sin(rad), np.cos(rad)

POL = ["pm25_ug_m3", "pm10_ug_m3", "so2_ug_m3", "co_ug_m3", "o3_ug_m3", "no2_ug_m3"]
CUM = ["temperature_c", "humidity_pct", "wind_speed_ms"]
NAMA = {"pm25_ug_m3": "pm25_mean", "pm10_ug_m3": "pm10_mean", "so2_ug_m3": "so2_mean",
        "co_ug_m3": "co_mean", "o3_ug_m3": "o3_mean", "no2_ug_m3": "no2_mean",
        "temperature_c": "temp_mean", "humidity_pct": "rh_mean", "wind_speed_ms": "ws_mean"}

g = C.groupby(["stasiun", "date"])
cnt = g[POL + CUM].count()
D = g[POL + CUM].mean().where(cnt >= MIN_JAM_HARIAN).rename(columns=NAMA)
D["n_jam_pm25"] = cnt["pm25_ug_m3"]
D["pm25_max_jam"] = g["pm25_ug_m3"].max()
wd = np.degrees(np.arctan2(g["_s"].mean(), g["_c"].mean())) % 360
D["wd_mean"] = wd.where(wd < 360, 0.0).where(g["wind_direction_deg"].count() >= MIN_JAM_HARIAN)
D["n_jam_flatline"] = g["flatline"].sum().astype(int)
D["n_jam_999"] = g["tersaturasi_999"].sum().astype(int)
D["dikonversi"] = g["dikonversi"].max()

D["ispu_pm25"] = pd.cut(D["pm25_mean"], bins=ISPU_BATAS, labels=ISPU_LABEL)
def bendera(seri, batas):
    return (seri > batas).astype("boolean").mask(seri.isna())
D["pm25_di_atas_bm_nasional"] = bendera(D["pm25_mean"], BM_NASIONAL_PM25_24J)
D["pm25_di_atas_who"] = bendera(D["pm25_mean"], PEDOMAN_WHO_PM25_24J)
D["pm10_di_atas_bm_nasional"] = bendera(D["pm10_mean"], BM_NASIONAL_PM10_24J)
D["pm10_di_atas_who"] = bendera(D["pm10_mean"], PEDOMAN_WHO_PM10_24J)
D = D.reset_index()
D.to_parquet(OUT / "air_daily.parquet", index=False)

n_hari = D.groupby("stasiun").size()
print("1. TABEL HARIAN UDARA")
print("baris:", len(D), "| stasiun:", D["stasiun"].nunique(), "| hari per stasiun (min/maks):", n_hari.min(), "/", n_hari.max())

# ---------- 2. Ringkasan per stasiun ----------
valid = D[D["pm25_mean"].notna()]
rk = pd.DataFrame({"hari_valid": valid.groupby("stasiun").size()}).reindex(n_hari.index).fillna(0).astype(int)
rk["persen_valid"] = (rk["hari_valid"] / n_hari * 100).round(1)
kat = (pd.crosstab(valid["stasiun"], valid["ispu_pm25"], normalize="index") * 100).round(1)
kat = kat.reindex(columns=ISPU_LABEL, fill_value=0.0)
rk = rk.join(kat)
rk["di_atas_BM_55_%"] = (valid.groupby("stasiun")["pm25_di_atas_bm_nasional"].mean() * 100).round(1)
rk["di_atas_WHO_15_%"] = (valid.groupby("stasiun")["pm25_di_atas_who"].mean() * 100).round(1)
rk["pm25_rata2_harian"] = valid.groupby("stasiun")["pm25_mean"].mean().round(1)
print("\n2. HARI VALID (>= 18 jam) DAN SEBARAN KATEGORI ISPU PM2.5 PER STASIUN (persen dari hari valid)")
print(rk.to_string())

# ---------- 3. Rata-rata tahunan vs baku mutu tahunan ----------
print("\n3. RATA-RATA PM2.5 TAHUNAN (hanya jika >= 274 hari valid) vs baku mutu tahunan 15 ug/m3 (PP 22/2021)")
valid = valid.assign(tahun=valid["date"].dt.year)
th = valid[valid["tahun"].isin([2024, 2025])].groupby(["stasiun", "tahun"])["pm25_mean"].agg(["mean", "size"])
th["mean"] = th["mean"].where(th["size"] >= MIN_HARI_TAHUNAN)
print(th["mean"].unstack().round(1).to_string())

# ---------- 4. Gabungkan dengan data kesehatan dan dimensi ----------
H = pd.read_parquet(OUT / "health_daily_clean.parquet")
dim = pd.read_parquet(OUT / "dim_stasiun.parquet")
H["date"] = H["date"].astype("datetime64[ns]")
A = H.merge(D, on=["stasiun", "date"], how="left", validate="one_to_one")
A = A.merge(dim[["stasiun", "pulau", "provinsi", "site_type", "kelas_faskes",
                 "kapasitas_tempat_tidur", "status_operasional", "baseline_pm25"]],
            on="stasiun", how="left", validate="many_to_one")
A.to_parquet(OUT / "analysis_daily.parquet", index=False)
print("\n4. TABEL ANALISIS (kesehatan + udara harian + dimensi)")
print("baris:", len(A), "| baris dengan PM2.5 harian valid:", int(A["pm25_mean"].notna().sum()),
      f"({A['pm25_mean'].notna().mean() * 100:.1f}%)")
print("hari kesehatan tanpa PM2.5 valid per rumah sakit (hanya yang > 10%):")
tanpa = (A["pm25_mean"].isna().groupby(A["hospital_id"]).mean() * 100).round(1)
print(tanpa[tanpa > 10].sort_values(ascending=False).to_string())
print("\nTersimpan:", OUT / "air_daily.parquet", "|", OUT / "analysis_daily.parquet")