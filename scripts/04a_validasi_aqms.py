from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "raw" / "package"
OUT = ROOT / "data" / "processed"

GRID = pd.date_range("2024-01-01 00:00:00", "2026-10-14 23:00:00", freq="h")   # samakan dengan 04a_clean_aqms.py
POL = ["pm25_ug_m3", "pm10_ug_m3", "so2_ug_m3", "co_ug_m3", "o3_ug_m3", "no2_ug_m3"]
CU = ["temperature_c", "humidity_pct", "wind_speed_ms", "wind_direction_deg"]

C = pd.read_parquet(OUT / "aqms_hourly_clean.parquet")

# ---------- 1. Struktur ----------
print("1. STRUKTUR")
n_st = C["stasiun"].nunique()
harap = n_st * len(GRID)
print(f"baris: {len(C)} | stasiun: {n_st} | harapan: {harap} -> {'OK' if len(C) == harap else 'PERIKSA'}")
print("duplikat (stasiun, ts_jam):", int(C.duplicated(["stasiun", "ts_jam"]).sum()))
per = C.groupby("stasiun")["ts_jam"].agg(["size", "min", "max"])
print("stasiun dengan jumlah jam != grid:", per[per["size"] != len(GRID)].index.tolist())
print("rentang ts_jam:", C["ts_jam"].min(), "->", C["ts_jam"].max())
print("jam tidak tepat di jam bulat:", int((C["ts_jam"].dt.minute + C["ts_jam"].dt.second != 0).sum()))

# ---------- 2. Cakupan ----------
print("\n" + "=" * 80)
print("2. CAKUPAN (% jam tersedia) PER STASIUN DAN VARIABEL")
cov = C[POL + CU].notna().groupby(C["stasiun"]).mean().mul(100).round(1)
print(cov.to_string())
print("\nJam bertanda khusus per stasiun (hanya yang > 0):")
tanda = C.groupby("stasiun")[["flatline", "tersaturasi_999"]].sum()
print(tanda[tanda.sum(axis=1) > 0].to_string())

print("\nCakupan PM2.5 per bulan (%) untuk stasiun bercakupan rendah:")
rendah = cov.index[cov["pm25_ug_m3"] < 90].tolist()
Z = C[C["stasiun"].isin(rendah)].copy()
Z["bulan"] = Z["ts_jam"].dt.to_period("M").astype(str)
Z["ada"] = Z["pm25_ug_m3"].notna()
print((Z.pivot_table(index="bulan", columns="stasiun", values="ada", aggfunc="mean") * 100).round(0).to_string())

# ---------- 3. Rentang dan konsistensi ----------
print("\n" + "=" * 80)
print("3. RENTANG DAN KONSISTENSI SETELAH CLEANING")
rows = []
for c in POL + CU:
    s = C[c]
    rows.append({"variabel": c, "min": s.min(), "p50": s.median(), "p99": s.quantile(.99),
                 "maks": s.max(), "negatif": int((s < 0).sum())})
print(pd.DataFrame(rows).round(2).to_string(index=False))
wd = C["wind_direction_deg"]
print("humidity_pct di luar 0-100:", int(((C["humidity_pct"] < 0) | (C["humidity_pct"] > 100)).sum()))
print("wind_direction_deg di luar [0,360]:", int(((wd < 0) | (wd > 360)).sum()),
      "| bernilai tepat 360 (setara 0):", int((wd == 360).sum()))
print("temperature_c di luar 5-45:", int(((C["temperature_c"] < 5) | (C["temperature_c"] > 45)).sum()))
lebih = C[C["pm25_ug_m3"] > C["pm10_ug_m3"]]
print("PM2.5 > PM10:", len(lebih), "|", lebih["stasiun"].value_counts().to_dict() if len(lebih) else "")

# ---------- 4. Sebelum vs sesudah (PM2.5, ug/m3) ----------
print("\n" + "=" * 80)
print("4. PM2.5 SEBELUM vs SESUDAH CLEANING (sebelum: semua baris mentah; stasiun ppm x1000)")
rows = []
for p in sorted(PKG.glob("*_AQMS.csv")):
    kode = p.name.split("_")[0]
    hdr = pd.read_csv(p, nrows=0).columns
    col = "pm25_ug_m3" if "pm25_ug_m3" in hdr else "pm25_ppm"
    raw = pd.to_numeric(pd.read_csv(p, usecols=[col], dtype=str, keep_default_na=False)[col], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    if col == "pm25_ppm":
        raw = raw * 1000
    cl = C.loc[C["stasiun"] == kode, "pm25_ug_m3"].dropna()
    rows.append({"stasiun": kode, "n_sebelum": len(raw), "n_sesudah": len(cl),
                 "median_sebelum": raw.median(), "median_sesudah": cl.median(),
                 "p95_sebelum": raw.quantile(.95), "p95_sesudah": cl.quantile(.95),
                 "maks_sebelum": raw.max(), "maks_sesudah": cl.max()})
print(pd.DataFrame(rows).round(2).to_string(index=False))

# ---------- 5. Ulangi uji satuan ppm pada data bersih ----------
print("\n" + "=" * 80)
print("5. ULANG UJI PPM PADA DATA BERSIH")
sites = pd.read_csv(list(PKG.rglob("sites.csv"))[0]).set_index("station_code")["baseline_pm25"]
med = C.groupby("stasiun")[POL].median()
tab = pd.DataFrame({"dikonversi": C.groupby("stasiun")["dikonversi"].first(),
                    "n_jam": C.groupby("stasiun")["pm25_ug_m3"].count(),
                    "pm25_median": med["pm25_ug_m3"]})
tab["baseline_pm25"] = [sites.get(f"AQMS-{k}") for k in tab.index]
tab["rasio_median_baseline"] = tab["pm25_median"] / tab["baseline_pm25"]
print(tab.round(3).to_string())
ug = tab[(~tab["dikonversi"]) & (tab["n_jam"] >= 0.5 * len(GRID))]
r = ug["rasio_median_baseline"]
print(f"\nRasio stasiun ug_m3 (cakupan >= 50%): min {r.min():.2f} | median {r.median():.2f} | maks {r.max():.2f}")
print("Rasio stasiun ppm (sudah dikonversi):", tab.loc[tab["dikonversi"], "rasio_median_baseline"].round(2).to_dict())
ref = med.loc[ug.index]
print("\nGas: rentang median stasiun ug_m3 vs median stasiun ppm yang dikonversi")
ppm_st = tab.index[tab["dikonversi"]]
bandingan = pd.DataFrame({"min_ug": ref.min(), "median_ug": ref.median(), "maks_ug": ref.max()})
print(bandingan.round(2).to_string())
print(med.loc[ppm_st].round(2).to_string())

# ---------- 6. Episode ekstrem (jam, bukan baris) ----------
print("\n" + "=" * 80)
print("6. JAM DENGAN PM2.5 >= 150 ug/m3 PER STASIUN DAN TAHUN (ambang eksplorasi)")
ek = C[C["pm25_ug_m3"] >= 150]
if len(ek):
    print(ek.groupby(["stasiun", ek["ts_jam"].dt.year]).size().unstack().fillna(0).astype(int).to_string())
else:
    print("Tidak ada.")