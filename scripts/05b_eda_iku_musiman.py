from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

# =================== KONFIGURASI ===================
REF_SO2, REF_NO2 = 20.0, 40.0     # nilai referensi EU pada rumus IKU (Permen LHK 27/2021)
KELENGKAPAN_IKU = 0.80            # AQMS tetap: minimal 80% data harian per tahun (292 dari 365 hari)
BATAS_BM_24J = 55.0               # baku mutu PM2.5 24 jam, PP 22/2021
JENDELA_AKHIR = (10, 14)          # bulan-hari terakhir data (14 Okt) untuk perbandingan tahun yang adil
# ===================================================

D = pd.read_parquet(OUT / "air_daily.parquet")
dim = pd.read_parquet(OUT / "dim_stasiun.parquet")
D = D.merge(dim[["stasiun", "pulau"]], on="stasiun", how="left", validate="many_to_one")
D["tahun"] = D["date"].dt.year
D["bulan"] = D["date"].dt.month

# ---------- A. IKU per stasiun per tahun (hanya SO2 dan NO2) ----------
print("A. IKU = 100 - (50/0,9) x (Ieu - 0,1); Ieu = rata2(SO2/20 ; NO2/40); rata-rata tahunan dari rata-rata harian")
rows = []
for (st, th), g in D.groupby(["stasiun", "tahun"]):
    n_total = len(g)
    gg = g.dropna(subset=["so2_mean", "no2_mean"])
    perlu = int(np.ceil(KELENGKAPAN_IKU * n_total))
    ok = len(gg) >= perlu
    so2, no2 = gg["so2_mean"].mean(), gg["no2_mean"].mean()
    ieu = (so2 / REF_SO2 + no2 / REF_NO2) / 2
    pm = g["pm25_mean"].dropna()
    rows.append({"stasiun": st, "tahun": th, "hari_valid": len(gg), "hari_perlu": perlu,
                 "SO2": so2 if ok else np.nan, "NO2": no2 if ok else np.nan,
                 "Ieu": ieu if ok else np.nan,
                 "IKU": (100 - (50 / 0.9) * (ieu - 0.1)) if ok else np.nan,
                 "PM25_tahunan": pm.mean() if len(pm) >= perlu else np.nan,
                 "dikonversi": bool(g["dikonversi"].max())})
R = pd.DataFrame(rows)
print(R.round(2).to_string(index=False))
print("\nCatatan: 2026 hanya sampai 14 Okt (parsial). Kolom hari_perlu = 80% dari jumlah hari pada tahun itu.")
luar = R[(R["IKU"] < 0) | (R["IKU"] > 100)]
print("IKU di luar 0-100 (jangan dipotong sendiri; periksa lampiran peraturan):", len(luar))

# ---------- B. Pola bulanan (kalender): rata-rata PM2.5 harian ----------
print("\n" + "=" * 80)
print("B1. RATA-RATA PM2.5 HARIAN PER BULAN KALENDER (semua tahun digabung), per stasiun")
print(D.groupby(["stasiun", "bulan"])["pm25_mean"].mean().unstack("bulan").round(0).to_string())

print("\nB2. PERSEN HARI VALID DENGAN PM2.5 > 55 per pulau dan bulan kalender")
V = D[D["pm25_mean"].notna()].copy()
V["di_atas"] = (V["pm25_mean"] > BATAS_BM_24J)
print((V.groupby(["pulau", "bulan"])["di_atas"].mean().unstack("bulan") * 100).round(0).to_string())

# ---------- C. Perbandingan tahun pada jendela yang sama (1 Jan - 14 Okt) ----------
print("\n" + "=" * 80)
print("C. RATA-RATA PM2.5 HARIAN PADA JENDELA 1 JAN - 14 OKT TIAP TAHUN (hanya jika >= 80% hari valid)")
md = D["date"].dt.month * 100 + D["date"].dt.day
W = D[md <= JENDELA_AKHIR[0] * 100 + JENDELA_AKHIR[1]]
rows = []
for (st, th), g in W.groupby(["stasiun", "tahun"]):
    n_total = len(g)
    p = g["pm25_mean"].dropna()
    rows.append({"stasiun": st, "tahun": th, "pm25": p.mean() if len(p) >= np.ceil(0.8 * n_total) else np.nan})
T = pd.DataFrame(rows).pivot(index="stasiun", columns="tahun", values="pm25")
if 2024 in T.columns and 2026 in T.columns:
    T["selisih_2026_vs_2024"] = T[2026] - T[2024]
print(T.round(1).to_string())

# ---------- D. Ringkasan per pulau ----------
print("\n" + "=" * 80)
print("D. RINGKASAN PER PULAU (hari valid semua stasiun digabung)")
V["tidak_sehat_atau_lebih"] = V["ispu_pm25"].isin(["Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"])
ring = V.groupby("pulau").agg(stasiun=("stasiun", "nunique"), hari_stasiun=("pm25_mean", "size"),
                              pm25_rata2=("pm25_mean", "mean"),
                              persen_di_atas_55=("di_atas", "mean"),
                              persen_tidak_sehat_atau_lebih=("tidak_sehat_atau_lebih", "mean"))
ring["persen_di_atas_55"] *= 100
ring["persen_tidak_sehat_atau_lebih"] *= 100
print(ring.round(1).to_string())