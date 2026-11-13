from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "raw" / "package"

POL = ["pm25", "pm10", "so2", "co", "o3", "no2"]
CU = ["temperature_c", "humidity_pct", "wind_speed_ms", "wind_direction_deg"]
VAR = POL + CU
KOSONG = ["", "nan", "NaN", "NA", "N/A", "null", "NULL", "None"]

def num(s):
    s = s.str.strip()
    return pd.to_numeric(s.where(~s.isin(KOSONG)), errors="coerce")

# ---------- Tabel kerja untuk audit (tidak disimpan, tidak mengubah data mentah) ----------
frames = []
for p in sorted(PKG.glob("*_AQMS.csv")):
    kode = p.name.split("_")[0]
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    out = pd.DataFrame(index=d.index)
    out["file"] = kode
    out["satuan"] = "ppm" if "pm25_ppm" in d.columns else "ug_m3"
    out["qc"] = d["qc_flag"].str.strip().replace("", "(kosong)")
    out["ts"] = pd.to_datetime(d["timestamp_start"], errors="coerce")
    for pol in POL:
        col = f"{pol}_ug_m3" if f"{pol}_ug_m3" in d.columns else f"{pol}_ppm"
        out[pol] = num(d[col])
    for c in CU:
        out[c] = num(d[c])
    frames.append(out)
A = pd.concat(frames, ignore_index=True)
A[VAR] = A[VAR].replace([np.inf, -np.inf], np.nan)

# HIPOTESIS (hanya untuk audit ini): PM bersatuan 'ppm' = mg/m3 -> x1000 = ug/m3
A["pm25_ug"] = A["pm25"] * np.where(A["satuan"] == "ppm", 1000.0, 1.0)
A["tahun"] = A["ts"].dt.year

# JMB-02 dikeluarkan (timestamp 1970); urutkan per stasiun dan waktu
B = A[A["file"] != "JMB-02"].sort_values(["file", "ts"]).reset_index(drop=True)

# ---------- A. Detektor flatline: nilai PM2.5 identik berturut-turut ----------
print("A. RANGKAIAN NILAI PM2.5 IDENTIK BERTURUT-TURUT (>= 12 baris), SEMUA FLAG")
rows = []
for kode, g in B.groupby("file"):
    s = g["pm25"]
    awal = (s != s.shift()) | s.isna()          # awal rangkaian baru
    run = awal.cumsum()
    h = g.assign(run=run)[s.notna()]
    agg = h.groupby("run").agg(
        n=("pm25", "size"), nilai=("pm25", "first"),
        mulai=("ts", "min"), selesai=("ts", "max"),
        qc_utama=("qc", lambda x: x.value_counts().index[0]))
    agg = agg[agg["n"] >= 12]
    if len(agg):
        agg.insert(0, "file", kode)
        rows.append(agg)
if rows:
    R = pd.concat(rows)
    print("\nRingkasan per stasiun:")
    print(R.groupby("file").agg(jumlah_rangkaian=("n", "size"), total_baris=("n", "sum"),
                                terpanjang=("n", "max")).to_string())
    print("\n40 rangkaian terpanjang:")
    print(R.sort_values("n", ascending=False).head(40).to_string(index=False))
else:
    print("Tidak ada.")

# ---------- B. BJB-01: rincian blok plafon ----------
print("\n" + "=" * 80)
print("B. BJB-01: BLOK BARIS DI NILAI MAKSIMUM PM2.5")
g = A[A["file"] == "BJB-01"].sort_values("ts")
cap = g["pm25"].max()
p = g[g["pm25"] == cap]
rentang_jam = int((p["ts"].max() - p["ts"].min()) / pd.Timedelta("1h")) + 1
dalam = int(((g["ts"] >= p["ts"].min()) & (g["ts"] <= p["ts"].max())).sum())
print(f"baris di maksimum: {len(p)} | dari {p['ts'].min()} sampai {p['ts'].max()}")
print(f"jumlah jam pada rentang itu: {rentang_jam} | jumlah baris BJB-01 pada rentang itu: {dalam}")
print("\nVariabel lain pada baris blok tersebut (min / median / maks / std):")
print(p[POL + CU].describe().T[["min", "50%", "max", "std"]].round(4).to_string())

# ---------- C. SUSPECT vs tetangga VALID (kontras lokal) ----------
print("\n" + "=" * 80)
print("C. NILAI BARIS SUSPECT DIBANDINGKAN MEDIAN BARIS VALID DI SEKITARNYA (+-6 jam)")
print("Rasio ~1 = sejalan dengan kondisi sekitar; jauh dari 1 = lonjakan/anomali.")

def ringkas(r, nama):
    r = r.dropna()
    q = r.quantile([.05, .25, .5, .75, .95]).round(2).tolist() if len(r) else []
    print(f"  {nama:<28} n={len(r):>5}  kuantil 5/25/50/75/95 = {q}")

for kode in ["MDN-01", "MER-01", "SM2-01"]:
    g = B[B["file"] == kode]
    idx = pd.DatetimeIndex(g["ts"])
    val = pd.Series(g["pm25_ug"].values, index=idx)
    valid = pd.Series((g["qc"] == "VALID").values, index=idx)
    ref = val.where(valid).rolling("13h", center=True, min_periods=5).median()
    rasio = val / ref
    sus = pd.Series((g["qc"] == "SUSPECT").values, index=idx)
    print(f"\n{kode}:")
    ringkas(rasio[sus & (val <= 0)], "SUSPECT bernilai <= 0")
    ringkas(rasio[sus & (val == 999)], "SUSPECT bernilai 999")
    ringkas(rasio[sus & (val > 0) & (val != 999)], "SUSPECT lainnya")

# ---------- D. Episode ekstrem (VALID, PM2.5 >= 150) per tahun ----------
print("\n" + "=" * 80)
print("D. EPISODE EKSTREM PER TAHUN (VALID, PM2.5 >= 150 ug/m3; BJB-01 blok plafon dikeluarkan)")
V = B[B["qc"] == "VALID"]
cap_bjb = A.loc[A["file"] == "BJB-01", "pm25"].max()
V = V[~((V["file"] == "BJB-01") & (V["pm25"] == cap_bjb))]
ek = V[V["pm25_ug"] >= 150]
if len(ek):
    t = ek.groupby(["file", "tahun"]).agg(n_baris=("pm25_ug", "size"),
                                          pertama=("ts", "min"), terakhir=("ts", "max"),
                                          maks=("pm25_ug", "max")).round(1)
    print(t.to_string())
    print("\nCatatan: PLB-01 memakai baris 15 menit pada Sep-Okt, jadi n_baris tidak sama dengan jumlah jam.")
else:
    print("Tidak ada.")