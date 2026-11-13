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

# HIPOTESIS (hanya untuk audit ini): kolom PM bersatuan 'ppm' sebenarnya mg/m3 -> x1000 = ug/m3
A["pm25_ug"] = A["pm25"] * np.where(A["satuan"] == "ppm", 1000.0, 1.0)
A["bulan_kal"] = A["ts"].dt.month
A["ym"] = A["ts"].dt.to_period("M").astype(str)

B = A[A["file"] != "JMB-02"].copy()     # JMB-02 dikeluarkan: timestamp 1970 merusak analisis bulan

# ---------- A. Profil PM2.5 per stasiun dan per flag ----------
print("A. PM2.5 (ug/m3; stasiun ppm sudah x1000) PER STASIUN DAN qc_flag")
FLAGS = ["VALID", "SUSPECT", "CALIBRATING", "WARMUP"]
t = (B[B["qc"].isin(FLAGS)].groupby(["file", "qc"])["pm25_ug"]
       .agg(n="size", median="median", p95=lambda s: s.quantile(.95), maks="max").round(2))
print(t.unstack("qc").swaplevel(0, 1, axis=1).sort_index(axis=1).to_string())

# ---------- B. SUSPECT menurut bulan kalender ----------
print("\n" + "=" * 80)
print("B. PERSEN BARIS SUSPECT PER BULAN KALENDER (1=Jan ... 12=Des), per stasiun")
B["suspect"] = B["qc"].eq("SUSPECT")
print(B.groupby(["file", "bulan_kal"])["suspect"].mean().mul(100).round(0)
        .unstack("bulan_kal").to_string())

# ---------- C. Kandidat episode ekstrem pada baris VALID ----------
print("\n" + "=" * 80)
print("C. BARIS VALID DENGAN PM2.5 >= 150 ug/m3 (ambang EKSPLORASI, bukan ambang resmi)")
Vv = B[B["qc"] == "VALID"]
ek = Vv[Vv["pm25_ug"] >= 150]
if len(ek):
    tab = ek.groupby("file").agg(n=("pm25_ug", "size"), maks=("pm25_ug", "max"))
    tab["persen_dari_valid"] = (tab["n"] / Vv.groupby("file").size() * 100).round(2)
    print(tab.sort_values("n", ascending=False).round(1).to_string())
    print("\nJumlah baris ekstrem per stasiun x bulan kalender:")
    print(ek.groupby(["file", "bulan_kal"]).size().unstack("bulan_kal").fillna(0).astype(int).to_string())
else:
    print("Tidak ada.")

# ---------- D. BJB-01: nilai plafon ----------
print("\n" + "=" * 80)
print("D. BJB-01: BARIS DI NILAI MAKSIMUM (nilai mentah, satuan ppm)")
g = A[A["file"] == "BJB-01"]
if len(g):
    cap = {v: float(g[v].max()) for v in ["pm25", "pm10", "so2", "co"]}
    di_maks = pd.DataFrame({v: g[v] == cap[v] for v in cap})
    print("nilai maksimum:", cap)
    print("baris di maksimum per variabel:", di_maks.sum().to_dict())
    print("baris dengan keempatnya di maksimum:", int(di_maks.all(axis=1).sum()))
    print("qc_flag pada baris di maksimum PM2.5:", g.loc[di_maks["pm25"], "qc"].value_counts().to_dict())
    print("\nBaris di maksimum PM2.5 per tahun-bulan:")
    print(g.loc[di_maks["pm25"]].groupby("ym").size().to_string())

# ---------- E. Nilai pada baris CALIBRATING ----------
print("\n" + "=" * 80)
print("E. NILAI POLUTAN PADA BARIS CALIBRATING (nilai mentah)")
for kode in ["MER-01", "MDN-01", "SM2-01", "JKT-01"]:
    c = A[(A["file"] == kode) & (A["qc"] == "CALIBRATING")]
    if len(c) == 0:
        continue
    print(f"\n{kode}: {len(c)} baris CALIBRATING (satuan {c['satuan'].iloc[0]})")
    print(c[POL].describe().T[["count", "min", "50%", "max"]].round(4).to_string())

# ---------- F. SM2-01: nilai 999 ----------
print("\n" + "=" * 80)
print("F. SM2-01: NILAI 999")
s = A[A["file"] == "SM2-01"]
print("pm25==999:", int((s["pm25"] == 999).sum()), "| pm10==999:", int((s["pm10"] == 999).sum()),
      "| keduanya:", int(((s["pm25"] == 999) & (s["pm10"] == 999)).sum()))
m = s[(s["pm25"] == 999) | (s["pm10"] == 999)]
print("qc_flag baris 999:", m["qc"].value_counts().to_dict())
print("\nBaris 999 per tahun-bulan:")
print(m.groupby("ym").size().to_string())
sus = s[(s["qc"] == "SUSPECT") & (s["pm25"] != 999)]["pm25"]
print("\nSUSPECT bukan-999: n =", len(sus), "| median =", round(sus.median(), 2),
      "| p95 =", round(sus.quantile(.95), 2), "| maks =", round(sus.max(), 2))
val = s[s["qc"] == "VALID"]["pm25"]
print("VALID: n =", len(val), "| median =", round(val.median(), 2),
      "| p95 =", round(val.quantile(.95), 2), "| maks =", round(val.max(), 2))