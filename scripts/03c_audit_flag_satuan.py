from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "raw" / "package"

sites = pd.read_csv(list(PKG.rglob("sites.csv"))[0])
baseline = sites.set_index("station_code")["baseline_pm25"]

POL = ["pm25", "pm10", "so2", "co", "o3", "no2"]
CU = ["temperature_c", "humidity_pct", "wind_speed_ms", "wind_direction_deg"]
VAR = POL + CU
KOSONG = ["", "nan", "NaN", "NA", "N/A", "null", "NULL", "None"]

def num(s):
    s = s.str.strip()
    return pd.to_numeric(s.where(~s.isin(KOSONG)), errors="coerce")

# ---------- Gabungkan 27 file ke satu tabel kerja (HANYA untuk audit ini) ----------
frames = []
for p in sorted(PKG.glob("*_AQMS.csv")):
    kode = p.name.split("_")[0]
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    out = pd.DataFrame(index=d.index)
    out["file"] = kode
    out["satuan"] = "ppm" if "pm25_ppm" in d.columns else "ug_m3"
    out["qc"] = d["qc_flag"].str.strip().replace("", "(kosong)")
    out["ts"] = pd.to_datetime(d["timestamp_start"], errors="coerce")
    out["te"] = pd.to_datetime(d["timestamp_end"], errors="coerce")
    for pol in POL:
        col = f"{pol}_ug_m3" if f"{pol}_ug_m3" in d.columns else f"{pol}_ppm"
        out[pol] = num(d[col])
    for c in CU:
        out[c] = num(d[c])
    frames.append(out)
A = pd.concat(frames, ignore_index=True)
A["menit"] = ((A["te"] - A["ts"]).dt.total_seconds() / 60).round(0)

# nilai tak hingga (inf) dicatat dulu, lalu dianggap kosong HANYA dalam audit ini
n_inf = np.isinf(A[VAR]).groupby(A["file"]).sum()
n_inf = n_inf.loc[n_inf.sum(axis=1) > 0]
print("A. JUMLAH NILAI TAK HINGGA (inf) PER STASIUN DAN KOLOM")
print(n_inf.to_string() if len(n_inf) else "Tidak ada.")
A[VAR] = A[VAR].replace([np.inf, -np.inf], np.nan)

# ---------- B. Hubungan qc_flag dengan kondisi nilai ----------
def profil(df):
    df = df.assign(
        pm25_kosong=df["pm25"].isna(), pm25_nol=df["pm25"] == 0,
        pm25_negatif=df["pm25"] < 0, pm25_999=df["pm25"] == 999,
        semua_polutan_kosong=df[POL].isna().all(axis=1),
        semua_10_kosong=df[VAR].isna().all(axis=1))
    kol = ["pm25_kosong", "pm25_nol", "pm25_negatif", "pm25_999",
           "semua_polutan_kosong", "semua_10_kosong"]
    t = df.groupby("qc")[kol].mean().mul(100).round(2)
    t.insert(0, "n", df.groupby("qc").size())
    t["pm25_median_ug"] = df.loc[df["satuan"] == "ug_m3"].groupby("qc")["pm25"].median().round(2)
    return t

print("\nB1. qc_flag vs KONDISI NILAI (semua stasiun KECUALI JMB-02; persen per qc_flag)")
print(profil(A[A["file"] != "JMB-02"]).to_string())
print("\nB2. qc_flag vs KONDISI NILAI (JMB-02 saja)")
print(profil(A[A["file"] == "JMB-02"]).to_string())

print("\nB3. JMB-02: statistik baris berstatus VALID saja")
J = A[(A["file"] == "JMB-02") & (A["qc"] == "VALID")]
print(J[VAR].describe().T[["count", "min", "50%", "max"]].round(3).to_string())

# ---------- C. Satuan ppm ----------
print("\n" + "=" * 80)
print("C. SATUAN PPM: PEMBANDING KE STASIUN ug/m3 (baris VALID saja, presisi penuh)")
V = A[(A["qc"] == "VALID") & (A["file"] != "JMB-02")]
ppm_med = V[V["satuan"] == "ppm"].groupby("file")[POL].median()
ug_med = V[V["satuan"] == "ug_m3"].groupby("file")[POL].median()
print("\nMedian stasiun ppm:")
print(ppm_med.round(6).to_string())
print("\nMedian dari median antar-stasiun ug_m3 (referensi kasar):")
print(ug_med.median().round(3).to_string())
print("\nRasio referensi ug_m3 / median ppm (faktor yang 'tersirat'):")
print((ug_med.median() / ppm_med).round(1).to_string())

MW = {"so2": 64.06, "co": 28.01, "o3": 48.00, "no2": 46.01}
print("\nFaktor konversi teoritis gas ppm -> ug/m3 (25 C, 1 atm = MW/24.45*1000):")
print({k: round(v / 24.45 * 1000, 1) for k, v in MW.items()})
print("Untuk PM: bila kolom 'ppm' ternyata mg/m3, faktornya 1000.")

print("\nPerbandingan median PM2.5 (VALID) dengan baseline_pm25 di sites.csv")
rows = []
for kode, g in V.groupby("file"):
    rows.append({"file": kode, "satuan": g["satuan"].iloc[0], "n_valid": len(g),
                 "pm25_median": g["pm25"].median(),
                 "baseline_pm25": baseline.get(f"AQMS-{kode}")})
R = pd.DataFrame(rows)
R["rasio_median_per_baseline"] = R["pm25_median"] / R["baseline_pm25"]
print(R.round(4).to_string(index=False))

# ---------- D. Konsistensi fisik: PM2.5 tidak boleh melebihi PM10 ----------
print("\n" + "=" * 80)
print("D. PM2.5 > PM10 (secara fisik tidak mungkin)")
L = A[A["file"] != "JMB-02"].dropna(subset=["pm25", "pm10"])
L = L.assign(lebih=L["pm25"] > L["pm10"])
t = L.groupby("file")["lebih"].agg(jumlah="sum", dari="size")
t["persen"] = (t["jumlah"] / t["dari"] * 100).round(2)
print(t[t["jumlah"] > 0].to_string() if (t["jumlah"] > 0).any() else "Tidak ada.")
if L["lebih"].any():
    print("\nSebaran qc_flag pada baris yang melanggar:")
    print(L[L["lebih"]].groupby(["file", "qc"]).size().to_string())

# ---------- E. Plafon nilai ----------
print("\n" + "=" * 80)
print("E. NILAI MAKSIMUM YANG MUNCUL BERULANG (>= 10 baris), indikasi plafon/clipping")
rows = []
for kode, g in A[A["file"] != "JMB-02"].groupby("file"):
    for v in POL:
        s = g[v].dropna()
        if len(s) == 0:
            continue
        mx = s.max()
        n = int((s == mx).sum())
        if n >= 10:
            rows.append({"file": kode, "variabel": v, "maks": mx, "n_baris_di_maks": n})
print(pd.DataFrame(rows).to_string(index=False) if rows else "Tidak ada.")

# ---------- F. PLB-01: resolusi x qc_flag ----------
print("\n" + "=" * 80)
print("F. PLB-01: JENDELA (menit) x qc_flag")
b = A[A["file"] == "PLB-01"]
print(pd.crosstab(b["menit"], b["qc"]).to_string())

# ---------- G. Pola bulanan: persen baris bukan VALID ----------
print("\n" + "=" * 80)
print("G. PERSEN BARIS BUKAN VALID PER BULAN (stasiun bermasalah + 1 stasiun pembanding)")
Z = A[A["file"].isin(["MER-01", "PLB-01", "MDN-01", "SM2-01", "JKT-01"])].copy()
Z["bulan"] = Z["ts"].dt.to_period("M").astype(str)
Z["bukan_valid"] = Z["qc"].ne("VALID")
print((Z.pivot_table(index="bulan", columns="file", values="bukan_valid", aggfunc="mean") * 100).round(1).to_string())