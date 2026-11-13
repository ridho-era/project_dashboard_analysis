from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

# =================== KONFIGURASI ===================
HARI_TOTAL = 1018
BATAS_EFEKTIF = 75.0             # stasiun 'efektif' bila >= 75% hari punya PM2.5 harian valid
BATAS_BM = 55.0                  # baku mutu PM2.5 24 jam (PP 22/2021)
BATAS_SANGAT_TIDAK_SEHAT = 150.4 # ISPU PM2.5 24 jam
MIN_HARI_PASANGAN = 200          # korelasi antar-stasiun dihitung bila >= 200 hari bersama
# ===================================================

dim = pd.read_parquet(OUT / "dim_stasiun.parquet")
D = pd.read_parquet(OUT / "air_daily.parquet")
A = pd.read_parquet(OUT / "analysis_daily.parquet")
Dv = D[D["pm25_mean"].notna()]

# ---------- 1. Tabel stasiun: nominal vs efektif ----------
S = dim.set_index("stasiun")[["provinsi", "pulau", "status_operasional", "latitude_dd", "longitude_dd"]].copy()
S["hari_valid"] = Dv.groupby("stasiun").size().reindex(S.index).fillna(0).astype(int)
S["persen_valid"] = (S["hari_valid"] / HARI_TOTAL * 100).round(1)
S["efektif"] = S["persen_valid"] >= BATAS_EFEKTIF
S["hari_di_atas_55_%"] = (Dv.assign(a=Dv["pm25_mean"] > BATAS_BM).groupby("stasiun")["a"].mean() * 100).round(1)
S["hari_sangat_tidak_sehat_atau_lebih"] = (Dv["pm25_mean"] > BATAS_SANGAT_TIDAK_SEHAT).groupby(Dv["stasiun"]).sum().reindex(S.index).fillna(0).astype(int)
S["resp_rata2_per_hari_RS"] = A.groupby("stasiun")["resp"].mean().round(1)
print("1. STASIUN NOMINAL vs EFEKTIF (efektif = >= 75% hari valid)")
print(S.sort_values("persen_valid").to_string())
print(f"\nStasiun nominal: {len(S)} | efektif: {int(S['efektif'].sum())} | terbatas: {int((~S['efektif']).sum())}")

# ---------- 2. Jarak dan keterwakilan antar-stasiun ----------
lat = np.radians(S["latitude_dd"].to_numpy(dtype=float))
lon = np.radians(S["longitude_dd"].to_numpy(dtype=float))
dl, dn = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
a = np.sin(dl / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dn / 2) ** 2
dist = 2 * 6371 * np.arcsin(np.sqrt(a))
np.fill_diagonal(dist, np.inf)
P = D.pivot(index="date", columns="stasiun", values="pm25_mean")
Pm = P - P.groupby(P.index.month).transform("mean")
rows = []
for i, st in enumerate(S.index):
    j = int(np.argmin(dist[i]))
    tj = S.index[j]
    ok = P[[st, tj]].dropna()
    okm = Pm[[st, tj]].dropna()
    rows.append({"stasiun": st, "tetangga_terdekat": tj, "jarak_km": round(dist[i, j]),
                 "n_hari_bersama": len(ok),
                 "korelasi_mentah": ok[st].corr(ok[tj]) if len(ok) >= MIN_HARI_PASANGAN else np.nan,
                 "korelasi_tanpa_musim": okm[st].corr(okm[tj]) if len(okm) >= MIN_HARI_PASANGAN else np.nan})
N = pd.DataFrame(rows).set_index("stasiun")
print("\n" + "=" * 80)
print("2. TETANGGA TERDEKAT: jarak dan keserupaan PM2.5 harian (mentah dan tanpa pola musiman bulanan)")
print(N.round(2).to_string())

# ---------- 3. Ringkasan per provinsi ----------
print("\n" + "=" * 80)
print("3. RINGKASAN PER PROVINSI")
prov = S.groupby("provinsi").agg(stasiun=("persen_valid", "size"), efektif=("efektif", "sum"),
                                 hari_valid=("hari_valid", "sum"),
                                 sangat_tidak_sehat_atau_lebih=("hari_sangat_tidak_sehat_atau_lebih", "sum"))
Dp = Dv.merge(dim[["stasiun", "provinsi"]], on="stasiun")
prov["persen_hari_di_atas_55"] = (Dp.assign(a=Dp["pm25_mean"] > BATAS_BM).groupby("provinsi")["a"].mean() * 100).round(1)
prov["pm25_harian_maks"] = Dp.groupby("provinsi")["pm25_mean"].max().round(0)
prov["pulau"] = S.groupby("provinsi")["pulau"].first()
prov["resp_rata2_per_hari_RS"] = A.groupby("provinsi")["resp"].mean().round(1)
def tingkat(r):
    cakupan = r["hari_valid"] / (r["stasiun"] * HARI_TOTAL)
    if cakupan < 0.10:
        return "BUTA: cakupan data < 10%"
    terpapar = (r["persen_hari_di_atas_55"] >= 20) or (r["sangat_tidak_sehat_atau_lebih"] > 0)
    if r["efektif"] == 0:
        return "TERBATAS: tidak ada stasiun efektif" + (" + terpapar" if terpapar else "")
    if terpapar and r["efektif"] <= 1:
        return "A: terpapar, hanya <=1 stasiun efektif"
    if terpapar and r["efektif"] < r["stasiun"]:
        return "B: terpapar, ada stasiun terbatas"
    return "C: tidak ada tanda kekurangan"
prov["tingkat"] = prov.apply(tingkat, axis=1)
print("Aturan: terpapar = (>= 20% hari di atas 55) ATAU (ada hari > 150,4). Aturan ini sederhana dan dapat Anda ubah; bukan rekomendasi final.")
print(prov.sort_values(["tingkat", "persen_hari_di_atas_55"], ascending=[True, False]).to_string())