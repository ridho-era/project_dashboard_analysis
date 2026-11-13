from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

# =================== KONFIGURASI ===================
MIN_PERSEN_HARI_VALID = 60       # RS dipakai jika >= 60% harinya punya PM2.5 harian valid
HASIL = ["resp_admissions_under5", "resp_admissions_over60", "asthma_emergency_cases",
         "copd_admissions", "resp", "cvd", "death"]
SINGKAT = {"resp_admissions_under5": "ISPA_balita", "resp_admissions_over60": "ISPA_lansia",
           "asthma_emergency_cases": "asma", "copd_admissions": "PPOK", "resp": "resp_umum",
           "cvd": "kardiovask", "death": "kematian"}
LAG_MAKS = 7
PER = 10.0                       # perubahan per +10 ug/m3 PM2.5
KECUALIKAN_RS = ["RS-XXX-23", "RS-XXX-27"]   # tanpa paparan andal (lihat catatan C-33)
# ===================================================

def poisson_fit(X, y, iters=100, tol=1e-9):
    """Regresi Poisson (IRLS). Mengembalikan koefisien dan matriks 'bread' (X'WX)^-1."""
    mu = y + 0.5
    eta = np.log(mu)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        z = eta + (y - mu) / mu
        XtW = X.T * mu
        beta_baru = np.linalg.solve(XtW @ X, XtW @ z)
        eta = X @ beta_baru
        mu = np.exp(eta)
        selesai = np.max(np.abs(beta_baru - beta)) < tol
        beta = beta_baru
        if selesai:
            break
    bread = np.linalg.inv((X.T * mu) @ X)
    return beta, mu, bread

def cov_klaster(X, y, mu, bread, klaster):
    """Kovarians robust terhadap klaster (rumah sakit): sandwich."""
    skor = X * (y - mu)[:, None]
    S = pd.DataFrame(skor).groupby(np.asarray(klaster)).sum().to_numpy()
    meat = S.T @ S
    G = S.shape[0]
    return bread @ meat @ bread * (G / (G - 1))

A = pd.read_parquet(OUT / "analysis_daily.parquet")
A = A.sort_values(["hospital_id", "date"]).reset_index(drop=True)

persen = A["pm25_mean"].notna().groupby(A["hospital_id"]).mean() * 100
dipakai = [h for h in persen.index if persen[h] >= MIN_PERSEN_HARI_VALID and h not in KECUALIKAN_RS]
print("RS dipakai:", len(dipakai), "dari", A["hospital_id"].nunique())
print("RS tidak dipakai (persen hari valid < batas atau dikecualikan):",
      {h: round(float(persen[h]), 1) for h in persen.index if h not in dipakai})
A = A[A["hospital_id"].isin(dipakai)].reset_index(drop=True)

# ---- paparan: jeda 0..7 hari dan rata-rata bergulir (dalam rumah sakit yang sama) ----
g = A.groupby("hospital_id")["pm25_mean"]
for L in range(LAG_MAKS + 1):
    A[f"x_lag{L}"] = g.shift(L)
for k in (2, 6):
    A[f"x_ma0_{k}"] = g.transform(lambda s, k=k: s.rolling(k + 1, min_periods=k + 1).mean())
paparan = [f"x_lag{L}" for L in range(LAG_MAKS + 1)] + ["x_ma0_2", "x_ma0_6"]

# ---- kovariat: efek tetap RS, hari dalam minggu, bulan, tahun, suhu, kelembapan ----
dasar = pd.concat([
    pd.get_dummies(A["hospital_id"], prefix="rs", drop_first=True),
    pd.get_dummies(A["dow"], prefix="dow", drop_first=True),
    pd.get_dummies(A["date"].dt.month, prefix="bln", drop_first=True),
    pd.get_dummies(A["date"].dt.year, prefix="thn", drop_first=True),
], axis=1).astype(float)
dasar["temp_c"] = A["temp_c"].astype(float)
dasar["rhum_pct"] = A["rhum_pct"].astype(float)
dasar.insert(0, "konstanta", 1.0)
Xb = dasar.to_numpy()

print("\nModel: Poisson, efek tetap RS + hari-minggu + bulan + tahun + suhu + kelembapan;")
print("SE robust per RS. Angka = perubahan persen kasus per +10 ug/m3 PM2.5 (* = interval 95% tidak memuat 0).\n")

hasil = []
for out in HASIL:
    y_all = A[out].to_numpy(dtype=float)
    for px in paparan:
        x = A[px].to_numpy(dtype=float)
        m = ~np.isnan(x)
        X = np.column_stack([Xb[m], x[m] / PER])
        y = y_all[m]
        beta, mu, bread = poisson_fit(X, y)
        cov = cov_klaster(X, y, mu, bread, A.loc[m, "hospital_id"].to_numpy())
        b, se = beta[-1], np.sqrt(cov[-1, -1])
        hasil.append({"hasil": SINGKAT[out], "paparan": px, "n": int(m.sum()),
                      "persen": (np.exp(b) - 1) * 100,
                      "bawah": (np.exp(b - 1.96 * se) - 1) * 100,
                      "atas": (np.exp(b + 1.96 * se) - 1) * 100})
R = pd.DataFrame(hasil)
R["tanda"] = np.where((R["bawah"] > 0) | (R["atas"] < 0), "*", "")
R["teks"] = R.apply(lambda r: f"{r['persen']:+.1f}{r['tanda']}", axis=1)
nama = {f"x_lag{L}": f"jeda {L} hari" for L in range(LAG_MAKS + 1)}
nama.update({"x_ma0_2": "rata2 jeda 0-2", "x_ma0_6": "rata2 jeda 0-6"})
T = R.pivot(index="paparan", columns="hasil", values="teks").reindex(list(nama))[list(SINGKAT.values())]
T.index = [nama[i] for i in T.index]
print("PERUBAHAN PERSEN KASUS PER +10 ug/m3 PM2.5")
print(T.to_string())

print("\nINTERVAL 95% (bawah ; atas) untuk JEDA TUNGGAL (0-7 hari) dengan estimasi tertinggi per hasil:")
print("(rata-rata bergulir tidak ikut diperingkat: perubahan 10 ug/m3 pada rata-rata beberapa hari adalah perubahan yang lebih berkelanjutan)")
for h, gg in R[R["paparan"].str.startswith("x_lag")].groupby("hasil"):
    top = gg.sort_values("persen", ascending=False).iloc[0]
    print(f"  {h:<12} {nama[top['paparan']]:<16} {top['persen']:+.2f}%  [{top['bawah']:+.2f} ; {top['atas']:+.2f}]  n={top['n']}")

print("\nJumlah observasi per paparan (hasil pertama):")
print(R[R["hasil"] == SINGKAT[HASIL[0]]].set_index("paparan")["n"].rename(index=nama).to_string())