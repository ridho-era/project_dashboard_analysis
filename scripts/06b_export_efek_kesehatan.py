from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
DASH = ROOT / "dashboard" / "data"
DASH.mkdir(parents=True, exist_ok=True)

# =================== KONFIGURASI (sama dengan 05d2) ===================
MIN_PERSEN_HARI_VALID = 60
HASIL = ["resp_admissions_under5", "resp_admissions_over60", "asthma_emergency_cases",
         "copd_admissions", "resp", "cvd", "death"]
SINGKAT = {"resp_admissions_under5": "ISPA_balita", "resp_admissions_over60": "ISPA_lansia",
           "asthma_emergency_cases": "asma", "copd_admissions": "PPOK", "resp": "resp_umum",
           "cvd": "kardiovask", "death": "kematian"}
LAG_MAKS, PER, JENDELA_LAMBAT = 7, 10.0, 29
KECUALIKAN_RS = ["RS-XXX-23", "RS-XXX-27"]
BATAS_KAT = (55.4, 150.4)
# ======================================================================

def poisson_fit(X, y, iters=100, tol=1e-9):
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
    return beta, mu, np.linalg.inv((X.T * mu) @ X)

def cov_klaster(X, y, mu, bread, klaster):
    skor = X * (y - mu)[:, None]
    S = pd.DataFrame(skor).groupby(np.asarray(klaster)).sum().to_numpy()
    G = S.shape[0]
    return bread @ (S.T @ S) @ bread * (G / (G - 1))

def pct(b, se):
    return (np.exp(b) - 1) * 100, (np.exp(b - 1.96 * se) - 1) * 100, (np.exp(b + 1.96 * se) - 1) * 100

A = pd.read_parquet(OUT / "analysis_daily.parquet").sort_values(["hospital_id", "date"]).reset_index(drop=True)
persen = A["pm25_mean"].notna().groupby(A["hospital_id"]).mean() * 100
dipakai = [h for h in persen.index if persen[h] >= MIN_PERSEN_HARI_VALID and h not in KECUALIKAN_RS]
A = A[A["hospital_id"].isin(dipakai)].reset_index(drop=True)
g = A.groupby("hospital_id")["pm25_mean"]
A["lp"] = g.transform(lambda s: s.rolling(JENDELA_LAMBAT, center=True, min_periods=JENDELA_LAMBAT // 2).mean())
A["hp"] = A["pm25_mean"] - A["lp"]
gh = A.groupby("hospital_id")["hp"]

dasar = pd.concat([pd.get_dummies(A["hospital_id"], prefix="rs", drop_first=True),
                   pd.get_dummies(A["dow"], prefix="dow", drop_first=True),
                   pd.get_dummies(A["date"].dt.month, prefix="bln", drop_first=True),
                   pd.get_dummies(A["date"].dt.year, prefix="thn", drop_first=True)], axis=1).astype(float)
dasar["temp_c"], dasar["rhum_pct"] = A["temp_c"].astype(float), A["rhum_pct"].astype(float)
dasar.insert(0, "konstanta", 1.0)
Xb = dasar.to_numpy()
kl = A["hospital_id"].to_numpy()

cepat, lambat = [], []
for out in HASIL:
    y_all = A[out].to_numpy(dtype=float)
    for L in range(LAG_MAKS + 1):
        x_hp, x_lp = gh.shift(L).to_numpy(dtype=float), A["lp"].to_numpy(dtype=float)
        m = ~np.isnan(x_hp) & ~np.isnan(x_lp)
        X = np.column_stack([Xb[m], x_lp[m] / PER, x_hp[m] / PER])
        y = y_all[m]
        beta, mu, bread = poisson_fit(X, y)
        cov = cov_klaster(X, y, mu, bread, kl[m])
        p, lo, hi = pct(beta[-1], np.sqrt(cov[-1, -1]))
        cepat.append({"hasil": SINGKAT[out], "jeda_hari": L, "persen": p, "bawah": lo, "atas": hi, "n": int(m.sum())})
        if L == 0:
            p2, lo2, hi2 = pct(beta[-2], np.sqrt(cov[-2, -2]))
            lambat.append({"hasil": SINGKAT[out], "persen": p2, "bawah": lo2, "atas": hi2, "n": int(m.sum())})

b1, b2 = BATAS_KAT
kat, kat_n = [], []
for nama_p, x in [("PM2.5 hari itu", A["pm25_mean"].to_numpy(dtype=float)),
                  ("rata-rata 7 hari", g.transform(lambda s: s.rolling(7, min_periods=7).mean()).to_numpy(dtype=float))]:
    m = ~np.isnan(x)
    c1, c2 = ((x > b1) & (x <= b2)).astype(float), (x > b2).astype(float)
    kat_n += [{"paparan": nama_p, "kategori": "<=55,4", "hari_RS": int((m & (c1 == 0) & (c2 == 0)).sum())},
              {"paparan": nama_p, "kategori": "55,5-150,4", "hari_RS": int(c1[m].sum())},
              {"paparan": nama_p, "kategori": ">150,4", "hari_RS": int(c2[m].sum())}]
    pakai = [(n, c) for n, c in [("55,5-150,4", c1), (">150,4", c2)] if int(c[m].sum()) >= 30]
    for out in HASIL:
        y = A[out].to_numpy(dtype=float)[m]
        X = np.column_stack([Xb[m]] + [c[m] for _, c in pakai])
        beta, mu, bread = poisson_fit(X, y)
        cov = cov_klaster(X, y, mu, bread, kl[m])
        for k, (nama_k, _) in enumerate(pakai):
            idx = -len(pakai) + k
            p, lo, hi = pct(beta[idx], np.sqrt(cov[idx, idx]))
            kat.append({"paparan": nama_p, "kategori": nama_k, "hasil": SINGKAT[out], "persen": p, "bawah": lo, "atas": hi})

for df, nama in [(pd.DataFrame(cepat), "efek_komponen_cepat.csv"), (pd.DataFrame(lambat), "efek_komponen_lambat.csv"),
                 (pd.DataFrame(kat), "efek_kategori.csv"), (pd.DataFrame(kat_n), "efek_kategori_n.csv")]:
    df.round(2).to_csv(DASH / nama, index=False)
    print(f"{nama:<28} {len(df):>4} baris")
print("RS dipakai:", len(dipakai))
print("Cuplikan komponen cepat, ISPA_balita:")
print(pd.DataFrame(cepat).query("hasil == 'ISPA_balita'")[["jeda_hari", "persen", "bawah", "atas"]].round(2).to_string(index=False))