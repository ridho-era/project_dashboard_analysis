from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

# =================== KONFIGURASI ===================
MIN_PERSEN_HARI_VALID = 60
HASIL = ["resp_admissions_under5", "resp_admissions_over60", "asthma_emergency_cases",
         "copd_admissions", "resp", "cvd", "death"]
SINGKAT = {"resp_admissions_under5": "ISPA_balita", "resp_admissions_over60": "ISPA_lansia",
           "asthma_emergency_cases": "asma", "copd_admissions": "PPOK", "resp": "resp_umum",
           "cvd": "kardiovask", "death": "kematian"}
LAG_MAKS = 7
PER = 10.0
JENDELA_LAMBAT = 29              # hari; komponen lambat = rata-rata bergulir terpusat 29 hari
KECUALIKAN_RS = ["RS-XXX-23", "RS-XXX-27"]
BATAS_KAT = (55.4, 150.4)        # batas ISPU PM2.5 24 jam: <=55,4 | 55,5-150,4 | >150,4
# ===================================================

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
    bread = np.linalg.inv((X.T * mu) @ X)
    return beta, mu, bread

def cov_klaster(X, y, mu, bread, klaster):
    skor = X * (y - mu)[:, None]
    S = pd.DataFrame(skor).groupby(np.asarray(klaster)).sum().to_numpy()
    meat = S.T @ S
    G = S.shape[0]
    return bread @ meat @ bread * (G / (G - 1))

def pct(b, se):
    return (np.exp(b) - 1) * 100, (np.exp(b - 1.96 * se) - 1) * 100, (np.exp(b + 1.96 * se) - 1) * 100

A = pd.read_parquet(OUT / "analysis_daily.parquet").sort_values(["hospital_id", "date"]).reset_index(drop=True)
persen = A["pm25_mean"].notna().groupby(A["hospital_id"]).mean() * 100
dipakai = [h for h in persen.index if persen[h] >= MIN_PERSEN_HARI_VALID and h not in KECUALIKAN_RS]
A = A[A["hospital_id"].isin(dipakai)].reset_index(drop=True)
print("RS dipakai:", len(dipakai))

g = A.groupby("hospital_id")["pm25_mean"]

# ---------- 1. Mengapa profil jeda datar? Autokorelasi PM2.5 ----------
print("\n1. AUTOKORELASI HARIAN PM2.5 (rata-rata antar-RS) pada jeda 1..7 hari")
ac = pd.DataFrame({L: A.groupby("hospital_id")["pm25_mean"].apply(lambda s, L=L: s.autocorr(L)) for L in range(1, 8)})
print(ac.mean().round(3).to_string())

# ---------- 2. Komponen lambat vs cepat ----------
A["lp"] = g.transform(lambda s: s.rolling(JENDELA_LAMBAT, center=True, min_periods=JENDELA_LAMBAT // 2).mean())
A["hp"] = A["pm25_mean"] - A["lp"]
gh = A.groupby("hospital_id")["hp"]
print(f"\nSD komponen lambat (rata-rata {JENDELA_LAMBAT} hari): {A['lp'].std():.1f} ug/m3 | SD komponen cepat (selisih harian): {A['hp'].std():.1f} ug/m3")
ac2 = pd.DataFrame({L: A.groupby("hospital_id")["hp"].apply(lambda s, L=L: s.autocorr(L)) for L in range(1, 8)})
print("Autokorelasi komponen cepat jeda 1..7:", ac2.mean().round(3).to_dict())

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
klaster_all = A["hospital_id"].to_numpy()

print("\n" + "=" * 80)
print("2. KOMPONEN CEPAT (selisih harian dari rata-rata 29 hari) pada jeda 0-7, dengan komponen lambat sebagai pengendali")
print("Angka = perubahan persen kasus per +10 ug/m3 pada komponen CEPAT (* = selang 95% tidak memuat 0)")
rows, lambat = [], []
for out in HASIL:
    y_all = A[out].to_numpy(dtype=float)
    for L in range(LAG_MAKS + 1):
        x_hp = gh.shift(L).to_numpy(dtype=float)
        x_lp = A["lp"].to_numpy(dtype=float)
        m = ~np.isnan(x_hp) & ~np.isnan(x_lp)
        X = np.column_stack([Xb[m], x_lp[m] / PER, x_hp[m] / PER])
        y = y_all[m]
        beta, mu, bread = poisson_fit(X, y)
        cov = cov_klaster(X, y, mu, bread, klaster_all[m])
        p, lo, hi = pct(beta[-1], np.sqrt(cov[-1, -1]))
        rows.append({"hasil": SINGKAT[out], "jeda": L, "teks": f"{p:+.1f}{'*' if lo > 0 or hi < 0 else ''}"})
        if L == 0:
            p2, lo2, hi2 = pct(beta[-2], np.sqrt(cov[-2, -2]))
            lambat.append({"hasil": SINGKAT[out], "teks": f"{p2:+.1f} [{lo2:+.1f} ; {hi2:+.1f}]"})
T = pd.DataFrame(rows).pivot(index="jeda", columns="hasil", values="teks")[list(SINGKAT.values())]
T.index = [f"jeda {i} hari" for i in T.index]
print(T.to_string())
print("\nKomponen LAMBAT (rata-rata 29 hari terpusat), per +10 ug/m3, pada model jeda 0:")
print(pd.DataFrame(lambat).set_index("hasil").T.to_string())

# ---------- 3. Kategori ISPU (hubungan tidak linear) ----------
print("\n" + "=" * 80)
print("3. KATEGORI PAPARAN vs HARI DENGAN PM2.5 <= 55,4 (Baik+Sedang): perubahan persen kasus [selang 95%]")
b1, b2 = BATAS_KAT
for nama_p, x in [("PM2.5 hari itu", A["pm25_mean"].to_numpy(dtype=float)),
                  ("rata-rata 7 hari (hari ini + 6 hari sebelumnya)",
                   g.transform(lambda s: s.rolling(7, min_periods=7).mean()).to_numpy(dtype=float))]:
    m = ~np.isnan(x)
    c1 = ((x > b1) & (x <= b2)).astype(float)
    c2 = (x > b2).astype(float)
    n_ref = int((m & (c1 == 0) & (c2 == 0)).sum())
    print(f"\nPaparan: {nama_p} | hari-RS per kategori: <=55,4 = {n_ref}, "
          f"55,5-150,4 = {int(c1[m].sum())}, >150,4 = {int(c2[m].sum())}")
    kat = [(n, c) for n, c in [("55,5-150,4", c1), (">150,4", c2)] if int(c[m].sum()) >= 30]
    hasil_kat = {n: {} for n, _ in kat}
    for out in HASIL:
        y = A[out].to_numpy(dtype=float)[m]
        X = np.column_stack([Xb[m]] + [c[m] for _, c in kat])
        beta, mu, bread = poisson_fit(X, y)
        cov = cov_klaster(X, y, mu, bread, klaster_all[m])
        for k, (nama_k, _) in enumerate(kat):
            idx = -len(kat) + k
            p_, lo, hi = pct(beta[idx], np.sqrt(cov[idx, idx]))
            hasil_kat[nama_k][SINGKAT[out]] = f"{p_:+.1f} [{lo:+.1f};{hi:+.1f}]"
    if kat:
        print(pd.DataFrame(hasil_kat).T[list(SINGKAT.values())].to_string())
    for nama_k in ["55,5-150,4", ">150,4"]:
        if nama_k not in hasil_kat:
            print(f"(kategori {nama_k} dilewati: < 30 hari-RS)")