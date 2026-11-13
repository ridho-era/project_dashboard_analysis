from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "raw" / "package"
OUT = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
OUT.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

# =====================================================================
# KONFIGURASI: semua keputusan cleaning ada di sini (ubah di sini saja)
# =====================================================================
GRID_MULAI = "2024-01-01 00:00:00"
GRID_AKHIR = "2026-10-14 23:00:00"
POL = ["pm25", "pm10", "so2", "co", "o3", "no2"]
CU = ["temperature_c", "humidity_pct", "wind_speed_ms", "wind_direction_deg"]
QC_DIPAKAI = {"VALID"}              # flag yang dianggap data andal
QC_KHUSUS = {}                      # sensitivitas SM2-01: {"SM2-01": {"VALID", "SUSPECT"}}
SENTINEL_PM = 999                   # PM2.5/PM10 = 999 dianggap penanda, bukan pengukuran
FLATLINE_MIN_BARIS = 12             # rangkaian PM2.5 identik >= 12 baris = alat macet
MIN_SUBBARIS_15MENIT = 3            # jam dari baris 15 menit wajib punya >= 3 sub-baris andal
KONVERSI_PPM = True                 # False = stasiun ppm dikosongkan (uji sensitivitas)
PM_FAKTOR = 1000.0                  # kolom PM 'ppm' diasumsikan mg/m3 -> ug/m3
MW = {"so2": 64.06, "co": 28.01, "o3": 48.00, "no2": 46.01}
FAKTOR_GAS = {k: v / 24.45 * 1000 for k, v in MW.items()}   # ppm -> ug/m3 pada 25 C, 1 atm
KOSONG = ["", "nan", "NaN", "NA", "N/A", "null", "NULL", "None"]
# =====================================================================

GRID = pd.date_range(GRID_MULAI, GRID_AKHIR, freq="h")

def num(s):
    s = s.str.strip()
    return pd.to_numeric(s.where(~s.isin(KOSONG)), errors="coerce")

def perbaiki_timestamp(d, log):
    """Tukar start/end yang terbalik; rekonstruksi timestamp tahun 1970 dari posisi baris.
    Rekonstruksi HANYA dilakukan bila terbukti: measurement_id = posisi+1 dan baris normal sejajar posisi."""
    ts = pd.to_datetime(d["timestamp_start"], errors="coerce")
    te = pd.to_datetime(d["timestamp_end"], errors="coerce")
    tukar = (te < ts) & (ts.dt.year >= 2000)
    ts2, te2 = ts.copy(), te.copy()
    ts2[tukar], te2[tukar] = te[tukar], ts[tukar]
    epoch = ts2.dt.year < 2000
    log["ts_ditukar"], log["ts_epoch_direkonstruksi"] = int(tukar.sum()), int(epoch.sum())
    if epoch.any() or tukar.any():
        pos = pd.Series(np.arange(len(d)), index=d.index)
        no = d["measurement_id"].str.extract(r"(\d+)$")[0].astype(float)
        assert (no == pos + 1).all(), "measurement_id bukan posisi+1: rekonstruksi tidak sah"
        basis = ts2[~epoch] - pd.to_timedelta(pos[~epoch], unit="h")
        assert basis.nunique() == 1, "baris normal tidak sejajar dengan posisi: rekonstruksi tidak sah"
        harapan = basis.iloc[0] + pd.to_timedelta(pos, unit="h")
        ts2[epoch] = harapan[epoch]
        te2[epoch] = ts2[epoch] + pd.Timedelta(hours=1)
    return ts2, te2

def proses_file(p):
    kode = p.name.split("_")[0]
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    log = {"stasiun": kode, "baris_mentah": len(d)}

    # --- 1. Waktu ---
    ts, te = perbaiki_timestamp(d, log)
    log["urut_setelah_perbaikan"] = bool(ts.is_monotonic_increasing)
    menit = ((te - ts).dt.total_seconds() / 60).round()
    is_sub = menit < 60                                   # baris 15 menit
    ts_jam = ts.dt.floor("h").where(is_sub, ts.dt.round("h"))

    # --- 2. Nilai angka (teks error dan inf menjadi kosong) ---
    satuan = "ppm" if "pm25_ppm" in d.columns else "ug_m3"
    X = pd.DataFrame(index=d.index)
    for pol in POL:
        col = f"{pol}_ug_m3" if f"{pol}_ug_m3" in d.columns else f"{pol}_ppm"
        X[pol] = num(d[col])
    for c in CU:
        X[c] = num(d[c])
    X = X.replace([np.inf, -np.inf], np.nan)

    # --- 3. Aturan flag dan sentinel ---
    qc = d["qc_flag"].str.strip()
    for f, n in qc.value_counts().items():
        log[f"n_{f}"] = int(n)
    qc_ok = qc.isin(QC_KHUSUS.get(kode, QC_DIPAKAI))
    sentinel = ((X["pm25"] == SENTINEL_PM) | (X["pm10"] == SENTINEL_PM))
    valid = qc_ok & ~sentinel
    neg = (X[POL] < 0).any(axis=1) & valid               # seharusnya 0
    valid = valid & ~neg
    log["baris_dibuang_flag"] = int((~qc_ok).sum())
    log["baris_sentinel_999"] = int(sentinel.sum())
    log["baris_negatif_lolos_flag"] = int(neg.sum())

    # --- 4. Detektor flatline (hanya pada baris andal) ---
    s = X["pm25"].where(valid)
    run = ((s != s.shift()) | s.isna()).cumsum()
    ukuran = s.groupby(run).transform("size")
    flat = s.notna() & (ukuran >= FLATLINE_MIN_BARIS)
    log["flatline_baris"] = int(flat.sum())
    if flat.any():
        nun = X.loc[flat, POL].groupby(run[flat]).nunique(dropna=False)
        log["flatline_run"] = int(len(nun))
        log["flatline_run_semua_polutan_konstan"] = int((nun.max(axis=1) == 1).sum())
    valid_pol = valid & ~flat                             # polutan: buang flatline
    P = X[POL].copy()
    P.loc[~valid_pol] = np.nan
    W = X[CU].copy()
    W.loc[~valid] = np.nan                                # cuaca: flatline tetap dipakai

    # --- 5. Satuan ppm -> ug/m3 ---
    dikonversi = satuan == "ppm"
    if dikonversi:
        if KONVERSI_PPM:
            P["pm25"] *= PM_FAKTOR
            P["pm10"] *= PM_FAKTOR
            for gas, f in FAKTOR_GAS.items():
                P[gas] *= f
        else:
            P[:] = np.nan

    # --- 6. Agregasi ke jam (baris 15 menit -> rata-rata jam) ---
    D = pd.concat([P, W], axis=1)
    D["ts_jam"] = ts_jam
    D["is_sub"] = is_sub
    D["tersaturasi"] = sentinel
    D["flat"] = flat
    rad = np.deg2rad(D["wind_direction_deg"])
    D["_sin"], D["_cos"] = np.sin(rad), np.cos(rad)
    D = D[D["ts_jam"].notna()]
    log["jam_di_luar_grid"] = int((~pd.DatetimeIndex(D["ts_jam"].unique()).isin(GRID)).sum())

    g = D.groupby("ts_jam")
    n_rows, n_sub = g.size(), g["is_sub"].sum()
    req = pd.Series(np.where(n_sub > 0, MIN_SUBBARIS_15MENIT, 1), index=n_rows.index)
    kol_rata = POL + ["temperature_c", "humidity_pct", "wind_speed_ms"]
    H = g[kol_rata].mean()
    cnt = g[kol_rata].count()
    ok = cnt.ge(req, axis=0)
    log["jam_ditolak_aturan_subbaris"] = int(((cnt["pm25"] > 0) & ~ok["pm25"]).sum())
    H = H.where(ok)
    wd = np.degrees(np.arctan2(g["_sin"].mean(), g["_cos"].mean())) % 360
    H["wind_direction_deg"] = wd.where(g["wind_direction_deg"].count() >= req)
    H["tersaturasi_999"] = g["tersaturasi"].max()
    H["flatline"] = g["flat"].max()
    log["jam_dari_baris_15menit"] = int((n_sub > 0).sum())
    log["jam_ganda_non15menit"] = int(((n_rows > 1) & (n_sub == 0)).sum())

    # --- 7. Grid jam lengkap: jam hilang menjadi kosong yang eksplisit ---
    H = H.reindex(GRID)
    H["tersaturasi_999"] = H["tersaturasi_999"].fillna(False).astype(bool)
    H["flatline"] = H["flatline"].fillna(False).astype(bool)
    H.index.name = "ts_jam"
    H.insert(0, "stasiun", kode)
    H.insert(1, "sensor_id", d["sensor_id"].iloc[0])
    H["dikonversi"] = dikonversi
    H = H.rename(columns={p: f"{p}_ug_m3" for p in POL})
    log["jam_grid"] = len(GRID)
    log["jam_pm25_tersedia"] = int(H["pm25_ug_m3"].notna().sum())
    log["cakupan_pm25_persen"] = round(H["pm25_ug_m3"].notna().mean() * 100, 1)
    return H.reset_index(), log

hasil, logs = [], []
for p in sorted(PKG.glob("*_AQMS.csv")):
    df, lg = proses_file(p)
    hasil.append(df)
    logs.append(lg)
    print(f"{lg['stasiun']:<8} cakupan PM2.5 = {lg['cakupan_pm25_persen']:>5}% "
          f"| flatline={lg['flatline_baris']} | ditukar={lg['ts_ditukar']} | epoch={lg['ts_epoch_direkonstruksi']}")

semua = pd.concat(hasil, ignore_index=True)
semua.to_parquet(OUT / "aqms_hourly_clean.parquet", index=False)
LOG = pd.DataFrame(logs).fillna(0)
for c in LOG.columns:
    if c.startswith(("n_", "baris", "jam", "flatline", "ts_")):
        LOG[c] = LOG[c].astype(int)
LOG.to_csv(DOCS / "cleaning_log_aqms_auto.csv", index=False)
print("\nTersimpan:", OUT / "aqms_hourly_clean.parquet", "| baris:", len(semua))
print("Log otomatis:", DOCS / "cleaning_log_aqms_auto.csv")
print("\nRINGKASAN LOG")
print(LOG.T.to_string(header=False))