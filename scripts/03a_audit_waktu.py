from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "raw" / "package"

sensors = pd.read_csv(list(PKG.rglob("sensors.csv"))[0])
status_by_sensor = sensors.set_index("sensor_id")["status_operasional"]

# Grid jam yang SEHARUSNYA ada (acuan: metadata_dates)
GRID = pd.date_range("2024-01-01 00:00:00", "2026-10-14 23:00:00",
                     freq="h").astype("datetime64[ns]")
BATAS_AWAL = pd.Timestamp("2023-12-31")
BATAS_AKHIR = pd.Timestamp("2026-10-15")
print("Jumlah jam seharusnya pada grid:", len(GRID))

rekap, anomali = [], []

for p in sorted(PKG.glob("*_AQMS.csv")):
    kode = p.name.split("_")[0]
    d = pd.read_csv(p, usecols=["sensor_id", "timestamp_start", "timestamp_end"])
    sid = d["sensor_id"].dropna().iloc[0]

    ts = pd.to_datetime(d["timestamp_start"], errors="coerce").astype("datetime64[ns]")
    te = pd.to_datetime(d["timestamp_end"], errors="coerce").astype("datetime64[ns]")

    luar = (ts < BATAS_AWAL) | (ts > BATAS_AKHIR)       # di luar rentang wajar
    tr = ts.dt.round("h")                                # dibulatkan ke jam terdekat
    off = (ts - tr).dt.total_seconds()                   # jitter (detik)
    dur = (te - ts).dt.total_seconds() / 3600            # durasi jendela (jam)

    ok = ~luar & ts.notna()
    tr_ok = tr[ok]
    uniq = pd.DatetimeIndex(tr_ok.unique())

    # selisih antar jam berurutan (setelah dibulatkan)
    dif = (tr_ok.sort_values().diff().dropna().dt.total_seconds() / 3600).round(2)
    dif_txt = "; ".join(f"{k:g}j x{v}" for k, v in dif.value_counts().head(4).items())
    dur_txt = "; ".join(f"{k:g}j x{v}" for k, v in
                        dur.round(2).value_counts(dropna=False).head(3).items())

    # jam yang hilang dan celah (run) berurutan
    hilang = GRID.difference(uniq)
    n_celah, celah_max = 0, 0
    if len(hilang):
        s = pd.Series(hilang)
        run = (s.diff() != pd.Timedelta(hours=1)).cumsum()
        runs = (s.groupby(run).agg(mulai="min", selesai="max", n="size")
                  .sort_values("n", ascending=False))
        n_celah, celah_max = len(runs), int(runs["n"].iloc[0])
        anomali.append((kode, "CELAH JAM HILANG (5 terpanjang)", runs.head(5)))

    if luar.any():
        anomali.append((kode, f"TIMESTAMP DI LUAR RENTANG (5 contoh dari {int(luar.sum())})",
                        d.loc[luar, ["timestamp_start", "timestamp_end"]].head(5)))

    dup_mask = tr_ok.duplicated(keep=False)
    if dup_mask.any():
        idx = tr_ok[dup_mask].sort_values().index[:6]
        anomali.append((kode, f"JAM YANG SAMA MUNCUL >1 KALI "
                              f"(6 contoh; baris terlibat: {int(dup_mask.sum())})",
                        d.loc[idx, ["timestamp_start", "timestamp_end"]]))

    o = off[ok].round(3)
    rekap.append({
        "file": kode, "status": status_by_sensor.get(sid), "baris": len(d),
        "ts_gagal": int(ts.isna().sum()), "te_gagal": int(te.isna().sum()),
        "di_luar_rentang": int(luar.sum()),
        "urut": bool(ts.is_monotonic_increasing),
        "dup_ts_persis": int(ts.duplicated().sum()),
        "baris_lebih_per_jam": int(tr_ok.duplicated().sum()),
        "jam_unik_ada": len(uniq), "jam_hilang": len(hilang),
        "n_celah": n_celah, "celah_maks_jam": celah_max,
        "offset_min_s": o.min(), "offset_med_s": o.median(),
        "offset_maks_s": o.max(), "n_offset_unik": o.nunique(),
        "selisih_jam_terbanyak": dif_txt, "durasi_terbanyak": dur_txt,
    })

R = pd.DataFrame(rekap)

print("\nA. STRUKTUR DAN PARSING")
print(R[["file", "status", "baris", "ts_gagal", "te_gagal", "di_luar_rentang",
         "urut", "dup_ts_persis", "baris_lebih_per_jam"]].to_string(index=False))

print("\nB. KELENGKAPAN GRID JAM DAN JITTER")
print(R[["file", "status", "jam_unik_ada", "jam_hilang", "n_celah", "celah_maks_jam",
         "offset_min_s", "offset_med_s", "offset_maks_s", "n_offset_unik"]].to_string(index=False))

print("\nC. SELISIH ANTAR JAM DAN DURASI JENDELA")
print(R[["file", "selisih_jam_terbanyak", "durasi_terbanyak"]].to_string(index=False))

for kode, judul, tabel in anomali:
    print(f"\n[{kode}] {judul}")
    print(tabel.to_string())