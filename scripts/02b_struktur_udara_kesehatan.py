from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "raw" / "package"

# ---------- BAGIAN 1: 27 file AQMS ----------
file_udara = sorted(PKG.glob("*_AQMS.csv"))
print("Jumlah file AQMS:", len(file_udara))

ringkas, kolom_per_file, dtype_per_file = [], {}, {}

for p in file_udara:
    df = pd.read_csv(p)
    ts = pd.to_datetime(df["timestamp_start"], errors="coerce")
    kolom_per_file[p.name] = list(df.columns)
    dtype_per_file[p.name] = df.dtypes.astype(str).to_dict()
    ringkas.append({
        "file": p.name.replace("_AQMS.csv", ""),
        "baris": len(df),
        "n_sensor": df["sensor_id"].nunique(),
        "sensor_id": ", ".join(map(str, df["sensor_id"].dropna().unique()[:3])),
        "ts_min": ts.min(),
        "ts_max": ts.max(),
        "ts_gagal_parse": int(ts.isna().sum()),
    })

print("\n--- Ringkasan per stasiun ---")
print(pd.DataFrame(ringkas).to_string(index=False))

# Apakah semua file punya kolom yang sama?
acuan = kolom_per_file[file_udara[0].name]
print("\n--- Kolom file acuan:", file_udara[0].name, "---")
print(acuan)

beda = {n: k for n, k in kolom_per_file.items() if k != acuan}
print("\nFile dengan kolom berbeda dari acuan:", len(beda))
for n, k in beda.items():
    print(" ", n, "| tambahan:", set(k) - set(acuan), "| hilang:", set(acuan) - set(k))

# Apakah tipe data tiap kolom konsisten antarfile?
print("\n--- Kolom dengan dtype tidak konsisten antarfile ---")
ada = False
for kol in acuan:
    tipe = {n: d.get(kol) for n, d in dtype_per_file.items()}
    if len(set(tipe.values())) > 1:
        ada = True
        print(f"\nKolom '{kol}':")
        for t in sorted(set(tipe.values()), key=str):
            files = [n.replace('_AQMS.csv', '') for n, v in tipe.items() if v == t]
            print(f"  {t}: {files}")
if not ada:
    print("Tidak ada.")

# ---------- BAGIAN 2: File kesehatan ----------
print("\n" + "=" * 90)
h = pd.read_csv(PKG / "HLT-01_health_outcomes_daily.csv")
print("HLT-01  ->", h.shape[0], "baris,", h.shape[1], "kolom")
print(pd.DataFrame({"dtype": h.dtypes.astype(str), "n_unik": h.nunique()}).to_string())

d = pd.to_datetime(h["date"], errors="coerce")
print("\nRentang tanggal:", d.min(), "->", d.max(), "| gagal parse:", int(d.isna().sum()))
print("Jumlah hospital_id unik:", h["hospital_id"].nunique())
print("\nBaris per rumah sakit:")
print(h["hospital_id"].value_counts().sort_index().to_string())
print("\nContoh 3 baris:")
print(h.head(3).T.to_string())