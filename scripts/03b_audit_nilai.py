from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "raw" / "package"

sensors = pd.read_csv(list(PKG.rglob("sensors.csv"))[0])
status_by_sensor = sensors.set_index("sensor_id")["status_operasional"]

POLUTAN = ["pm25", "pm10", "so2", "co", "o3", "no2"]
CUACA = ["temperature_c", "humidity_pct", "wind_speed_ms", "wind_direction_deg"]
SENTINEL = [-999, -9999, 9999, 99999, 999]          # nilai 'pengganti kosong' yang lazim di logger
KOSONG = ["", "nan", "NaN", "NA", "N/A", "null", "NULL", "None"]

def jadi_angka(seri_teks):
    """Teks -> angka. Mengembalikan (angka, penanda_kosong, penanda_non_numerik)."""
    s = seri_teks.str.strip()
    kosong = s.isin(KOSONG)
    angka = pd.to_numeric(s.where(~kosong), errors="coerce")
    non_num = angka.isna() & ~kosong
    return angka, kosong, non_num

temuan = []
def catat(kode, kolom, jenis, jumlah):
    if jumlah > 0:
        temuan.append({"file": kode, "kolom": kolom, "jenis": jenis, "jumlah": int(jumlah)})

rows_kosong, rows_qc, rows_token, rows_med, rows_max = [], [], [], [], []

for p in sorted(PKG.glob("*_AQMS.csv")):
    kode = p.name.split("_")[0]
    d = pd.read_csv(p, dtype=str, keep_default_na=False)     # semua dibaca apa adanya sebagai teks
    status = status_by_sensor.get(d["sensor_id"].iloc[0])

    peta = {}
    for pol in POLUTAN:
        if f"{pol}_ug_m3" in d.columns:
            peta[pol] = f"{pol}_ug_m3"
        elif f"{pol}_ppm" in d.columns:
            peta[pol] = f"{pol}_ppm"
        else:
            peta[pol] = None
    for c in CUACA:
        peta[c] = c if c in d.columns else None
    satuan = "ppm" if any(v and v.endswith("_ppm") for v in peta.values()) else "ug_m3"

    baris = {"file": kode, "status": status, "satuan": satuan, "n": len(d)}
    angka_all, nonnum_total = {}, 0

    for var, col in peta.items():
        if col is None:
            baris[var] = np.nan
            continue
        angka, kosong, non_num = jadi_angka(d[col])
        angka_all[var] = angka
        baris[var] = round(kosong.mean() * 100, 2)
        nonnum_total += int(non_num.sum())
        if non_num.any():
            for tok, n in d.loc[non_num, col].str.strip().value_counts().head(8).items():
                rows_token.append({"file": kode, "kolom": col, "isi_teks": tok, "jumlah": int(n)})

        if var in POLUTAN:
            catat(kode, col, "negatif", (angka < 0).sum())
            catat(kode, col, "tepat_nol", (angka == 0).sum())
        catat(kode, col, "nilai_sentinel", angka.isin(SENTINEL).sum())
        if var == "humidity_pct":
            catat(kode, col, "di_luar_0_100", ((angka < 0) | (angka > 100)).sum())
        if var == "wind_direction_deg":
            catat(kode, col, "di_luar_0_360", ((angka < 0) | (angka > 360)).sum())
        if var == "wind_speed_ms":
            catat(kode, col, "negatif", (angka < 0).sum())
        if var == "temperature_c":
            catat(kode, col, "di_luar_5_45C", ((angka < 5) | (angka > 45)).sum())

    baris["non_numerik_total"] = nonnum_total
    rows_kosong.append(baris)

    qc = d["qc_flag"].str.strip().replace("", "(kosong)")
    rows_qc.append(qc.value_counts().rename(kode))

    med, mx = {"file": kode, "satuan": satuan}, {"file": kode, "satuan": satuan}
    for pol in POLUTAN:
        a = angka_all.get(pol)
        med[pol] = round(a.median(), 3) if a is not None else np.nan
        mx[pol] = round(a.max(), 3) if a is not None else np.nan
    rows_med.append(med)
    rows_max.append(mx)

print("A. PERSEN NILAI KOSONG PER VARIABEL (kolom polutan = ug_m3 atau ppm, sesuai file)")
print(pd.DataFrame(rows_kosong).to_string(index=False))

print("\nA2. ISI TEKS NON-NUMERIK (maks 8 teratas per kolom)")
print(pd.DataFrame(rows_token).to_string(index=False) if rows_token else "Tidak ada.")

print("\nB. qc_flag PER STASIUN (jumlah baris)")
print(pd.DataFrame(rows_qc).fillna(0).astype(int).to_string())

print("\nC. TEMUAN RENTANG (hanya yang jumlahnya > 0)")
print(pd.DataFrame(temuan).to_string(index=False) if temuan else "Tidak ada.")

print("\nD1. MEDIAN POLUTAN PER STASIUN")
print(pd.DataFrame(rows_med).to_string(index=False))
print("\nD2. MAKSIMUM POLUTAN PER STASIUN")
print(pd.DataFrame(rows_max).to_string(index=False))

# ---------- E. JMB-02: apakah urutan baris = urutan jam? ----------
print("\n" + "=" * 80)
print("E. JMB-02: URUTAN BARIS vs JAM")
fj = list(PKG.glob("JMB-02_*_AQMS.csv"))
assert len(fj) == 1
j = pd.read_csv(fj[0], usecols=["measurement_id", "timestamp_start", "timestamp_end"])
j["no_id"] = j["measurement_id"].str.extract(r"(\d+)$")[0].astype(int)
pos = pd.Series(range(len(j)), index=j.index)
print("measurement_id = posisi baris + 1 untuk SEMUA baris?", bool((j["no_id"] == pos + 1).all()))

ts = pd.to_datetime(j["timestamp_start"])
te = pd.to_datetime(j["timestamp_end"])
off = pd.Timedelta(seconds=312.1)
harapan = pd.Series(pd.date_range("2024-01-01", periods=len(j), freq="h") + off, index=j.index)
epoch = ts.dt.year == 1970
tukar = te < ts
normal = ~epoch & ~tukar
print(f"baris normal      : {int(normal.sum())} | start == jam harapan posisinya: {int((ts[normal] == harapan[normal]).sum())}")
print(f"baris start>end   : {int(tukar.sum())} | end   == jam harapan posisinya: {int((te[tukar] == harapan[tukar]).sum())}")
print(f"baris tahun 1970  : {int(epoch.sum())} | posisi baris 5 pertama: {pos[epoch].head(5).tolist()}")

# ---------- F. PLB-01: sebaran resolusi ----------
print("\n" + "=" * 80)
print("F. PLB-01: RESOLUSI JENDELA PENGUKURAN")
fb = list(PKG.glob("PLB-01_*_AQMS.csv"))
assert len(fb) == 1
b = pd.read_csv(fb[0], usecols=["timestamp_start", "timestamp_end"])
tsb = pd.to_datetime(b["timestamp_start"])
teb = pd.to_datetime(b["timestamp_end"])
menit = ((teb - tsb).dt.total_seconds() / 60).round(2)
seg = (menit != menit.shift()).cumsum()
runs = b.assign(menit=menit).groupby(seg).agg(
    mulai=("timestamp_start", "first"), selesai=("timestamp_start", "last"),
    menit=("menit", "first"), n_baris=("menit", "size"))
print("jumlah segmen resolusi:", len(runs))
print(runs.head(20).to_string())
print("\nBanyak baris per jam-kalender (kolom kiri = baris per jam, kanan = berapa jam):")
print(tsb.dt.floor("h").value_counts().value_counts().sort_index().to_string())