from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "raw" / "package"
OUT = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

# =================== KONFIGURASI ===================
TANGGAL_MULAI = "2024-01-01"     # awal jendela data udara; baris kesehatan sebelum ini dikeluarkan
TANGGAL_AKHIR = "2026-10-14"
CNT = ["resp_admissions_under5", "resp_admissions_over60", "asthma_emergency_cases",
       "copd_admissions", "resp", "cvd", "death"]
ENV = ["temp_c", "dptp_c", "rhum_pct"]
PULAU = {
    "DKI Jakarta": "Jawa", "Jawa Barat": "Jawa", "D.I. Yogyakarta": "Jawa", "Jawa Timur": "Jawa",
    "Aceh": "Sumatera", "Sumatera Utara": "Sumatera", "Sumatera Barat": "Sumatera", "Riau": "Sumatera",
    "Kepulauan Riau": "Sumatera", "Jambi": "Sumatera", "Sumatera Selatan": "Sumatera",
    "Bengkulu": "Sumatera", "Lampung": "Sumatera",
    "Kalimantan Barat": "Kalimantan", "Kalimantan Tengah": "Kalimantan",
    "Kalimantan Timur": "Kalimantan", "Kalimantan Selatan": "Kalimantan",
    "Sulawesi Selatan": "Sulawesi", "Sulawesi Tengah": "Sulawesi",
    "Papua Barat Daya": "Papua", "Papua Selatan": "Papua",
}
# ===================================================

def cari(nama):
    hasil = list(PKG.rglob(nama))
    assert len(hasil) == 1, f"{nama}: ditemukan {len(hasil)} file"
    return hasil[0]

def pilih(df, kolom, nama):
    ada = [c for c in kolom if c in df.columns]
    hilang = [c for c in kolom if c not in df.columns]
    if hilang:
        print(f"PERHATIAN: kolom tidak ditemukan di {nama}: {hilang}")
    return df[ada].copy()

# ---------- A. Tabel dimensi stasiun (27 baris) ----------
sites = pd.read_csv(cari("sites.csv"))
sensors = pd.read_csv(cari("sensors.csv"))
hosp = pd.read_csv(cari("hospitals.csv"))

sites_s = pilih(sites, ["location_id", "station_code", "provinsi", "kota_kabupaten", "kecamatan",
                        "karakteristik_wilayah", "latitude_dd", "longitude_dd", "baseline_pm25",
                        "region_group", "site_type", "distance_to_road_m", "traffic_intensity_per_h"], "sites.csv")
sens_s = pilih(sensors, ["sensor_id", "location_id", "model_sensor", "tipe_pemasangan",
                         "status_operasional", "tanggal_instalasi"], "sensors.csv")
hosp_s = pilih(hosp, ["hospital_id", "nama_rumah_sakit", "kelas_faskes", "kapasitas_tempat_tidur"], "hospitals.csv")

sens_s["no"] = sens_s["sensor_id"].str.extract(r"(\d+)$")[0].astype(int)
hosp_s["no"] = hosp_s["hospital_id"].str.extract(r"(\d+)$")[0].astype(int)
dim = (sens_s.merge(sites_s, on="location_id", how="left", validate="one_to_one")
             .merge(hosp_s, on="no", how="outer", validate="one_to_one"))
dim["stasiun"] = dim["station_code"].str.replace("AQMS-", "", regex=False)
tak_terpetakan = sorted(set(dim["provinsi"]) - set(PULAU))
assert not tak_terpetakan, f"provinsi belum dipetakan ke pulau: {tak_terpetakan}"
dim["pulau"] = dim["provinsi"].map(PULAU)
dim = dim.drop(columns=["no"]).sort_values("stasiun").reset_index(drop=True)
dim.to_parquet(OUT / "dim_stasiun.parquet", index=False)

print("A. TABEL DIMENSI STASIUN")
print("baris:", len(dim), "| stasiun unik:", dim["stasiun"].nunique(), "| rumah sakit unik:", dim["hospital_id"].nunique())
print("sel kosong pada kolom kunci:", int(dim[["stasiun", "sensor_id", "hospital_id", "provinsi", "pulau"]].isna().sum().sum()))
print("stasiun per pulau:", dim["pulau"].value_counts().to_dict())
print(dim[["stasiun", "hospital_id", "provinsi", "pulau", "status_operasional", "kelas_faskes"]].to_string(index=False))

# ---------- B. Cleaning data kesehatan ----------
raw = pd.read_csv(cari("HLT-01_health_outcomes_daily.csv"), dtype=str, keep_default_na=False)
H = pd.DataFrame({"date": pd.to_datetime(raw["date"], errors="coerce"),
                  "hospital_id": raw["hospital_id"].str.strip(),
                  "dow": raw["dow"].str.strip()})
for c in ENV + CNT + ["time"]:
    H[c] = pd.to_numeric(raw[c].str.strip(), errors="coerce")
assert H[ENV + CNT + ["time", "date"]].notna().all().all(), "ada nilai kosong/gagal baca: periksa audit 3F"
assert ((H[CNT] % 1) == 0).all().all() and (H[CNT] >= 0).all().all(), "kasus bukan bilangan bulat/negatif"
for c in CNT + ["time"]:
    H[c] = H[c].astype("int64")
H = H.rename(columns={"time": "hari_ke_rs"})

n_mentah = H.groupby("hospital_id").size()
keluar = H["date"] < pd.Timestamp(TANGGAL_MULAI)
n_keluar = H[keluar].groupby("hospital_id").size().reindex(n_mentah.index).fillna(0).astype(int)
H = H[~keluar].copy()
H["resp_inkonsisten"] = H["resp"] < (H["resp_admissions_under5"] + H["resp_admissions_over60"])
H = H.merge(dim[["hospital_id", "stasiun"]], on="hospital_id", how="left", validate="many_to_one")
H = H.sort_values(["hospital_id", "date"]).reset_index(drop=True)
H.to_parquet(OUT / "health_daily_clean.parquet", index=False)

LOG = pd.DataFrame({"baris_mentah": n_mentah, "baris_dibuang_sebelum_2024": n_keluar,
                    "hari_tersisa": H.groupby("hospital_id").size(),
                    "resp_inkonsisten": H.groupby("hospital_id")["resp_inkonsisten"].sum().astype(int)})
LOG.to_csv(DOCS / "cleaning_log_kesehatan_auto.csv")

hari_harap = (pd.Timestamp(TANGGAL_AKHIR) - pd.Timestamp(TANGGAL_MULAI)).days + 1
print("\n" + "=" * 80)
print("B. DATA KESEHATAN SETELAH CLEANING")
print(f"baris mentah: {int(n_mentah.sum())} | dibuang: {int(n_keluar.sum())} | tersisa: {len(H)}")
print("duplikat (hospital_id, date):", int(H.duplicated(['hospital_id', 'date']).sum()))
print(f"hospital dengan hari != {hari_harap}:", LOG.index[LOG['hari_tersisa'] != hari_harap].tolist())
print("rentang tanggal:", H["date"].min().date(), "->", H["date"].max().date())
print("baris tanpa pasangan stasiun:", int(H["stasiun"].isna().sum()))
print("resp_inkonsisten total:", int(H["resp_inkonsisten"].sum()))
print("\nresp_inkonsisten per rumah sakit (hanya > 0):")
ri = LOG["resp_inkonsisten"]
print(ri[ri > 0].sort_values(ascending=False).to_string())

# ---------- C. Uji ulang pemetaan: suhu harian rumah sakit vs suhu stasiun (data udara BERSIH) ----------
print("\n" + "=" * 80)
print("C. KORELASI SUHU HARIAN RS vs STASIUN (data udara bersih; hari dengan >= 4 jam valid)")
aq_path = OUT / "aqms_hourly_clean.parquet"
if aq_path.exists():
    A = pd.read_parquet(aq_path, columns=["stasiun", "ts_jam", "temperature_c"]).dropna(subset=["temperature_c"])
    A["tgl"] = A["ts_jam"].dt.normalize()
    daily = A.groupby(["stasiun", "tgl"])["temperature_c"].agg(["mean", "size"])
    daily = daily[daily["size"] >= 4]["mean"]
    rows = []
    for _, r in dim.iterrows():
        h = H[H["hospital_id"] == r["hospital_id"]].set_index("date")["temp_c"]
        if r["stasiun"] not in daily.index.get_level_values(0):
            rows.append({"hospital_id": r["hospital_id"], "stasiun": r["stasiun"], "n_hari": 0})
            continue
        s = daily.loc[r["stasiun"]]
        row = {"hospital_id": r["hospital_id"], "stasiun": r["stasiun"]}
        for k in (-1, 0, 1):
            x = s.copy()
            x.index = x.index + pd.Timedelta(days=k)
            j = pd.concat([h, x], axis=1, join="inner").dropna()
            row[f"r_geser_{k:+d}"] = round(j.iloc[:, 0].corr(j.iloc[:, 1]), 3) if len(j) >= 30 else np.nan
            row["n_hari"] = len(j)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
else:
    print("aqms_hourly_clean.parquet tidak ditemukan; jalankan 04a_clean_aqms.py lebih dulu.")

print("\nTersimpan:", OUT / "dim_stasiun.parquet", "|", OUT / "health_daily_clean.parquet")
print("Log:", DOCS / "cleaning_log_kesehatan_auto.csv")