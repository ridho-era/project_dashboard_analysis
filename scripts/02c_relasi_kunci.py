from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "raw" / "package"

def cari(nama):
    hasil = list(PKG.rglob(nama))
    assert len(hasil) == 1, f"{nama}: ditemukan {len(hasil)} file"
    return hasil[0]

sites   = pd.read_csv(cari("sites.csv"))
sensors = pd.read_csv(cari("sensors.csv"))
hosp    = pd.read_csv(cari("hospitals.csv"))
mdates  = pd.read_csv(cari("metadata_dates.csv"))
hlt     = pd.read_csv(cari("HLT-01_health_outcomes_daily.csv"))

# ---------- 1. Keunikan kandidat primary key ----------
print("=" * 80); print("1. KEUNIKAN KANDIDAT PRIMARY KEY")
kandidat = {
    "sites.location_id": sites["location_id"],
    "sites.station_code": sites["station_code"],
    "sensors.sensor_id": sensors["sensor_id"],
    "sensors.location_id": sensors["location_id"],
    "hospitals.hospital_id": hosp["hospital_id"],
    "metadata_dates.location_id": mdates["location_id"],
}
for nama, s in kandidat.items():
    print(f"{nama:<30} baris={len(s):>3} unik={s.nunique():>3} "
          f"kosong={int(s.isna().sum())} duplikat={int(s.duplicated().sum())}")

# ---------- 2. Integritas foreign key ----------
print("\n" + "=" * 80); print("2. INTEGRITAS FOREIGN KEY (selisih himpunan)")
def bandingkan(label, a, b, na, nb):
    a, b = set(a.dropna()), set(b.dropna())
    print(label)
    print(f"   hanya di {na}: {sorted(a - b)}")
    print(f"   hanya di {nb}: {sorted(b - a)}")

bandingkan("sensors.location_id vs sites.location_id",
           sensors["location_id"], sites["location_id"], "sensors", "sites")
bandingkan("metadata_dates.location_id vs sites.location_id",
           mdates["location_id"], sites["location_id"], "metadata_dates", "sites")
bandingkan("HLT.hospital_id vs hospitals.hospital_id",
           hlt["hospital_id"], hosp["hospital_id"], "HLT", "hospitals")

# ---------- 3. Tabel jembatan stasiun <-> rumah sakit ----------
print("\n" + "=" * 80); print("3. PEMETAAN STASIUN <-> RUMAH SAKIT (berdasarkan nomor urut)")
sensors["no"] = sensors["sensor_id"].str.extract(r"(\d+)$")[0].astype(int)
hosp["no"]    = hosp["hospital_id"].str.extract(r"(\d+)$")[0].astype(int)

st = sensors.merge(sites, on="location_id", how="left", validate="one_to_one")
st = st[["no", "sensor_id", "location_id", "kecamatan", "kota_kabupaten", "provinsi"]]
peta = st.merge(hosp[["no", "hospital_id", "kecamatan", "kota_kabupaten", "provinsi"]],
                on="no", how="outer", suffixes=("_stasiun", "_rs"), validate="one_to_one")
peta["provinsi_sama"] = (peta["provinsi_stasiun"].str.strip().str.lower()
                         == peta["provinsi_rs"].str.strip().str.lower())
print(peta.sort_values("no").to_string(index=False))
print("\nProvinsi tidak sama:", int((~peta["provinsi_sama"]).sum()), "dari", len(peta))

# ---------- 4. File AQMS: kesesuaian sensor, stasiun, measurement_id ----------
print("\n" + "=" * 80); print("4. KESESUAIAN FILE AQMS")
loc_by_sensor  = sensors.set_index("sensor_id")["location_id"]
code_by_loc    = sites.set_index("location_id")["station_code"]

rekap, semua_id = [], []
for p in sorted(PKG.glob("*_AQMS.csv")):
    d = pd.read_csv(p, usecols=["measurement_id", "sensor_id", "timestamp_start"])
    kode = p.name.split("_")[0]                     # contoh: JKT-01
    sid_unik = d["sensor_id"].dropna().unique().tolist()
    sid = sid_unik[0] if len(sid_unik) == 1 else None
    loc = loc_by_sensor.get(sid) if sid else None
    station = code_by_loc.get(loc) if loc else None
    kode_id = (d["measurement_id"].str.replace(r"^MEAS-", "", regex=True)
                                  .str.replace(r"-\d+$", "", regex=True)
                                  .dropna().unique().tolist())
    rekap.append({
        "file": kode,
        "baris": len(d),
        "id_unik": d["measurement_id"].nunique(),
        "id_duplikat": int(d["measurement_id"].duplicated().sum()),
        "id_kosong": int(d["measurement_id"].isna().sum()),
        "sensor_id": sid,
        "sensor_ada": sid in loc_by_sensor.index if sid else False,
        "kode_stasiun_cocok": station == f"AQMS-{kode}",
        "kode_di_measurement_id": ",".join(kode_id[:3]),
    })
    semua_id.append(d["measurement_id"])

print(pd.DataFrame(rekap).to_string(index=False))
ids = pd.concat(semua_id)
print("\nmeasurement_id duplikat LINTAS file:", int(ids.dropna().duplicated().sum()))

# ---------- 5. Kelengkapan tanggal data kesehatan ----------
print("\n" + "=" * 80); print("5. DATA KESEHATAN: KELENGKAPAN TANGGAL")
hlt["date_dt"] = pd.to_datetime(hlt["date"], errors="coerce")
print("Duplikat (hospital_id, date):", int(hlt.duplicated(["hospital_id", "date"]).sum()))
per_rs = hlt.groupby("hospital_id").agg(
    n_baris=("date", "size"), n_tgl_unik=("date", "nunique"),
    tgl_awal=("date_dt", "min"), tgl_akhir=("date_dt", "max"))
per_rs["hari_seharusnya"] = (per_rs["tgl_akhir"] - per_rs["tgl_awal"]).dt.days + 1
per_rs["hari_hilang"] = per_rs["hari_seharusnya"] - per_rs["n_tgl_unik"]
print(per_rs.to_string())

# ---------- 6. Kategori yang perlu dilihat (hanya melihat) ----------
print("\n" + "=" * 80); print("6. DAFTAR NILAI KATEGORI")
for judul, s in [("sites.site_type", sites["site_type"]),
                 ("hospitals.kelas_faskes", hosp["kelas_faskes"]),
                 ("sensors.status_operasional", sensors["status_operasional"]),
                 ("metadata_dates.notes", mdates["notes"])]:
    print(f"\n{judul}")
    print(s.value_counts(dropna=False).to_string())

print("\nhospitals.kapasitas kosong:", int(hosp["kapasitas_tempat_tidur"].isna().sum()))
inst = pd.to_datetime(sensors["tanggal_instalasi"], errors="coerce")
print("tanggal_instalasi gagal parse:", int(inst.isna().sum()),
      "| terawal:", inst.min(), "| terakhir:", inst.max(),
      "| setelah 2024-01-01:", int((inst > "2024-01-01").sum()))