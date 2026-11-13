from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "raw" / "package"

KOSONG = ["", "nan", "NaN", "NA", "N/A", "null", "NULL", "None"]
def num(s):
    s = s.str.strip()
    return pd.to_numeric(s.where(~s.isin(KOSONG)), errors="coerce")

raw = pd.read_csv(PKG / "HLT-01_health_outcomes_daily.csv", dtype=str, keep_default_na=False)
print("Jumlah baris:", len(raw), "| kolom:", list(raw.columns))

CNT = ["resp_admissions_under5", "resp_admissions_over60", "asthma_emergency_cases",
       "copd_admissions", "resp", "cvd", "death"]
ENV = ["temp_c", "dptp_c", "rhum_pct"]
NUM = ["time"] + ENV + CNT
hilang = [c for c in NUM + ["date", "hospital_id", "dow"] if c not in raw.columns]
if hilang:
    raise SystemExit(f"KOLOM TIDAK DITEMUKAN: {hilang}  (kirim daftar kolom di atas ke mentor)")

H = raw[["date", "hospital_id", "dow"]].copy()
for c in NUM:
    H[c] = num(raw[c])
H["date_dt"] = pd.to_datetime(H["date"], errors="coerce")

# ---------- A. Nilai kosong dan non-numerik ----------
print("\nA. NILAI KOSONG DAN NON-NUMERIK")
rows = []
for c in NUM:
    s = raw[c].str.strip()
    kosong = s.isin(KOSONG)
    nn = H[c].isna() & ~kosong
    rows.append({"kolom": c, "kosong": int(kosong.sum()), "kosong_%": round(kosong.mean() * 100, 2),
                 "non_numerik": int(nn.sum())})
print(pd.DataFrame(rows).to_string(index=False))
print("tanggal gagal parse:", int(H["date_dt"].isna().sum()),
      "| dow kosong:", int((H["dow"].str.strip() == "").sum()))
m = H[NUM].isna().groupby(H["hospital_id"]).mean().mul(100).round(1)
m = m[m.sum(axis=1) > 0]
print("\nPersen kosong per rumah sakit (hanya yang punya kosong):")
print(m.to_string() if len(m) else "Tidak ada.")

# ---------- B. Rentang dan logika ----------
print("\n" + "=" * 80)
print("B. RENTANG DAN LOGIKA")
desc = H[CNT].describe(percentiles=[.5, .95]).T[["min", "50%", "95%", "max"]]
desc["negatif"] = (H[CNT] < 0).sum()
desc["bukan_bulat"] = H[CNT].apply(lambda s: int(((s.dropna() % 1) != 0).sum()))
desc["nol_%"] = (H[CNT] == 0).mean().mul(100).round(1)
print(desc.round(2).to_string())
print("\nVariabel lingkungan:")
print(H[ENV].describe().T[["min", "50%", "max"]].round(2).to_string())
print("rhum_pct di luar 0-100:", int(((H["rhum_pct"] < 0) | (H["rhum_pct"] > 100)).sum()))
print("dptp_c > temp_c (titik embun tidak boleh melebihi suhu):", int((H["dptp_c"] > H["temp_c"]).sum()))

nama_hari = H["date_dt"].dt.day_name()
print("\nbaris dengan dow tidak cocok dengan tanggal:", int((nama_hari != H["dow"].str.strip()).sum()))
print("nilai dow:", H["dow"].str.strip().value_counts().to_dict())

ok = H[["resp", "resp_admissions_under5", "resp_admissions_over60"]].dropna()
print("\nresp < under5 + over60:", int((ok["resp"] < ok["resp_admissions_under5"] + ok["resp_admissions_over60"]).sum()),
      "| resp < under5:", int((ok["resp"] < ok["resp_admissions_under5"]).sum()),
      "| resp < over60:", int((ok["resp"] < ok["resp_admissions_over60"]).sum()), "| dari", len(ok))

# ---------- C. Profil rumah sakit ----------
print("\n" + "=" * 80)
print("C. PROFIL RATA-RATA HARIAN PER RUMAH SAKIT (dengan kelas faskes dan kapasitas)")
hosp = pd.read_csv(list(PKG.rglob("hospitals.csv"))[0])[["hospital_id", "kelas_faskes", "kapasitas_tempat_tidur"]]
pendek = {"resp_admissions_under5": "u5", "resp_admissions_over60": "o60",
          "asthma_emergency_cases": "asma", "copd_admissions": "ppok"}
prof = H.groupby("hospital_id")[CNT].mean().round(2).rename(columns=pendek)
prof = prof.join(hosp.set_index("hospital_id"))
prof["resp_per_100_tt"] = (prof["resp"] / prof["kapasitas_tempat_tidur"] * 100).round(2)
print(prof.sort_values("resp", ascending=False).to_string())

# ---------- D. Indeks 'time' dan hari pertama ----------
print("\n" + "=" * 80)
print("D. KESESUAIAN 'time' DENGAN TANGGAL DAN HARI PERTAMA PER RUMAH SAKIT")
H["time_harapan"] = (H["date_dt"] - pd.Timestamp("2023-12-31")).dt.days + 1
H["selisih_time"] = H["time"] - H["time_harapan"]
print("selisih (time - harapan) per rumah sakit:")
print(H.groupby("hospital_id")["selisih_time"].agg(["min", "max", "nunique"]).to_string())
lima = ["RS-XXX-12", "RS-XXX-13", "RS-XXX-16", "RS-XXX-19", "RS-XXX-20", "RS-XXX-01", "RS-XXX-02"]
awal = H.sort_values("date_dt").groupby("hospital_id").head(3)
print("\n3 baris pertama (5 rumah sakit mulai 2023-12-31 + 2 pembanding):")
print(awal[awal["hospital_id"].isin(lima)]
      [["hospital_id", "date", "time", "dow", "temp_c", "resp", "death"]]
      .sort_values(["hospital_id", "date"]).to_string(index=False))

# ---------- E. Uji penyelarasan tanggal lewat suhu ----------
print("\n" + "=" * 80)
print("E. KORELASI SUHU HARIAN RUMAH SAKIT vs SUHU HARIAN STASIUN (geser -1/0/+1 hari)")
print("(menguji pemetaan dan penyelarasan tanggal; JMB-02/RS-27 dilewati)")
rows = []
for p in sorted(PKG.glob("*_AQMS.csv")):
    kode = p.name.split("_")[0]
    if kode == "JMB-02":
        continue
    d = pd.read_csv(p, usecols=["sensor_id", "timestamp_start", "temperature_c"],
                    dtype=str, keep_default_na=False)
    no = int(d["sensor_id"].iloc[0].split("-")[-1])
    rs = f"RS-XXX-{no:02d}"
    ts = pd.to_datetime(d["timestamp_start"], errors="coerce").dt.round("h")
    daily = num(d["temperature_c"]).groupby(ts.dt.normalize()).mean()
    h = H[H["hospital_id"] == rs].set_index("date_dt")["temp_c"]
    row = {"hospital_id": rs, "stasiun": kode}
    for k in (-1, 0, 1):
        x = daily.copy()
        x.index = x.index + pd.Timedelta(days=k)
        j = pd.concat([h, x], axis=1, join="inner").dropna()
        row[f"r_geser_{k:+d}"] = round(j.iloc[:, 0].corr(j.iloc[:, 1]), 3) if len(j) > 30 else np.nan
        row["n_hari"] = len(j)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))