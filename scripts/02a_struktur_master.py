from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 45)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "data" / "raw" / "package"

def cari(nama):
    hasil = list(PKG.rglob(nama))
    assert len(hasil) == 1, f"{nama}: ditemukan {len(hasil)} file"
    return hasil[0]

for nama in ["sites.csv", "sensors.csv", "hospitals.csv",
             "metadata_sensors.csv", "metadata_dates.csv"]:
    df = pd.read_csv(cari(nama))
    print("=" * 90)
    print(f"{nama}  ->  {df.shape[0]} baris, {df.shape[1]} kolom")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unik": df.nunique(),
    })
    print(info.to_string())
    print("\nContoh 3 baris pertama (ditransposisi):")
    print(df.head(3).T.to_string())
    print()