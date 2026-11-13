from pathlib import Path
import zipfile, hashlib

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "raw" / "Dataset_DA_Gelar_Rasa_2026.zip"
OUT = ROOT / "data" / "raw" / "package"

if not ZIP.exists():
    raise SystemExit(f"Zip tidak ditemukan: {ZIP}")

# Sidik jari zip, bukti bahwa data mentah tidak berubah
sha = hashlib.sha256(ZIP.read_bytes()).hexdigest()
print("SHA256 zip:", sha)

if not OUT.exists():
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(OUT)
    print("Diekstrak ke:", OUT)
else:
    print("Folder package sudah ada, ekstraksi dilewati.")

def kelompok(p: Path) -> str:
    if p.name.endswith("_AQMS.csv"): return "1. Data Udara"
    if p.name.startswith("HLT-"):    return "2. Data Kesehatan"
    if p.name in {"sites.csv", "sensors.csv", "hospitals.csv"}: return "3. Tabel Master"
    if p.parent.name == "metadata":  return "4. Metadata"
    return "9. LAINNYA (periksa!)"

files = sorted(OUT.rglob("*.csv"))
print(f"\nTotal file CSV: {len(files)}\n")

rekap = {}
for p in files:
    rekap.setdefault(kelompok(p), []).append(p)

for k in sorted(rekap):
    total_mb = sum(p.stat().st_size for p in rekap[k]) / 1e6
    print(f"{k:<22} {len(rekap[k]):>3} file  {total_mb:>7.1f} MB")

print("\nDetail file AQMS:")
for p in rekap.get("1. Data Udara", []):
    print(f"  {p.name:<40} {p.stat().st_size/1e6:5.2f} MB")