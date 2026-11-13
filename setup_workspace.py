from pathlib import Path

ROOT = Path(__file__).resolve().parent
folders = [
    "data/raw", "data/processed", "data/final",
    "notebooks", "scripts", "dashboard", "reports", "docs",
]
for f in folders:
    (ROOT / f).mkdir(parents=True, exist_ok=True)
    print("OK :", f)

readme = ROOT / "README.md"
if not readme.exists():
    readme.write_text("# DA Gelar Rasa 2026 - AQMS\n\nCatatan proyek.\n", encoding="utf-8")