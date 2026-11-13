"""Halaman 6: data, pembersihan, definisi, metode, keterbatasan, dan acuan."""
import pandas as pd
import streamlit as st

from core.data import muat
from core.tema import judul_halaman, temuan, kartu

judul_halaman("Metodologi", "Dari mana angka di dashboard ini berasal, bagaimana data dibersihkan, dan apa batas penafsirannya.", "Transparansi · 6 dari 6")

S = muat("stasiun.csv")
k = S["kategori_data"].value_counts()

# ---------- Data ----------
st.subheader("Data")
c1, c2, c3 = st.columns(3)
c1.metric("Stasiun AQMS", len(S))
c2.metric("Rumah sakit pasangan", S["hospital_id"].nunique())
c3.metric("Periode", "1 Jan 2024 – 14 Okt 2026")
st.markdown(
    "Dataset sintetis resmi panitia DA Gelar Rasa 2026: 27 berkas pengukuran polutan dan cuaca per jam, satu berkas "
    "kasus kesehatan harian per rumah sakit, tiga tabel master (lokasi, sensor, rumah sakit), dan dua metadata. "
    "Setiap stasiun dipasangkan dengan satu rumah sakit lewat nomor urut sensor dan rumah sakit; pasangan ini cocok "
    "kecamatan, kota, dan provinsinya pada 27 dari 27 pasangan.")

# ---------- Pembersihan ----------
st.subheader("Pembersihan data")
st.markdown("Masalah dicari lebih dulu (audit), baru ditangani satu per satu. Data mentah tidak pernah ditimpa, "
            "dan seluruh langkah dijalankan ulang oleh skrip Python yang sama.")
langkah = pd.DataFrame([
    ("Hanya baris berstatus VALID yang dipakai", "Semua stasiun",
     "Baris MISSING/MAINTENANCE kosong; CALIBRATING berisi nilai uji kalibrasi; SUSPECT di MDN-01, MER-01, SM2-01 menyimpang 1,7–5× dari kondisi sekitar."),
    ("Sensor macet dikosongkan", "BJB-01",
     "7.329 jam (30%) keenam polutan bernilai konstan walau berlabel VALID; cuaca tetap dipakai karena masih berubah."),
    ("Nilai 999 dianggap penanda, bukan pengukuran", "SM2-01",
     "1.039 jam PM2.5 dan PM10 = 999, sekitar 4,5× kondisi sekitar."),
    ("Satuan ppm dikonversi ke µg/m³", "BJB-01, MDN-01, PSW-01",
     "PM ×1.000 (diasumsikan mg/m³); gas memakai berat molekul pada 25 °C, 1 atm. Hasilnya sejalan dengan baseline lokasi."),
    ("Data 15 menit dirata-ratakan per jam", "PLB-01",
     "Sep–Okt tiap tahun tercatat per 15 menit; tanpa agregasi, bulan kabut asap terbobot 4×. Satu jam sah bila ≥3 dari 4 sub-baris valid."),
    ("Timestamp rusak direkonstruksi", "JMB-02",
     "132 baris bertanggal 1970 dan 92 baris dengan awal/akhir tertukar, dipulihkan dari urutan baris setelah terbukti sejajar."),
    ("Waktu dibulatkan ke jam terdekat", "17 stasiun",
     "Jam logger bergeser tetap hingga 11,7 menit."),
    ("Jam hilang dibiarkan kosong", "Semua stasiun",
     "Tidak ada imputasi: mengisi dengan rata-rata akan meratakan episode pencemaran."),
    ("5 baris 31 Des 2023 dikeluarkan", "Data kesehatan",
     "Di luar jendela data udara."),
], columns=["Langkah", "Berlaku untuk", "Alasan"])
st.dataframe(langkah, hide_index=True, width="stretch")
st.caption("Hasil validasi: 659.664 jam-stasiun tanpa duplikat, tanpa nilai negatif, dan tanpa PM2.5 > PM10. "
           "88,4% jam punya PM2.5 andal.")

# ---------- Definisi ----------
st.subheader("Definisi yang dipakai")
st.markdown(f"""
- **Hari valid:** rata-rata 24 jam dihitung bila minimal 18 dari 24 jam (75%) punya data andal, sesuai Permen LHK P.14/2020.
- **Kelengkapan data stasiun:** *Efektif* ≥75% hari valid ({int(k.get('Efektif', 0))} stasiun), *Terbatas* 10–75% ({int(k.get('Terbatas', 0))}), *Buta* <10% ({int(k.get('Buta', 0))}).
- **Kategori ISPU PM2.5 (rata-rata 24 jam, µg/m³):** Baik ≤15,5; Sedang 15,6–55,4; Tidak Sehat 55,5–150,4; Sangat Tidak Sehat 150,5–250,4; Berbahaya >250,4. Warna kategori di dashboard adalah pilihan desain tim.
- **Baku mutu PM2.5 (PP 22/2021):** 55 µg/m³ rata-rata 24 jam dan 15 µg/m³ rata-rata tahunan. Rata-rata tahunan hanya dihitung bila ≥80% hari valid.
- **Tren:** rata-rata 1 Januari–14 Oktober tiap tahun, agar 2026 yang terpotong tetap sebanding.
- **IKU (Permen LHK 27/2021):** IKU = 100 − (50/0,9) × (Ieu − 0,1), dengan Ieu = rata-rata dari SO₂/20 dan NO₂/40. Dihitung per stasiun per tahun bila ≥80% hari valid.
""")

# ---------- Metode kesehatan ----------
st.subheader("Metode analisis kesehatan")
st.markdown("""
- **Model:** regresi Poisson jumlah kasus harian, dengan efek tetap per rumah sakit serta pengendali hari dalam minggu, bulan, tahun, suhu, dan kelembapan. Galat baku robust per rumah sakit.
- **Rumah sakit:** 24 dari 27. RS-23 dan RS-27 tidak punya data paparan andal; RS-24 hanya 36,4% hari valid.
- **Efek jeda:** PM2.5 hari ini sangat mirip dengan seminggu lalu (autokorelasi 0,89–0,82), sehingga PM2.5 dipisah menjadi komponen lambat (rata-rata 29 hari) dan komponen cepat (selisih harian). Jeda diuji pada komponen cepat. Estimasi negatif pada jeda 5–7 adalah artefak pemisahan ini dan tidak ditafsirkan.
- **Bentuk hubungan:** hari dikelompokkan menurut kategori ISPU dan dibandingkan dengan hari ≤55,4 µg/m³.
""")

# ---------- Keterbatasan ----------
st.subheader("Keterbatasan")
st.markdown("""
- Data **sintetis** dan **historis**: dashboard ini tidak memantau secara waktu nyata, dan pola yang tampak (misalnya episode yang dimulai tepat tanggal 1) mencerminkan cara data dibuat.
- Hasil kesehatan adalah **hubungan statistik, bukan sebab-akibat**. Pengendali musim berlaku umum, bukan per rumah sakit. Kategori >150,4 hanya berdasar 239 hari-rumah sakit.
- **Konversi ppm** pada tiga stasiun adalah asumsi yang didukung data, bukan fakta yang dikonfirmasi.
- **Puncak bisa terpotong:** beberapa polutan berhenti di nilai maksimum yang sama (misalnya O₃ = 38), dan jam kabut asap terparah di SM2-01 tidak punya nilai andal, sehingga rata-rata musimannya kemungkinan terlalu rendah.
- **IKU** di sini adalah proksi dari satu stasiun AQMS, bukan metode resmi (passive sampler di empat lokasi per kota).
- **Jaringan:** hanya 21 provinsi yang punya stasiun. Kebutuhan pemantauan di provinsi lain tidak dapat dinilai dari data ini.
- **Kategori ISPU** dihitung per hari kalender, sebagai penyederhanaan dari perhitungan resmi yang memakai jendela 24 jam bergulir.
""")

# ---------- Acuan ----------
st.subheader("Acuan")
st.markdown("""
- Peraturan Pemerintah Republik Indonesia Nomor 22 Tahun 2021 tentang Penyelenggaraan Perlindungan dan Pengelolaan Lingkungan Hidup.
- Peraturan Menteri Lingkungan Hidup dan Kehutanan Nomor P.14/MENLHK/SETJEN/KUM.1/7/2020 tentang Indeks Standar Pencemar Udara.
- Peraturan Menteri Lingkungan Hidup dan Kehutanan Nomor 27 Tahun 2021 tentang Indeks Kualitas Lingkungan Hidup.
- World Health Organization. (2021). *WHO global air quality guidelines*. WHO.
""")
st.caption("Kode sumber dan skrip pembersihan tersedia di repositori proyek (tautan di laporan).")
