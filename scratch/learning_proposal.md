# Proposal Pembelajaran (/learn): Multi-Sensor Satellite Precipitation Downscaling & Cross-Comparison

Berdasarkan investigasi mendalam 25 tahun (2001--2025) komparasi **GSMaP (Passive Microwave)** vs **CHIRPS (Thermal Infrared)** serta perancangan model **Fused Multi-Sensor Ensemble**, diajukan penambahan aturan operasional baru di `.agents/AGENTS.md`:

---

## 1. Klasifikasi Pola
* **Tipe:** `Project Rule` (Penambahan Aturan Nomor 17 pada `AGENTS.md`)
* **Kategori:** Komparasi Multi-Sensor Satelit, Fisika Penginderaan Jauh Presipitasi, dan Fusi Model Spasial Resolusi Tinggi.

---

## 2. Rincian Aturan yang Diusulkan (Rule 17)

### Rule 17: Multi-Sensor Precipitation Cross-Comparison & Fusion Downscaling Architecture
1. **Pembedaan Fisika Sensor (PMW vs TIR):**
   - Agen wajib membedakan prinsip deteksi fisik antara sensor **Thermal Infrared (TIR)** seperti CHIRPS (berbasis *Cold Cloud Duration* / suhu puncak awan) dan sensor **Passive Microwave (PMW)** seperti GSMaP (berbasis emisi cairan dan hamburan kristal es kolom awan).
   - Jangan pernah mengasumsikan kedua satelit menghasilkan nilai absolut yang sama: CHIRPS secara sistematis lebih tinggi di daerah tropis karena kalibrasi stasiun permukaan stasioner (CHPclim), sedangkan GSMaP cenderung mencatat volume lebih rendah (defisit $\sim 20-25\%$) karena ambang batas sensitivitas radar DPR antariksa terhadap gerimis halus (*drizzle*).
2. **Karakterisasi Dinamika Musiman & Rasio Sensor:**
   - Dalam analisis komparatif multi-dekade, hitung korelasi temporal bulanan ($r$ dan $\rho$) serta rasio musiman ($\text{GSMaP}/\text{CHIRPS}$).
   - Rasio satelit bervariasi mengikuti siklus monsun: rasio tertinggi terjadi pada puncak musim hujan (DJF, konveksi tebal), sedangkan rasio terendah terjadi pada musim peralihan/pancaroba (MAM, variabilitas konveksi mikro lokal).
3. **Analisis Gradien Orografis Transekt:**
   - Evaluasi respon orografis wajib menggunakan profil transekt kontinu Utara--Selatan melintasi zona morfometri kritis (misal: Pegunungan Sadang/Karangsambung $\rightarrow$ Dataran Aluvial $\rightarrow$ Garis Pantai Samudera Hindia).
   - Sensor gelombang mikro (GSMaP) menangkap gradien lereng curam lebih responsif dibanding inframerah termal.
4. **Arsitektur Fusi Multi-Sensor (Fused Ensemble):**
   - Untuk mencapai akurasi hidrometeorologi optimal, implementasikan model **Fused Multi-Sensor Ensemble** (misalnya *Fused XGBoost*) yang memadukan fitur presipitasi TIR dan PMW secara simultan bersama data reanalisis atmosferik (ERA5-Land) dan topografi lokal (DEM 250m).
   - Model fusi terbukti memitigasi kelemahan intrinsik masing-masing sensor tunggal dan menghasilkan estimasi dengan bias terendah terhadap pengamatan stasiun darat permukaan.

---

## 3. Rencana Penerapan
Jika disetujui, aturan di atas akan ditambahkan ke `.agents/AGENTS.md` sebagai **Rule 17: Multi-Sensor Precipitation Cross-Comparison & Fusion Downscaling Architecture**.
