# Learning Proposal: Standarisasi Deep Learning Sekuensial & Downscaling Spasial 250m

## 1. Identifikasi Kebutuhan Pembelajaran & Rationale
Pengguna meminta evaluasi penggunaan algoritma deret waktu lanjutan (misalnya **LSTM / Deep Learning**) serta metode **Spatial Downscaling** untuk meningkatkan resolusi prediksi musim hujan ke skala spasial mikro.

Untuk memastikan bahwa arsitektur ini terdokumentasi dan dapat digunakan kembali secara konsisten di seluruh sesi dan agen, diusulkan penambahan dua aturan baru ke dalam [`.agents/AGENTS.md`](file:///d:/Github/Projek_Downscale/.agents/AGENTS.md):
1. **Rule 19: Sequential Deep Learning (LSTM/GRU) for Dasarian Climate Modeling**
   Menstandarkan pembentukan tensor 3-dimensi `(batch, time_steps, n_features)` tanpa kebocoran temporal (*no future leakage*), normalisasi input, arsitektur multi-head (klasifikasi kategori AMH + regresi dasarian), dan protokol pengujian *Leave-One-Year-Out (LOYO)*.
2. **Rule 20: Topographic Downscaling of Seasonal Predictions to 250m Continuous Raster Grids**
   Menstandarkan penggunaan DEM topografi resolusi tinggi, dekomposisi aspek ortogonal ($\sin/\cos$), jarak pantai metrik, kalibrasi pelestarian massa regional (*regional mass conservation calibration*), dan format ekspor GeoTIFF.

---

## 2. Klasifikasi Pembelajaran
- **Tipe:** Update Existing Configuration (Menambahkan Rule Baru ke [`.agents/AGENTS.md`](file:///d:/Github/Projek_Downscale/.agents/AGENTS.md))
- **Target File:** [`.agents/AGENTS.md`](file:///d:/Github/Projek_Downscale/.agents/AGENTS.md)
- **Target Nomor Rule:** Rule 19 & Rule 20

---

## 3. Rincian Usulan Teks Rule Baru (Proposed Additions)

```markdown
## 19. Sequential Deep Learning (LSTM/GRU) for Dasarian Climate Modeling
- **3D Temporal Sequence Tensor Construction:**
  - Input deret waktu pra-musim harus dibentuk sebagai tensor 3D berdimensi `(n_samples, time_steps, n_features)`.
  - Rentang sekuens waktu standar: Dasarian 1 (Jan I) hingga Dasarian 27 (Sep III) (27 time-steps per tahun per wilayah).
  - Fitur per time-step mencakup presipitasi dasarian, suhu 2m, komponen angin zonal $u_{10}$, kelembapan relatif $RH$, serta interpolasi indeks ENSO (ONI) dan IOD (DMI).
  - Skala fitur harus dinormalisasi per fold training tanpa menyertakan tahun uji untuk mencegah kebocoran data (*zero data leakage*).
- **Multi-Head Task Architecture:**
  - Model sekuensial (LSTM / Bi-LSTM / GRU) harus mengekstrak representasi tersembunyi bersama (*shared latent representation*), kemudian dicabangkan ke multi-head:
    1. *Kategori AMH Head:* Klasifikasi 3-kelas (Maju, Sama, Mundur) dengan aktivasi Softmax.
    2. *Dasarian AMH Eksak Head:* Regresi linier nilai dasarian.
    3. *Sifat Hujan Head:* Klasifikasi 3-kelas (Atas Normal, Normal, Bawah Normal).
- **Evaluasi LOYO Wajib:**
  - Pelatihan model deep learning harus dievaluasi dengan Leave-One-Year-Out (LOYO) cross-validation melintasi seluruh rentang tahun historis.

## 20. Topographic Downscaling of Seasonal Predictions to 250m Continuous Raster Grids
- **Multi-Covariate Topographic Mesh (250m Grid):**
  - Grid kontinu 250m diekstraksi dari DEM resolusi tinggi (`DEM_Topografi_Kebumen_30m.tif`), mencakup:
    1. Elevasi topografi ($Z$, mdpl).
    2. Kemiringan lereng (*slope*, derajat).
    3. Dekomposisi aspek ortogonal: $\sin(\text{aspect})$ dan $\cos(\text{aspect})$ (Rule 12).
    4. Jarak Euclidean ke garis pantai Samudra Hindia ($D_{\text{coast}}$, km).
    5. Topographic Position Index (TPI 250m).
- **Penegakan Konservasi Massa Air Regional (Rule 15):**
  - Setiap raster hasil downscaling harus dikalibrasi volume total presipitasinya agar presisi terhadap observasi satelit referensi:
    $$P_{\text{downscaled, calibrated}}(x, y) = P_{\text{downscaled}}(x, y) \times \left( \frac{\iint_{\Omega} P_{\text{satellite}}(x, y) \, dx dy}{\iint_{\Omega} P_{\text{downscaled}}(x, y) \, dx dy} \right)$$
- **Format Ekspor Standar:**
  - Hasil downscaling harus diekspor dalam format GeoTIFF terproyeksi (EPSG:4326/WGS84) dengan kompresi LZW serta divisualisasikan dalam rasio aspek standar 16:9 (300 DPI).
```

---

## 4. Konfirmasi Pengguna
Apakah Anda menyetujui penambahan **Rule 19 dan Rule 20** ini ke dalam [`.agents/AGENTS.md`](file:///d:/Github/Projek_Downscale/.agents/AGENTS.md)?
