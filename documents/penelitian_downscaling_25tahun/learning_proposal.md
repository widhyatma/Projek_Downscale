# Proposal Pembelajaran Sistem: Downscaling Presipitasi Multi-Sensor 25 Tahun (2001--2025)

Dokumen ini mendokumentasikan kaidah dan wawasan ilmiah pivotal yang diperoleh selama pelaksanaan riset *downscaling* presipitasi 25 tahun di Kabupaten Kebumen, untuk ditambahkan ke pedoman repositori (`AGENTS.md`).

---

## 1. Identifikasi Wawasan Kunci & Titik Kritis (*Key Learnings*)

### A. Verifikasi Rentang Satuan Suhu ERA5-Land & Masking NoData ($\ge 9000$)
- **Gejala Masalah**: Pada perhitungan Kelembapan Relatif ($\text{RH}$ \%) menggunakan persamaan Magnus-Tetens, terjadi peringatan numerik `RuntimeWarning: overflow encountered in exp` dan nilai $\text{NaN}$.
- **Akar Masalah**:
  1. Pada dataset lokal NetCDF ERA5-Land tertentu, variabel `temperature_2m` dan `dewpoint_temperature_2m` telah tersimpan dalam satuan Celsius ($19.5^\circ\text{C} - 30.5^\circ\text{C}$), bukan Kelvin. Pengurangan $273.15$ kembali menghasilkan suhu $-248^\circ\text{C}$, yang menyebabkan penyebut persamaan Magnus $(243.04 + T) \to -5$ dan eksponen $\exp(>600)$ mengalami *overflow* float64.
  2. Komponen angin $u_{10}$ dan $v_{10}$ memiliki *FillValue* / NoData sebesar $9999.0$ pada sel laut/tepi batas.
- **Kaidah Solutif**:
  - Selalu lakukan inspeksi awal terhadap rentang nilai variabel atmosferik. Jika $\min(T) > 0$ dan $\max(T) < 50$, data sudah berupa Celsius.
  - Maskir nilai $\ge 9000.0$ atau $\le -9000.0$ menjadi `NaN` sebelum melakukan operasi rata-rata zonal (`np.nanmean`).

---

### B. Isolasi Koordinat Unik per Langkah Waktu pada Interpolasi Spasial Multi-Bulan
- **Gejala Masalah**: Evaluasi geostatistik (Bilinear RBF, Kriging residu, ANUSPLIN) pada dataset gabungan multi-bulan memicu galat `numpy.linalg.LinAlgError: A singular matrix detected: slice(s) [0] are singular`.
- **Akar Masalah**: Jika observasi dari 30 stasiun digabungkan di beberapa bulan berbeda ke dalam satu matriks latih, koordinat $(X, Y)$ stasiun yang identik muncul berkali-kali dengan nilai presipitasi berbeda. Hal ini menyebabkan jarak antar titik bernilai 0 dan matriks kovarian jarak menjadi singular.
- **Kaidah Solutif**:
  - Model Machine Learning (XGBoost, LightGBM, Random Forest, SVR, MLP) dilatih pada seluruh observasi gabungan.
  - Model interpolasi spasial murni (Bilinear, Spline ANUSPLIN, Kriging residu) wajib dievaluasi secara terisolasi per langkah waktu (`year_month`), di mana setiap stasiun hanya memiliki 1 observasi unik pada koordinatnya.

---

### C. Fusi Multi-Satelit (CHIRPS + GSMaP) & Penegakan Konservasi Massa Regional
- **Temuan Saintifik**:
  - CHIRPS ($89.41\%$ kontribusi) memberikan fondasi spasio-temporal inframerah termal jangka panjang yang sangat stabil.
  - GSMaP ($9.26\%$ kontribusi) memberikan koreksi esensial terhadap inti badai konvektif gelombang mikro.
  - Penggabungan keduanya bersama variabel atmosferik ERA5-Land dan DEM menghasilkan akurasi tertinggi ($KGE = 0.9973$, $RMSE = 5.14\text{ mm}$, $R^2 = 0.9988$).
  - Penegakan rasio konservasi massa regional menjamin volume total air terestrial tetap lestari tanpa adanya defisit atau surplus artifisial.

---

## 2. Usulan Tambahan Aturan pada `AGENTS.md` (Rule 15)

```markdown
## 15. Multi-Sensor Reanalysis Fusion & Numeric Guardrails
- **Atmospheric Variable Unit Inspection:**
  - Before applying thermodynamic formulas (e.g. Magnus-Tetens RH), inspect the empirical range of $T_{2m}$ and $T_{dew}$.
  - If values fall within $[15, 45]$, they are already in Celsius; do NOT subtract $273.15$.
  - Always mask out reanalysis missing values ($\ge 9000$ or $\le -9000$) with NaN before spatial aggregation.
- **Time-Slice Grouping for Geostatistical Interpolators:**
  - When applying spatial RBF splines, ANUSPLIN, or Kriging residual interpolation on multi-temporal observational tables, process spatial interpolation per time slice (`year_month`) to ensure station coordinates $(X, Y)$ are strictly unique, preventing singular matrix errors.
- **Regional Mass Conservation Preservation:**
  - Always rescale the high-resolution downscaled grid so that the areal total precipitation volume equals the fused satellite areal volume:
    $$\iint_{\Omega} P_{\text{downscaled}}(x,y) \, dx\,dy = \iint_{\Omega} P_{\text{satellite}}(x,y) \, dx\,dy$$
```
