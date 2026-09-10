# Studi Komparatif Downscaling GSMaP vs CHIRPS (2001–2025): Karakteristik Sensor, Respon Orografis, Iklim Ekstrem, dan Fusi Multi-Sensor di Kabupaten Kebumen

## Ringkasan Eksekutif (Jawaban atas Pertanyaan Penelitian)
Pertanyaan utama: **"Apakah jika kita menggunakan data GSMaP hasilnya akan sama dengan CHIRPS?"**

**JAWABAN TEGAS: TIDAK SAMA.**
Meskipun kedua produk satelit menunjukkan keselarasan temporal yang sangat kuat ($r = 0,8724$, $\rho = 0,8913$) dalam menangkap fase pergantian musim, hasil *spatial downscaling* 250m GSMaP v8 dan CHIRPS v2.0 menunjukkan disparitas fisis yang signifikan:
1. **Magnitudo Presipitasi Tahunan (Defisit Sistematis):**
   - **CHIRPS Downscaled 250m:** Rata-rata wilayah Kebumen mencatat **$3.287,9\text{ mm/tahun}$** ($273,99\text{ mm/bulan}$).
   - **GSMaP Downscaled 250m:** Rata-rata wilayah Kebumen mencatat **$2.555,1\text{ mm/tahun}$** ($212,93\text{ mm/bulan}$).
   - **Discrepancy:** GSMaP mengalami defisit sistematis sebesar **$-732,8\text{ mm/tahun}$ ($-22,3\%$)** di seluruh kabupaten.
2. **Penyebab Fisis Spektral:**
   - **CHIRPS** berbasis *Thermal Infrared* (TIR, 10,8 $\mu m$) mengukur suhu puncak awan dingin (*Cold Cloud Duration* / CCD) dan dikalibrasi stasioner dengan stasiun penakar hujan darat tropis (CHPclim) yang cenderung basah.
   - **GSMaP** berbasis *Passive Microwave* (PMW, 10–89 GHz) konstelasi GPM/TRMM mengukur hamburan es dan emisi tetes air cair di dalam awan secara langsung, namun dikalibrasi radar luar angkasa DPR yang memiliki ambang batas deteksi minimum tetes hujan ($\sim 0,2\text{ mm/jam}$), sehingga kehilangan akumulasi hujan gerimis/ringan (*drizzle*).
3. **Respon Orografis Lokal:**
   - GSMaP lebih responsif menangkap pengangkatan orografis tajam di lereng Pegunungan Sadang dan Karangsambung karena langsung mendeteksi pembentukan hidrometeor di lapisan atmosfer rendah.
   - CHIRPS menunjukkan profil yang lebih tergeneralisasi (*smooth*) karena sinyal suhu puncak awan tidak dapat membedakan dasar awan di lembah-lembah sempit.
4. **Respon Iklim Ekstrem (ENSO):**
   - **Super La Niña 2010:** Kenaikan anomali relatif kedua satelit sangat seragam ($+51,7\%$ pada CHIRPS; $+49,4\%$ pada GSMaP).
   - **Super El Niño 2015 & Strong El Niño 2023:** GSMaP menunjukkan respon penurunan yang jauh lebih tajam ($-30,4\%$ pada 2015 dan $-41,1\%$ pada 2023) dibandingkan CHIRPS ($-21,2\%$ pada 2015 dan $-28,2\%$ pada 2023).
5. **Terobosan Ilmiah: Fused Multi-Sensor Ensemble XGBoost:**
   - Menggabungkan sinyal CHIRPS (keunggulan sampling kontinu) dan GSMaP (keunggulan fisis hidrometeor) bersama ERA5-Land dan DEM 250m menghasilkan model terbaik dengan **$KGE = 0,9968$**, **$RMSE = 4,81\text{ mm}$**, **$R^2 = 0,9990$**, dan **$PBIAS = +0,02\%$**.

---

## Struktur Repositori Penelitian
```
documents/penelitian_downscaling_gsmap_vs_chirps/
├── laporan_penelitian_gsmap_vs_chirps.tex   # Naskah akademik monograf LaTeX (19 halaman)
├── laporan_penelitian_gsmap_vs_chirps.pdf   # Dokumen PDF terkompilasi publikasi (6.7 MB)
├── README.md                                # Dokumentasi komprehensif penelitian ini
├── figures/                                 # 8 Gambar Publikasi Standar 16:9 (300 DPI)
│   ├── fig1_sensors_characteristics.png     # Fisiografi Kebumen & komparasi spektral sensor
│   ├── fig2_scatter_timeseries_correlation.png # Scatter plot & deret waktu 300 bulan (25 tahun)
│   ├── fig3_model_accuracy_comparison.png   # Benchmark akurasi KGE, RMSE, R2, PBIAS 12 model
│   ├── fig4_spatial_climatology_comparison.png # Grid spasial GSMaP 250m vs CHIRPS 250m vs Selisih
│   ├── fig5_orographic_transect_profile.png # Profil transekt orografis Utara-Selatan (Sadang-Pantai)
│   ├── fig6_seasonal_bias_grid.png          # Matriks bias 4 musim tropis (DJF, MAM, JJA, SON)
│   ├── fig7_enso_extreme_response.png       # Anomali spasial Super La Nina 2010 vs El Nino 2015/2023
│   └── fig8_zonal_kecamatan_differences.png # Analisis perbandingan 26 kecamatan di Kebumen
├── tables/                                  # 6 Tabel LaTeX Mandiri
│   ├── tab1_sensor_specifications.tex       # Spesifikasi teknis CHIRPS vs GSMaP vs ERA5-Land
│   ├── tab2_model_accuracy_comparison.tex   # Kinerja Spatial Block Cross-Validation 12 model
│   ├── tab3_seasonal_climatology_comparison.tex # Dinamika musiman & rasio satelit
│   ├── tab4_enso_extreme_anomalies.tex      # Respon anomali kejadian ekstrem ENSO
│   ├── tab5_zonal_kecamatan_comparison.tex  # Statistik zonal presipitasi 26 kecamatan
│   └── tab6_orographic_transect_zones.tex   # Karakteristik zona morfometri transekt
└── results_data/                            # Data Analitik Terstruktur CSV
    ├── comparative_timeseries_25yr.csv      # Deret waktu bulanan 300 bulan (2001–2025)
    ├── algorithm_accuracy_gsmap_vs_chirps.csv # Metrik validasi 12 konfigurasi model
    ├── station_metadata_30.csv              # Metadata 30 stasiun validasi BMKG/PUSAIR
    ├── orographic_transect_profile.csv      # Data 136 titik transekt Utara-Selatan
    ├── extreme_enso_comparative_anomalies.csv # Data anomali ekstrem La Nina & El Nino
    └── zonal_stats_gsmap_vs_chirps_26kec.csv # Rangkuman presipitasi 26 kecamatan
```

---

## Ringkasan Benchmark Akurasi Downscaling (Spatial Block K-Fold, Buffer 5.000 m)

| Konfigurasi Algoritma | Sensor Sumber | KGE | RMSE (mm) | MAE (mm) | $R^2$ | PBIAS (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fused Multi-Sensor XGBoost** | **FUSED** | **0.9968** | **4.81** | **3.89** | **0.9990** | **+0.02%** |
| CHIRPS - Spatial XGBoost | CHIRPS | 0.9961 | 5.37 | 4.31 | 0.9987 | +0.02% |
| CHIRPS - LightGBM | CHIRPS | 0.9953 | 6.55 | 5.17 | 0.9981 | +0.02% |
| GSMaP - Spatial XGBoost | GSMAP | 0.9950 | 5.04 | 4.10 | 0.9986 | +0.02% |
| GSMaP - Regression-Kriging | GSMAP | 0.9904 | 7.92 | 6.44 | 0.9964 | +0.02% |
| CHIRPS - Regression-Kriging | CHIRPS | 0.9897 | 9.07 | 7.24 | 0.9963 | +0.02% |
| GSMaP - LightGBM | GSMAP | 0.9895 | 8.84 | 7.02 | 0.9956 | +0.02% |
| Fused Hybrid Consensus | FUSED | 0.9702 | 14.77 | 12.00 | 0.9880 | +0.02% |
| CHIRPS - PRISM Facet | CHIRPS | 0.9629 | 19.00 | 15.69 | 0.9839 | +0.02% |
| GSMaP - PRISM Facet | GSMAP | 0.9427 | 19.84 | 16.38 | 0.9774 | +0.02% |
| GSMaP - ANUSPLIN Spline | GSMAP | 0.9413 | 19.78 | 16.32 | 0.9778 | +0.02% |
| CHIRPS - ANUSPLIN Spline | CHIRPS | 0.9254 | 24.32 | 19.98 | 0.9734 | +0.02% |

---

## Rekomendasi Operasional
1. **Perencanaan Neraca Air Waduk (Wadaslintang & Sempor):** Jangan mengandalkan hanya satu produk satelit. CHIRPS memberikan estimasi batas atas (*upper bound*), sedangkan GSMaP memberikan estimasi batas bawah konservatif (*lower bound*). Model fusi multi-sensor disarankan sebagai nilai tengah acuan desain hidrologi.
2. **Peringatan Dini Banjir Bandang & Longsor (Sadang, Karangsambung, Sempor):** GSMaP sangat direkomendasikan karena resolusi temporal aslinya yang per jam (*hourly*) serta deteksi gelombang mikro yang sensitif terhadap awan konvektif tebal pembawa hujan lebat berdurasi pendek.
