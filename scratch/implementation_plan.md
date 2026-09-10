# Rencana Implementasi: Penelitian Komparatif Downscaling GSMaP vs CHIRPS 25 Tahun (2001--2025) Kabupaten Kebumen

## 1. Ringkasan Masalah dan Tujuan Riset

Penelitian sebelumnya di [`documents/penelitian_downscaling_25tahun`](file:///d:/Github/Projek_Downscale/documents/penelitian_downscaling_25tahun) dan [`documents/penelitian_downscaling_2025`](file:///d:/Github/Projek_Downscale/documents/penelitian_downscaling_2025) berfokus pada downscaling CHIRPS (atau CHIRPS sebagai jangkar dominan). Pengguna mengajukan pertanyaan saintifik fundamental:
> *"Bagaimana jika kita coba untuk data GSMaP juga, apakah hasilnya sama?"*

Secara fisik dan instrumentasi penginderaan jauh satelit:
1. **CHIRPS v2.0** berbasis sensor *Thermal Infrared* (TIR) yang mengukur suhu puncak awan (*cloud-top brightness temperature*), kemudian dikalibrasi dengan klimatologi stasiun jangka panjang.
2. **GSMaP v8 MVK** berbasis konstelasi sensor *Passive Microwave* (PMW) yang mengukur emisi dan hamburan gelombang mikro oleh hidrometeor es/butir hujan di dalam kolom atmosfer secara langsung, dengan pembaruan per-jam.

Oleh karena itu, **hasilnya secara fisik diprediksi tidak akan persis sama**, terutama pada:
- **Respon Orografis Lereng Terjal:** Sensor gelombang mikro (GSMaP) memiliki kemampuan lebih baik mendeteksi presipitasi konvektif intensif di lereng pegunungan, sedangkan TIR (CHIRPS) rentan meremehkan (*underestimate*) awan konvektif orografis hangat yang puncaknya tidak terlalu dingin.
- **Bias Musiman Kemarau (JJAS):** CHIRPS cenderung lebih stabil pada musim kemarau karena jangkar klimatologi BMKG, sedangkan GSMaP dapat mengalami *overestimation* atau hamburan permukaan kering.
- **Respon Ekstrem ENSO:** Apakah fluktuasi anomali El Niño 2015 dan La Niña 2010 tercatat sama kuat di antara kedua sensor?

Tujuan penelitian ini adalah menjawab pertanyaan tersebut secara saintifik, kuantitatif, dan spasial melalui komparasi *head-to-head* selama 25 tahun penuh (2001--2025 / 300 bulan).

---

## 2. Metodologi dan Tahapan Pelaksanaan

Penelitian akan dieksekusi secara bertahap melalui 5 fase terstruktur:

### Fase 1: Ekstraksi dan Penyelarasan Dataset Komparatif GSMaP vs CHIRPS (2001--2025)
- Memproses 300 bulan data GSMaP v8 (`data/gsmap/`) dan CHIRPS v2.0 (`data/chirps/chirps_rnl/`) secara selaras pada domain 250m Kabupaten Kebumen ($219 \times 185$ piksel, $21.704$ sel aktif).
- Menghitung statistik temporal bulanan, kuartalan (DJF, MAM, JJA, SON), dan tahunan untuk kedua satelit.
- Menyelaraskan dengan data observasi 30 pos penakar hujan daratan Kebumen (`downscale_curah_hujan/data/raw/rain_gauges_daily.csv`).
- Menyimpan hasil ekstraksi ke `data/processed_gsmap_vs_chirps/comparative_timeseries_25yr.csv`.

### Fase 2: Pelatihan dan Validasi Spasial 12 Algoritma Downscaling pada GSMaP vs CHIRPS
- Menerapkan protokol **Spatial Block Cross-Validation (4 Kuadran)** dengan **Buffer Zone 5.000m** pada koordinat metrik UTM Zona 49S (Rule 12).
- Melatih suite algoritma inti secara terpisah pada:
  1. **Trek A: Downscaling GSMaP Murni** (PRISM, ANUSPLIN, Regression-Kriging, XGBoost, LightGBM).
  2. **Trek B: Downscaling CHIRPS Murni** (PRISM, ANUSPLIN, Regression-Kriging, XGBoost, LightGBM).
  3. **Trek C: Fusi Multi-Sensor Hybrid** (CHIRPS + GSMaP + ERA5-Land + MODIS NDVI).
- Mengevaluasi metrik akurasi kuantitatif terhadap 30 stasiun: $KGE$, $RMSE$, $MAE$, $R^2$, dan $PBIAS$ (\%).
- Membandingkan apakah algoritma tertentu bekerja lebih optimal pada GSMaP vs CHIRPS.

### Fase 3: Rekonstruksi Klimatologi Spasial 25 Tahun & Analisis Selisih Spasial (GSMaP - CHIRPS)
- Merekonstruksi peta rata-rata tahunan 25 tahun dan 12 bulan siklus klimatologis untuk:
  - GSMaP Downscaled 250m.
  - CHIRPS Downscaled 250m.
  - Selisih Spasial Absolut ($\Delta P = P_{\text{GSMaP}} - P_{\text{CHIRPS}}$ mm/tahun) dan Relatif ($\%$) untuk mengungkap zona di mana GSMaP lebih basah/kering dibandingkan CHIRPS.
- Analisis pengangkatan orografis: Membandingkan rasio presipitasi pegunungan utara (Sadang/Karangsambung) terhadap dataran pantai selatan (Petanahan/Ambal).

### Fase 4: Analisis Sensitivitas Ekstrem ENSO & Statistik Zonal 26 Kecamatan
- Mengevaluasi respon kedua sensor pada tahun ekstrem:
  - Super La Niña 2010 \& Triple-dip La Niña 2022 (Surplus).
  - Super El Niño 2015 \& Strong El Niño 2023 (Defisit).
- Menghitung statistik zonal komparatif (Mean, Min, Max, Bias GSMaP vs CHIRPS) untuk seluruh 26 kecamatan di Kebumen.
- Menyimpan data terstruktur ke CSV di `data/processed_gsmap_vs_chirps/`.

### Fase 5: Visualisasi Publikasi 16:9, Naskah Monograf LaTeX, dan Kompilasi PDF
- Menghasilkan 8 gambar visualisasi publikasi resolusi tinggi (300 DPI, rasio 16:9):
  1. `fig1_sensors_characteristics.png`: Karakteristik spektral TIR vs PMW & domain studi Kebumen.
  2. `fig2_scatter_timeseries_correlation.png`: Scatter plot dan deret waktu 25 tahun GSMaP vs CHIRPS.
  3. `fig3_model_accuracy_comparison.png`: Perbandingan metrik KGE/RMSE algoritma downscaling GSMaP vs CHIRPS.
  4. `fig4_spatial_climatology_comparison.png`: Peta komparasi spasial 25 tahun (GSMaP vs CHIRPS vs Selisih).
  5. `fig5_orographic_transect_profile.png`: Profil transekt orografis Utara-Selatan (pegunungan ke pantai).
  6. `fig6_seasonal_bias_grid.png`: Matriks spasial 4 musim perbandingan bias GSMaP vs CHIRPS.
  7. `fig7_enso_extreme_response.png`: Respon anomali ENSO ekstrem kedua satelit (2010 vs 2015/2023).
  8. `fig8_zonal_kecamatan_differences.png`: Grafik komparasi zonal selisih GSMaP vs CHIRPS 26 kecamatan.
- Menyusun naskah monograf akademik LaTeX komprehensif di folder baru:
  [`documents/penelitian_downscaling_gsmap_vs_chirps/laporan_penelitian_gsmap_vs_chirps.tex`](file:///d:/Github/Projek_Downscale/documents/penelitian_downscaling_gsmap_vs_chirps/laporan_penelitian_gsmap_vs_chirps.tex).
- Mengompilasi PDF 2-Pass dengan MiKTeX `pdflatex` hingga tuntas dengan nol galat.

---

## 3. Rencana Berkas & Struktur Folder Baru

Semua berkas akan diisolasi secara rapi di dalam subdirektori modular sesuai Rule 16:
```
documents/penelitian_downscaling_gsmap_vs_chirps/
├── README.md
├── laporan_penelitian_gsmap_vs_chirps.pdf
├── laporan_penelitian_gsmap_vs_chirps.tex
├── figures/
│   ├── fig1_sensors_characteristics.png
│   ├── fig2_scatter_timeseries_correlation.png
│   ├── fig3_model_accuracy_comparison.png
│   ├── fig4_spatial_climatology_comparison.png
│   ├── fig5_orographic_transect_profile.png
│   ├── fig6_seasonal_bias_grid.png
│   ├── fig7_enso_extreme_response.png
│   └── fig8_zonal_kecamatan_differences.png
├── tables/
│   ├── tab1_sensor_specifications.tex
│   ├── tab2_accuracy_gsmap_vs_chirps.tex
│   ├── tab3_orographic_sensitivity_transect.tex
│   ├── tab4_seasonal_bias_summary.tex
│   ├── tab5_zonal_26kec_comparison.tex
│   └── tab6_enso_extreme_anomalies.tex
└── results_data/
    ├── comparative_timeseries_25yr.csv
    ├── algorithm_accuracy_gsmap_vs_chirps.csv
    ├── zonal_stats_gsmap_vs_chirps_26kec.csv
    └── extreme_enso_comparative_anomalies.csv
```
