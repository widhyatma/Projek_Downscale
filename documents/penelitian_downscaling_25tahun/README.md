# Arsip Dokumen Penelitian: Downscaling Presipitasi 25 Tahun (2001--2025)

Folder ini berisi seluruh kumpulan dokumen, naskah akademik LaTeX, gambar visualisasi publikasi resolusi tinggi (16:9, 300 DPI), tabel saintifik, dan data analitik dari penelitian:

**Judul Penelitian:**
> *Fusi Multi-Satelit, Reanalisis Atmosferik, dan Topografi Resolusi Tinggi untuk Downscaling Presipitasi 25 Tahun (2001--2025) serta Analisis Iklim Ekstrem Kabupaten Kebumen*

## Struktur Folder:
- `laporan_penelitian_downscaling_25tahun.pdf`: Laporan monograf akademik final 23 halaman.
- `laporan_penelitian_downscaling_25tahun.tex`: Sumber kode LaTeX monograf lengkap.
- `figures/`: 8 gambar publikasi resolusi tinggi (300 DPI, rasio 16:9):
  - `fig1_study_area_terrain.png`
  - `fig2_multisource_correlation.png`
  - `fig3_model_benchmark_comparison.png`
  - `fig4_climatology_annual_mean.png`
  - `fig5_monthly_climatological_cycle.png`
  - `fig6_climate_trend_sens_slope.png`
  - `fig7_extreme_enso_anomalies.png`
  - `fig8_zonal_kecamatan_comparison.png`
- `tables/`: 6 tabel format LaTeX standar publikasi ilmiah (`tab1` s.d. `tab6`).
- `results_data/`: Data analitik terstruktur (metrik akurasi 12 model, feature importance, uji VIF, statistik zonal 26 kecamatan, anomali ENSO).
- `learning_proposal.md`: Dokumen kaidah pembelajaran teknis sistem.

## Kompilasi Ulang:
Untuk mengompilasi ulang dokumen PDF di folder ini:
```bash
D:\MiKTeX\miktex\bin\x64\pdflatex.exe -interaction=nonstopmode laporan_penelitian_downscaling_25tahun.tex
```
