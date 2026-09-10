"""
Organize completed 25-Year Downscaling Research Documents into a new dedicated folder
=====================================================================================
Creates documents/penelitian_downscaling_25tahun/ with all necessary assets:
- TeX source and compiled PDF
- Dedicated figures/ directory
- Dedicated tables/ directory
- Key results CSV files in results_data/
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import shutil
from pathlib import Path

def run():
    print("Mengorganisasi kumpulan dokumen penelitian 25 tahun ke folder baru...")
    base_doc = Path("documents")
    target_dir = base_doc / "penelitian_downscaling_25tahun"
    target_fig = target_dir / "figures"
    target_tab = target_dir / "tables"
    target_res = target_dir / "results_data"

    target_fig.mkdir(parents=True, exist_ok=True)
    target_tab.mkdir(parents=True, exist_ok=True)
    target_res.mkdir(parents=True, exist_ok=True)

    # 1. Salin file dokumen utama
    doc_files = [
        "laporan_penelitian_downscaling_25tahun.tex",
        "laporan_penelitian_downscaling_25tahun.pdf",
        "laporan_penelitian_downscaling_25tahun.aux",
        "laporan_penelitian_downscaling_25tahun.toc",
        "laporan_penelitian_downscaling_25tahun.lof",
        "laporan_penelitian_downscaling_25tahun.lot",
        "laporan_penelitian_downscaling_25tahun.out",
        "laporan_penelitian_downscaling_25tahun.log"
    ]
    for fname in doc_files:
        src = base_doc / fname
        if src.exists():
            shutil.copy2(src, target_dir / fname)
            print(f"  ✓ Tersalin: {fname}")

    # 2. Salin gambar-gambar penelitian 25 tahun (fig1 s.d. fig8)
    fig_files = [f"fig{i}_{suffix}" for i, suffix in [
        (1, "study_area_terrain.png"),
        (2, "multisource_correlation.png"),
        (3, "model_benchmark_comparison.png"),
        (4, "climatology_annual_mean.png"),
        (5, "monthly_climatological_cycle.png"),
        (6, "climate_trend_sens_slope.png"),
        (7, "extreme_enso_anomalies.png"),
        (8, "zonal_kecamatan_comparison.png")
    ]]
    for fname in fig_files:
        src = base_doc / "figures" / fname
        if src.exists():
            shutil.copy2(src, target_fig / fname)
            print(f"  ✓ Tersalin figure: {fname}")

    # 3. Salin tabel-tabel LaTeX (tab1 s.d. tab6)
    tab_files = [
        "tab1_dataset_summary.tex",
        "tab2_vif_analysis.tex",
        "tab3_benchmark_models.tex",
        "tab4_feature_importance.tex",
        "tab5_zonal_climatology_26kec.tex",
        "tab6_extreme_enso_impact.tex"
    ]
    for fname in tab_files:
        src = base_doc / "tables" / fname
        if src.exists():
            shutil.copy2(src, target_tab / fname)
            print(f"  ✓ Tersalin table: {fname}")

    # 4. Salin file hasil analitik CSV
    res_dir = Path("data/processed_25yr")
    res_files = [
        "benchmark_model_metrics.csv",
        "feature_importance.csv",
        "vif_analysis.csv",
        "zonal_stats_26_kecamatan_25yr.csv",
        "extreme_enso_zonal_anomalies.csv",
        "multivariate_25yr_timeseries_monthly.csv"
    ]
    for fname in res_files:
        src = res_dir / fname
        if src.exists():
            shutil.copy2(src, target_res / fname)
            print(f"  ✓ Tersalin data analitik: {fname}")

    # Salin proposal pembelajaran
    shutil.copy2("scratch/learning_proposal.md", target_dir / "learning_proposal.md")
    print(f"  ✓ Tersalin: learning_proposal.md")

    # Buat README.md penjelas di dalam folder baru
    readme_content = """# Arsip Dokumen Penelitian: Downscaling Presipitasi 25 Tahun (2001--2025)

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
D:\\MiKTeX\\miktex\\bin\\x64\\pdflatex.exe -interaction=nonstopmode laporan_penelitian_downscaling_25tahun.tex
```
"""
    with open(target_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("  ✓ Dibuat: README.md")

    print("\n" + "=" * 80)
    print(f"🎉 SUKSES! Seluruh dokumen penelitian 25 tahun tersimpan di: {target_dir}")
    print("=" * 80)

if __name__ == "__main__":
    run()
