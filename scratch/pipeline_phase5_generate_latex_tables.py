"""
Generate Clean LaTeX Tables for Academic Monograph
===================================================
Produces publication-grade LaTeX tables in documents/tables/.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import pandas as pd
import numpy as np

def run():
    print("Mengekspor tabel-tabel LaTeX ke documents/tables/...")
    out_dir = Path("documents/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/processed_25yr")

    # Table 1: Multi-Dataset Summary
    tab1 = r"""\begin{table}[htbp]
\centering
\small
\caption{Karakteristik Komprehensif Multi-Dataset Satelit, Reanalisis Atmosferik, dan Topografi (2001--2025)}
\label{tab:dataset_summary}
\begin{tabular}{lccccc}
\hline
\textbf{Nama Produk / Sumber} & \textbf{Tipe Data} & \textbf{Resolusi Asli} & \textbf{Domain Waktu} & \textbf{Variabel Utama} & \textbf{Format} \\
\hline
CHIRPS v2.0 (UCSB/USGS) & Satelit IR + Stasiun & $0.05^\circ$ ($\approx 5.5$ km) & 2001--2025 (300 bln) & Presipitasi Harian & NetCDF4 \\
GSMaP v8 MVK (JAXA) & Satelit Microwave & $0.10^\circ$ ($\approx 11$ km) & 2001--2025 (300 bln) & Presipitasi Per-Jam & NetCDF4 \\
ERA5-Land (ECMWF) & Reanalisis Atmosfer & $0.10^\circ$ ($\approx 11$ km) & 2001--2025 (300 bln) & $T_{2m}$, $T_{dew}$, $u/v$, $P_s$ & NetCDF4 \\
MODIS Terra (MOD13A3) & Reflektansi Satelit & 1 km resampled 250m & 2001--2025 (300 bln) & NDVI Rata-rata Bulanan & GeoTIFF \\
SRTM DEM (NASA/USGS) & Radar Topografi & 30m target 250m & Statis (Baseline) & Elevasi, Slope, Aspect & GeoTIFF \\
BPS / BIG Kebumen & Vektor Administrasi & Skala 1:25.000 & 2024 (Terkini) & Batas 26 Kecamatan & GeoJSON \\
\hline
\end{tabular}
\end{table}
"""
    with open(out_dir / "tab1_dataset_summary.tex", "w", encoding="utf-8") as f:
        f.write(tab1)

    # Table 2: VIF Analysis
    df_vif = pd.read_csv(data_dir / "vif_analysis.csv")
    tab2 = r"""\begin{table}[htbp]
\centering
\small
\caption{Evaluasi Multikolinieritas dan Nilai Variance Inflation Factor (VIF) Prediktor Lingkungan}
\label{tab:vif_analysis}
\begin{tabular}{lccc}
\hline
\textbf{Variabel Prediktor Lingkungan} & \textbf{Nilai VIF} & \textbf{Ambang Batas Guardrail} & \textbf{Status Multikolinieritas} \\
\hline
"""
    for _, r in df_vif.iterrows():
        p_name = r['predictor'].replace('_', r'\_')
        v_val = f"{r['vif']:.2f}"
        status = "Memenuhi Syarat (Lolos)" if r['vif'] < 5.0 else "Batas Toleransi Orografis"
        tab2 += f"{p_name} & {v_val} & $< 5.00$ & {status} \\\\\n"
    tab2 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(out_dir / "tab2_vif_analysis.tex", "w", encoding="utf-8") as f:
        f.write(tab2)

    # Table 3: Benchmark 12 Models
    df_met = pd.read_csv(data_dir / "benchmark_model_metrics.csv")
    tab3 = r"""\begin{table}[htbp]
\centering
\small
\caption{Peringkat Evaluasi Akurasi 12 Algoritma Downscaling Berdasarkan Spatial Block Cross-Validation (Buffer 5.000m)}
\label{tab:benchmark_models}
\begin{tabular}{clccccc}
\hline
\textbf{Peringkat} & \textbf{Nama Algoritma / Arsitektur} & \textbf{KGE} & \textbf{RMSE (mm)} & \textbf{MAE (mm)} & \textbf{$R^2$} & \textbf{PBIAS (\%)} \\
\hline
"""
    for rank, (_, r) in enumerate(df_met.iterrows(), 1):
        m_name = r['Model'].replace('&', r'\&').replace('_', r'\_')
        if ". " in m_name:
            m_name = m_name.split(". ")[1]
        kge_s = f"{r['KGE']:.4f}"
        rmse_s = f"{r['RMSE (mm)']:.2f}"
        mae_s = f"{r['MAE (mm)']:.2f}"
        r2_s = f"{r['R2']:.4f}"
        pbias_s = f"{r['PBIAS (%)']:+.2f}"
        bold_pre = r"\textbf{" if rank <= 3 else ""
        bold_post = r"}" if rank <= 3 else ""
        tab3 += f"{rank} & {bold_pre}{m_name}{bold_post} & {bold_pre}{kge_s}{bold_post} & {rmse_s} & {mae_s} & {r2_s} & {pbias_s} \\\\\n"
    tab3 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(out_dir / "tab3_benchmark_models.tex", "w", encoding="utf-8") as f:
        f.write(tab3)

    # Table 4: Feature Importance
    df_feat = pd.read_csv(data_dir / "feature_importance.csv")
    tab4 = r"""\begin{table}[htbp]
\centering
\small
\caption{Kontribusi Relatif dan Feature Importance Prediktor pada Model Multi-Sensor Hybrid XGBoost}
\label{tab:feature_importance}
\begin{tabular}{llcc}
\hline
\textbf{Kategori Prediktor} & \textbf{Nama Fitur Lingkungan} & \textbf{Gain Relatif} & \textbf{Kontribusi Persentase (\%)} \\
\hline
"""
    cat_map = {
        'chirps_raw': 'Satelit Presipitasi (Thermal IR)',
        'gsmap_raw': 'Satelit Presipitasi (Passive Microwave)',
        't2m': 'Atmosfer Reanalisis ERA5-Land',
        'rh': 'Atmosfer Reanalisis ERA5-Land',
        'ws': 'Atmosfer Reanalisis ERA5-Land',
        'sp': 'Atmosfer Reanalisis ERA5-Land',
        'ndvi': 'Biosfer / Indeks Vegetasi (MODIS)',
        'elev': 'Topografi Mikro (SRTM DEM)',
        'slope': 'Topografi Mikro (SRTM DEM)',
        'coast': 'Morfologi Geografis Pesisir',
        'sin_aspect': 'Topografi Mikro (Aspek Sinus)',
        'cos_aspect': 'Topografi Mikro (Aspek Cosinus)'
    }
    for _, r in df_feat.iterrows():
        f_name = r['Feature']
        cat = cat_map.get(f_name, 'Prediktor Lingkungan')
        f_name_esc = f_name.replace('_', r'\_')
        gain = f"{r['Importance']:.6f}"
        pct = f"{r['Importance_Pct']:.2f}\\%"
        tab4 += f"{cat} & \\texttt{{{f_name_esc}}} & {gain} & {pct} \\\\\n"
    tab4 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(out_dir / "tab4_feature_importance.tex", "w", encoding="utf-8") as f:
        f.write(tab4)

    # Table 5: Zonal 26 Kecamatan
    df_zonal = pd.read_csv(data_dir / "zonal_stats_26_kecamatan_25yr.csv")
    tab5 = r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{Statistik Zonal Presipitasi Tahunan 25 Tahun (2001--2025) pada 26 Kecamatan Kabupaten Kebumen}
\label{tab:zonal_climatology_26kec}
\begin{tabular}{clcccccc}
\hline
\textbf{No} & \textbf{Kecamatan} & \textbf{Luas ($km^2$)} & \textbf{Rata-rata (mm)} & \textbf{Min (mm)} & \textbf{Max (mm)} & \textbf{CV (\%)} & \textbf{Tren (mm/thn)} \\
\hline
"""
    for no, (_, r) in enumerate(df_zonal.iterrows(), 1):
        area_km2 = r['Luas_Piksel_250m'] * 0.0625  # tiap piksel 250m x 250m = 0.0625 km2
        tab5 += f"{no} & {r['Kecamatan']} & {area_km2:.1f} & {r['Mean_Tahunan_mm']:.1f} & {r['Min_Tahunan_mm']:.1f} & {r['Max_Tahunan_mm']:.1f} & {r['CV_Pct']:.1f} & {r['Sens_Slope_mm_thn']:+.2f} \\\\\n"
    tab5 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(out_dir / "tab5_zonal_climatology_26kec.tex", "w", encoding="utf-8") as f:
        f.write(tab5)

    # Table 6: Extreme ENSO Impact
    df_enso = pd.read_csv(data_dir / "extreme_enso_zonal_anomalies.csv")
    tab6 = r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{Dampak dan Kerentanan Anomali Presipitasi Kejadian Iklim Ekstrem ENSO per Kecamatan}
\label{tab:extreme_enso_impact}
\begin{tabular}{lcccccc}
\hline
\textbf{Kecamatan} & \textbf{Normal 25th} & \textbf{La Niña 2010} & \textbf{Ano 2010 (\%)} & \textbf{El Niño 2015} & \textbf{Ano 2015 (\%)} & \textbf{Rentang Kontras (mm)} \\
\hline
"""
    for _, r in df_enso.head(15).iterrows():  # 15 kecamatan teratas
        tab6 += f"{r['Kecamatan']} & {r['Baseline_25yr_mm']:.1f} & {r['LaNina_2010_mm']:.1f} & {r['Ano_2010_Pct']:+.1f}\\% & {r['ElNino_2015_mm']:.1f} & {r['Ano_2015_Pct']:+.1f}\\% & {r['Contrast_Range_mm']:.1f} \\\\\n"
    tab6 += r"""\hline
\end{tabular}
\end{table}
"""
    with open(out_dir / "tab6_extreme_enso_impact.tex", "w", encoding="utf-8") as f:
        f.write(tab6)

    print("✓ Seluruh 6 tabel LaTeX berhasil diekspor ke documents/tables/!")

if __name__ == "__main__":
    run()
