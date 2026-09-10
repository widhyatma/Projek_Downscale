"""
GSMaP vs CHIRPS Comparative Downscaling - Phase 5B: LaTeX Tables & Data Packaging
=================================================================================
Generates 6 publication-ready LaTeX tables and copies analytic CSV results to:
documents/penelitian_downscaling_gsmap_vs_chirps/tables/
documents/penelitian_downscaling_gsmap_vs_chirps/results_data/
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

def run():
    t0 = time.time()
    print("=" * 85)
    print("🚀 FASE 5B: GENERASI 6 TABEL LATEX & PACKAGING DATA HASIL PENELITIAN")
    print("=" * 85)

    tab_dir = Path("documents/penelitian_downscaling_gsmap_vs_chirps/tables")
    res_dir = Path("documents/penelitian_downscaling_gsmap_vs_chirps/results_data")
    src_dir = Path("data/processed_gsmap_vs_chirps")
    tab_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy Analytical CSVs to results_data/
    csv_files = [
        "comparative_timeseries_25yr.csv",
        "station_metadata_30.csv",
        "algorithm_accuracy_gsmap_vs_chirps.csv",
        "orographic_transect_profile.csv",
        "extreme_enso_comparative_anomalies.csv",
        "zonal_stats_gsmap_vs_chirps_26kec.csv"
    ]
    print("\n[Langkah 1/2] Menyalin File Data Analitik CSV ke results_data/...")
    for f in csv_files:
        src_f = src_dir / f
        dst_f = res_dir / f
        if src_f.exists():
            shutil.copy2(src_f, dst_f)
            print(f"  ✓ Tersalin: {f}")

    # Load Data
    df_acc = pd.read_csv(src_dir / "algorithm_accuracy_gsmap_vs_chirps.csv")
    df_ts = pd.read_csv(src_dir / "comparative_timeseries_25yr.csv")
    df_enso = pd.read_csv(src_dir / "extreme_enso_comparative_anomalies.csv")
    df_zonal = pd.read_csv(src_dir / "zonal_stats_gsmap_vs_chirps_26kec.csv")
    df_transect = pd.read_csv(src_dir / "orographic_transect_profile.csv")

    # -------------------------------------------------------------
    # TABEL 1: Spesifikasi Sensor Presipitasi & Data Pendukung
    # -------------------------------------------------------------
    print("\n[Langkah 2/2] Menghasilkan 6 Tabel LaTeX...")
    t1_content = r"""\begin{table}[htbp]
\centering
\caption{Spesifikasi Teknis Sensor Presipitasi Satelit dan Kovariat Lingkungan}
\label{tab:sensor_specifications}
\resizebox{\linewidth}{!}{%
\begin{tabular}{p{3.2cm}p{3.8cm}p{4.2cm}p{3.2cm}}
\hline
\textbf{Atribut / Parameter} & \textbf{CHIRPS v2.0 (USGS/CHC)} & \textbf{GSMaP v8 (JAXA/EORC)} & \textbf{ERA5-Land (ECMWF)} \\
\hline
Konstelasi / Satelit & GEO Satellites (GOES, Meteosat) & GPM Core, TRMM, DMSP, NOAA & Reanalisis Global (IFS Cy45r1) \\
Domain Sensor & Thermal Infrared (TIR, 10.8 $\mu\text{m}$) & Passive Microwave (PMW) + TIR & Asimilasi Multi-Sensor Fisik \\
Prinsip Estimasi & Cold Cloud Duration (CCD, $T_b < 235\text{K}$) & Scattering/Emisi Hidrometeor & Persamaan Termodinamika Lahan \\
Resolusi Spasial Asli & $0.05^\circ \times 0.05^\circ$ ($\sim 5.5\text{ km}$) & $0.10^\circ \times 0.10^\circ$ ($\sim 11.1\text{ km}$) & $0.10^\circ \times 0.10^\circ$ ($\sim 9.0\text{ km}$) \\
Resolusi Temporal Asli & Bulanan / Pentad / Harian & Per Jam (Hourly) / Harian & Per Jam / Bulanan \\
Kalibrasi Permukaan & Regresi Stasiun Global (CHPclim) & Dual-Frequency Precip Radar (DPR) & Asimilasi Stasiun WMO Global \\
Sensitivitas Orografis & Rentan Underestimate Hujan Hangat & Deteksi Es \& Tetes Awan Curang & Represif terhadap Lembah Sempit \\
Target Downscale & $250\text{ m} \times 250\text{ m}$ (UTM 49S) & $250\text{ m} \times 250\text{ m}$ (UTM 49S) & Resampled Bilinear $250\text{ m}$ \\
Rentang Waktu & 25 Tahun (2001--2025, 300 bln) & 25 Tahun (2001--2025, 300 bln) & 25 Tahun (2001--2025, 300 bln) \\
\hline
\end{tabular}%
}
\end{table}
"""
    with open(tab_dir / "tab1_sensor_specifications.tex", "w", encoding="utf-8") as f:
        f.write(t1_content)
    print("  ✓ Selesai tab1_sensor_specifications.tex")

    # -------------------------------------------------------------
    # TABEL 2: Komparasi Akurasi 12 Konfigurasi Downscaling
    # -------------------------------------------------------------
    t2_rows = []
    for _, r in df_acc.iterrows():
        t2_rows.append(
            f"{r['Config_Name']} & {r['Sensor_Source']} & {r['KGE']:.4f} & {r['RMSE_mm']:.2f} & {r['MAE_mm']:.2f} & {r['R2']:.4f} & {r['PBIAS_pct']:+.2f}\\% \\\\"
        )
    t2_body = "\n".join(t2_rows)

    t2_tmpl = r"""\begin{table}[htbp]
\centering
\small
\caption{Perbandingan Kinerja Validasi Silang Spasial Berblok (\textit{Spatial Block K-Fold}, Buffer 5.000~m) Seluruh Konfigurasi Algoritma}
\label{tab:model_accuracy_comparison}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llccccc}
\hline
\textbf{Konfigurasi Algoritma} & \textbf{Sensor} & \textbf{KGE} & \textbf{RMSE (mm)} & \textbf{MAE (mm)} & \textbf{$R^2$} & \textbf{PBIAS (\%)} \\
\hline
__BODY__
\hline
\end{tabular}%
}
\end{table}
"""
    with open(tab_dir / "tab2_model_accuracy_comparison.tex", "w", encoding="utf-8") as f:
        f.write(t2_tmpl.replace("__BODY__", t2_body))
    print("  ✓ Selesai tab2_model_accuracy_comparison.tex")

    # -------------------------------------------------------------
    # TABEL 3: Dinamika Musiman Presipitasi & Bias
    # -------------------------------------------------------------
    seas_groups = df_ts.groupby('season')
    t3_rows = []
    for s_name, grp in seas_groups:
        c_mean = grp['chirps_raw_mm'].mean()
        g_mean = grp['gsmap_raw_mm'].mean()
        diff = grp['diff_mm'].mean()
        ratio = grp['ratio_gsmap_chirps'].mean()
        corr = grp['chirps_raw_mm'].corr(grp['gsmap_raw_mm'])
        t3_rows.append(
            f"{s_name} & {len(grp)} & {c_mean:.1f} & {g_mean:.1f} & {diff:+.1f} & {ratio:.3f} & {corr:.3f} \\\\"
        )
    # Total
    c_tot = df_ts['chirps_raw_mm'].mean()
    g_tot = df_ts['gsmap_raw_mm'].mean()
    d_tot = df_ts['diff_mm'].mean()
    r_tot = df_ts['ratio_gsmap_chirps'].mean()
    cr_tot = df_ts['chirps_raw_mm'].corr(df_ts['gsmap_raw_mm'])
    t3_rows.append(
        f"\\textbf{{Rata-rata 25 Tahun}} & \\textbf{{{len(df_ts)}}} & \\textbf{{{c_tot:.1f}}} & \\textbf{{{g_tot:.1f}}} & \\textbf{{{d_tot:+.1f}}} & \\textbf{{{r_tot:.3f}}} & \\textbf{{{cr_tot:.3f}}} \\\\"
    )
    t3_body = "\n".join(t3_rows)

    t3_tmpl = r"""\begin{table}[htbp]
\centering
\small
\caption[Dinamika Presipitasi Rata-rata Musiman CHIRPS vs GSMaP]{Dinamika Presipitasi Rata-rata Musiman dan Karakteristik \textit{Discrepancy} CHIRPS vs GSMaP (2001--2025)}
\label{tab:seasonal_climatology_comparison}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccccc}
\hline
\textbf{Musim Tropis} & \textbf{Jumlah Bulan} & \textbf{CHIRPS (mm)} & \textbf{GSMaP (mm)} & \textbf{Selisih (mm)} & \textbf{Rasio GSMaP/CHIRPS} & \textbf{Korelasi ($r$)} \\
\hline
__BODY__
\hline
\end{tabular}%
}
\end{table}
"""
    with open(tab_dir / "tab3_seasonal_climatology_comparison.tex", "w", encoding="utf-8") as f:
        f.write(t3_tmpl.replace("__BODY__", t3_body))
    print("  ✓ Selesai tab3_seasonal_climatology_comparison.tex")

    # -------------------------------------------------------------
    # TABEL 4: Respon Anomali Ekstrem ENSO (La Nina & El Nino)
    # -------------------------------------------------------------
    t4_rows = []
    for _, r in df_enso.iterrows():
        t4_rows.append(
            f"{r['Event']} & {r['GSMaP_Total_mm']:.1f} & {r['GSMaP_Ano_mm']:+.1f} ({r['GSMaP_Ano_Pct']:+.1f}\\%) & {r['CHIRPS_Total_mm']:.1f} & {r['CHIRPS_Ano_mm']:+.1f} ({r['CHIRPS_Ano_Pct']:+.1f}\\%) \\\\"
        )
    t4_body = "\n".join(t4_rows)

    t4_tmpl = r"""\begin{table}[htbp]
\centering
\small
\caption{Respon Komparatif Presipitasi Satelit terhadap Kejadian Iklim Ekstrem ENSO di Kabupaten Kebumen}
\label{tab:enso_extreme_anomalies}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccc}
\hline
\textbf{Kejadian Iklim Ekstrem} & \textbf{GSMaP Total (mm)} & \textbf{GSMaP Anomali} & \textbf{CHIRPS Total (mm)} & \textbf{CHIRPS Anomali} \\
\hline
__BODY__
\hline
\end{tabular}%
}
\end{table}
"""
    with open(tab_dir / "tab4_enso_extreme_anomalies.tex", "w", encoding="utf-8") as f:
        f.write(t4_tmpl.replace("__BODY__", t4_body))
    print("  ✓ Selesai tab4_enso_extreme_anomalies.tex")

    # -------------------------------------------------------------
    # TABEL 5: Statistik Zonal 26 Kecamatan (GSMaP vs CHIRPS vs Fused)
    # -------------------------------------------------------------
    t5_rows = []
    for idx, r in df_zonal.iterrows():
        t5_rows.append(
            f"{idx+1} & {r['Kecamatan']} & {r['GSMaP_Mean_mm']:.1f} & {r['CHIRPS_Mean_mm']:.1f} & {r['Fused_Mean_mm']:.1f} & {r['Diff_GSMaP_CHIRPS_mm']:+.1f} & {r['Diff_Pct']:+.1f}\\% & {r['Contrast_GSMaP_mm']:.1f} & {r['Contrast_CHIRPS_mm']:.1f} \\\\"
        )
    t5_body = "\n".join(t5_rows)

    t5_tmpl = r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{Statistik Zonal Rata-rata Presipitasi Tahunan (2001--2025) pada 26 Kecamatan di Kabupaten Kebumen}
\label{tab:zonal_kecamatan_comparison}
\resizebox{\linewidth}{!}{%
\begin{tabular}{clccccccc}
\hline
\textbf{No} & \textbf{Kecamatan} & \textbf{GSMaP (mm)} & \textbf{CHIRPS (mm)} & \textbf{Fused (mm)} & \textbf{Selisih (mm)} & \textbf{Selisih (\%)} & \textbf{Kontras GSMaP} & \textbf{Kontras CHIRPS} \\
\hline
__BODY__
\hline
\end{tabular}%
}
\end{table}
"""
    with open(tab_dir / "tab5_zonal_kecamatan_comparison.tex", "w", encoding="utf-8") as f:
        f.write(t5_tmpl.replace("__BODY__", t5_body))
    print("  ✓ Selesai tab5_zonal_kecamatan_comparison.tex")

    # -------------------------------------------------------------
    # TABEL 6: Karakteristik Zona Morfometri Sepanjang Transekt
    # -------------------------------------------------------------
    # Sesuai Rule 1: Di Belahan Bumi Selatan, lintang utara bernilai lebih tinggi (kurang negatif)
    z_north = df_transect[df_transect['latitude'] >= -7.56]
    z_trans = df_transect[(df_transect['latitude'] < -7.56) & (df_transect['latitude'] >= -7.63)]
    z_alluv = df_transect[(df_transect['latitude'] < -7.63) & (df_transect['latitude'] >= -7.72)]
    z_coast = df_transect[df_transect['latitude'] < -7.72]

    zones = [
        ("Zona Pegunungan Utara (Sadang/Karangsambung)", z_north),
        ("Zona Perbukitan Transisi (Alian/Pejagoan)", z_trans),
        ("Zona Dataran Aluvial Tengah (Kebumen/Kutowinangun)", z_alluv),
        ("Zona Pesisir Pantai Selatan (Samudera Hindia)", z_coast)
    ]

    t6_rows = []
    for z_name, z_df in zones:
        elev_m = z_df['elevation_m'].mean()
        g_m = z_df['gsmap_mm'].mean()
        c_m = z_df['chirps_mm'].mean()
        f_m = z_df['fused_mm'].mean()
        d_m = z_df['diff_mm'].mean()
        t6_rows.append(
            f"{z_name} & {elev_m:.0f} & {g_m:.1f} & {c_m:.1f} & {f_m:.1f} & {d_m:+.1f} \\\\"
        )
    t6_body = "\n".join(t6_rows)

    t6_tmpl = r"""\begin{table}[htbp]
\centering
\small
\caption[Karakteristik Zona Morfometri Transekt]{Karakteristik Gradien Presipitasi Sepanjang Profil Morfometri Transekt Utara--Selatan Kebumen}
\label{tab:orographic_transect_zones}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lccccc}
\hline
\textbf{Zona Morfologi Transekt} & \textbf{Elevasi Rerata (m)} & \textbf{GSMaP (mm)} & \textbf{CHIRPS (mm)} & \textbf{Fused (mm)} & \textbf{Selisih (mm)} \\
\hline
__BODY__
\hline
\end{tabular}%
}
\end{table}
"""
    with open(tab_dir / "tab6_orographic_transect_zones.tex", "w", encoding="utf-8") as f:
        f.write(t6_tmpl.replace("__BODY__", t6_body))
    print("  ✓ Selesai tab6_orographic_transect_zones.tex")

    print("\n" + "=" * 85)
    print(f"🎉 FASE 5B BERHASIL: 6 TABEL LATEX & DATA ANALITIK TERSUSUN DALAM {time.time() - t0:.2f} DETIK!")
    print("=" * 85)

if __name__ == "__main__":
    run()
