"""
GSMaP vs CHIRPS Comparative Downscaling - Phase 5A: Publication Figures (16:9, 300 DPI)
======================================================================================
Generates 8 high-impact scientific figures formatted strictly according to Rule 8:
- 16:9 Aspect Ratio (figsize=(16, 9), 300 DPI)
- Clean, publication-grade aesthetics
- Output: documents/penelitian_downscaling_gsmap_vs_chirps/figures/
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from rasterio.transform import Affine

def set_style():
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 16

def run():
    t0 = time.time()
    set_style()
    print("=" * 85)
    print("🚀 FASE 5A: GENERASI 8 GAMBAR PUBLIKASI 16:9 (GSMaP vs CHIRPS DOWNSCALING)")
    print("=" * 85)

    fig_dir = Path("documents/penelitian_downscaling_gsmap_vs_chirps/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path("data/processed_gsmap_vs_chirps")

    # Load Data
    print("\n[Memuat Data Analitik...]")
    df_ts = pd.read_csv(out_dir / "comparative_timeseries_25yr.csv")
    df_st = pd.read_csv(out_dir / "station_metadata_30.csv")
    df_acc = pd.read_csv(out_dir / "algorithm_accuracy_gsmap_vs_chirps.csv")
    df_transect = pd.read_csv(out_dir / "orographic_transect_profile.csv")
    df_enso = pd.read_csv(out_dir / "extreme_enso_comparative_anomalies.csv")
    df_zonal = pd.read_csv(out_dir / "zonal_stats_gsmap_vs_chirps_26kec.csv")

    npz_data = np.load(out_dir / "reconstructed_gsmap_vs_chirps_250m.npz")
    npz_terrain = np.load("data/processed_25yr/terrain_covariates_250m.npz")

    mean_ann_g = npz_data['mean_ann_g']
    mean_ann_c = npz_data['mean_ann_c']
    mean_ann_f = npz_data['mean_ann_f']
    diff_ann_mm = npz_data['diff_ann_mm']
    diff_ann_pct = npz_data['diff_ann_pct']
    annual_g = npz_data['annual_g']
    annual_c = npz_data['annual_c']
    monthly_g = npz_data['monthly_g']
    monthly_c = npz_data['monthly_c']
    mask_keb = npz_data['mask_kebumen']
    years = list(npz_data['years'])

    elev = npz_terrain['elev']
    xs = npz_terrain['xs']
    ys = npz_terrain['ys']
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]

    gdf_kec = gpd.read_file("33.05_kecamatan.geojson")
    gdf_kec['geometry'] = gdf_kec['geometry'].apply(shapely.force_2d)

    # -------------------------------------------------------------
    # FIG 1: Karakteristik Sensor Satelit & Setting Fisiografi Kebumen
    # -------------------------------------------------------------
    print("\n[Membuat Fig 1] fig1_sensors_characteristics.png...")
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.18)

    # Panel A: DEM & Stasiun Validasi Kebumen
    ax1 = fig.add_subplot(gs[0])
    elev_masked = np.where(mask_keb, elev, np.nan)
    im1 = ax1.imshow(elev_masked, extent=extent, cmap='terrain', vmin=0, vmax=650, origin='upper')
    gdf_kec.boundary.plot(ax=ax1, color='#222222', linewidth=0.7, alpha=0.7)
    ax1.scatter(df_st['longitude'], df_st['latitude'], color='red', edgecolor='white', s=60, linewidth=1.2, label='30 Stasiun Validasi (BMKG/PUSAIR)', zorder=5)
    ax1.set_title("(a) Fisiografi Topografi Kebumen & Jaringan Stasiun Validasi", fontweight='bold')
    ax1.set_xlabel("Bujur (Longitude °E)")
    ax1.set_ylabel("Lintang (Latitude °S)")
    cb1 = fig.colorbar(im1, ax=ax1, orientation='horizontal', fraction=0.046, pad=0.08)
    cb1.set_label("Elevasi Digital Elevation Model (m dpl)")
    ax1.legend(loc='lower left', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Panel B: Perbandingan Spektral & Konseptual Satelit
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    col_labels = ['Parameter Teknis', 'CHIRPS v2.0 (USGS/CHC)', 'GSMaP v8 (JAXA/EORC)']
    cell_data = [
        ['Konstelasi Satelit', 'Meteosat, GOES, GMS/MTSAT', 'GPM Core Observatory, TRMM, DMSP'],
        ['Instrumen Sensor', 'Thermal Infrared (TIR) 10.8 µm', 'Passive Microwave (PMW) + TIR Morphing'],
        ['Prinsip Deteksi', 'Suhu Puncak Awan Dingin (CCD)', 'Direct Hydrometeor Emission & Scattering'],
        ['Resolusi Spasial Asli', '0.05° (~5.5 km)', '0.10° (~11.1 km)'],
        ['Resolusi Temporal Asli', 'Harian / Pentad / Bulanan', 'Per Jam (Hourly) / Harian'],
        ['Metode Kalibrasi', 'Regresi Stasiun Stasioner Global', 'Dual-Frequency Precip Radar (DPR)'],
        ['Sensitivitas Awan Tipis', 'Rentan Overestimasi Awan Cirrus', 'Kebal Cirrus (Mendeteksi Tetes Es/Air)'],
        ['Sensitivitas Hujan Hangat', 'Rentan Underestimate Awan Rendah', 'Menangkap Emisi Lapisan Rendah'],
        ['Grid Target Downscale', '250 m x 250 m (Tersinkronisasi)', '250 m x 250 m (Tersinkronisasi)'],
        ['Domain Penelitian', 'Kabupaten Kebumen (2001–2025)', 'Kabupaten Kebumen (2001–2025)']
    ]
    table = ax2.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.95)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1f4e79')
        else:
            if row % 2 == 1:
                cell.set_facecolor('#f2f4f8')
            else:
                cell.set_facecolor('#ffffff')
    ax2.set_title("(b) Perbandingan Karakteristik Fisik Sensor: Inframerah vs Gelombang Mikro", fontweight='bold', pad=15)

    plt.suptitle("Gambar 1: Kerangka Fisiografi Kebumen dan Komparasi Sensor Presipitasi Satelit (CHIRPS vs GSMaP)", fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(fig_dir / "fig1_sensors_characteristics.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Selesai Fig 1")

    # -------------------------------------------------------------
    # FIG 2: Scatter Plot & Timeseries Correlation (300 Bulan)
    # -------------------------------------------------------------
    print("\n[Membuat Fig 2] fig2_scatter_timeseries_correlation.png...")
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.0, 1.4], height_ratios=[1.2, 0.8], wspace=0.18, hspace=0.28)

    # Panel A: Scatter Plot
    ax1 = fig.add_subplot(gs[:, 0])
    season_colors = {
        'DJF (Hujan)': '#1f77b4',
        'MAM (Peralihan)': '#2ca02c',
        'JJA (Kemarau)': '#d62728',
        'SON (Peralihan)': '#ff7f0e'
    }
    for s_name, color in season_colors.items():
        s_data = df_ts[df_ts['season'] == s_name]
        ax1.scatter(s_data['chirps_raw_mm'], s_data['gsmap_raw_mm'], color=color, alpha=0.75, s=45, label=s_name, edgecolors='none')

    max_val = max(df_ts['chirps_raw_mm'].max(), df_ts['gsmap_raw_mm'].max()) + 30
    ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Garis 1:1 (Ideal)')
    # Trendline
    m, b = np.polyfit(df_ts['chirps_raw_mm'], df_ts['gsmap_raw_mm'], 1)
    x_vals = np.linspace(0, max_val, 100)
    ax1.plot(x_vals, m * x_vals + b, color='darkblue', linewidth=2.0, label=f'Regresi Linier (y = {m:.2f}x + {b:.1f})')

    r_corr = df_ts['chirps_raw_mm'].corr(df_ts['gsmap_raw_mm'])
    rho_corr = df_ts['chirps_raw_mm'].corr(df_ts['gsmap_raw_mm'], method='spearman')
    mean_bias = df_ts['diff_mm'].mean()

    stats_text = (
        f"Korelasi Pearson (r): {r_corr:.3f}\n"
        f"Korelasi Spearman (ρ): {rho_corr:.3f}\n"
        f"Rerata Bias (GSMaP - CHIRPS): {mean_bias:.1f} mm/bln\n"
        f"Rasio Rerata GSMaP/CHIRPS: {df_ts['ratio_gsmap_chirps'].mean():.3f}\n"
        f"Total Sampel: 300 Bulan (2001–2025)"
    )
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#ffffea', edgecolor='#cccccc', alpha=0.95), fontsize=10)

    ax1.set_title("(a) Scatter Plot Presipitasi Bulanan: CHIRPS vs GSMaP (2001–2025)", fontweight='bold')
    ax1.set_xlabel("Presipitasi CHIRPS (mm/bulan)")
    ax1.set_ylabel("Presipitasi GSMaP (mm/bulan)")
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.legend(loc='lower right', framealpha=0.9, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Panel B: Timeseries 25 Tahun
    ax2 = fig.add_subplot(gs[0, 1])
    time_idx = np.arange(len(df_ts))
    ax2.plot(time_idx, df_ts['chirps_raw_mm'], color='#1f77b4', linewidth=1.4, label='CHIRPS (Inframerah / CCD)', alpha=0.85)
    ax2.plot(time_idx, df_ts['gsmap_raw_mm'], color='#d62728', linewidth=1.4, label='GSMaP (Gelombang Mikro / PMW)', alpha=0.85)
    ax2.fill_between(time_idx, df_ts['chirps_raw_mm'], df_ts['gsmap_raw_mm'], color='gray', alpha=0.25, label='Discrepancy (Selisih)')
    ax2.set_title("(b) Dinamika Deret Waktu Presipitasi Bulanan 25 Tahun (Kebumen)", fontweight='bold')
    ax2.set_ylabel("Curah Hujan (mm/bulan)")
    # X-Ticks every 5 years
    tick_locs = [i * 60 for i in range(6)]
    tick_labels = [str(2001 + i * 5) for i in range(6)]
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels(tick_labels)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # Panel C: Discrepancy & 12-Month Rolling Mean
    ax3 = fig.add_subplot(gs[1, 1])
    diff_series = df_ts['diff_mm']
    rolling_diff = diff_series.rolling(12, center=True).mean()
    ax3.bar(time_idx, diff_series, color=np.where(diff_series >= 0, '#2ca02c', '#d62728'), alpha=0.55, width=1.0, label='Selisih Bulanan (GSMaP - CHIRPS)')
    ax3.plot(time_idx, rolling_diff, color='black', linewidth=2.0, label='Tren 12-Bulan (Rata-rata Bergerak)')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=1.0)
    ax3.set_title("(c) Anomali Discrepancy Bulanan (GSMaP Menunjukkan Underestimate Sistematis)", fontweight='bold')
    ax3.set_xlabel("Tahun Pengamatan")
    ax3.set_ylabel("Selisih (mm/bulan)")
    ax3.set_xticks(tick_locs)
    ax3.set_xticklabels(tick_labels)
    ax3.legend(loc='lower right', framealpha=0.9, fontsize=9.5)
    ax3.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle("Gambar 2: Korelasi Temporal dan Analisis Discrepancy Presipitasi Satelit CHIRPS vs GSMaP (2001–2025)", fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(fig_dir / "fig2_scatter_timeseries_correlation.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Selesai Fig 2")

    # -------------------------------------------------------------
    # FIG 3: Model Accuracy Comparison (KGE, RMSE, R2, PBIAS)
    # -------------------------------------------------------------
    print("\n[Membuat Fig 3] fig3_model_accuracy_comparison.png...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9), dpi=300)
    plt.subplots_adjust(wspace=0.18, hspace=0.28)

    algo_keys = ['prism', 'anusplin', 'rk', 'xgboost', 'lightgbm']
    algo_labels = ['PRISM', 'ANUSPLIN', 'Regression-Kriging', 'Spatial XGBoost', 'LightGBM']
    
    # Filter accuracies
    acc_c = df_acc[df_acc['Sensor_Source'] == 'CHIRPS'].set_index('Algorithm').loc[algo_keys]
    acc_g = df_acc[df_acc['Sensor_Source'] == 'GSMAP'].set_index('Algorithm').loc[algo_keys]
    acc_f = df_acc[df_acc['Sensor_Source'] == 'FUSED'].iloc[0]

    x = np.arange(len(algo_keys))
    w = 0.35

    # Panel 1: KGE
    ax1.bar(x - w/2, acc_c['KGE'], width=w, label='CHIRPS Downscaled', color='#1f77b4', edgecolor='black')
    ax1.bar(x + w/2, acc_g['KGE'], width=w, label='GSMaP Downscaled', color='#d62728', edgecolor='black')
    ax1.axhline(acc_f['KGE'], color='#2ca02c', linestyle='--', linewidth=2.0, label=f"Fused XGBoost (KGE={acc_f['KGE']:.4f})")
    ax1.set_title("(a) Kling-Gupta Efficiency (KGE) - Semakin Mendekati 1.0 Semakin Unggul", fontweight='bold')
    ax1.set_ylabel("Skor KGE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(algo_labels, rotation=15)
    ax1.set_ylim(0.90, 1.005)
    ax1.legend(loc='lower right', framealpha=0.9, fontsize=9.5)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Panel 2: RMSE
    ax2.bar(x - w/2, acc_c['RMSE_mm'], width=w, label='CHIRPS Downscaled', color='#1f77b4', edgecolor='black')
    ax2.bar(x + w/2, acc_g['RMSE_mm'], width=w, label='GSMaP Downscaled', color='#d62728', edgecolor='black')
    ax2.axhline(acc_f['RMSE_mm'], color='#2ca02c', linestyle='--', linewidth=2.0, label=f"Fused XGBoost (RMSE={acc_f['RMSE_mm']:.2f} mm)")
    ax2.set_title("(b) Root Mean Square Error (RMSE) - Semakin Rendah Semakin Presisi", fontweight='bold')
    ax2.set_ylabel("RMSE (mm/bulan)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(algo_labels, rotation=15)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # Panel 3: R2
    ax3.bar(x - w/2, acc_c['R2'], width=w, label='CHIRPS Downscaled', color='#1f77b4', edgecolor='black')
    ax3.bar(x + w/2, acc_g['R2'], width=w, label='GSMaP Downscaled', color='#d62728', edgecolor='black')
    ax3.axhline(acc_f['R2'], color='#2ca02c', linestyle='--', linewidth=2.0, label=f"Fused XGBoost (R²={acc_f['R2']:.4f})")
    ax3.set_title("(c) Koefisien Determinasi (R²) - Variansi yang Mampu Dijelaskan", fontweight='bold')
    ax3.set_ylabel("Skor R²")
    ax3.set_xticks(x)
    ax3.set_xticklabels(algo_labels, rotation=15)
    ax3.set_ylim(0.95, 1.002)
    ax3.legend(loc='lower right', framealpha=0.9, fontsize=9.5)
    ax3.grid(True, linestyle='--', alpha=0.4)

    # Panel 4: Percent Bias (PBIAS)
    ax4.bar(x - w/2, acc_c['PBIAS_pct'], width=w, label='CHIRPS Downscaled', color='#1f77b4', edgecolor='black')
    ax4.bar(x + w/2, acc_g['PBIAS_pct'], width=w, label='GSMaP Downscaled', color='#d62728', edgecolor='black')
    ax4.axhline(acc_f['PBIAS_pct'], color='#2ca02c', linestyle='--', linewidth=2.0, label=f"Fused XGBoost (PBIAS={acc_f['PBIAS_pct']:.2f}%)")
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_title("(d) Percent Bias (PBIAS %) - Deviasi Total Akumulasi Volume", fontweight='bold')
    ax4.set_ylabel("PBIAS (%)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(algo_labels, rotation=15)
    ax4.legend(loc='lower right', framealpha=0.9, fontsize=9.5)
    ax4.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle("Gambar 3: Benchmark Akurasi Spatial Cross-Validation Algoritma Downscaling: CHIRPS vs GSMaP vs Fused", fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(fig_dir / "fig3_model_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Selesai Fig 3")

    # -------------------------------------------------------------
    # FIG 4: Spatial Climatology Comparison (GSMaP vs CHIRPS vs Diff)
    # -------------------------------------------------------------
    print("\n[Membuat Fig 4] fig4_spatial_climatology_comparison.png...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 9), dpi=300)
    plt.subplots_adjust(wspace=0.15, top=0.88, bottom=0.12)

    g_masked = np.where(mask_keb, mean_ann_g, np.nan)
    c_masked = np.where(mask_keb, mean_ann_c, np.nan)
    diff_masked = np.where(mask_keb, diff_ann_mm, np.nan)

    vmin_rain = 2200
    vmax_rain = 3600
    norm_rain = Normalize(vmin=vmin_rain, vmax=vmax_rain)

    # Panel A: GSMaP 250m
    im1 = ax1.imshow(g_masked, extent=extent, cmap='YlGnBu', norm=norm_rain, origin='upper')
    gdf_kec.boundary.plot(ax=ax1, color='#333333', linewidth=0.6, alpha=0.7)
    ax1.set_title("(a) GSMaP Downscaled 250m\n(Rerata: 2.555,1 mm/tahun)", fontweight='bold')
    ax1.set_xlabel("Bujur (°E)")
    ax1.set_ylabel("Lintang (°S)")
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Panel B: CHIRPS 250m
    im2 = ax2.imshow(c_masked, extent=extent, cmap='YlGnBu', norm=norm_rain, origin='upper')
    gdf_kec.boundary.plot(ax=ax2, color='#333333', linewidth=0.6, alpha=0.7)
    ax2.set_title("(b) CHIRPS Downscaled 250m\n(Rerata: 3.287,9 mm/tahun)", fontweight='bold')
    ax2.set_xlabel("Bujur (°E)")
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Panel C: Spatial Difference (GSMaP - CHIRPS)
    norm_diff = TwoSlopeNorm(vcenter=0, vmin=-1100, vmax=200)
    im3 = ax3.imshow(diff_masked, extent=extent, cmap='coolwarm', norm=norm_diff, origin='upper')
    gdf_kec.boundary.plot(ax=ax3, color='#333333', linewidth=0.6, alpha=0.7)
    ax3.set_title("(c) Selisih Spasial (GSMaP - CHIRPS)\n(Rerata: -732,8 mm/tahun [-22,3%])", fontweight='bold')
    ax3.set_xlabel("Bujur (°E)")
    ax3.grid(True, linestyle='--', alpha=0.3)

    # Colorbars
    cb_rain = fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', fraction=0.046, pad=0.12)
    cb_rain.set_label("Presipitasi Rata-rata Tahunan 25 Tahun (mm/tahun)")

    cb_diff = fig.colorbar(im3, ax=ax3, orientation='horizontal', fraction=0.046, pad=0.12)
    cb_diff.set_label("Selisih Curah Hujan (mm/tahun)")

    plt.suptitle("Gambar 4: Distribusi Spasial Klimatologi Presipitasi 25 Tahun (2001–2025) pada Resolusi 250m", fontsize=15, fontweight='bold', y=0.96)
    fig.savefig(fig_dir / "fig4_spatial_climatology_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Selesai Fig 4")

    # -------------------------------------------------------------
    # FIG 5: Orographic Transect Profile (North-South)
    # -------------------------------------------------------------
    print("\n[Membuat Fig 5] fig5_orographic_transect_profile.png...")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 9), dpi=300, sharex=True, gridspec_kw={'height_ratios': [1.0, 1.3]})
    plt.subplots_adjust(hspace=0.12)

    lats = df_transect['latitude']

    # Upper panel: Topography Elevation Profile
    ax_top.fill_between(lats, df_transect['elevation_m'], color='#8c510a', alpha=0.35)
    ax_top.plot(lats, df_transect['elevation_m'], color='#543005', linewidth=2.0, label='Elevasi DEM (m dpl)')
    ax_top.set_title("(a) Profil Morfometri dan Elevasi Transekt Utara-Selatan Kebumen (Bujur 109.65°E)", fontweight='bold')
    ax_top.set_ylabel("Elevasi (m dpl)")
    ax_top.set_ylim(0, 600)
    ax_top.grid(True, linestyle='--', alpha=0.4)
    ax_top.legend(loc='upper right', framealpha=0.9)

    # Annotations for zones
    ax_top.annotate("Zona Pegunungan Utara\n(Kec. Sadang / Karangsambung)", xy=(-7.53, 480), xytext=(-7.56, 520),
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontweight='bold', fontsize=10)
    ax_top.annotate("Dataran Aluvial Tengah\n(Kec. Kebumen / Pejagoan)", xy=(-7.67, 30), xytext=(-7.66, 250),
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontweight='bold', fontsize=10)
    ax_top.annotate("Garis Pantai Selatan\n(Samudera Hindia)", xy=(-7.76, 5), xytext=(-7.75, 180),
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2), fontweight='bold', fontsize=10)

    # Lower panel: Rainfall profiles
    ax_bot.plot(lats, df_transect['chirps_mm'], color='#1f77b4', linewidth=2.5, label='CHIRPS Downscaled 250m (TIR/CCD)')
    ax_bot.plot(lats, df_transect['gsmap_mm'], color='#d62728', linewidth=2.5, label='GSMaP Downscaled 250m (PMW/GPM)')
    ax_bot.plot(lats, df_transect['fused_mm'], color='#2ca02c', linewidth=2.2, linestyle='--', label='Fused Multi-Sensor Ensemble 250m')
    ax_bot.fill_between(lats, df_transect['chirps_mm'], df_transect['gsmap_mm'], color='gray', alpha=0.2, label='Discrepancy (Selisih Satelit)')

    ax_bot.set_title("(b) Gradien Orografis Presipitasi Tahunan Sepanjang Profil Transekt", fontweight='bold')
    ax_bot.set_xlabel("Lintang Geografis (Latitude °S) [Utara ➔ Selatan]")
    ax_bot.set_ylabel("Curah Hujan Tahunan (mm/tahun)")
    ax_bot.set_xlim(lats.min(), lats.max())
    ax_bot.grid(True, linestyle='--', alpha=0.4)
    ax_bot.legend(loc='lower left', framealpha=0.9, fontsize=10)

    plt.suptitle("Gambar 5: Analisis Transekt Orografis Utara-Selatan: Pengangkatan Orografis Sadang ke Dataran Pesisir", fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(fig_dir / "fig5_orographic_transect_profile.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Selesai Fig 5")

    # -------------------------------------------------------------
    # FIG 6: Seasonal Bias Grid (DJF, MAM, JJA, SON)
    # -------------------------------------------------------------
    print("\n[Membuat Fig 6] fig6_seasonal_bias_grid.png...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)
    plt.subplots_adjust(wspace=0.18, hspace=0.25, top=0.90, bottom=0.10)

    seasons_meta = [
        ('DJF', 'Puncak Musim Hujan (Des-Jan-Feb)', [11, 0, 1], axes[0, 0]),
        ('MAM', 'Peralihan Barat-Timur (Mar-Apr-Mei)', [2, 3, 4], axes[0, 1]),
        ('JJA', 'Musim Kemarau (Jun-Jul-Agu)', [5, 6, 7], axes[1, 0]),
        ('SON', 'Peralihan Timur-Barat (Sep-Okt-Nov)', [8, 9, 10], axes[1, 1])
    ]

    norm_seas = TwoSlopeNorm(vcenter=0, vmin=-160, vmax=40)

    for code, title_s, m_indices, ax_s in seasons_meta:
        seas_g = np.mean(monthly_g[m_indices], axis=0)
        seas_c = np.mean(monthly_c[m_indices], axis=0)
        seas_diff = seas_g - seas_c
        seas_diff_masked = np.where(mask_keb, seas_diff, np.nan)
        mean_diff_val = np.nanmean(seas_diff_masked)

        im_s = ax_s.imshow(seas_diff_masked, extent=extent, cmap='coolwarm', norm=norm_seas, origin='upper')
        gdf_kec.boundary.plot(ax=ax_s, color='#333333', linewidth=0.5, alpha=0.7)
        ax_s.set_title(f"({code}) {title_s}\nRerata Selisih: {mean_diff_val:.1f} mm/bulan", fontweight='bold', fontsize=11)
        ax_s.set_xlabel("Bujur (°E)")
        ax_s.set_ylabel("Lintang (°S)")
        ax_s.grid(True, linestyle='--', alpha=0.3)

    cb_seas = fig.colorbar(im_s, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.038, pad=0.08)
    cb_seas.set_label("Selisih Presipitasi Rata-rata Musiman: GSMaP - CHIRPS (mm/bulan)")

    plt.suptitle("Gambar 6: Variasi Musiman Discrepancy Spasial (GSMaP - CHIRPS) Sepanjang 4 Pola Siklus Iklim Tropis", fontsize=15, fontweight='bold', y=0.97)
    fig.savefig(fig_dir / "fig6_seasonal_bias_grid.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Selesai Fig 6")

    # -------------------------------------------------------------
    # FIG 7: Extreme ENSO Response (La Nina 2010 vs El Nino 2015/2023)
    # -------------------------------------------------------------
    print("\n[Membuat Fig 7] fig7_enso_extreme_response.png...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
    plt.subplots_adjust(wspace=0.15, hspace=0.22, top=0.88, bottom=0.12)

    events_meta = [
        (2010, 'Super La Niña (2010)', axes[0, 0], axes[1, 0]),
        (2015, 'Super El Niño (2015)', axes[0, 1], axes[1, 1]),
        (2023, 'Strong El Niño (2023)', axes[0, 2], axes[1, 2])
    ]

    norm_ano = TwoSlopeNorm(vcenter=0, vmin=-60, vmax=60)

    for yr, event_name, ax_top, ax_bot in events_meta:
        idx = years.index(yr)
        ano_c_pct = ((annual_c[idx] - mean_ann_c) / mean_ann_c) * 100.0
        ano_g_pct = ((annual_g[idx] - mean_ann_g) / mean_ann_g) * 100.0

        ano_c_masked = np.where(mask_keb, ano_c_pct, np.nan)
        ano_g_masked = np.where(mask_keb, ano_g_pct, np.nan)

        im_c = ax_top.imshow(ano_c_masked, extent=extent, cmap='BrBG', norm=norm_ano, origin='upper')
        gdf_kec.boundary.plot(ax=ax_top, color='#333333', linewidth=0.5, alpha=0.7)
        ax_top.set_title(f"CHIRPS: {event_name}\n(Anomali: {np.nanmean(ano_c_masked):+.1f}%)", fontweight='bold', fontsize=11)
        ax_top.set_ylabel("Lintang (°S)")
        ax_top.grid(True, linestyle='--', alpha=0.3)

        im_g = ax_bot.imshow(ano_g_masked, extent=extent, cmap='BrBG', norm=norm_ano, origin='upper')
        gdf_kec.boundary.plot(ax=ax_bot, color='#333333', linewidth=0.5, alpha=0.7)
        ax_bot.set_title(f"GSMaP: {event_name}\n(Anomali: {np.nanmean(ano_g_masked):+.1f}%)", fontweight='bold', fontsize=11)
        ax_bot.set_xlabel("Bujur (°E)")
        ax_bot.set_ylabel("Lintang (°S)")
        ax_bot.grid(True, linestyle='--', alpha=0.3)

    cb_ano = fig.colorbar(im_c, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.040, pad=0.09)
    cb_ano.set_label("Persentase Anomali Presipitasi Relatif terhadap Klimatologi 25 Tahun (%)")

    plt.suptitle("Gambar 7: Respon Spasial Satelit terhadap Kejadian Iklim Ekstrem ENSO (La Niña 2010 vs El Niño 2015 & 2023)", fontsize=15, fontweight='bold', y=0.97)
    fig.savefig(fig_dir / "fig7_enso_extreme_response.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Selesai Fig 7")

    # -------------------------------------------------------------
    # FIG 8: Zonal Kecamatan Differences (26 Kecamatan)
    # -------------------------------------------------------------
    print("\n[Membuat Fig 8] fig8_zonal_kecamatan_differences.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=300, gridspec_kw={'width_ratios': [1.2, 0.9]})
    plt.subplots_adjust(wspace=0.22, top=0.92, bottom=0.08)

    y_pos = np.arange(len(df_zonal))
    bar_h = 0.28

    # Panel A: Absolute Rainfall GSMaP vs CHIRPS vs Fused
    ax1.barh(y_pos - bar_h, df_zonal['CHIRPS_Mean_mm'], height=bar_h, color='#1f77b4', label='CHIRPS (TIR)', edgecolor='black', alpha=0.85)
    ax1.barh(y_pos, df_zonal['GSMaP_Mean_mm'], height=bar_h, color='#d62728', label='GSMaP (PMW)', edgecolor='black', alpha=0.85)
    ax1.barh(y_pos + bar_h, df_zonal['Fused_Mean_mm'], height=bar_h, color='#2ca02c', label='Fused Multi-Sensor', edgecolor='black', alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_zonal['Kecamatan'], fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Rata-rata Presipitasi Tahunan (mm/tahun)")
    ax1.set_title("(a) Perbandingan Volume Presipitasi Tahunan per Kecamatan", fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.9, fontsize=9.5)
    ax1.grid(True, linestyle='--', alpha=0.4, axis='x')

    # Panel B: Discrepancy Percentage (GSMaP vs CHIRPS)
    diff_pcts = df_zonal['Diff_Pct']
    ax2.barh(y_pos, diff_pcts, height=0.6, color='#b2182b', edgecolor='black', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.axvline(diff_pcts.mean(), color='black', linestyle='--', linewidth=1.5, label=f"Rata-rata Selisih ({diff_pcts.mean():.1f}%)")
    ax2.set_xlabel("Persentase Selisih [(GSMaP - CHIRPS) / CHIRPS %]")
    ax2.set_title("(b) Konsistensi Defisit Estimasi GSMaP terhadap CHIRPS", fontweight='bold')
    ax2.set_xlim(-26, -18)
    ax2.legend(loc='lower right', framealpha=0.9, fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='x')

    plt.suptitle("Gambar 8: Analisis Statistik Zonal Komparatif 26 Kecamatan di Kabupaten Kebumen", fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(fig_dir / "fig8_zonal_kecamatan_differences.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Selesai Fig 8")

    print("\n" + "=" * 85)
    print(f"🎉 FASE 5A BERHASIL: 8 GAMBAR PUBLIKASI 16:9 SELESAI DALAM {time.time() - t0:.2f} DETIK!")
    print("=" * 85)

if __name__ == "__main__":
    run()
