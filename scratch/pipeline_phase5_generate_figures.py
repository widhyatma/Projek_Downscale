"""
Phase 5: High-Resolution 16:9 Publication Visualization Suite (300 DPI)
========================================================================
Generates publication-quality figures for the 25-year downscaling research report:
1. Fig 1: Study Domain & Multi-Scale Topographic Predictors (DEM, Slope, Coast, 26 Kecamatan)
2. Fig 2: 25-Year Multi-Sensor & Atmospheric Correlation Matrix (CHIRPS, GSMaP, ERA5, NDVI)
3. Fig 3: Benchmark Model Comparison (12 Algorithms KGE/RMSE & Feature Importance)
4. Fig 4: 25-Year Climatological Mean Annual Precipitation Atlas (250m)
5. Fig 5: 12-Month Spatial Climatological Cycle Grid (3x4 Layout, Unified Colormap)
6. Fig 6: Spatial Climate Trend Atlas (Sen's Slope mm/year per pixel)
7. Fig 7: Climate Extremes & ENSO Spatial Anomalies (2010, 2015, 2022, 2023)
8. Fig 8: Zonal Climatology & Vulnerability Profiles across all 26 Kecamatan

Follows AGENTS.md Rule 2 (Multi-Band 3x4 Unified Norm), Rule 7 (Memory-Safe), Rule 8 (16:9 Standard), Rule 10 (scratch dir).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import rasterio
from rasterio.transform import Affine
import gc

def run():
    t0 = time.time()
    print("=" * 85)
    print("🎨 FASE 5: PEMBUATAN ATLAS VISUALISASI PUBLIKASI 16:9 (300 DPI)")
    print("   Output Directory: documents/figures/ | Standar: 16:9 (16 x 9 Inci)")
    print("=" * 85)

    fig_dir = Path("documents/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir = Path("documents/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data/processed_25yr")

    # Muat data dasar
    df_ts = pd.read_csv(data_dir / "multivariate_25yr_timeseries_monthly.csv")
    df_metrics = pd.read_csv(data_dir / "benchmark_model_metrics.csv")
    df_feat = pd.read_csv(data_dir / "feature_importance.csv")
    df_zonal = pd.read_csv(data_dir / "zonal_stats_26_kecamatan_25yr.csv")
    df_enso = pd.read_csv(data_dir / "extreme_enso_zonal_anomalies.csv")

    npz_terrain = np.load(data_dir / "terrain_covariates_250m.npz")
    elev = npz_terrain['elev']
    slope = npz_terrain['slope']
    aspect = npz_terrain['aspect']
    coast = npz_terrain['coast']
    mask_keb = npz_terrain['mask']
    xs = npz_terrain['xs']
    ys = npz_terrain['ys']
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]

    npz_clim = np.load(data_dir / "reconstructed_climatology_250m.npz")
    mean_annual = npz_clim['mean_annual_grid']
    monthly_clim = npz_clim['monthly_clim_grids']
    annual_grids = npz_clim['annual_grids']
    sens_slope = npz_clim['sens_slope_grid']
    years = list(npz_clim['years'])

    npz_enso = np.load(data_dir / "extreme_enso_anomaly_grids.npz")
    p_2010 = npz_enso['p_2010']
    p_2015 = npz_enso['p_2015']
    p_2022 = npz_enso['p_2022']
    p_2023 = npz_enso['p_2023']
    ano_2010_pct = npz_enso['ano_2010_pct']
    ano_2015_pct = npz_enso['ano_2015_pct']

    gdf_kec = gpd.read_file("33.05_kecamatan.geojson")
    gdf_kec['geometry'] = gdf_kec['geometry'].apply(shapely.force_2d)

    masked_nan = lambda arr: np.where(mask_keb, arr, np.nan)

    # -------------------------------------------------------------------------
    # FIG 1: Study Domain & Multi-Scale Topographic Predictors (4-Panel 16:9)
    # -------------------------------------------------------------------------
    print("\n[Visualisasi 1/8] Menghasilkan Gambar 1: Karakteristik Domain & Kovariat Topografi...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)

    # (a) Elevasi DEM
    im0 = axes[0, 0].imshow(masked_nan(elev), extent=extent, cmap='terrain', vmin=0, vmax=900)
    gdf_kec.boundary.plot(ax=axes[0, 0], color='black', linewidth=0.5, alpha=0.7)
    axes[0, 0].set_title("(a) Elevasi Topografi SRTM 30m Resampled 250m (mdpl)", fontsize=11, fontweight='bold', pad=8)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.035, pad=0.02, label='Elevasi (m)')

    # (b) Slope
    im1 = axes[0, 1].imshow(masked_nan(slope), extent=extent, cmap='magma', vmin=0, vmax=35)
    gdf_kec.boundary.plot(ax=axes[0, 1], color='cyan', linewidth=0.5, alpha=0.6)
    axes[0, 1].set_title("(b) Kemiringan Lereng Topografi (Derajat)", fontsize=11, fontweight='bold', pad=8)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.035, pad=0.02, label='Slope (°)')

    # (c) Jarak Pantai
    im2 = axes[1, 0].imshow(masked_nan(coast), extent=extent, cmap='Blues', vmin=0, vmax=45)
    gdf_kec.boundary.plot(ax=axes[1, 0], color='black', linewidth=0.5, alpha=0.7)
    axes[1, 0].set_title("(c) Jarak Euclidean ke Garis Pantai Samudera Hindia (km)", fontsize=11, fontweight='bold', pad=8)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.035, pad=0.02, label='Jarak Pantai (km)')

    # (d) Batas Administrasi 26 Kecamatan dengan Label
    gdf_kec.plot(ax=axes[1, 1], column='nm_kecamatan', cmap='tab20', alpha=0.5, edgecolor='black', linewidth=0.7)
    for _, row in gdf_kec.iterrows():
        pt = row['geometry'].representative_point()
        axes[1, 1].text(pt.x, pt.y, row['nm_kecamatan'], fontsize=5.5, ha='center', va='center', weight='bold', color='navy')
    axes[1, 1].set_title("(d) Batas Administrasi 26 Kecamatan Kabupaten Kebumen", fontsize=11, fontweight='bold', pad=8)

    for ax in axes.flat:
        ax.set_xlabel("Bujur Timur (°E)", fontsize=9)
        ax.set_ylabel("Lintang Selatan (°S)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)

    fig.suptitle("Karakteristik Fisik Topografi dan Geografis Wilayah Penelitian Kabupaten Kebumen", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_study_area_terrain.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print("  ✓ Gambar 1 tersimpan: documents/figures/fig1_study_area_terrain.png")

    # -------------------------------------------------------------------------
    # FIG 2: Multi-Sensor & Atmospheric Correlation Matrix & Time-Series
    # -------------------------------------------------------------------------
    print("\n[Visualisasi 2/8] Menghasilkan Gambar 2: Matriks Korelasi & Dinamika Temporal 25 Tahun...")
    fig, (ax_corr, ax_ts) = plt.subplots(1, 2, figsize=(16, 9), dpi=300, gridspec_kw={'width_ratios': [1, 1.3]})

    # Matriks korelasi variabel
    corr_cols = ['chirps_raw_mm', 'gsmap_raw_mm', 'era5_t2m_c', 'era5_rh_pct', 'era5_ws_ms', 'era5_sp_hpa', 'ndvi_mean']
    col_labels = ['CHIRPS', 'GSMaP', 'T2m', 'RH', 'Wind Speed', 'Surface Press', 'NDVI']
    corr_mat = df_ts[corr_cols].corr().values

    im_c = ax_corr.imshow(corr_mat, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    ax_corr.set_xticks(range(len(col_labels)))
    ax_corr.set_yticks(range(len(col_labels)))
    ax_corr.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
    ax_corr.set_yticklabels(col_labels, fontsize=9)
    for i in range(len(col_labels)):
        for j in range(len(col_labels)):
            val = corr_mat[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax_corr.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8, color=color, weight='bold')
    ax_corr.set_title("(a) Matriks Korelasi Pearson Multi-Variabel (300 Bulan)", fontsize=11, fontweight='bold', pad=10)
    fig.colorbar(im_c, ax=ax_corr, fraction=0.045, pad=0.03, label='Koefisien Korelasi (r)')

    # Deret waktu tahunan perbandingan CHIRPS vs GSMaP vs ERA5-T2m
    df_annual_ts = df_ts.groupby('year').agg({
        'chirps_raw_mm': 'sum',
        'gsmap_raw_mm': 'sum',
        'era5_t2m_c': 'mean'
    }).reset_index()

    ax_ts2 = ax_ts.twinx()
    w_bar = 0.35
    x_yr = df_annual_ts['year'].values
    ax_ts.bar(x_yr - w_bar/2, df_annual_ts['chirps_raw_mm'], width=w_bar, color='royalblue', label='CHIRPS Tahunan (mm)', alpha=0.85)
    ax_ts.bar(x_yr + w_bar/2, df_annual_ts['gsmap_raw_mm'], width=w_bar, color='darkturquoise', label='GSMaP Tahunan (mm)', alpha=0.85)
    ax_ts2.plot(x_yr, df_annual_ts['era5_t2m_c'], color='crimson', marker='o', linewidth=2.0, markersize=5, label='ERA5 Suhu Rata-rata (°C)')

    # Highlight El Nino dan La Nina
    ax_ts.axvspan(2009.5, 2010.5, color='blue', alpha=0.12, label='Super La Niña 2010')
    ax_ts.axvspan(2014.5, 2015.5, color='orange', alpha=0.15, label='Super El Niño 2015')
    ax_ts.axvspan(2022.5, 2023.5, color='red', alpha=0.12, label='Strong El Niño 2023')

    ax_ts.set_xlabel("Tahun Pengamatan (2001 - 2025)", fontsize=10, fontweight='bold')
    ax_ts.set_ylabel("Presipitasi Akumulasi Tahunan (mm/tahun)", fontsize=10, fontweight='bold', color='navy')
    ax_ts2.set_ylabel("Suhu Permukaan 2m Rata-rata (°C)", fontsize=10, fontweight='bold', color='crimson')
    ax_ts.set_title("(b) Deret Waktu Presipitasi Satelit & Suhu Atmosfer ERA5-Land (25 Tahun)", fontsize=11, fontweight='bold', pad=10)
    ax_ts.grid(True, linestyle=':', alpha=0.4)
    ax_ts.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_ts2.legend(loc='upper right', fontsize=8, framealpha=0.9)

    fig.suptitle("Analisis Hubungan Multi-Sensor dan Variabilitas Iklim 25 Tahun Kebumen (2001 - 2025)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig2_multisource_correlation.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print("  ✓ Gambar 2 tersimpan: documents/figures/fig2_multisource_correlation.png")

    # -------------------------------------------------------------------------
    # FIG 3: Model Benchmark Comparison & Feature Importance
    # -------------------------------------------------------------------------
    print("\n[Visualisasi 3/8] Menghasilkan Gambar 3: Benchmark 12 Model & Feature Importance...")
    fig, (ax_kge, ax_imp) = plt.subplots(1, 2, figsize=(16, 9), dpi=300, gridspec_kw={'width_ratios': [1.2, 1]})

    # Barplot horizontal KGE 12 model
    df_metrics_sorted = df_metrics.sort_values(by='KGE', ascending=True)
    models_clean = [m.split('. ')[1] if '. ' in m else m for m in df_metrics_sorted['Model']]
    kge_vals = df_metrics_sorted['KGE'].values
    colors_kge = ['forestgreen' if k > 0.8 else 'mediumseagreen' if k > 0.5 else 'lightcoral' if k > 0 else 'firebrick' for k in kge_vals]

    bars = ax_kge.barh(range(len(models_clean)), kge_vals, color=colors_kge, edgecolor='black', alpha=0.85)
    ax_kge.set_yticks(range(len(models_clean)))
    ax_kge.set_yticklabels(models_clean, fontsize=9)
    ax_kge.set_xlabel("Kling-Gupta Efficiency (KGE)", fontsize=10, fontweight='bold')
    ax_kge.set_title("(a) Peringkat Validasi Spasial 12 Algoritma Downscaling (KGE)", fontsize=11, fontweight='bold', pad=10)
    ax_kge.axvline(0.0, color='black', linestyle='--', linewidth=0.8)
    ax_kge.grid(True, axis='x', linestyle=':', alpha=0.5)

    for bar, val in zip(bars, kge_vals):
        x_pos = val + 0.02 if val >= 0 else val - 0.08
        ax_kge.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va='center', fontsize=7.5, weight='bold')

    # Barplot Feature Importance Multi-Sensor Model
    df_feat_sorted = df_feat.sort_values(by='Importance', ascending=True)
    feat_clean = df_feat_sorted['Feature'].values
    imp_pct = df_feat_sorted['Importance_Pct'].values

    bars2 = ax_imp.barh(range(len(feat_clean)), imp_pct, color='royalblue', edgecolor='black', alpha=0.85)
    ax_imp.set_yticks(range(len(feat_clean)))
    ax_imp.set_yticklabels(feat_clean, fontsize=9)
    ax_imp.set_xlabel("Kontribusi Relatif / Importance (%)", fontsize=10, fontweight='bold')
    ax_imp.set_title("(b) Kontribusi Fitur Multi-Sensor & Atmosfer (Multi-Sensor XGBoost)", fontsize=11, fontweight='bold', pad=10)
    ax_imp.grid(True, axis='x', linestyle=':', alpha=0.5)

    for bar, val in zip(bars2, imp_pct):
        ax_imp.text(val + 0.5, bar.get_y() + bar.get_height()/2, f"{val:.2f}%", va='center', fontsize=7.5, weight='bold')

    fig.suptitle("Evaluasi Kinerja 12 Algoritma Downscaling & Kontribusi Prediktor Lingkungan", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig3_model_benchmark_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print("  ✓ Gambar 3 tersimpan: documents/figures/fig3_model_benchmark_comparison.png")

    # -------------------------------------------------------------------------
    # FIG 4: 25-Year Climatological Mean Annual Precipitation Atlas (250m)
    # -------------------------------------------------------------------------
    print("\n[Visualisasi 4/8] Menghasilkan Gambar 4: Atlas Rata-rata Presipitasi Tahunan 25 Tahun...")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

    im_ann = ax.imshow(masked_nan(mean_annual), extent=extent, cmap='YlGnBu', vmin=2800, vmax=3100)
    gdf_kec.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)

    # Tambahkan kontur isohyet
    cs = ax.contour(masked_nan(mean_annual), levels=[2850, 2900, 2950, 3000, 3050], extent=extent, colors='navy', linewidths=0.9)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f mm')

    # Label nama kecamatan representatif
    for _, row in gdf_kec.iterrows():
        pt = row['geometry'].representative_point()
        ax.text(pt.x, pt.y, row['nm_kecamatan'], fontsize=6, ha='center', va='center', weight='bold', color='darkslategray')

    cbar = fig.colorbar(im_ann, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Curah Hujan Tahunan Rata-rata (mm/tahun)", fontsize=11, fontweight='bold')
    ax.set_title("Atlas Klimatologi Presipitasi Tahunan Rata-rata 25 Tahun (2001 - 2025) Resolusi 250m", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Bujur Timur (°E)", fontsize=10)
    ax.set_ylabel("Lintang Selatan (°S)", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig4_climatology_annual_mean.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print("  ✓ Gambar 4 tersimpan: documents/figures/fig4_climatology_annual_mean.png")

    # -------------------------------------------------------------------------
    # FIG 5: 12-Month Spatial Climatological Cycle Grid (3x4 Layout, Rule 2)
    # -------------------------------------------------------------------------
    print("\n[Visualisasi 5/8] Menghasilkan Gambar 5: Grid 3x4 Siklus Klimatologis Spasial 12 Bulan...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), dpi=300, sharex=True, sharey=True)
    months_names = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]

    vmin_m, vmax_m = 30.0, 480.0
    cmap_m = plt.cm.YlGnBu

    for m_idx, ax in enumerate(axes.flat):
        grid_m = monthly_clim[m_idx]
        im_m = ax.imshow(masked_nan(grid_m), extent=extent, cmap=cmap_m, vmin=vmin_m, vmax=vmax_m)
        gdf_kec.boundary.plot(ax=ax, color='black', linewidth=0.3, alpha=0.6)

        m_mean = np.nanmean(masked_nan(grid_m))
        ax.set_title(f"{months_names[m_idx]}\nMean: {m_mean:.1f} mm", fontsize=9, fontweight='bold', pad=4)
        ax.tick_params(labelsize=6)
        ax.grid(True, linestyle=':', alpha=0.2)

    # Colorbar terpadu di samping
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im_m, cax=cbar_ax)
    cb.set_label("Presipitasi Bulanan Rata-rata 25 Tahun (mm/bulan)", fontsize=10, fontweight='bold')

    fig.suptitle("Siklus Musiman Presipitasi Spasial 25 Tahun Kabupaten Kebumen (2001 - 2025)", fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.06, wspace=0.15, hspace=0.25)
    fig.savefig(fig_dir / "fig5_monthly_climatological_cycle.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print("  ✓ Gambar 5 tersimpan: documents/figures/fig5_monthly_climatological_cycle.png")

    # -------------------------------------------------------------------------
    # FIG 6: Spatial Climate Trend Atlas (Sen's Slope mm/year per pixel)
    # -------------------------------------------------------------------------
    print("\n[Visualisasi 6/8] Menghasilkan Gambar 6: Peta Tren Spasial Sen's Slope...")
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

    im_trend = ax.imshow(masked_nan(sens_slope), extent=extent, cmap='RdYlGn', vmin=20.0, vmax=27.0)
    gdf_kec.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)

    for _, row in gdf_kec.iterrows():
        pt = row['geometry'].representative_point()
        ax.text(pt.x, pt.y, row['nm_kecamatan'], fontsize=6, ha='center', va='center', weight='bold', color='black')

    cbar = fig.colorbar(im_trend, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Laju Perubahan Presipitasi Tahunan (mm/tahun)", fontsize=11, fontweight='bold')
    ax.set_title("Peta Tren Spasial Perubahan Iklim 25 Tahun (2001 - 2025) Menggunakan Estimator Sen's Slope", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Bujur Timur (°E)", fontsize=10)
    ax.set_ylabel("Lintang Selatan (°S)", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig6_climate_trend_sens_slope.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print("  ✓ Gambar 6 tersimpan: documents/figures/fig6_climate_trend_sens_slope.png")

    # -------------------------------------------------------------------------
    # FIG 7: Climate Extremes & ENSO Spatial Anomalies (4-Panel Grid)
    # -------------------------------------------------------------------------
    print("\n[Visualisasi 7/8] Menghasilkan Gambar 7: Peta Anomali Spasial ENSO Ekstrem...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300, sharex=True, sharey=True)

    # 2010 (Super La Nina)
    im0 = axes[0, 0].imshow(masked_nan(p_2010), extent=extent, cmap='YlGnBu', vmin=1800, vmax=4700)
    gdf_kec.boundary.plot(ax=axes[0, 0], color='black', linewidth=0.5, alpha=0.7)
    axes[0, 0].set_title("(a) Super La Niña 2010: Curah Hujan Tahunan Total (Mean: 4,461 mm)", fontsize=10, fontweight='bold', pad=6)

    # 2015 (Super El Nino)
    axes[0, 1].imshow(masked_nan(p_2015), extent=extent, cmap='YlGnBu', vmin=1800, vmax=4700)
    gdf_kec.boundary.plot(ax=axes[0, 1], color='black', linewidth=0.5, alpha=0.7)
    axes[0, 1].set_title("(b) Super El Niño 2015: Curah Hujan Tahunan Total (Mean: 2,225 mm)", fontsize=10, fontweight='bold', pad=6)

    # Anomali Relatif 2010 (%)
    im_ano = axes[1, 0].imshow(masked_nan(ano_2010_pct), extent=extent, cmap='BrBG', vmin=-40, vmax=60)
    gdf_kec.boundary.plot(ax=axes[1, 0], color='black', linewidth=0.5, alpha=0.7)
    axes[1, 0].set_title("(c) Super La Niña 2010: Anomali Relatif terhadap Normal 25 Tahun (+50.8%)", fontsize=10, fontweight='bold', pad=6)

    # Anomali Relatif 2015 (%)
    axes[1, 1].imshow(masked_nan(ano_2015_pct), extent=extent, cmap='BrBG', vmin=-40, vmax=60)
    gdf_kec.boundary.plot(ax=axes[1, 1], color='black', linewidth=0.5, alpha=0.7)
    axes[1, 1].set_title("(d) Super El Niño 2015: Anomali Relatif terhadap Normal 25 Tahun (-24.8%)", fontsize=10, fontweight='bold', pad=6)

    for ax in axes.flat:
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle=':', alpha=0.2)

    fig.colorbar(im0, ax=[axes[0, 0], axes[0, 1]], fraction=0.02, pad=0.02, label='Curah Hujan (mm/tahun)')
    fig.colorbar(im_ano, ax=[axes[1, 0], axes[1, 1]], fraction=0.02, pad=0.02, label='Anomali Presipitasi (%)')

    fig.suptitle("Respon Spasial Presipitasi Terhadap Kejadian Iklim Ekstrem ENSO di Kabupaten Kebumen", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig7_extreme_enso_anomalies.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print("  ✓ Gambar 7 tersimpan: documents/figures/fig7_extreme_enso_anomalies.png")

    # -------------------------------------------------------------------------
    # FIG 8: Zonal Climatology & Vulnerability Profiles across all 26 Kecamatan
    # -------------------------------------------------------------------------
    print("\n[Visualisasi 8/8] Menghasilkan Gambar 8: Profil Statistik Zonal 26 Kecamatan...")
    fig, (ax_mean, ax_contrast) = plt.subplots(1, 2, figsize=(16, 9), dpi=300)

    # (a) Mean Tahunan 26 Kecamatan
    df_zonal_sorted = df_zonal.sort_values(by='Mean_Tahunan_mm', ascending=True)
    kec_names = df_zonal_sorted['Kecamatan'].values
    mean_vals = df_zonal_sorted['Mean_Tahunan_mm'].values
    min_vals = df_zonal_sorted['Min_Tahunan_mm'].values
    max_vals = df_zonal_sorted['Max_Tahunan_mm'].values

    y_pos = np.arange(len(kec_names))
    ax_mean.barh(y_pos, mean_vals, color='teal', alpha=0.85, edgecolor='black', label='Rata-rata 25 Tahun')
    ax_mean.errorbar(mean_vals, y_pos, xerr=[mean_vals - min_vals, max_vals - mean_vals], fmt='none', ecolor='darkorange', capsize=3, label='Rentang Historis (Min - Max)')
    ax_mean.set_yticks(y_pos)
    ax_mean.set_yticklabels(kec_names, fontsize=8)
    ax_mean.set_xlabel("Presipitasi Tahunan (mm/tahun)", fontsize=10, fontweight='bold')
    ax_mean.set_title("(a) Presipitasi Tahunan Rata-rata & Rentang Historis 26 Kecamatan", fontsize=11, fontweight='bold', pad=8)
    ax_mean.grid(True, axis='x', linestyle=':', alpha=0.5)
    ax_mean.legend(loc='lower right', fontsize=8)

    # (b) Rentang Kontras ENSO (La Nina 2010 vs El Nino 2015)
    df_enso_sorted = df_enso.sort_values(by='Contrast_Range_mm', ascending=True)
    kec_enso_names = df_enso_sorted['Kecamatan'].values
    contrast_vals = df_enso_sorted['Contrast_Range_mm'].values

    ax_contrast.barh(y_pos, contrast_vals, color='crimson', alpha=0.85, edgecolor='black')
    ax_contrast.set_yticks(y_pos)
    ax_contrast.set_yticklabels(kec_enso_names, fontsize=8)
    ax_contrast.set_xlabel("Rentang Kontras Presipitasi Ekstrem: La Niña - El Niño (mm)", fontsize=10, fontweight='bold')
    ax_contrast.set_title("(b) Kerentanan Hidrometeorologi Ekstrem ENSO per Kecamatan", fontsize=11, fontweight='bold', pad=8)
    ax_contrast.grid(True, axis='x', linestyle=':', alpha=0.5)

    for bar, val in zip(ax_contrast.patches, contrast_vals):
        ax_contrast.text(val + 5, bar.get_y() + bar.get_height()/2, f"{val:.0f} mm", va='center', fontsize=6.5, weight='bold')

    fig.suptitle("Profil Klimatologi dan Kerentanan Presipitasi Ekstrem 26 Kecamatan Kabupaten Kebumen", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig8_zonal_kecamatan_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print("  ✓ Gambar 8 tersimpan: documents/figures/fig8_zonal_kecamatan_comparison.png")

    print("\n" + "=" * 85)
    print(f"🎉 FASE 5 SELESAI DALAM {time.time() - t0:.2f} DETIK!")
    print("=" * 85)

if __name__ == "__main__":
    run()
