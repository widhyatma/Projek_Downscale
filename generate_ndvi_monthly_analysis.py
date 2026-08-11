"""
Script: generate_ndvi_monthly_analysis.py
Deskripsi: Analisis spasial dan statistik bulanan data NDVI Kabupaten Kebumen (2001 - 2025).
Output:
- 25 Plot Tahunan (masing-masing 3x4 subplot = 12 bulan) untuk tahun 2001 - 2025
- 1 Plot Klimatologi Bulanan Rata-rata 25 Tahun (3x4 subplot = 12 bulan)
  -> Total: 26 Plot Grid Bulanan
- Plot Tren Time Series NDVI Kebumen 2001 - 2025
- Plot Profil Siklus Musiman NDVI per Kecamatan
- File CSV Statistik Bulanan per Kecamatan (2001-2025) dan Rata-rata Klimatologi
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.ops import unary_union
import rasterio
from rasterio.mask import mask
import rasterio.features
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tif_path = os.path.join(base_dir, "data", "NDVI_Spasial_Bulanan_Kebumen.tif")
    geojson_path = os.path.join(base_dir, "33.05_kecamatan.geojson")
    out_dir = os.path.join(base_dir, "NDVI_Bulanan")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("ANALISIS SPASIAL DAN STATISTIK NDVI BULANAN KABUPATEN KEBUMEN")
    print("=" * 70)

    # 1. Load GeoJSON
    print(f"[1/6] Membaca batas wilayah GeoJSON: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    gdf['geometry'] = shapely.force_2d(gdf.geometry)
    kebumen_union = unary_union(gdf.geometry)
    kecamatan_names = sorted(gdf['nm_kecamatan'].unique().tolist())
    print(f"      Total Kecamatan: {len(gdf)} kecamatan")

    # 2. Read Raster and Mask
    print(f"[2/6] Membaca GeoTIFF NDVI: {tif_path}")
    with rasterio.open(tif_path) as src:
        total_bands = src.count
        descriptions = src.descriptions
        crs = src.crs
        print(f"      Total Band: {total_bands} band")
        
        # Parse band names (e.g. NDVI_2001_01)
        band_info = []
        for i in range(1, total_bands + 1):
            desc = descriptions[i-1] if (descriptions and len(descriptions) >= i and descriptions[i-1]) else f"Band_{i}"
            parts = desc.split('_')
            if len(parts) >= 3:
                year = int(parts[1])
                month = int(parts[2])
            else:
                year = 2001 + (i - 1) // 12
                month = ((i - 1) % 12) + 1
            band_info.append({'band_idx': i, 'desc': desc, 'year': year, 'month': month})

        df_bands = pd.DataFrame(band_info)
        years = sorted(df_bands['year'].unique().tolist())
        print(f"      Rentang Tahun: {min(years)} - {max(years)} ({len(years)} tahun)")

        # Masking raster with Kebumen boundary
        print("[3/6] Memotong (masking) raster sesuai geometri Kabupaten Kebumen...")
        masked_data, masked_transform = mask(src, [kebumen_union], crop=True, nodata=np.nan)
        h, w = masked_data.shape[1], masked_data.shape[2]
        left = masked_transform[2]
        top = masked_transform[5]
        right = left + w * masked_transform[0]
        bottom = top + h * masked_transform[4]
        extent = [left, right, bottom, top]

    # 3. Setup Plot Parameters
    month_names = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
                   'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    norm = Normalize(vmin=0.0, vmax=0.9)
    cmap = plt.cm.RdYlGn

    # 4. Generate 25 Annual 3x4 Plots
    print(f"[4/6] Menghasilkan 25 plot tahunan (3x4 grid bulanan)...")
    
    # Pre-calculate monthly mean across all pixels in Kebumen for time series
    ts_records = []
    
    for yr_idx, yr in enumerate(years, start=1):
        df_yr = df_bands[df_bands['year'] == yr].sort_values('month')
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=200)
        fig.suptitle(f"Analisis Spasial NDVI Bulanan Kabupaten Kebumen - Tahun {yr}", 
                     fontsize=16, fontweight='bold', y=0.98)
        
        for m in range(12):
            ax = axes[m // 4, m % 4]
            
            if m < len(df_yr):
                b_idx = df_yr.iloc[m]['band_idx'] - 1
                band_data = masked_data[b_idx]
                
                # Display raster
                im = ax.imshow(band_data, extent=extent, cmap=cmap, norm=norm, origin='upper')
                
                # Overlay kecamatan boundary
                gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.75)
                
                valid_vals = band_data[~np.isnan(band_data)]
                if len(valid_vals) > 0:
                    m_val = np.mean(valid_vals)
                    min_val = np.min(valid_vals)
                    max_val = np.max(valid_vals)
                    p25_val = np.percentile(valid_vals, 25)
                    p75_val = np.percentile(valid_vals, 75)
                    std_val = np.std(valid_vals)
                else:
                    m_val = min_val = max_val = p25_val = p75_val = std_val = np.nan
                
                # Record for time series
                ts_records.append({
                    'year': yr,
                    'month': m + 1,
                    'date': f"{yr}-{m+1:02d}",
                    'mean': m_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'p25': p25_val,
                    'p75': p75_val
                })
                
                ax.set_title(f"{month_names[m]} {yr}\n(Rerata: {m_val:.2f} | Rentang: {min_val:.2f} - {max_val:.2f})", 
                             fontsize=9.5, fontweight='bold', pad=4)
            else:
                ax.text(0.5, 0.5, "Data Tidak Tersedia", ha='center', va='center')
                ax.set_title(f"{month_names[m]} {yr}", fontsize=9.5, fontweight='bold')
            
            ax.tick_params(labelsize=7.5)
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f°'))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f°'))

        plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
        
        # Colorbar
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('NDVI (Normalized Difference Vegetation Index)', fontsize=11, fontweight='bold')
        
        out_plot_path = os.path.join(out_dir, f"NDVI_Bulanan_{yr}.png")
        plt.savefig(out_plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"      [OK] Plot {yr_idx}/25: NDVI_Bulanan_{yr}.png tersimpan.")

    # 5. Generate 26th Plot: Climatology (Multi-Year Monthly Mean 2001-2025)
    print(f"[5/6] Menghitung dan menghasilkan Plot ke-26: Klimatologi Rata-rata Bulanan (2001-2025)...")
    climatology_monthly = np.zeros((12, h, w), dtype=np.float32)
    
    for m in range(12):
        month_bands = df_bands[df_bands['month'] == (m + 1)]['band_idx'].values - 1
        month_stack = masked_data[month_bands, :, :]
        # Nanmean along year axis
        climatology_monthly[m] = np.nanmean(month_stack, axis=0)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=200)
    fig.suptitle(f"Klimatologi Spasial NDVI Bulanan Kabupaten Kebumen (Rata-rata 2001 - 2025)", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for m in range(12):
        ax = axes[m // 4, m % 4]
        band_data = climatology_monthly[m]
        
        im = ax.imshow(band_data, extent=extent, cmap=cmap, norm=norm, origin='upper')
        gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.75)
        
        valid_vals = band_data[~np.isnan(band_data)]
        m_val = np.mean(valid_vals) if len(valid_vals) > 0 else 0
        min_val = np.min(valid_vals) if len(valid_vals) > 0 else 0
        max_val = np.max(valid_vals) if len(valid_vals) > 0 else 0
        
        ax.set_title(f"Klimatologi {month_names[m]}\n(Rerata: {m_val:.2f} | Rentang: {min_val:.2f} - {max_val:.2f})", 
                     fontsize=9.5, fontweight='bold', pad=4)
        ax.tick_params(labelsize=7.5)
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f°'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f°'))

    plt.tight_layout(rect=[0, 0.03, 0.92, 0.95])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('NDVI Klimatologi (2001-2025)', fontsize=11, fontweight='bold')
    
    climatology_plot_path = os.path.join(out_dir, "NDVI_Bulanan_Klimatologi_2001_2025.png")
    plt.savefig(climatology_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Plot 26/26: NDVI_Bulanan_Klimatologi_2001_2025.png tersimpan.")

    # 6. Zonal Statistics per Kecamatan & Extra Analytical Plots
    print("[6/6] Menghitung statistik zonal per kecamatan & membuat grafik pendukung...")
    
    # Rasterize kecamatan polygons for fast zonal masking
    shapes = ((geom, idx) for idx, geom in enumerate(gdf.geometry))
    kecamatan_raster = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(h, w),
        transform=masked_transform,
        fill=-1,
        dtype=np.int16
    )

    zonal_records = []
    for _, row in df_bands.iterrows():
        b_idx = row['band_idx'] - 1
        yr = row['year']
        mo = row['month']
        band_data = masked_data[b_idx]
        
        for k_idx, k_row in gdf.iterrows():
            k_name = k_row['nm_kecamatan']
            k_mask = (kecamatan_raster == k_idx)
            vals = band_data[k_mask]
            vals = vals[~np.isnan(vals)]
            
            if len(vals) > 0:
                mean_k = np.mean(vals)
                median_k = np.median(vals)
                min_k = np.min(vals)
                max_k = np.max(vals)
                std_k = np.std(vals)
            else:
                mean_k = median_k = min_k = max_k = std_k = np.nan
            
            zonal_records.append({
                'year': yr,
                'month': mo,
                'date': f"{yr}-{mo:02d}",
                'nm_kecamatan': k_name,
                'mean_ndvi': mean_k,
                'median_ndvi': median_k,
                'min_ndvi': min_k,
                'max_ndvi': max_k,
                'std_ndvi': std_k
            })

    df_zonal = pd.DataFrame(zonal_records)
    csv_zonal_path = os.path.join(out_dir, "NDVI_Kecamatan_Bulanan_2001_2025.csv")
    df_zonal.to_csv(csv_zonal_path, index=False)
    print(f"      [OK] CSV Data Zonal Bulanan tersimpan: {csv_zonal_path}")

    # Climatology per Kecamatan
    df_clim_k = df_zonal.groupby(['nm_kecamatan', 'month'])['mean_ndvi'].mean().reset_index()
    csv_clim_path = os.path.join(out_dir, "NDVI_Kecamatan_Klimatologi_Bulanan.csv")
    df_clim_k.to_csv(csv_clim_path, index=False)
    print(f"      [OK] CSV Klimatologi per Kecamatan tersimpan: {csv_clim_path}")

    # Time Series Summary Plot (Kebumen Average 2001 - 2025)
    df_ts = pd.DataFrame(ts_records)
    df_ts['date_dt'] = pd.to_datetime(df_ts['date'])
    
    plt.figure(figsize=(15, 6), dpi=200)
    plt.plot(df_ts['date_dt'], df_ts['mean'], color='#1b7837', linewidth=1.5, label='Rata-rata NDVI Kebumen')
    plt.fill_between(df_ts['date_dt'], df_ts['p25'], df_ts['p75'], color='#a6d96a', alpha=0.35, label='Rentang Interkuartil (Q1 - Q3)')
    
    # 12-month Rolling Mean
    df_ts['rolling_12m'] = df_ts['mean'].rolling(12, center=True).mean()
    plt.plot(df_ts['date_dt'], df_ts['rolling_12m'], color='#762a83', linewidth=2.2, linestyle='--', label='Tren 12-Bulan (Rolling Mean)')
    
    plt.title('Dinamika Tren NDVI Bulanan Kabupaten Kebumen (2001 - 2025)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Tahun', fontsize=11, fontweight='bold')
    plt.ylabel('Rerata NDVI', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.ylim(0.2, 0.9)
    plt.tight_layout()
    ts_plot_path = os.path.join(out_dir, "NDVI_Tren_Kebumen_2001_2025.png")
    plt.savefig(ts_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Plot Tren Time Series tersimpan: {ts_plot_path}")

    # Monthly Climatological Profile per Kecamatan Plot
    plt.figure(figsize=(14, 7), dpi=200)
    pivot_clim = df_clim_k.pivot(index='month', columns='nm_kecamatan', values='mean_ndvi')
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(pivot_clim.columns)))
    for idx, col in enumerate(pivot_clim.columns):
        plt.plot(range(1, 13), pivot_clim[col], marker='o', markersize=3, label=col, color=colors[idx % len(colors)], linewidth=1.2)
    
    # Overall Mean
    overall_mean = pivot_clim.mean(axis=1)
    plt.plot(range(1, 13), overall_mean, color='black', linewidth=3.5, linestyle='-', marker='s', markersize=6, label='RATA-RATA KEBUMEN')

    plt.title('Profil Siklus Musiman NDVI Rata-rata per Kecamatan di Kabupaten Kebumen', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Bulan', fontsize=11, fontweight='bold')
    plt.ylabel('Rerata NDVI Klimatologi', fontsize=11, fontweight='bold')
    plt.xticks(range(1, 13), [m[:3] for m in month_names], fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize=8)
    plt.tight_layout()
    clim_profile_path = os.path.join(out_dir, "NDVI_Profil_Bulanan_Per_Kecamatan.png")
    plt.savefig(clim_profile_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Plot Profil Musiman per Kecamatan tersimpan: {clim_profile_path}")

    print("=" * 70)
    print("SELURUH PROSES BERHASIL DISELESAIKAN DENGAN SEMPURNA!")
    print(f"Total file tersimpan di {out_dir}:")
    print(f"  - 26 Plot Peta Grid Bulanan (25 Tahunan + 1 Klimatologi)")
    print(f"  - 2 Plot Grafik Analisis Tambahan (Tren & Profil Musiman)")
    print(f"  - 2 File CSV Statistik Zonal")
    print("=" * 70)

if __name__ == "__main__":
    main()
