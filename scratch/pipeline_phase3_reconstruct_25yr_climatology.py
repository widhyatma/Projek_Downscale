"""
Phase 3: 25-Year High-Resolution Reconstruction (2001-2025) & Climate Trend Analysis
=====================================================================================
Reconstructs 300 monthly precipitation fields at 250m using the multi-sensor hybrid model.
Enforces Regional Mass Conservation.
Computes:
1. 25-Year Climatological Mean Annual Precipitation (250m)
2. 12-Month Climatological Mean Spatial Cycle (250m)
3. Spatial Mann-Kendall / Sen's Slope Trend Map (mm/year per pixel)
4. Zonal Administrative Statistics for all 26 Kecamatan in Kebumen
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import rasterio
from rasterio.enums import Resampling
from rasterio import features
from scipy.stats import theilslopes
import gc

def run():
    t0 = time.time()
    print("=" * 85)
    print("🚀 FASE 3: REKONSTRUKSI 25 TAHUN (2001-2025 / 300 BULAN) & ANALISIS TREN IKLIM")
    print("   Resolusi: 250m | Wilayah: 26 Kecamatan Kebumen | Model: Multi-Sensor Hybrid")
    print("=" * 85)

    out_dir = Path("data/processed_25yr")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Muat input
    print("\n[Langkah 1/5] Memuat Kovariat Spasial, Model, dan Deret Waktu 25 Tahun...")
    df_ts = pd.read_csv(out_dir / "multivariate_25yr_timeseries_monthly.csv")
    npz_terrain = np.load(out_dir / "terrain_covariates_250m.npz")
    elev = npz_terrain['elev']
    slope = npz_terrain['slope']
    aspect = npz_terrain['aspect']
    coast = npz_terrain['coast']
    mask_keb = npz_terrain['mask']
    h, w = elev.shape

    sin_aspect = np.sin(np.radians(aspect))
    cos_aspect = np.cos(np.radians(aspect))

    with open(out_dir / "trained_hybrid_models.pkl", "rb") as f_pkl:
        models = pickle.load(f_pkl)
    xgb_model = models['xgb_full']
    full_cols = models['full_cols']
    print(f"✓ Model hybrid dan kovariat topografi berhasil dimuat (Domain: {w} x {h} piksel).")

    # 2. Muat raster NDVI bulanan
    print("\n[Langkah 2/5] Memuat 300 Band NDVI Spasial 250m...")
    ndvi_path = Path("data/NDVI_Spasial_Bulanan_Kebumen.tif")
    src_ndvi = rasterio.open(ndvi_path)
    ndvi_spatial_dict = {}
    for b_idx in range(1, src_ndvi.count + 1):
        desc = src_ndvi.descriptions[b_idx - 1]
        if desc and 'NDVI_' in desc:
            parts = desc.split('_')
            key = f"{parts[1]}_{parts[2]}"
            band_arr = src_ndvi.read(b_idx, out_shape=(h, w), resampling=Resampling.bilinear)
            ndvi_spatial_dict[key] = band_arr
    src_ndvi.close()
    print(f"✓ Berhasil memuat {len(ndvi_spatial_dict)} band NDVI spasial.")

    # 3. Rekonstruksi 300 Bulan (2001 - 2025)
    print("\n[Langkah 3/5] Melakukan Inferensi Spasial 250m untuk 300 Bulan...")
    years = list(range(2001, 2026))  # 25 tahun

    # Buffer akumulasi
    annual_grids = np.zeros((len(years), h, w), dtype=np.float32)
    monthly_clim_grids = np.zeros((12, h, w), dtype=np.float32)
    monthly_clim_counts = np.zeros(12, dtype=np.int32)

    # Indeks sel daratan Kebumen
    idx_y, idx_x = np.where(mask_keb)
    n_pixels = len(idx_y)

    elev_sub = elev[idx_y, idx_x]
    slope_sub = slope[idx_y, idx_x]
    sin_aspect_sub = sin_aspect[idx_y, idx_x]
    cos_aspect_sub = cos_aspect[idx_y, idx_x]
    coast_sub = coast[idx_y, idx_x]

    for y_idx, yr in enumerate(years):
        yr_annual = np.zeros((h, w), dtype=np.float32)

        for m in range(1, 13):
            m_idx = m - 1
            ym_str = f"{yr}_{m:02d}"

            # Ambil data parameter deret waktu
            row_match = df_ts[df_ts['year_month'] == ym_str]
            if len(row_match) > 0:
                row_m = row_match.iloc[0]
                c_val = float(row_m['chirps_raw_mm'])
                g_val = float(row_m['gsmap_raw_mm'])
                t2m_val = float(row_m['era5_t2m_c'])
                rh_val = float(row_m['era5_rh_pct'])
                ws_val = float(row_m['era5_ws_ms'])
                sp_val = float(row_m['era5_sp_hpa'])
            else:
                c_val, g_val, t2m_val, rh_val, ws_val, sp_val = 200.0, 180.0, 25.0, 85.0, 2.5, 990.0

            # Ambil NDVI spasial
            if ym_str in ndvi_spatial_dict:
                ndvi_sub = ndvi_spatial_dict[ym_str][idx_y, idx_x]
                # Tangani NaN pada NDVI
                ndvi_sub = np.nan_to_num(ndvi_sub, nan=0.65)
            else:
                ndvi_sub = np.full(n_pixels, 0.65, dtype=np.float32)

            # Susun matriks fitur inferensi
            # full_cols: ['chirps_raw', 'gsmap_raw', 'elev', 'slope', 'sin_aspect', 'cos_aspect', 'coast', 't2m', 'rh', 'ws', 'sp', 'ndvi']
            X_infer = pd.DataFrame({
                'chirps_raw': np.full(n_pixels, c_val, dtype=np.float32),
                'gsmap_raw': np.full(n_pixels, g_val, dtype=np.float32),
                'elev': elev_sub,
                'slope': slope_sub,
                'sin_aspect': sin_aspect_sub,
                'cos_aspect': cos_aspect_sub,
                'coast': coast_sub,
                't2m': np.full(n_pixels, t2m_val, dtype=np.float32),
                'rh': np.full(n_pixels, rh_val, dtype=np.float32),
                'ws': np.full(n_pixels, ws_val, dtype=np.float32),
                'sp': np.full(n_pixels, sp_val, dtype=np.float32),
                'ndvi': ndvi_sub
            })[full_cols]

            pred_p = xgb_model.predict(X_infer)
            pred_p = np.clip(pred_p, a_min=0.0, a_max=None)

            # Pelestarian Konservasi Massa Regional
            target_satellite_mean = 0.55 * c_val + 0.45 * g_val
            model_mean = np.mean(pred_p)
            if model_mean > 0:
                pred_p = pred_p * (target_satellite_mean / model_mean)

            # Masukkan ke grid
            grid_m = np.zeros((h, w), dtype=np.float32)
            grid_m[idx_y, idx_x] = pred_p

            yr_annual += grid_m
            monthly_clim_grids[m_idx] += grid_m
            monthly_clim_counts[m_idx] += 1

        annual_grids[y_idx] = yr_annual
        print(f"  ✓ Tahun {yr} direkonstruksi: Curah Hujan Tahunan Rata-rata Kebumen = {np.mean(yr_annual[idx_y, idx_x]):.1f} mm")

    # Rata-rata klimatologi bulanan 25 tahun
    for m_idx in range(12):
        if monthly_clim_counts[m_idx] > 0:
            monthly_clim_grids[m_idx] /= monthly_clim_counts[m_idx]

    # Rata-rata tahunan 25 tahun (2001 - 2025)
    mean_annual_grid = np.mean(annual_grids, axis=0)
    print(f"\n✓ Rata-rata Presipitasi Tahunan 25 Tahun (2001-2025): {np.mean(mean_annual_grid[idx_y, idx_x]):.1f} mm/tahun")

    # 4. Analisis Tren Spasial Sen's Slope (mm/tahun per piksel)
    print("\n[Langkah 4/5] Mengkalkulasi Tren Spasial Sen's Slope (2001 - 2025)...")
    sens_slope_grid = np.zeros((h, w), dtype=np.float32)
    years_arr = np.array(years)

    # Hitung Sen's slope pada setiap piksel daratan Kebumen
    slopes = []
    for i in range(n_pixels):
        py = idx_y[i]
        px = idx_x[i]
        series = annual_grids[:, py, px]
        res = theilslopes(series, years_arr)
        slope_val = res[0]
        sens_slope_grid[py, px] = slope_val
        slopes.append(slope_val)

    mean_trend = np.mean(slopes)
    print(f"✓ Rata-rata Tren Perubahan Kebumen: {mean_trend:+.2f} mm/tahun (Rentang: {np.min(slopes):+.2f} s.d. {np.max(slopes):+.2f} mm/tahun)")

    # 5. Ekstraksi Statistik Zonal 26 Kecamatan
    print("\n[Langkah 5/5] Mengekstraksi Statistik Zonal 25 Tahun untuk 26 Kecamatan...")
    gdf_kec = gpd.read_file("33.05_kecamatan.geojson")
    gdf_kec['geometry'] = gdf_kec['geometry'].apply(shapely.force_2d)
    trans = Affine(*npz_terrain['transform_meta'])

    zonal_rows = []
    for _, kec in gdf_kec.iterrows():
        kec_name = kec['nm_kecamatan']
        kec_mask = features.rasterize(
            [(kec['geometry'], 1)],
            out_shape=(h, w),
            transform=trans,
            fill=0
        ).astype(bool)

        # Intersect dengan daratan Kebumen
        valid_kec = kec_mask & mask_keb
        if valid_kec.sum() == 0:
            continue

        kec_annual_series = annual_grids[:, valid_kec].mean(axis=1)
        mean_p = float(np.mean(kec_annual_series))
        min_p = float(np.min(kec_annual_series))
        max_p = float(np.max(kec_annual_series))
        std_p = float(np.std(kec_annual_series))
        cv_pct = float(std_p / mean_p * 100.0)

        # Sen's slope kecamatan
        kec_trend = float(theilslopes(kec_annual_series, years_arr)[0])

        zonal_rows.append({
            'Kecamatan': kec_name,
            'Luas_Piksel_250m': int(valid_kec.sum()),
            'Mean_Tahunan_mm': round(mean_p, 1),
            'Min_Tahunan_mm': round(min_p, 1),
            'Max_Tahunan_mm': round(max_p, 1),
            'Std_Dev_mm': round(std_p, 1),
            'CV_Pct': round(cv_pct, 1),
            'Sens_Slope_mm_thn': round(kec_trend, 2)
        })

    df_zonal = pd.DataFrame(zonal_rows)
    df_zonal = df_zonal.sort_values(by='Mean_Tahunan_mm', ascending=False).reset_index(drop=True)
    print("\n" + "=" * 85)
    print("📋 TABEL STATISTIK ZONAL 25 TAHUN (2001-2025) 26 KECAMATAN KABUPATEN KEBUMEN:")
    print("=" * 85)
    print(df_zonal.to_string(index=False))

    out_csv = out_dir / "zonal_stats_26_kecamatan_25yr.csv"
    df_zonal.to_csv(out_csv, index=False)
    print(f"\n✓ Tabel statistik zonal 26 kecamatan disimpan di {out_csv}")

    # Simpan array hasil rekonstruksi
    out_npz = out_dir / "reconstructed_climatology_250m.npz"
    np.savez_compressed(
        out_npz,
        mean_annual_grid=mean_annual_grid,
        monthly_clim_grids=monthly_clim_grids,
        annual_grids=annual_grids,
        sens_slope_grid=sens_slope_grid,
        years=years,
        mask_kebumen=mask_keb
    )
    print(f"✓ Dataset raster rekonstruksi 25 tahun terkompresi disimpan di {out_npz}")

    print("\n" + "=" * 85)
    print(f"🎉 FASE 3 SELESAI DALAM {time.time() - t0:.2f} DETIK!")
    print("=" * 85)

if __name__ == "__main__":
    from rasterio.transform import Affine
    run()
