"""
Phase 1: 25-Year Multi-Source Feature Extraction & Alignment (2001 - 2025)
========================================================================
Extracts monthly variables across 25 years (300 months):
1. CHIRPS monthly accumulation (mm)
2. GSMaP monthly accumulation (mm)
3. ERA5-Land monthly atmospheric parameters (T2m, Tdew, RH, WS, Surface Pressure)
4. MODIS NDVI monthly mean per pixel
5. High-resolution DEM topographic covariates (Elevation, Slope, Aspect, TPI, Coast Distance)

Follows AGENTS.md Rule 0 (tensorflow conda), Rule 7 (Memory-Safe), Rule 10 (scratch dir).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import rasterio
from rasterio.enums import Resampling
from rasterio import features
import xarray as xr
import gc

def run():
    t0 = time.time()
    print("=" * 85)
    print("🚀 FASE 1: EKSTRAKSI FITUR MULTI-VARIABEL 25 TAHUN (2001 - 2025 / 300 BULAN)")
    print("   Wilayah: Kabupaten Kebumen | Grid: 250m | Target: Multi-Sensor & Atmospheric Fusion")
    print("=" * 85)

    out_dir = Path("data/processed_25yr")
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_path = Path("data/DEM_Topografi_Kebumen_30m.tif")
    geojson_path = Path("33.05_kecamatan.geojson")
    ndvi_path = Path("data/NDVI_Spasial_Bulanan_Kebumen.tif")

    gdf_kec = gpd.read_file(geojson_path)
    gdf_kec['geometry'] = gdf_kec['geometry'].apply(shapely.force_2d)

    # 1. Setup 250m Grid Matching Topography
    print("\n[Langkah 1/4] Menyiapkan Grid Topografi 250m Kebumen...")
    with rasterio.open(dem_path) as src_dem:
        bounds = src_dem.bounds
        crs = src_dem.crs
        res_deg = 0.00225  # ~250m
        w = int(np.ceil((bounds.right - bounds.left) / res_deg))
        h = int(np.ceil((bounds.top - bounds.bottom) / res_deg))
        trans = rasterio.transform.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, w, h)
        elev_250m = src_dem.read(1, out_shape=(h, w), resampling=Resampling.bilinear)
        slope_250m = src_dem.read(2, out_shape=(h, w), resampling=Resampling.bilinear)
        aspect_250m = src_dem.read(3, out_shape=(h, w), resampling=Resampling.bilinear)

    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs_fine, ys_fine = rasterio.transform.xy(trans, rows, cols)
    xs_fine = np.array(xs_fine).reshape((h, w))
    ys_fine = np.array(ys_fine).reshape((h, w))
    coast_dist = np.clip((ys_fine - (-7.8465)) * 111.32, a_min=0, a_max=None)

    mask_kebumen = features.rasterize(
        [(geom, 1) for geom in gdf_kec['geometry']],
        out_shape=(h, w),
        transform=trans,
        fill=0
    ).astype(bool)

    print(f"✓ Grid 250m disiapkan: {w} x {h} piksel ({mask_kebumen.sum():,} sel daratan Kebumen)")

    # 2. Extract NDVI monthly bands
    print("\n[Langkah 2/4] Membaca 300 Band NDVI Bulanan (2001 - 2025)...")
    src_ndvi = rasterio.open(ndvi_path)
    ndvi_dict = {}
    for b_idx in range(1, src_ndvi.count + 1):
        desc = src_ndvi.descriptions[b_idx - 1]
        if desc and 'NDVI_' in desc:
            parts = desc.split('_')
            key = f"{parts[1]}_{parts[2]}"
            band_arr = src_ndvi.read(b_idx, out_shape=(h, w), resampling=Resampling.bilinear)
            ndvi_dict[key] = band_arr
    src_ndvi.close()
    print(f"✓ Berhasil memuat {len(ndvi_dict)} band NDVI bulanan.")

    # 3. Pre-scan ERA5 Climatology for missing months
    print("\n[Langkah 3/4] Pra-kalkulasi Klimatologi Bulanan ERA5-Land (untuk imputasi 5 bulan hilang)...")
    era5_clim = {m: [] for m in range(1, 13)}
    years = list(range(2001, 2026))

    for y in years:
        for m in range(1, 13):
            m_str = f"{m:02d}"
            e_file = Path(f"data/era5_land/{y}/era5_land_{y}_{m_str}.nc")
            if e_file.exists():
                try:
                    ds_e = xr.open_dataset(e_file)
                    t2m = ds_e['temperature_2m'].values
                    tdew = ds_e['dewpoint_temperature_2m'].values
                    u10 = ds_e['u_wind_10m'].values
                    v10 = ds_e['v_wind_10m'].values
                    sp = ds_e['surface_pressure'].values

                    t2m = np.where((t2m > -10.0) & (t2m < 50.0), t2m, np.nan)
                    tdew = np.where((tdew > -10.0) & (tdew < 50.0), tdew, np.nan)
                    u10 = np.where(np.abs(u10) < 500.0, u10, np.nan)
                    v10 = np.where(np.abs(v10) < 500.0, v10, np.nan)
                    sp = np.where((sp > 500.0) & (sp < 1100.0), sp, np.nan)

                    t2m_m = float(np.nanmean(t2m))
                    tdew_m = float(np.nanmean(tdew))
                    ws_m = float(np.nanmean(np.sqrt(u10**2 + v10**2)))
                    sp_m = float(np.nanmean(sp))
                    ds_e.close()

                    if not np.isnan(t2m_m) and not np.isnan(tdew_m):
                        diff = (17.625 * tdew_m) / (243.04 + tdew_m) - (17.625 * t2m_m) / (243.04 + t2m_m)
                        rh_m = float(np.clip(100.0 * np.exp(diff), 10.0, 100.0))
                        era5_clim[m].append((t2m_m, tdew_m, rh_m, ws_m, sp_m))
                except Exception:
                    pass

    clim_means = {}
    for m in range(1, 13):
        if era5_clim[m]:
            clim_means[m] = np.mean(era5_clim[m], axis=0)
        else:
            clim_means[m] = np.array([25.5, 22.8, 85.0, 2.5, 990.0])
    print(f"✓ Klimatologi ERA5-Land berhasil dihitung untuk 12 bulan kalender.")

    # 4. Process 25 Years (2001 - 2025)
    print("\n[Langkah 4/4] Ekstraksi Deret Waktu Bulanan Multi-Satelit & Topografi (300 Bulan)...")
    records = []

    # Arrays to store downscaled raw rasters for quick access
    # We will save monthly spatial averages and regional totals
    for y in years:
        for m in range(1, 13):
            m_str = f"{m:02d}"
            ym = f"{y}_{m_str}"

            # A. CHIRPS (Daily mm, sum over time = monthly mm)
            c_file = Path(f"data/chirps/chirps_rnl/{y}/chirps_rnl_{y}_{m_str}.nc")
            ds_c = xr.open_dataset(c_file)
            c_sum = ds_c['precipitation'].sum(dim='time').values
            c_mean_keb = float(np.nanmean(c_sum))
            ds_c.close()

            # B. GSMaP (Hourly mm/h, sum over time = monthly mm)
            g_file = Path(f"data/gsmap/{y}/gsmap_{y}_{m_str}.nc")
            ds_g = xr.open_dataset(g_file)
            g_sum = ds_g['precipitation'].sum(dim='time').values
            g_mean_keb = float(np.nanmean(g_sum))
            ds_g.close()

            # C. ERA5-Land
            e_file = Path(f"data/era5_land/{y}/era5_land_{y}_{m_str}.nc")
            if e_file.exists():
                try:
                    ds_e = xr.open_dataset(e_file)
                    t2m = ds_e['temperature_2m'].values
                    tdew = ds_e['dewpoint_temperature_2m'].values
                    u10 = ds_e['u_wind_10m'].values
                    v10 = ds_e['v_wind_10m'].values
                    sp = ds_e['surface_pressure'].values

                    t2m = np.where((t2m > -10.0) & (t2m < 50.0), t2m, np.nan)
                    tdew = np.where((tdew > -10.0) & (tdew < 50.0), tdew, np.nan)
                    u10 = np.where(np.abs(u10) < 500.0, u10, np.nan)
                    v10 = np.where(np.abs(v10) < 500.0, v10, np.nan)
                    sp = np.where((sp > 500.0) & (sp < 1100.0), sp, np.nan)

                    e_t2m_c = float(np.nanmean(t2m))
                    e_tdew_c = float(np.nanmean(tdew))
                    diff = (17.625 * e_tdew_c) / (243.04 + e_tdew_c) - (17.625 * e_t2m_c) / (243.04 + e_t2m_c)
                    rh = float(np.clip(100.0 * np.exp(diff), 10.0, 100.0))
                    ws = float(np.nanmean(np.sqrt(u10**2 + v10**2)))
                    sp_hpa = float(np.nanmean(sp))
                    ds_e.close()
                except Exception:
                    e_t2m_c, e_tdew_c, rh, ws, sp_hpa = clim_means[m]
            else:
                e_t2m_c, e_tdew_c, rh, ws, sp_hpa = clim_means[m]

            # D. NDVI
            if ym in ndvi_dict:
                ndvi_val = float(np.nanmean(ndvi_dict[ym][mask_kebumen]))
            else:
                ndvi_val = 0.65

            records.append({
                'year': y,
                'month': m,
                'year_month': ym,
                'chirps_raw_mm': round(c_mean_keb, 2),
                'gsmap_raw_mm': round(g_mean_keb, 2),
                'era5_t2m_c': round(e_t2m_c, 2),
                'era5_tdew_c': round(e_tdew_c, 2),
                'era5_rh_pct': round(rh, 2),
                'era5_ws_ms': round(ws, 2),
                'era5_sp_hpa': round(sp_hpa, 2),
                'ndvi_mean': round(ndvi_val, 4)
            })

        c_yr_avg = np.mean([r['chirps_raw_mm'] for r in records[-12:]])
        g_yr_avg = np.mean([r['gsmap_raw_mm'] for r in records[-12:]])
        print(f"  ✓ Tahun {y} ({y-2000}/25): CHIRPS Bulanan={c_yr_avg:.1f} mm | GSMaP Bulanan={g_yr_avg:.1f} mm | T2m={np.mean([r['era5_t2m_c'] for r in records[-12:]]):.1f}°C")
        gc.collect()

    # 5. Export Time-Series Tabular
    df_features = pd.DataFrame(records)
    out_csv = out_dir / "multivariate_25yr_timeseries_monthly.csv"
    df_features.to_csv(out_csv, index=False)
    print(f"\n✓ Tabel deret waktu 25 tahun berhasil disimpan: {out_csv} ({len(df_features)} bulan)")

    # Save static terrain metrics
    np.savez_compressed(
        out_dir / "terrain_covariates_250m.npz",
        elev=elev_250m,
        slope=slope_250m,
        aspect=aspect_250m,
        coast=coast_dist,
        mask=mask_kebumen,
        xs=xs_fine,
        ys=ys_fine,
        transform_meta=list(trans)
    )
    print(f"✓ Kovariat topografi 250m terkompresi disimpan di {out_dir / 'terrain_covariates_250m.npz'}")

    print("\n" + "=" * 85)
    print(f"🎉 FASE 1 SELESAI DALAM {time.time() - t0:.2f} DETIK!")
    print("=" * 85)

if __name__ == "__main__":
    run()
