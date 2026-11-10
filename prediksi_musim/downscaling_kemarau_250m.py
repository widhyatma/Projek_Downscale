#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Spatial Downscaling Topografi Resolusi Ultra-Tinggi (250m) Prediksi Musim Kemarau Kebumen
Memproyeksikan Awal Musim Kemarau (AMK), Akumulasi Presipitasi Kemarau, dan Indeks Kerawanan Kekeringan
ke grid raster kontinyu 250m dengan kalibrasi Pelestarian Konservasi Massa Regional (Rule 15).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestRegressor
from shapely import force_2d

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
RASTER_DIR = BASE_DIR / "prediksi_musim" / "rasters_kemarau_250m"
DEM_PATH = BASE_DIR / "data" / "DEM_Topografi_Kebumen_30m.tif"
GEOJSON_PATH = BASE_DIR / "33.05_kecamatan.geojson"

RASTER_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_DEG = 0.00225

def prepare_250m_topographic_mesh():
    """Mengekstrak dan me-resample 5 kovariat topografi ke grid 250m."""
    with rasterio.open(DEM_PATH) as src:
        bounds = src.bounds
        dem_raw = src.read(1).astype(np.float32)
        dem_raw[dem_raw < -100] = 0.0
        
    west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
    width = int(np.round((east - west) / PIXEL_SIZE_DEG))
    height = int(np.round((north - south) / PIXEL_SIZE_DEG))
    transform_250m = from_bounds(west, south, east, north, width, height)
    
    y_idx = np.linspace(0, dem_raw.shape[0] - 1, height).astype(int)
    x_idx = np.linspace(0, dem_raw.shape[1] - 1, width).astype(int)
    dem_250m = np.clip(dem_raw[np.ix_(y_idx, x_idx)], 0.0, 1100.0)
    
    dx_m = PIXEL_SIZE_DEG * 110000.0
    dy_m = PIXEL_SIZE_DEG * 111000.0
    grad_y, grad_x = np.gradient(dem_250m, dy_m, dx_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
    
    aspect_rad = np.arctan2(-grad_y, grad_x)
    sin_aspect = np.sin(aspect_rad)
    cos_aspect = np.cos(aspect_rad)
    
    lats = np.linspace(north, south, height)
    lons = np.linspace(west, east, width)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    dist_coast_km = np.maximum(0.0, (lat_mesh - (-7.82)) * 111.0)
    
    tpi_250m = dem_250m - uniform_filter(dem_250m, size=5)
    
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf['geometry'] = force_2d(gdf['geometry'])
    mask_kebumen = geometry_mask(gdf.geometry, transform=transform_250m, invert=True, out_shape=(height, width))
    
    return {
        'dem': dem_250m, 'slope': slope_deg, 'sin_aspect': sin_aspect, 'cos_aspect': cos_aspect,
        'dist_coast': dist_coast_km, 'tpi': tpi_250m, 'mask': mask_kebumen,
        'transform': transform_250m, 'width': width, 'height': height, 'gdf': gdf
    }

def run_spatial_downscaling_kemarau_2026():
    """Menjalankan proses downscaling spasial 250m musim kemarau 2026."""
    print("\n[INFO] Memulai Downscaling Spasial Topografi 250m Musim Kemarau...")
    mesh = prepare_250m_topographic_mesh()
    mask = mesh['mask']
    gdf = mesh['gdf']
    
    pred_file = DATA_DIR / "prakiraan_musim_kemarau_2026_kebumen.csv"
    norm_file = DATA_DIR / "normal_klimatologi_amk_kebumen.csv"
    
    if not pred_file.exists():
        print(f"File {pred_file} belum ada. Jalankan inferensi LSTM terlebih dahulu.")
        return
        
    df_pred = pd.read_csv(pred_file)
    df_norm = pd.read_csv(norm_file)
    df_pred = df_pred.merge(df_norm[['kecamatan', 'normal_total_hujan_mk_mm']], on='kecamatan', how='left')
    
    # Perkiraan total presipitasi kemarau 2026 berdasarkan sifat hujan
    df_pred['prakiraan_total_mk_mm'] = np.where(
        df_pred['prakiraan_sifat_hujan_mk'] == 'Bawah Normal',
        df_pred['normal_total_hujan_mk_mm'] * 0.75,
        np.where(df_pred['prakiraan_sifat_hujan_mk'] == 'Atas Normal',
                 df_pred['normal_total_hujan_mk_mm'] * 1.20,
                 df_pred['normal_total_hujan_mk_mm'])
    )
    
    gdf_pred = gdf.merge(df_pred, left_on='nm_kecamatan', right_on='kecamatan')
    
    # Rasterize target
    coarse_amk = np.full((mesh['height'], mesh['width']), np.nan, dtype=np.float32)
    coarse_tot = np.full((mesh['height'], mesh['width']), np.nan, dtype=np.float32)
    coarse_norm_tot = np.full((mesh['height'], mesh['width']), np.nan, dtype=np.float32)
    
    for _, row in gdf_pred.iterrows():
        k_mask = geometry_mask([row.geometry], transform=mesh['transform'], invert=True, out_shape=(mesh['height'], mesh['width']))
        coarse_amk[k_mask] = row['prakiraan_amk_dasarian']
        coarse_tot[k_mask] = row['prakiraan_total_mk_mm']
        coarse_norm_tot[k_mask] = row['normal_total_hujan_mk_mm']
        
    valid_train = mask & (~np.isnan(coarse_amk))
    
    X_train = np.column_stack([
        mesh['dem'][valid_train],
        mesh['slope'][valid_train],
        mesh['sin_aspect'][valid_train],
        mesh['cos_aspect'][valid_train],
        mesh['dist_coast'][valid_train],
        mesh['tpi'][valid_train]
    ])
    
    all_Z = mesh['dem'].flatten()
    all_slope = mesh['slope'].flatten()
    all_sin_a = mesh['sin_aspect'].flatten()
    all_cos_a = mesh['cos_aspect'].flatten()
    all_d_coast = mesh['dist_coast'].flatten()
    all_tpi = mesh['tpi'].flatten()
    X_all = np.column_stack([all_Z, all_slope, all_sin_a, all_cos_a, all_d_coast, all_tpi])
    
    # 1. Downscaling AMK Dasarian
    rf_amk = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_amk.fit(X_train, coarse_amk[valid_train])
    pred_amk_250m = rf_amk.predict(X_all).reshape(mesh['height'], mesh['width'])
    pred_amk_250m[~mask] = np.nan
    
    # 2. Downscaling Total Presipitasi Kemarau dengan Konservasi Massa Regional (Rule 15)
    rf_tot = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_tot.fit(X_train, coarse_tot[valid_train])
    pred_tot_raw = rf_tot.predict(X_all).reshape(mesh['height'], mesh['width'])
    
    sat_vol = np.sum(coarse_tot[valid_train])
    down_vol = np.sum(pred_tot_raw[valid_train])
    calib_factor = sat_vol / down_vol if down_vol > 0 else 1.0
    pred_tot_250m = pred_tot_raw * calib_factor
    pred_tot_250m[~mask] = np.nan
    
    # 3. Downscaling Normal Total Presipitasi
    rf_norm = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_norm.fit(X_train, coarse_norm_tot[valid_train])
    norm_tot_250m = rf_norm.predict(X_all).reshape(mesh['height'], mesh['width'])
    norm_tot_250m[~mask] = np.nan
    
    # Persentase terhadap Normal
    persen_normal_250m = (pred_tot_250m / np.maximum(norm_tot_250m, 1.0)) * 100.0
    persen_normal_250m[~mask] = np.nan
    
    # 4. Indeks Kerentanan Kekeringan Pertanian 250m (0..100)
    # Defisit curah hujan tinggi + elevasi rendah/pantai + lereng curam
    defisit_score = np.clip((100.0 - persen_normal_250m) * 2.0, 0.0, 60.0) # maks 60 poin
    pantai_score = np.clip((30.0 - mesh['dist_coast']) / 30.0 * 25.0, 0.0, 25.0) # maks 25 poin
    lereng_score = np.clip(mesh['slope'] / 40.0 * 15.0, 0.0, 15.0) # maks 15 poin
    indeks_kekeringan = defisit_score + pantai_score + lereng_score
    indeks_kekeringan[~mask] = np.nan
    
    # Ekspor GeoTIFF
    meta_gtiff = {
        'driver': 'GTiff', 'dtype': 'float32', 'nodata': -9999.0,
        'width': mesh['width'], 'height': mesh['height'], 'count': 1,
        'crs': 'EPSG:4326', 'transform': mesh['transform'], 'compress': 'lzw'
    }
    
    files_to_export = [
        ('AMK_Dasarian_250m_Kebumen_2026.tif', pred_amk_250m),
        ('Total_Hujan_MK_250m_Kebumen_2026.tif', pred_tot_250m),
        ('Sifat_Hujan_MK_Persen_Normal_250m.tif', persen_normal_250m),
        ('Indeks_Kerawanan_Kekeringan_250m.tif', indeks_kekeringan)
    ]
    
    for fname, arr in files_to_export:
        out_path = RASTER_DIR / fname
        with rasterio.open(out_path, 'w', **meta_gtiff) as dst:
            dst.write(np.nan_to_num(arr, nan=-9999.0).astype(np.float32), 1)
        print(f"  -> GeoTIFF tersimpan: {out_path.name}")
        
    print(f"\n[SUKSES] Seluruh raster downscaling spasial 250m tersimpan di: {RASTER_DIR}")
    print(f"Kalibrasi Konservasi Massa Regional: Faktor = {calib_factor:.4f} (Bias Volume = 0.00%)")

if __name__ == "__main__":
    run_spatial_downscaling_kemarau_2026()

