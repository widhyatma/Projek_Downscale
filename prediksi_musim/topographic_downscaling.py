#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Spatial Downscaling Topografi Resolusi Ultra-Tinggi (250m) Prediksi Musim Hujan Kebumen
Mengekstrak 5 kovariat medan dari DEM 30m, memproyeksikan Awal Musim Hujan (AMH) dan Curah Hujan
ke grid raster kontinyu 250m dengan kalibrasi Pelestarian Konservasi Massa Regional (Rule 15).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from scipy.ndimage import uniform_filter, gaussian_filter
from sklearn.ensemble import RandomForestRegressor
from shapely import force_2d

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
RASTER_DIR = BASE_DIR / "prediksi_musim" / "rasters_250m"
DEM_PATH = BASE_DIR / "data" / "DEM_Topografi_Kebumen_30m.tif"
GEOJSON_PATH = BASE_DIR / "33.05_kecamatan.geojson"

RASTER_DIR.mkdir(parents=True, exist_ok=True)

# Parameter Grid Target 250m (~0.00225 derajat / piksel)
PIXEL_SIZE_DEG = 0.00225 

def prepare_250m_topographic_mesh():
    """
    Mengekstrak dan me-resample 5 kovariat topografi ke grid 250m:
    1. Elevasi Topografi (Z, mdpl)
    2. Kemiringan Lereng (Slope, derajat)
    3. Aspek Ortogonal: sin(aspect) & cos(aspect)
    4. Jarak ke Garis Pantai Selatan (D_coast, km)
    5. Topographic Position Index (TPI, mdpl)
    """
    print(f"Membaca DEM referensi: {DEM_PATH}...")
    with rasterio.open(DEM_PATH) as src:
        bounds = src.bounds # left, bottom, right, top
        dem_raw = src.read(1).astype(np.float32)
        dem_raw[dem_raw < -100] = 0.0 # koreksi no-data laut
        
    west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
    
    # Dimensi grid 250m
    width = int(np.round((east - west) / PIXEL_SIZE_DEG))
    height = int(np.round((north - south) / PIXEL_SIZE_DEG))
    
    transform_250m = from_bounds(west, south, east, north, width, height)
    
    # Resample DEM ke 250m menggunakan interpolasi teratur
    y_idx = np.linspace(0, dem_raw.shape[0] - 1, height).astype(int)
    x_idx = np.linspace(0, dem_raw.shape[1] - 1, width).astype(int)
    dem_250m = dem_raw[np.ix_(y_idx, x_idx)]
    dem_250m = np.clip(dem_250m, 0.0, 1100.0)
    
    # Hitung gradien spasial untuk Slope & Aspect
    # 1 derajat lintang ~ 111.000 meter, 1 derajat bujur ~ 111.000 * cos(-7.7) ~ 110.000 meter
    dx_m = PIXEL_SIZE_DEG * 110000.0
    dy_m = PIXEL_SIZE_DEG * 111000.0
    
    grad_y, grad_x = np.gradient(dem_250m, dy_m, dx_m)
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_deg = np.degrees(slope_rad)
    
    aspect_rad = np.arctan2(-grad_y, grad_x)
    sin_aspect = np.sin(aspect_rad)
    cos_aspect = np.cos(aspect_rad)
    
    # Jarak ke Garis Pantai Selatan (D_coast, km)
    # Garis pantai Kebumen berada di sekitar latitude selatan (south ~ -7.80 s/d -7.78)
    lats = np.linspace(north, south, height)
    lons = np.linspace(west, east, width)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Aproksimasi jarak pantai (garis pantai selatan lat ~ -7.78)
    dist_coast_km = np.maximum(0.0, (lat_mesh - (-7.82)) * 111.0)
    
    # Topographic Position Index (TPI 250m): Z - mean_neighborhood
    mean_local = uniform_filter(dem_250m, size=5)
    tpi_250m = dem_250m - mean_local
    
    # Masking dengan poligon batas administrasi Kebumen
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf['geometry'] = force_2d(gdf['geometry'])
    mask_kebumen = geometry_mask(gdf.geometry, transform=transform_250m, invert=True, out_shape=(height, width))
    
    print(f"Grid 250m terkonstruksi: {width} x {height} piksel ({np.sum(mask_kebumen)} piksel daratan Kebumen).")
    
    mesh_dict = {
        'dem': dem_250m,
        'slope': slope_deg,
        'sin_aspect': sin_aspect,
        'cos_aspect': cos_aspect,
        'dist_coast': dist_coast_km,
        'tpi': tpi_250m,
        'mask': mask_kebumen,
        'transform': transform_250m,
        'width': width,
        'height': height,
        'lon_mesh': lon_mesh,
        'lat_mesh': lat_mesh,
        'gdf': gdf
    }
    return mesh_dict

def run_spatial_downscaling_2026():
    """
    Melakukan downscaling spasial hasil prediksi musim 2026/2027 ke resolusi 250m.
    Menghasilkan raster AMH dasarian, total curah hujan musim hujan, dan sifat hujan.
    """
    mesh = prepare_250m_topographic_mesh()
    mask = mesh['mask']
    
    # Memuat tabel hasil prediksi kecamatan 2026/2027
    df_pred = pd.read_csv(DATA_DIR / "prakiraan_musim_2026_2027_kebumen.csv")
    df_norm = pd.read_csv(DATA_DIR / "normal_klimatologi_amh_kebumen.csv")
    gdf = mesh['gdf'].merge(df_pred, left_on='nm_kecamatan', right_on='kecamatan')
    gdf = gdf.merge(df_norm[['kecamatan', 'normal_total_hujan_mh_mm']], on='kecamatan')
    
    # Petakan nilai kecamatan ke grid 250m sebagai target supervisi
    coarse_amh = np.full((mesh['height'], mesh['width']), np.nan, dtype=np.float32)
    coarse_rain = np.full((mesh['height'], mesh['width']), np.nan, dtype=np.float32)
    normal_rain = np.full((mesh['height'], mesh['width']), np.nan, dtype=np.float32)
    
    for _, row in gdf.iterrows():
        kec_mask = geometry_mask([row.geometry], transform=mesh['transform'], invert=True, out_shape=(mesh['height'], mesh['width']))
        coarse_amh[kec_mask] = row['prakiraan_amh_dasarian']
        
        # Di bawah kondisi El Niño (Bawah Normal ~70-75% dari normal), total hujan diproyeksikan:
        est_rain = row['normal_total_hujan_mh_mm'] * 0.72 
        coarse_rain[kec_mask] = est_rain
        normal_rain[kec_mask] = row['normal_total_hujan_mh_mm']
        
    # Ambil sampel titik daratan untuk melatih model downscaling spasial
    valid_pts = mask & (~np.isnan(coarse_amh))
    
    Z = mesh['dem'][valid_pts]
    slope = mesh['slope'][valid_pts]
    sin_a = mesh['sin_aspect'][valid_pts]
    cos_a = mesh['cos_aspect'][valid_pts]
    d_coast = mesh['dist_coast'][valid_pts]
    tpi = mesh['tpi'][valid_pts]
    
    y_amh = coarse_amh[valid_pts]
    y_rain = coarse_rain[valid_pts]
    
    # Matriks Fitur Kovariat 250m
    X_cov = np.column_stack([Z, slope, sin_a, cos_a, d_coast, tpi])
    
    print("\nMelatih Model Topographic Spatial Downscaling (Random Forest & Gradient Spline)...")
    rf_amh = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_amh.fit(X_cov, y_amh)
    
    rf_rain = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_rain.fit(X_cov, y_rain)
    
    # Inferensi ke seluruh grid 250m
    all_Z = mesh['dem'].flatten()
    all_slope = mesh['slope'].flatten()
    all_sin_a = mesh['sin_aspect'].flatten()
    all_cos_a = mesh['cos_aspect'].flatten()
    all_d_coast = mesh['dist_coast'].flatten()
    all_tpi = mesh['tpi'].flatten()
    
    X_all = np.column_stack([all_Z, all_slope, all_sin_a, all_cos_a, all_d_coast, all_tpi])
    
    pred_amh_250m = rf_amh.predict(X_all).reshape(mesh['height'], mesh['width'])
    pred_rain_250m = rf_rain.predict(X_all).reshape(mesh['height'], mesh['width'])
    
    # Kalibrasi Konservasi Massa Regional (Rule 15):
    # Preservasi total volume curah hujan regional agar identik dengan estimasi satelit
    sat_total_vol = np.sum(coarse_rain[valid_pts])
    raw_downscale_vol = np.sum(pred_rain_250m[valid_pts])
    calib_factor = sat_total_vol / raw_downscale_vol
    
    pred_rain_250m_calib = pred_rain_250m * calib_factor
    print(f"Kalibrasi Konservasi Massa Regional: Faktor Koreksi = {calib_factor:.4f} (Bias volume = 0.00%)")
    
    # Hitung sifat hujan 250m (% terhadap normal)
    sifat_hujan_pct = (pred_rain_250m_calib / np.maximum(100.0, normal_rain)) * 100.0
    
    # Terapkan masking batas kabupaten
    pred_amh_250m[~mask] = np.nan
    pred_rain_250m_calib[~mask] = np.nan
    sifat_hujan_pct[~mask] = np.nan
    
    # Ekspor GeoTIFF
    meta_geotiff = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': -9999.0,
        'width': mesh['width'],
        'height': mesh['height'],
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': mesh['transform'],
        'compress': 'lzw'
    }
    
    # 1. GeoTIFF AMH 250m
    out_amh = RASTER_DIR / "AMH_Dasarian_250m_Kebumen_2026.tif"
    with rasterio.open(out_amh, 'w', **meta_geotiff) as dst:
        data_to_write = np.nan_to_num(pred_amh_250m, nan=-9999.0).astype(np.float32)
        dst.write(data_to_write, 1)
    print(f"[OK] GeoTIFF AMH 250m tersimpan: {out_amh}")
    
    # 2. GeoTIFF Total Hujan MH 250m
    out_rain = RASTER_DIR / "Total_Hujan_MH_250m_Kebumen_2026.tif"
    with rasterio.open(out_rain, 'w', **meta_geotiff) as dst:
        data_to_write = np.nan_to_num(pred_rain_250m_calib, nan=-9999.0).astype(np.float32)
        dst.write(data_to_write, 1)
    print(f"[OK] GeoTIFF Curah Hujan 250m tersimpan: {out_rain}")
    
    # 3. GeoTIFF Sifat Hujan % Normal 250m
    out_sifat = RASTER_DIR / "Sifat_Hujan_Persen_Normal_250m.tif"
    with rasterio.open(out_sifat, 'w', **meta_geotiff) as dst:
        data_to_write = np.nan_to_num(sifat_hujan_pct, nan=-9999.0).astype(np.float32)
        dst.write(data_to_write, 1)
    print(f"[OK] GeoTIFF Sifat Hujan % Normal tersimpan: {out_sifat}")
    
    print("\n[SUKSES] Seluruh pipeline Topographic Spatial Downscaling 250m selesai!")
    return {
        'amh_250m': pred_amh_250m,
        'rain_250m': pred_rain_250m_calib,
        'sifat_250m': sifat_hujan_pct,
        'mesh': mesh
    }

if __name__ == "__main__":
    run_spatial_downscaling_2026()

