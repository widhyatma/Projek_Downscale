#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Atlas Normal Klimatologi Dasarian 250m Kabupaten Kebumen
Membangun 36 raster GeoTIFF baseline normal klimatologi dasarian (Januari I s/d Desember III)
dan 1 raster normal tahunan pada resolusi 250m berbasis fusi DEM topografi dan konservasi massa (Rule 15 & 20).
"""

import os
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
ATLAS_DIR = BASE_DIR / "prediksi_musim" / "atlas_normal_250m"
DEM_PATH = BASE_DIR / "data" / "DEM_Topografi_Kebumen_30m.tif"
GEOJSON_PATH = BASE_DIR / "33.05_kecamatan.geojson"

ATLAS_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_SIZE_DEG = 0.00225

def prepare_mesh():
    """Mengekstrak grid 250m dan kovariat topografi dari DEM."""
    with rasterio.open(DEM_PATH) as src:
        bounds = src.bounds
        dem_raw = src.read(1).astype(np.float32)
        dem_raw[dem_raw < -100] = 0.0
        
    west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
    width = int(np.round((east - west) / PIXEL_SIZE_DEG))
    height = int(np.round((north - south) / PIXEL_SIZE_DEG))
    transform = from_bounds(west, south, east, north, width, height)
    
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
    mask_kebumen = geometry_mask(gdf.geometry, transform=transform, invert=True, out_shape=(height, width))
    
    return {
        'dem': dem_250m, 'slope': slope_deg, 'sin_aspect': sin_aspect, 'cos_aspect': cos_aspect,
        'dist_coast': dist_coast_km, 'tpi': tpi_250m, 'mask': mask_kebumen,
        'transform': transform, 'width': width, 'height': height, 'gdf': gdf
    }

def generate_atlas_36_dasarians():
    """Membangun 36 file GeoTIFF normal dasarian 250m sepanjang tahun."""
    print("Mempersiapkan mesh topografi 250m...")
    mesh = prepare_mesh()
    mask = mesh['mask']
    gdf = mesh['gdf']
    
    df_das = pd.read_csv(DATA_DIR / "dasarian_curah_hujan_kebumen_2000_2026.csv")
    
    # Rata-rata normal per kecamatan per dasarian (1..36)
    df_norm_das = df_das[df_das['tahun'] <= 2025].groupby(['kecamatan', 'dasarian'])['curah_hujan'].mean().reset_index()
    
    meta_geotiff = {
        'driver': 'GTiff', 'dtype': 'float32', 'nodata': -9999.0,
        'width': mesh['width'], 'height': mesh['height'], 'count': 1,
        'crs': 'EPSG:4326', 'transform': mesh['transform'], 'compress': 'lzw'
    }
    
    all_Z = mesh['dem'].flatten()
    all_slope = mesh['slope'].flatten()
    all_sin_a = mesh['sin_aspect'].flatten()
    all_cos_a = mesh['cos_aspect'].flatten()
    all_d_coast = mesh['dist_coast'].flatten()
    all_tpi = mesh['tpi'].flatten()
    X_all = np.column_stack([all_Z, all_slope, all_sin_a, all_cos_a, all_d_coast, all_tpi])
    
    annual_sum_grid = np.zeros((mesh['height'], mesh['width']), dtype=np.float32)
    
    print("\nMenghasilkan 36 Raster Normal Dasarian 250m (Rule 15 & 20)...")
    
    for d in range(1, 37):
        sub_d = df_norm_das[df_norm_das['dasarian'] == d]
        gdf_d = gdf.merge(sub_d, left_on='nm_kecamatan', right_on='kecamatan')
        
        coarse_grid = np.full((mesh['height'], mesh['width']), np.nan, dtype=np.float32)
        for _, row in gdf_d.iterrows():
            k_mask = geometry_mask([row.geometry], transform=mesh['transform'], invert=True, out_shape=(mesh['height'], mesh['width']))
            coarse_grid[k_mask] = row['curah_hujan']
            
        valid_pts = mask & (~np.isnan(coarse_grid))
        
        X_train = np.column_stack([
            mesh['dem'][valid_pts],
            mesh['slope'][valid_pts],
            mesh['sin_aspect'][valid_pts],
            mesh['cos_aspect'][valid_pts],
            mesh['dist_coast'][valid_pts],
            mesh['tpi'][valid_pts]
        ])
        y_train = coarse_grid[valid_pts]
        
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        
        pred_250m = rf.predict(X_all).reshape(mesh['height'], mesh['width'])
        
        # Kalibrasi Konservasi Massa Regional (Rule 15)
        sat_vol = np.sum(coarse_grid[valid_pts])
        down_vol = np.sum(pred_250m[valid_pts])
        if down_vol > 0:
            calib_factor = sat_vol / down_vol
            pred_250m = pred_250m * calib_factor
            
        pred_250m[~mask] = np.nan
        annual_sum_grid += np.nan_to_num(pred_250m, nan=0.0)
        
        out_file = ATLAS_DIR / f"Normal_Dasarian_{d:02d}_250m.tif"
        with rasterio.open(out_file, 'w', **meta_geotiff) as dst:
            dst.write(np.nan_to_num(pred_250m, nan=-9999.0).astype(np.float32), 1)
            
        if d % 6 == 0 or d == 1:
            print(f"  -> Dasarian {d:02d}/36 selesai diekspor.")
            
    # Ekspor Normal Tahunan 250m
    annual_sum_grid[~mask] = np.nan
    out_annual = ATLAS_DIR / "Normal_Tahunan_250m_Kebumen.tif"
    with rasterio.open(out_annual, 'w', **meta_geotiff) as dst:
        dst.write(np.nan_to_num(annual_sum_grid, nan=-9999.0).astype(np.float32), 1)
        
    print(f"\n[SUKSES] Seluruh 36 GeoTIFF Atlas Normal Dasarian 250m tersimpan di: {ATLAS_DIR}")
    print(f"[SUKSES] GeoTIFF Normal Tahunan 250m tersimpan di: {out_annual}")

if __name__ == "__main__":
    generate_atlas_36_dasarians()

