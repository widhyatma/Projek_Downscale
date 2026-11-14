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
from scipy.ndimage import uniform_filter
from scipy.interpolate import RBFInterpolator
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
from shapely import force_2d
import pyproj

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
RASTER_DIR = BASE_DIR / "prediksi_musim" / "rasters_250m"
DEM_PATH = BASE_DIR / "data" / "DEM_Topografi_Kebumen_30m.tif"
GEOJSON_PATH = BASE_DIR / "kebumen_33.05_batas_kecamatan.geojson"

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
    
    # Koordinat Proyeksi Metrik UTM 49S (EPSG:32749, Rule 12)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
    grid_utm_x, grid_utm_y = transformer.transform(lon_mesh, lat_mesh)
    
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
        'grid_utm_x': grid_utm_x,
        'grid_utm_y': grid_utm_y,
        'gdf': gdf
    }
    return mesh_dict

def downscale_variable_multimodel(gdf_pts, val_col, mesh, var_name="Variabel", apply_mass_conservation=False):
    """
    Menjalankan Ensemble Multi-Model Downscaling (Rule 11):
    1. Regression-Kriging (RFK): RF Kovariat Medan + Gaussian Semivariogram Ordinary Kriging Residual
    2. ANUSPLIN / Thin-Plate Spline (TPS): C^2 continuous trivariate spline (X, Y, Z)
    3. PRISM: Topographic-Coastal Facet Regression (Wd, Wz, Wc)
    4. Multi-Model Consensus Ensemble: Rata-rata tertimbang + Peta Ketidakpastian Spasial (Sigma)
    5. Pelestarian Konservasi Massa Regional (Rule 15): Bias Volume = 0.00%
    """
    mask = mesh['mask']
    H, W = mesh['height'], mesh['width']
    
    # 1. Koordinat Metrik Titik Kontrol (UTM 49S, Rule 12)
    gdf_utm = gdf_pts.to_crs(epsg=32749)
    pts_x = gdf_utm.geometry.centroid.x.values
    pts_y = gdf_utm.geometry.centroid.y.values
    y_target = gdf_pts[val_col].values.astype(np.float32)
    
    inv_transform = ~mesh['transform']
    pt_Z, pt_slope, pt_sin_a, pt_cos_a, pt_d_coast, pt_tpi = [], [], [], [], [], []
    
    for _, row in gdf_pts.iterrows():
        c_pt = row.geometry.centroid
        col, r = inv_transform * (c_pt.x, c_pt.y)
        col = int(np.clip(col, 0, W - 1))
        r = int(np.clip(r, 0, H - 1))
        pt_Z.append(mesh['dem'][r, col])
        pt_slope.append(mesh['slope'][r, col])
        pt_sin_a.append(mesh['sin_aspect'][r, col])
        pt_cos_a.append(mesh['cos_aspect'][r, col])
        pt_d_coast.append(mesh['dist_coast'][r, col])
        pt_tpi.append(mesh['tpi'][r, col])
        
    X_pts = np.column_stack([pt_Z, pt_slope, pt_sin_a, pt_cos_a, pt_d_coast, pt_tpi])
    
    all_Z = mesh['dem'].flatten()
    all_slope = mesh['slope'].flatten()
    all_sin_a = mesh['sin_aspect'].flatten()
    all_cos_a = mesh['cos_aspect'].flatten()
    all_d_coast = mesh['dist_coast'].flatten()
    all_tpi = mesh['tpi'].flatten()
    X_all = np.column_stack([all_Z, all_slope, all_sin_a, all_cos_a, all_d_coast, all_tpi])
    
    # --- MODEL 1: Regression-Kriging (RFK) ---
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_pts, y_target)
    trend_pts = rf.predict(X_pts)
    res_pts = y_target - trend_pts
    
    ok = OrdinaryKriging(pts_x, pts_y, res_pts, variogram_model='gaussian', verbose=False, enable_plotting=False)
    kriged_res, _ = ok.execute('points', mesh['grid_utm_x'].flatten(), mesh['grid_utm_y'].flatten())
    kriged_res = kriged_res.reshape(H, W)
    pred_rfk = rf.predict(X_all).reshape(H, W) + kriged_res
    
    # --- MODEL 2: ANUSPLIN / Thin-Plate Spline (TPS) ---
    scale_xy, scale_z = 1000.0, 1000.0
    pts_tps = np.column_stack([pts_x / scale_xy, pts_y / scale_xy, np.array(pt_Z) / scale_z])
    grid_tps = np.column_stack([mesh['grid_utm_x'].flatten() / scale_xy, mesh['grid_utm_y'].flatten() / scale_xy, all_Z / scale_z])
    rbf = RBFInterpolator(pts_tps, y_target, kernel='thin_plate_spline', smoothing=0.1)
    pred_tps = rbf(grid_tps).reshape(H, W)
    
    # --- MODEL 3: PRISM Topographic-Coastal Facet Model ---
    valid_r, valid_c = np.where(mask)
    g_x = mesh['grid_utm_x'][valid_r, valid_c, np.newaxis]
    g_y = mesh['grid_utm_y'][valid_r, valid_c, np.newaxis]
    g_z = mesh['dem'][valid_r, valid_c, np.newaxis]
    g_c = mesh['dist_coast'][valid_r, valid_c, np.newaxis]
    
    p_x = pts_x[np.newaxis, :]
    p_y = pts_y[np.newaxis, :]
    p_z = np.array(pt_Z)[np.newaxis, :]
    p_c = np.array(pt_d_coast)[np.newaxis, :]
    
    d_km = np.sqrt((g_x - p_x)**2 + (g_y - p_y)**2) / 1000.0
    w_d = 1.0 / (d_km + 2.0)**2
    w_z = 1.0 / (np.abs(g_z - p_z) + 25.0)
    w_c = np.exp(-np.abs(g_c - p_c) / 10.0)
    w = w_d * w_z * w_c
    
    prism_vals = np.sum(w * y_target[np.newaxis, :], axis=1) / np.maximum(1e-6, np.sum(w, axis=1))
    pred_prism = np.full((H, W), np.nan, dtype=np.float32)
    pred_prism[valid_r, valid_c] = prism_vals
    
    # --- MODEL 4: Multi-Model Consensus Ensemble & Uncertainty Sigma ---
    ensemble_pred = (pred_rfk + pred_tps + pred_prism) / 3.0
    uncertainty_sigma = np.std([pred_rfk, pred_tps, pred_prism], axis=0)
    
    # --- Penegakan Konservasi Massa Regional (Rule 15) ---
    if apply_mass_conservation:
        calib_factor = np.mean(y_target) / np.mean(ensemble_pred[mask])
        ensemble_pred = ensemble_pred * calib_factor
        print(f"  -> Kalibrasi Konservasi Massa Regional ({var_name}): Faktor = {calib_factor:.4f} (Bias Volume = 0.00%)")
        
    ensemble_pred[~mask] = np.nan
    uncertainty_sigma[~mask] = np.nan
    
    print(f"  -> {var_name} Downscaled: Ensemble Rata-rata = {np.nanmean(ensemble_pred):.2f}, Uncertainty Sigma = {np.nanmean(uncertainty_sigma):.2f}")
    return ensemble_pred, uncertainty_sigma

def run_spatial_downscaling_2026():
    """
    Melakukan downscaling spasial multi-model hasil prediksi musim 2026/2027 ke resolusi 250m.
    Menghasilkan raster AMH dasarian, total curah hujan musim hujan, sifat hujan, dan ketidakpastian spasial.
    """
    print("\n" + "="*80)
    print("MEMULAI TOPOGRAPHIC SPATIAL DOWNSCALING 250M MULTI-MODEL ENSEMBLE (RULE 11)")
    print("Suite: Regression-Kriging (RFK) + PRISM Facet Model + ANUSPLIN Thin-Plate Spline")
    print("="*80)
    
    mesh = prepare_250m_topographic_mesh()
    mask = mesh['mask']
    
    # Memuat tabel hasil prediksi kecamatan 2026/2027
    df_pred = pd.read_csv(DATA_DIR / "prakiraan_musim_2026_2027_kebumen.csv")
    df_norm = pd.read_csv(DATA_DIR / "normal_klimatologi_amh_kebumen.csv")
    gdf = mesh['gdf'].merge(df_pred, left_on='nm_kecamatan', right_on='kecamatan')
    gdf = gdf.merge(df_norm[['kecamatan', 'normal_total_hujan_mh_mm']], on='kecamatan')
    
    # Proyeksi curah hujan total estimasi 2026/2027 (El Nino: Bawah Normal ~72%)
    gdf['prakiraan_total_mh_mm'] = gdf['normal_total_hujan_mh_mm'] * 0.72
    
    # 1. Downscaling AMH Dasarian Multi-Model
    print("\n[1/3] Menjalankan Downscaling Multi-Model untuk Awal Musim Hujan (AMH Dasarian)...")
    amh_250m, amh_sigma = downscale_variable_multimodel(gdf, 'prakiraan_amh_dasarian', mesh, "AMH Dasarian", apply_mass_conservation=False)
    
    # 2. Downscaling Total Curah Hujan Musim Hujan dengan Konservasi Massa Regional (Rule 15)
    print("\n[2/3] Menjalankan Downscaling Multi-Model Total Curah Hujan Musim Hujan (mm)...")
    rain_250m, rain_sigma = downscale_variable_multimodel(gdf, 'prakiraan_total_mh_mm', mesh, "Total Curah Hujan MH", apply_mass_conservation=True)
    
    # 3. Downscaling Normal Curah Hujan untuk Menghitung Sifat Hujan (% Normal)
    print("\n[3/3] Menjalankan Downscaling Normal Curah Hujan MH (mm)...")
    norm_rain_250m, _ = downscale_variable_multimodel(gdf, 'normal_total_hujan_mh_mm', mesh, "Normal Hujan MH", apply_mass_conservation=True)
    
    sifat_hujan_pct = (rain_250m / np.maximum(100.0, norm_rain_250m)) * 100.0
    sifat_hujan_pct[~mask] = np.nan
    
    # Ekspor GeoTIFF (EPSG:4326, LZW)
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
    
    files_to_export = [
        ("AMH_Dasarian_250m_Kebumen_2026.tif", amh_250m),
        ("Total_Hujan_MH_250m_Kebumen_2026.tif", rain_250m),
        ("Sifat_Hujan_Persen_Normal_250m.tif", sifat_hujan_pct),
        ("Uncertainty_Downscaling_MH_250m.tif", rain_sigma)
    ]
    
    print("\nMengekspor Hasil Raster GeoTIFF 250m ke 'prediksi_musim/rasters_250m/'...")
    for fname, arr in files_to_export:
        out_path = RASTER_DIR / fname
        with rasterio.open(out_path, 'w', **meta_geotiff) as dst:
            dst.write(np.nan_to_num(arr, nan=-9999.0).astype(np.float32), 1)
        print(f"  -> GeoTIFF tersimpan: {fname}")
        
    print("\n[SUKSES] Seluruh pipeline Topographic Spatial Downscaling Multi-Model 250m selesai!")
    return {
        'amh_250m': amh_250m,
        'rain_250m': rain_250m,
        'sifat_250m': sifat_hujan_pct,
        'uncertainty_sigma': rain_sigma,
        'mesh': mesh
    }

if __name__ == "__main__":
    run_spatial_downscaling_2026()

