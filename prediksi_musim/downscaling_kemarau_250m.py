#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Spatial Downscaling Topografi Resolusi Ultra-Tinggi (250m) Prediksi Musim Kemarau Kebumen
Memproyeksikan Awal Musim Kemarau (AMK), Akumulasi Presipitasi Kemarau, dan Indeks Kerawanan Kekeringan
ke grid raster kontinyu 250m dengan Multi-Model Ensemble (RFK + PRISM + ANUSPLIN) & Rule 15.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
sys.path.insert(0, str(BASE_DIR))

from prediksi_musim.topographic_downscaling import prepare_250m_topographic_mesh, downscale_variable_multimodel

DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
RASTER_DIR = BASE_DIR / "prediksi_musim" / "rasters_kemarau_250m"
RASTER_DIR.mkdir(parents=True, exist_ok=True)

def run_spatial_downscaling_kemarau_2026():
    """
    Menjalankan proses downscaling spasial multi-model (RFK + PRISM + ANUSPLIN) 250m musim kemarau 2026.
    Memproyeksikan AMK dasarian, akumulasi hujan kemarau, indeks kekeringan, dan ketidakpastian spasial.
    """
    print("\n" + "="*80)
    print("MEMULAI TOPOGRAPHIC SPATIAL DOWNSCALING 250M MUSIM KEMARAU MULTI-MODEL ENSEMBLE (RULE 11)")
    print("Suite: Regression-Kriging (RFK) + PRISM Facet Model + ANUSPLIN Thin-Plate Spline")
    print("="*80)
    
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
    
    # 1. Downscaling AMK Dasarian Multi-Model
    print("\n[1/3] Menjalankan Downscaling Multi-Model untuk Awal Musim Kemarau (AMK Dasarian)...")
    amk_250m, amk_sigma = downscale_variable_multimodel(gdf_pred, 'prakiraan_amk_dasarian', mesh, "AMK Dasarian", apply_mass_conservation=False)
    
    # 2. Downscaling Total Presipitasi Kemarau dengan Konservasi Massa Regional (Rule 15)
    print("\n[2/3] Menjalankan Downscaling Multi-Model Total Curah Hujan Kemarau (mm)...")
    tot_mk_250m, tot_mk_sigma = downscale_variable_multimodel(gdf_pred, 'prakiraan_total_mk_mm', mesh, "Total Hujan Kemarau", apply_mass_conservation=True)
    
    # 3. Downscaling Normal Total Presipitasi Kemarau
    print("\n[3/3] Menjalankan Downscaling Multi-Model Normal Total Presipitasi Kemarau (mm)...")
    norm_tot_250m, _ = downscale_variable_multimodel(gdf_pred, 'normal_total_hujan_mk_mm', mesh, "Normal Hujan Kemarau", apply_mass_conservation=True)
    
    # Persentase terhadap Normal
    persen_normal_250m = (tot_mk_250m / np.maximum(norm_tot_250m, 1.0)) * 100.0
    persen_normal_250m[~mask] = np.nan
    
    # 4. Indeks Kerentanan Kekeringan Pertanian 250m (0..100)
    # Defisit curah hujan tinggi + elevasi rendah/pantai + lereng curam
    defisit_score = np.clip((100.0 - persen_normal_250m) * 2.0, 0.0, 60.0) # maks 60 poin
    pantai_score = np.clip((30.0 - mesh['dist_coast']) / 30.0 * 25.0, 0.0, 25.0) # maks 25 poin
    lereng_score = np.clip(mesh['slope'] / 40.0 * 15.0, 0.0, 15.0) # maks 15 poin
    indeks_kekeringan = defisit_score + pantai_score + lereng_score
    indeks_kekeringan[~mask] = np.nan
    
    # Ekspor GeoTIFF (EPSG:4326, LZW)
    meta_gtiff = {
        'driver': 'GTiff', 'dtype': 'float32', 'nodata': -9999.0,
        'width': mesh['width'], 'height': mesh['height'], 'count': 1,
        'crs': 'EPSG:4326', 'transform': mesh['transform'], 'compress': 'lzw'
    }
    
    files_to_export = [
        ('AMK_Dasarian_250m_Kebumen_2026.tif', amk_250m),
        ('Total_Hujan_MK_250m_Kebumen_2026.tif', tot_mk_250m),
        ('Sifat_Hujan_MK_Persen_Normal_250m.tif', persen_normal_250m),
        ('Indeks_Kerawanan_Kekeringan_250m.tif', indeks_kekeringan),
        ('Uncertainty_Downscaling_MK_250m.tif', tot_mk_sigma)
    ]
    
    print("\nMengekspor Hasil Raster GeoTIFF 250m ke 'prediksi_musim/rasters_kemarau_250m/'...")
    for fname, arr in files_to_export:
        out_path = RASTER_DIR / fname
        with rasterio.open(out_path, 'w', **meta_gtiff) as dst:
            dst.write(np.nan_to_num(arr, nan=-9999.0).astype(np.float32), 1)
        print(f"  -> GeoTIFF tersimpan: {fname}")
        
    print(f"\n[SUKSES] Seluruh raster downscaling spasial multi-model 250m tersimpan di: {RASTER_DIR}")
    return {
        'amk_250m': amk_250m,
        'tot_mk_250m': tot_mk_250m,
        'persen_normal_250m': persen_normal_250m,
        'indeks_kekeringan': indeks_kekeringan,
        'uncertainty_sigma': tot_mk_sigma,
        'mesh': mesh
    }

if __name__ == "__main__":
    run_spatial_downscaling_kemarau_2026()
