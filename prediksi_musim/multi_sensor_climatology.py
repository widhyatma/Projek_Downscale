#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Multi-Sensor Climatology: Evaluasi Komparatif & Fusi Konsensus
Menghitung Baseline Normal Klimatologi Awal Musim Hujan (AMH) 2000-2025
lintas 4 dataset: CHIRPS v3.0, JAXA GSMaP v8, ECMWF ERA5-Land, dan NASA GPM IMERG.
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely import force_2d

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
GEOJSON_PATH = BASE_DIR / "33.05_kecamatan.geojson"

def get_dasarian_label(d_idx):
    d_round = int(round(d_idx))
    m = ((d_round - 1) // 3) + 1
    d_m = ((d_round - 1) % 3) + 1
    m_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    roman = ["I", "II", "III"]
    if 1 <= m <= 12:
        return f"{m_names[m-1]} {roman[d_m-1]}"
    return f"Dasarian {d_round}"

def extract_sensor_dasarians(sensor_name, base_folder, file_pattern, multiplier=1.0, is_imerg=False):
    """
    Mengekstrak curah hujan dasarian 22..36 (Agustus - Desember) 
    untuk 26 kecamatan di Kebumen selama 2000-2025.
    """
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf['geometry'] = force_2d(gdf['geometry'])
    centroids = gdf.geometry.centroid
    cents_x = xr.DataArray(centroids.x.values, dims="kecamatan", coords={"kecamatan": gdf['nm_kecamatan'].values})
    cents_y = xr.DataArray(centroids.y.values, dims="kecamatan", coords={"kecamatan": gdf['nm_kecamatan'].values})
    
    years = range(2000, 2026) # 2000..2025
    months = [8, 9, 10, 11, 12] # Agu s/d Des
    
    print(f"\n[INFO] Mengekstrak dasarian dari {sensor_name} ({len(years)} tahun x 5 bulan)...")
    
    records = []
    
    for yr in years:
        for m in months:
            fname = file_pattern.format(yr=yr, m=m)
            fpath = BASE_DIR / "data" / base_folder / str(yr) / fname
            
            if not fpath.exists():
                continue
                
            try:
                ds = xr.open_dataset(fpath, engine='netcdf4')
                p_var = ds['precipitation']
                n_t = p_var.shape[0]
                
                # Pembagian dasarian dalam bulan:
                # D1: 1-10 (n_t * 10/days_in_month)
                # D2: 11-20
                # D3: 21-end
                if m in [8, 10, 12]:
                    days = 31
                else:
                    days = 30
                    
                steps_per_day = n_t // days
                t1 = 10 * steps_per_day
                t2 = 20 * steps_per_day
                
                p_d1 = p_var[:t1].sum(dim='time') * multiplier
                p_d2 = p_var[t1:t2].sum(dim='time') * multiplier
                p_d3 = p_var[t2:].sum(dim='time') * multiplier
                
                # Interpolasi spasial ke 26 titik kecamatan
                val_d1 = p_d1.interp(x=cents_x, y=cents_y, method='linear').values
                val_d2 = p_d2.interp(x=cents_x, y=cents_y, method='linear').values
                val_d3 = p_d3.interp(x=cents_x, y=cents_y, method='linear').values
                
                ds.close()
                
                base_d = (m - 1) * 3
                for i, kec in enumerate(gdf['nm_kecamatan'].values):
                    records.append({'tahun': yr, 'kecamatan': kec, 'dasarian': base_d + 1, 'curah_hujan': float(val_d1[i])})
                    records.append({'tahun': yr, 'kecamatan': kec, 'dasarian': base_d + 2, 'curah_hujan': float(val_d2[i])})
                    records.append({'tahun': yr, 'kecamatan': kec, 'dasarian': base_d + 3, 'curah_hujan': float(val_d3[i])})
            except Exception as e:
                print(f"Warning: error membaca {fpath}: {e}")
                
    df_res = pd.DataFrame(records)
    print(f"Ekstraksi {sensor_name} selesai: {len(df_res)} baris observasi dasarian.")
    return df_res

def calculate_sensor_amh(df_das, sensor_threshold=50.0):
    """
    Menghitung Awal Musim Hujan (AMH) per tahun per kecamatan berdasarkan kriteria BMKG:
    Dasarian pertama dengan curah hujan >= threshold yang diikuti konfirmasi.
    """
    records_amh = []
    years = sorted(df_das['tahun'].unique())
    kecamatan_list = sorted(df_das['kecamatan'].unique())
    
    for yr in years:
        for kec in kecamatan_list:
            sub = df_das[(df_das['tahun'] == yr) & (df_das['kecamatan'] == kec)].sort_values('dasarian').reset_index(drop=True)
            
            amh_d = 31.0 # Default fallback Nov I
            for idx in range(len(sub) - 2):
                p0 = sub.loc[idx, 'curah_hujan']
                p1 = sub.loc[idx + 1, 'curah_hujan']
                p2 = sub.loc[idx + 2, 'curah_hujan']
                d0 = sub.loc[idx, 'dasarian']
                
                # Ambang batas kriteria BMKG
                if p0 >= sensor_threshold and (p1 >= sensor_threshold or p2 >= sensor_threshold):
                    amh_d = d0
                    break
                    
            records_amh.append({
                'tahun': yr,
                'kecamatan': kec,
                'amh_dasarian': amh_d
            })
            
    return pd.DataFrame(records_amh)

def build_multi_sensor_comparison():
    """
    Menjalankan komparasi 4 sensor dan menghasilkan tabel konsensus klimatologi.
    """
    # 1. CHIRPS (Baseline yang sudah ada)
    df_norm_chirps = pd.read_csv(DATA_DIR / "normal_klimatologi_amh_kebumen.csv")
    
    # 2. GSMaP v8 (Hourly mm/hr, multiplier 1.0, threshold proporsional 40mm karena defisit volume DPR ~25%)
    df_das_gsmap = extract_sensor_dasarians("GSMaP v8", "gsmap", "gsmap_{yr}_{m:02d}.nc", multiplier=1.0)
    df_amh_gsmap = calculate_sensor_amh(df_das_gsmap, sensor_threshold=40.0)
    norm_gsmap = df_amh_gsmap.groupby('kecamatan')['amh_dasarian'].median().rename('normal_amh_gsmap')
    
    # 3. ERA5-Land (Hourly mm/hr, multiplier 1.0, threshold 45mm)
    df_das_era5 = extract_sensor_dasarians("ERA5-Land", "era5_land", "era5_land_{yr}_{m:02d}.nc", multiplier=1.0)
    df_amh_era5 = calculate_sensor_amh(df_das_era5, sensor_threshold=45.0)
    norm_era5 = df_amh_era5.groupby('kecamatan')['amh_dasarian'].median().rename('normal_amh_era5')
    
    # 4. NASA GPM IMERG (Half-hourly mm/hr, multiplier 0.5 jam, threshold 50mm)
    df_das_imerg = extract_sensor_dasarians("NASA GPM IMERG", "imerg", "imerg_{yr}_{m:02d}.nc", multiplier=0.5, is_imerg=True)
    df_amh_imerg = calculate_sensor_amh(df_das_imerg, sensor_threshold=50.0)
    norm_imerg = df_amh_imerg.groupby('kecamatan')['amh_dasarian'].median().rename('normal_amh_imerg')
    
    # Gabungkan ke dalam satu tabel konsensus
    df_comp = df_norm_chirps[['kecamatan', 'normal_amh_dasarian', 'normal_amh_label']].rename(
        columns={'normal_amh_dasarian': 'normal_amh_chirps', 'normal_amh_label': 'normal_label_chirps'}
    )
    df_comp = df_comp.merge(norm_gsmap, on='kecamatan')
    df_comp = df_comp.merge(norm_era5, on='kecamatan')
    df_comp = df_comp.merge(norm_imerg, on='kecamatan')
    
    # Label masing-masing sensor
    df_comp['normal_label_gsmap'] = [get_dasarian_label(d) for d in df_comp['normal_amh_gsmap']]
    df_comp['normal_label_era5'] = [get_dasarian_label(d) for d in df_comp['normal_amh_era5']]
    df_comp['normal_label_imerg'] = [get_dasarian_label(d) for d in df_comp['normal_amh_imerg']]
    
    # Fused Multi-Sensor Consensus (Rata-rata 4 Sensor)
    df_comp['normal_amh_fused'] = np.round(
        (df_comp['normal_amh_chirps'] + df_comp['normal_amh_gsmap'] + df_comp['normal_amh_era5'] + df_comp['normal_amh_imerg']) / 4.0, 1
    )
    df_comp['normal_label_fused'] = [get_dasarian_label(d) for d in df_comp['normal_amh_fused']]
    
    # Dispersi / Selisih ketidakpastian antar-sensor (Standar deviasi dalam hari: 1 dasarian = 10 hari)
    das_matrix = df_comp[['normal_amh_chirps', 'normal_amh_gsmap', 'normal_amh_era5', 'normal_amh_imerg']].values
    df_comp['dispersi_hari_sigma'] = np.round(np.std(das_matrix, axis=1) * 10.0, 1)
    
    out_csv = DATA_DIR / "normal_klimatologi_komparasi_multi_sensor.csv"
    df_comp.to_csv(out_csv, index=False)
    
    print("\n" + "="*95)
    print("HASIL EVALUASI KOMPARATIF NORMAL KLIMATOLOGI AMH MULTI-SENSOR KEBUMEN (2000-2025)")
    print("="*95)
    print(df_comp[['kecamatan', 'normal_label_chirps', 'normal_label_gsmap', 'normal_label_era5', 'normal_label_imerg', 'normal_label_fused', 'dispersi_hari_sigma']].to_string(index=False))
    print("="*95)
    print(f"\n[SUKSES] Tabel komparasi multi-sensor tersimpan di: {out_csv}")
    return df_comp

if __name__ == "__main__":
    build_multi_sensor_comparison()

