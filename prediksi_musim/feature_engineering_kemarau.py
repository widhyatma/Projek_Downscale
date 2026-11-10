#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Feature Engineering Prediksi Musim Kemarau Kebumen
Mengintegrasikan Indeks Iklim Global (ENSO ONI DJF/JFM, IOD DMI Feb/Mar) 
dan Reanalisis Sirkulasi Atmosferik ERA5-Land (Januari-Februari-Maret)
dengan Riwayat AMK Historis 26 Tahun (2000-2025) untuk 26 Kecamatan.
"""

from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
import xarray as xr

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
ERA5_DIR = BASE_DIR / "data" / "era5_land"

def load_noaa_indices():
    """Memuat dan mengekstrak indeks NOAA ONI (DJF & JFM) serta NOAA DMI (Feb & Mar)."""
    # 1. ONI
    oni_file = DATA_DIR / "noaa_oni_raw.txt"
    with open(oni_file, "r") as f:
        text = f.read()
        
    lines = text.strip().split('\n')
    data_oni = []
    for l in lines[1:]:
        p = l.split()
        if len(p) >= 4:
            data_oni.append({'SEAS': p[0], 'tahun': int(p[1]), 'oni_anom': float(p[3])})
    df_oni = pd.DataFrame(data_oni)
    
    oni_djf = df_oni[df_oni['SEAS'] == 'DJF'].set_index('tahun')['oni_anom'].to_dict()
    oni_jfm = df_oni[df_oni['SEAS'] == 'JFM'].set_index('tahun')['oni_anom'].to_dict()
    
    # 2. DMI
    dmi_file = DATA_DIR / "noaa_dmi_raw.txt"
    with open(dmi_file, "r") as f:
        text_dmi = f.read()
        
    records_dmi = []
    for l in text_dmi.strip().split('\n'):
        p = l.split()
        if len(p) == 13:
            try:
                yr = int(p[0])
                if 1950 <= yr <= 2030:
                    for m_idx in range(1, 13):
                        val = float(p[m_idx])
                        if val > -90.0:
                            records_dmi.append({'tahun': yr, 'bulan': m_idx, 'dmi': val})
            except ValueError:
                continue
    df_dmi = pd.DataFrame(records_dmi)
    dmi_feb = df_dmi[df_dmi['bulan'] == 2].set_index('tahun')['dmi'].to_dict()
    dmi_mar = df_dmi[df_dmi['bulan'] == 3].set_index('tahun')['dmi'].to_dict()
    
    return oni_djf, oni_jfm, dmi_feb, dmi_mar

def extract_era5_jfm_predictors(start_year=2000, end_year=2026):
    """
    Mengekstrak prediktor atmosferik lokal di Kebumen pada bulan prakondisi kemarau (Januari, Februari, Maret):
    - u10 (kecepatan angin zonal)
    - rh (kelembapan relatif permukaan)
    - t2m (suhu udara permukaan)
    """
    records = []
    print("Mengekstrak prediktor sirkulasi atmosferik ERA5-Land (Jan-Feb-Mar 2000-2026)...")
    
    for yr in range(start_year, end_year + 1):
        for mo in [1, 2, 3]:
            nc_path = ERA5_DIR / str(yr) / f"era5_land_{yr}_{mo:02d}.nc"
            if not nc_path.exists():
                nc_path = ERA5_DIR / f"era5_land_{yr}_{mo:02d}.nc"
                
            if nc_path.exists():
                try:
                    with xr.open_dataset(nc_path, engine='netcdf4') as ds:
                        u_arr = ds['u_wind_10m'].values
                        u_valid = u_arr[np.abs(u_arr) < 900.0]
                        u10_val = float(np.nanmean(u_valid)) if len(u_valid) > 0 else 0.0
                        
                        t_arr = ds['temperature_2m'].values
                        t_valid = t_arr[np.abs(t_arr) < 900.0]
                        t2m_val = float(np.nanmean(t_valid)) if len(t_valid) > 0 else 26.0
                        
                        td_arr = ds['dewpoint_temperature_2m'].values
                        td_valid = td_arr[np.abs(td_arr) < 900.0]
                        tdew_val = float(np.nanmean(td_valid)) if len(td_valid) > 0 else 23.0
                        
                        # August-Roche-Magnus equation untuk RH %
                        es = 6.112 * np.exp((17.67 * t2m_val) / (t2m_val + 243.5))
                        e = 6.112 * np.exp((17.67 * tdew_val) / (tdew_val + 243.5))
                        rh_val = np.clip((e / es) * 100.0, 0.0, 100.0)
                        
                        records.append({
                            'tahun': yr,
                            'bulan': mo,
                            f'u10_m{mo}': round(u10_val, 3),
                            f'rh_m{mo}': round(rh_val, 2),
                            f't2m_m{mo}': round(t2m_val, 2)
                        })
                except Exception as e:
                    pass
                    
    df = pd.DataFrame(records)
    pivot_dict = {}
    for yr in range(start_year, end_year + 1):
        sub = df[df['tahun'] == yr]
        if not sub.empty:
            row = {'tahun': yr}
            for _, r in sub.iterrows():
                mo = int(r['bulan'])
                row[f'u10_m{mo}'] = r[f'u10_m{mo}']
                row[f'rh_m{mo}'] = r[f'rh_m{mo}']
                row[f't2m_m{mo}'] = r[f't2m_m{mo}']
            pivot_dict[yr] = row
            
    df_pivot = pd.DataFrame(list(pivot_dict.values()))
    return df_pivot

def build_training_matrix_kemarau():
    """
    Membangun dataset tabular training terpadu untuk Prediksi Musim Kemarau.
    """
    amk_path = DATA_DIR / "amk_historis_kebumen_2000_2025.csv"
    norm_path = DATA_DIR / "normal_klimatologi_amk_kebumen.csv"
    
    df_amk = pd.read_csv(amk_path)
    df_norm = pd.read_csv(norm_path)
    
    df_merged = df_amk.merge(
        df_norm[['kecamatan', 'normal_amk_dasarian', 'normal_total_hujan_mk_mm']],
        on='kecamatan',
        how='left'
    )
    
    # Selisih dasarian terhadap normal (negatif = Maju, positif = Mundur)
    df_merged['selisih_dasarian'] = df_merged['amk_dasarian'] - df_merged['normal_amk_dasarian']
    
    # Kategori AMK (Standar BMKG):
    # Maju: <= -1.0 dasarian | Sama: -1.0 s.d. +1.0 | Mundur: >= +1.0 dasarian
    def categorize_amk(diff):
        if diff <= -1.0:
            return "Maju"
        elif diff >= 1.0:
            return "Mundur"
        else:
            return "Sama"
            
    df_merged['kategori_amk'] = df_merged['selisih_dasarian'].apply(categorize_amk)
    
    # Sifat Hujan Kemarau:
    # Bawah Normal: < 85% (Kemarau Kering) | Normal: 85% - 115% | Atas Normal: > 115% (Kemarau Basah)
    df_merged['persentase_normal_hujan'] = (df_merged['total_hujan_mk_mm'] / df_merged['normal_total_hujan_mk_mm']) * 100.0
    def categorize_sifat(pct):
        if pct < 85.0:
            return "Bawah Normal"
        elif pct > 115.0:
            return "Atas Normal"
        else:
            return "Normal"
            
    df_merged['sifat_hujan_mk'] = df_merged['persentase_normal_hujan'].apply(categorize_sifat)
    
    # Prediktor Global
    oni_djf, oni_jfm, dmi_feb, dmi_mar = load_noaa_indices()
    df_merged['oni_djf'] = df_merged['tahun'].map(oni_djf)
    df_merged['oni_jfm'] = df_merged['tahun'].map(oni_jfm)
    df_merged['dmi_feb'] = df_merged['tahun'].map(dmi_feb)
    df_merged['dmi_mar'] = df_merged['tahun'].map(dmi_mar)
    
    # Prediktor Lokal ERA5-Land
    df_era5 = extract_era5_jfm_predictors(start_year=2000, end_year=2026)
    if not df_era5.empty:
        df_merged = df_merged.merge(df_era5, on='tahun', how='left')
        
    # Forward-fill / imputasi mean jika ada nilai kosong di tahun terbaru
    numeric_cols = [c for c in df_merged.columns if c.startswith(('oni_', 'dmi_', 'u10_', 'rh_', 't2m_'))]
    for c in numeric_cols:
        df_merged[c] = df_merged[c].fillna(df_merged[c].mean())
        
    out_file = DATA_DIR / "dataset_training_prediksi_kemarau.csv"
    df_merged.to_csv(out_file, index=False)
    print(f"\n[SUKSES] Dataset training musim kemarau berhasil dibangun: {len(df_merged)} baris x {len(df_merged.columns)} kolom.")
    print(f"Tersimpan di: {out_file}")
    return df_merged

if __name__ == "__main__":
    build_training_matrix_kemarau()

