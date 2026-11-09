#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Feature Engineering Prediksi Musim Hujan Kebumen
Mengintegrasikan Indeks Iklim Global (ENSO ONI, IOD DMI) dan Reanalisis ERA5-Land
dengan Riwayat AMH Historis 26 Tahun (2000-2025) untuk 26 Kecamatan.
"""

import os
import glob
import urllib.request
import calendar
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
ERA5_DIR = BASE_DIR / "data" / "era5_land"

def fetch_noaa_oni():
    """Mengambil indeks Oceanic Niño Index (ONI / Nino 3.4) resmi dari NOAA CPC."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    cache_file = DATA_DIR / "noaa_oni_raw.txt"
    
    text = ""
    if cache_file.exists():
        with open(cache_file, "r") as f:
            text = f.read()
    else:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                text = response.read().decode('utf-8')
            with open(cache_file, "w") as f:
                f.write(text)
        except Exception as e:
            print(f"Warning: Gagal download ONI online ({e}), menggunakan data lokal jika ada.")

    lines = text.strip().split('\n')
    data = []
    for l in lines[1:]:
        p = l.split()
        if len(p) >= 4:
            data.append({'SEAS': p[0], 'tahun': int(p[1]), 'oni_anom': float(p[3])})
    df = pd.DataFrame(data)
    return df

def fetch_noaa_dmi():
    """Mengambil indeks Dipole Mode Index (DMI / IOD) dari NOAA PSL."""
    url = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
    cache_file = DATA_DIR / "noaa_dmi_raw.txt"
    
    text = ""
    if cache_file.exists():
        with open(cache_file, "r") as f:
            text = f.read()
    else:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                text = response.read().decode('utf-8')
            with open(cache_file, "w") as f:
                f.write(text)
        except Exception as e:
            print(f"Warning: Gagal download DMI online ({e}), menggunakan data lokal jika ada.")

    lines = text.strip().split('\n')
    records = []
    # Baris data dimulai setelah baris header rentang tahun
    for l in lines:
        p = l.split()
        if len(p) == 13: # Tahun + 12 bulan
            try:
                yr = int(p[0])
                if 1950 <= yr <= 2030:
                    for m_idx in range(1, 13):
                        val = float(p[m_idx])
                        if val > -90.0: # Filter missing values (-999)
                            records.append({'tahun': yr, 'bulan': m_idx, 'dmi': val})
            except ValueError:
                continue
    return pd.DataFrame(records)

def extract_era5_predictors(start_year=2000, end_year=2026):
    """
    Mengekstrak prediktor atmosferik lokal di Kebumen pada bulan prakondisi (Juli, Agustus, September):
    - u10 (kecepatan angin zonal baratan/timuran)
    - rh (kelembapan relatif)
    - t2m (suhu udara permukaan)
    """
    records = []
    print("Mengekstrak prediktor sirkulasi atmosferik ERA5-Land (Juli-Sep 2000-2026)...")
    
    for yr in range(start_year, end_year + 1):
        for mo in [7, 8, 9]:
            nc_path = ERA5_DIR / str(yr) / f"era5_land_{yr}_{mo:02d}.nc"
            if not nc_path.exists():
                # Cek di folder utama jika ada
                nc_path = ERA5_DIR / f"era5_land_{yr}_{mo:02d}.nc"
            
            if nc_path.exists():
                try:
                    with xr.open_dataset(nc_path, engine='netcdf4') as ds:
                        # Rule 15: Masking nilai ocean/missing >= 900.0
                        u_arr = ds['u_wind_10m'].values
                        u_valid = u_arr[np.abs(u_arr) < 900.0]
                        u10_val = float(np.nanmean(u_valid)) if len(u_valid) > 0 else 0.0
                        
                        t_arr = ds['temperature_2m'].values
                        t_valid = t_arr[np.abs(t_arr) < 900.0]
                        t2m_val = float(np.nanmean(t_valid)) if len(t_valid) > 0 else 25.0
                        
                        td_arr = ds['dewpoint_temperature_2m'].values
                        td_valid = td_arr[np.abs(td_arr) < 900.0]
                        tdew_val = float(np.nanmean(td_valid)) if len(td_valid) > 0 else 22.0
                        
                        # August-Roche-Magnus equation untuk Kelembapan Relatif (RH %)
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
    # Pivot per tahun
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

def build_training_matrix():
    """
    Menggabungkan seluruh dataset fitur prediktor (ENSO, IOD, ERA5-Land)
    dengan label target AMH historis 26 kecamatan (2000-2025).
    """
    # 1. Load AMH Historis & Normal
    amh_path = DATA_DIR / "amh_historis_kebumen_2000_2025.csv"
    norm_path = DATA_DIR / "normal_klimatologi_amh_kebumen.csv"
    
    if not amh_path.exists():
        raise FileNotFoundError(f"File {amh_path} belum dibuat. Jalankan ekstraksi dasarian terlebih dahulu.")
        
    df_amh = pd.read_csv(amh_path)
    df_norm = pd.read_csv(norm_path)
    
    # Gabungkan dengan normal untuk menghitung anomali (Maju / Sama / Mundur)
    df_merged = df_amh.merge(
        df_norm[['kecamatan', 'normal_amh_dasarian', 'normal_total_hujan_mh_mm']],
        on='kecamatan',
        how='left'
    )
    
    # Selisih dasarian terhadap normal (negatif = Maju, positif = Mundur)
    df_merged['selisih_dasarian'] = df_merged['amh_dasarian'] - df_merged['normal_amh_dasarian']
    
    # Label Kategori AMH (BMKG standard):
    # Maju: <= -1.0 dasarian | Sama: -1.0 s.d. +1.0 | Mundur: >= +1.0 dasarian
    def categorize_amh(diff):
        if diff <= -1.0:
            return "Maju"
        elif diff >= 1.0:
            return "Mundur"
        else:
            return "Sama"
            
    df_merged['kategori_amh'] = df_merged['selisih_dasarian'].apply(categorize_amh)
    
    # Sifat Hujan Musim (BMKG standard):
    # Bawah Normal: < 85% | Normal: 85% - 115% | Atas Normal: > 115%
    df_merged['persentase_normal_hujan'] = (df_merged['total_hujan_mh_mm'] / df_merged['normal_total_hujan_mh_mm']) * 100.0
    def categorize_sifat(pct):
        if pct < 85.0:
            return "Bawah Normal"
        elif pct > 115.0:
            return "Atas Normal"
        else:
            return "Normal"
            
    df_merged['sifat_hujan'] = df_merged['persentase_normal_hujan'].apply(categorize_sifat)
    
    # 2. Ambil Prediktor Iklim Global
    df_oni = fetch_noaa_oni()
    df_dmi = fetch_noaa_dmi()
    
    # Ekstrak ONI periode JJA (Juni-Juli-Agu) dan JAS (Juli-Agu-Sep)
    oni_jas = df_oni[df_oni['SEAS'] == 'JAS'].set_index('tahun')['oni_anom'].to_dict()
    oni_jja = df_oni[df_oni['SEAS'] == 'JJA'].set_index('tahun')['oni_anom'].to_dict()
    
    # Ekstrak DMI bulan Agustus & September
    dmi_agu = df_dmi[df_dmi['bulan'] == 8].set_index('tahun')['dmi'].to_dict()
    dmi_sep = df_dmi[df_dmi['bulan'] == 9].set_index('tahun')['dmi'].to_dict()
    
    df_merged['oni_jas'] = df_merged['tahun_awal'].map(oni_jas)
    df_merged['oni_jja'] = df_merged['tahun_awal'].map(oni_jja)
    df_merged['dmi_agu'] = df_merged['tahun_awal'].map(dmi_agu)
    df_merged['dmi_sep'] = df_merged['tahun_awal'].map(dmi_sep)
    
    # 3. Ambil Prediktor Atmosferik Lokal ERA5-Land
    df_era5 = extract_era5_predictors(start_year=2000, end_year=2026)
    if not df_era5.empty:
        df_merged = df_merged.merge(df_era5, left_on='tahun_awal', right_on='tahun', how='left')
        if 'tahun' in df_merged.columns:
            df_merged = df_merged.drop(columns=['tahun'])
            
    # Forward-fill / mean impute jika ada prediktor yang belum lengkap di tahun terakhir
    numeric_cols = [c for c in df_merged.columns if c.startswith(('oni_', 'dmi_', 'u10_', 'rh_', 't2m_'))]
    for c in numeric_cols:
        df_merged[c] = df_merged[c].fillna(df_merged[c].mean())
        
    out_file = DATA_DIR / "dataset_training_prediksi_musim.csv"
    df_merged.to_csv(out_file, index=False)
    print(f"\n[SUKSES] Dataset training berhasil dibangun: {len(df_merged)} baris x {len(df_merged.columns)} kolom.")
    print(f"Tersimpan di: {out_file}")
    return df_merged

if __name__ == "__main__":
    build_training_matrix()
