#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Pembaruan Dasarian Otomatis (Incremental Dasarian Update)
Mengekstrak akumulasi curah hujan dasarian untuk 26 kecamatan di Kebumen
dari dataset presipitasi terbaru di data/ (CHIRPS RNL/SAT atau GSMaP).
"""

import os
import sys
import glob
import calendar
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely import force_2d

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
GEOJSON_PATH = BASE_DIR / "kebumen_33.05_batas_kecamatan.geojson"
CSV_PATH = DATA_DIR / "dasarian_curah_hujan_kebumen_2000_2026.csv"

def get_kecamatan_centroids():
    gdf = gpd.read_file(GEOJSON_PATH)
    if 'nm_kecamatan' not in gdf.columns and 'name' in gdf.columns:
        gdf['nm_kecamatan'] = gdf['name'].astype(str).str.strip()
    else:
        gdf['nm_kecamatan'] = gdf['nm_kecamatan'].astype(str).str.strip()
    # Rule 12: Hitung centroid dalam koordinat metrik terproyeksi (UTM 49S / EPSG:32749)
    gdf_utm = gdf.to_crs("EPSG:32749")
    cents_utm = gdf_utm.geometry.centroid
    cents_geo = gpd.GeoSeries(cents_utm, crs="EPSG:32749").to_crs("EPSG:4326")
    kec_names = gdf['nm_kecamatan'].tolist()
    return kec_names, cents_geo.x.values, cents_geo.y.values

def update_dasarian_dataset():
    if not CSV_PATH.exists():
        print(f"[ERROR] {CSV_PATH} tidak ditemukan.")
        return False

    df_das = pd.read_csv(CSV_PATH)
    max_year = int(df_das['tahun'].max())
    max_month = int(df_das[df_das['tahun'] == max_year]['bulan'].max())
    max_das = int(df_das[df_das['tahun'] == max_year]['dasarian'].max())

    print(f"[DASARIAN] Status saat ini: Terisi s.d. Tahun {max_year}, Bulan {max_month}, Dasarian {max_das}")

    kec_names, lons, lats = get_kecamatan_centroids()

    # Cek bulan-bulan berikutnya di data/
    new_rows = []
    
    # Periksa 2026 bulan 10 s/d 12
    for target_m in range(max_month + 1, 13):
        # Cari file presipitasi: prioritas CHIRPS RNL -> CHIRPS SAT -> GSMaP
        f_rnl = BASE_DIR / "data" / "chirps" / "chirps_rnl" / str(max_year) / f"chirps_rnl_{max_year}_{target_m:02d}.nc"
        f_sat = BASE_DIR / "data" / "chirps" / "chirps_sat" / str(max_year) / f"chirps_sat_{max_year}_{target_m:02d}.nc"
        f_gsmap = BASE_DIR / "data" / "gsmap" / str(max_year) / f"gsmap_{max_year}_{target_m:02d}.nc"

        chosen_file = None
        source_type = None

        if f_rnl.exists():
            chosen_file = f_rnl
            source_type = "CHIRPS_RNL"
        elif f_sat.exists():
            chosen_file = f_sat
            source_type = "CHIRPS_SAT"
        elif f_gsmap.exists():
            chosen_file = f_gsmap
            source_type = "GSMaP"

        if not chosen_file:
            print(f"[DASARIAN] Bulan {max_year}-{target_m:02d} belum memiliki file presipitasi utuh.")
            break

        print(f"[DASARIAN] Memproses {max_year}-{target_m:02d} dari sumber {source_type}: {chosen_file.name}")
        
        try:
            with xr.open_dataset(chosen_file, engine='netcdf4') as ds:
                p_var = ds['precipitation']
                n_t = p_var.shape[0]
                _, days_in_month = calendar.monthrange(max_year, target_m)

                # Cek apakah file lengkap
                if source_type.startswith("CHIRPS") and n_t < days_in_month:
                    print(f"  [SKIP] File CHIRPS {max_year}-{target_m:02d} belum penuh ({n_t}/{days_in_month} hari)")
                    break
                elif source_type == "GSMaP" and n_t < days_in_month * 24:
                    print(f"  [SKIP] File GSMaP {max_year}-{target_m:02d} belum penuh ({n_t}/{days_in_month*24} jam)")
                    break

                # Pembagian dasarian
                steps_per_day = n_t // days_in_month
                t1 = 10 * steps_per_day
                t2 = 20 * steps_per_day

                # Jika GSMaP, kalibrasi rasio orografis tipikal terhadap CHIRPS (~1.25) jika ada
                calib = 1.25 if source_type == "GSMaP" else 1.0

                p_d1 = p_var[:t1].sum(dim='time') * calib
                p_d2 = p_var[t1:t2].sum(dim='time') * calib
                p_d3 = p_var[t2:].sum(dim='time') * calib

                # Interpolasi koordinat kecamatan
                cx = xr.DataArray(lons, dims="kecamatan", coords={"kecamatan": kec_names})
                cy = xr.DataArray(lats, dims="kecamatan", coords={"kecamatan": kec_names})

                # Tentukan koordinat x/y atau lon/lat
                x_coord = 'x' if 'x' in p_var.coords else 'longitude'
                y_coord = 'y' if 'y' in p_var.coords else 'latitude'

                vals_d1 = p_d1.interp({x_coord: cx, y_coord: cy}, method='linear').values
                vals_d2 = p_d2.interp({x_coord: cx, y_coord: cy}, method='linear').values
                vals_d3 = p_d3.interp({x_coord: cx, y_coord: cy}, method='linear').values

                base_das = (target_m - 1) * 3
                for i, kec in enumerate(kec_names):
                    new_rows.append({
                        'tahun': max_year,
                        'bulan': target_m,
                        'dasarian': base_das + 1,
                        'dasarian_dalam_bulan': 1,
                        'kecamatan': kec,
                        'curah_hujan': round(float(vals_d1[i]), 2)
                    })
                    new_rows.append({
                        'tahun': max_year,
                        'bulan': target_m,
                        'dasarian': base_das + 2,
                        'dasarian_dalam_bulan': 2,
                        'kecamatan': kec,
                        'curah_hujan': round(float(vals_d2[i]), 2)
                    })
                    new_rows.append({
                        'tahun': max_year,
                        'bulan': target_m,
                        'dasarian': base_das + 3,
                        'dasarian_dalam_bulan': 3,
                        'kecamatan': kec,
                        'curah_hujan': round(float(vals_d3[i]), 2)
                    })

                print(f"  [OK] Dasarian {base_das+1}, {base_das+2}, {base_das+3} ({max_year}-{target_m:02d}) berhasil diekstrak untuk 26 kecamatan.")
        except Exception as e:
            print(f"  [ERROR] Gagal memproses {chosen_file}: {e}")
            break

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        # Gabungkan dan hilangkan duplikasi jika ada
        df_updated = pd.concat([df_das, df_new], ignore_index=True)
        df_updated.drop_duplicates(subset=['tahun', 'dasarian', 'kecamatan'], keep='last', inplace=True)
        df_updated.sort_values(by=['tahun', 'dasarian', 'kecamatan'], inplace=True)
        df_updated.to_csv(CSV_PATH, index=False)
        print(f"[SUKSES] Dataset dasarian diperbarui: +{len(new_rows)} baris. Total: {len(df_updated)} baris.")
        return True
    else:
        print("[DASARIAN] Tidak ada baris baru yang perlu ditambahkan.")
        return False

if __name__ == "__main__":
    update_dasarian_dataset()
