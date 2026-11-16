#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Ekstraksi Dasarian & Baseline Klimatologi Kabupaten Purworejo (2000-2026)
Stasiun Meteorologi Kelas III Jerukagung
Mengekstrak 36 dasarian per tahun untuk 16 kecamatan di Kabupaten Purworejo dari GEE CHIRPS v3,
menghitung AMH, AMK, Puncak Musim, dan Normal Klimatologi sesuai standar BMKG/WMO.
"""

import os
import sys
import calendar
import time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import force_2d
import ee

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
GEOJSON_KEC = BASE_DIR / "purworejo_33.06_batas_kecamatan.geojson"
CSV_DASARIAN = DATA_DIR / "dasarian_curah_hujan_purworejo_2000_2026.csv"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
FULL_MONTH_NAMES = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
ROMAN = ["I", "II", "III"]

def get_dasarian_label(d_idx):
    d_round = int(round(d_idx))
    m = ((d_round - 1) // 3) + 1
    d_m = ((d_round - 1) % 3) + 1
    if 1 <= m <= 12:
        return f"{MONTH_NAMES[m-1]} {ROMAN[d_m-1]}"
    return f"Dasarian {d_round}"

def get_month_name(d_idx):
    d_round = int(round(d_idx))
    m = ((d_round - 1) // 3) + 1
    if 1 <= m <= 12:
        return FULL_MONTH_NAMES[m-1]
    return "Januari"

def extract_all_dasarians():
    print(f"[GEE] Inisialisasi Google Earth Engine dengan project staklimjerukagung...")
    ee.Initialize(project='staklimjerukagung')

    gdf = gpd.read_file(GEOJSON_KEC)
    gdf['geometry'] = force_2d(gdf['geometry'])
    gdf_utm = gdf.to_crs(epsg=32749)
    cents_utm = gdf_utm.geometry.centroid
    cents_geo = gpd.GeoSeries(cents_utm, crs="EPSG:32749").to_crs("EPSG:4326")

    pts = []
    for idx, row in gdf.iterrows():
        pt = cents_geo.iloc[idx]
        pts.append(ee.Feature(ee.Geometry.Point([pt.x, pt.y]), {'kecamatan': row['name'], 'code': row['code']}))
    fc_pts = ee.FeatureCollection(pts)
    print(f"[GEE] Titik kontrol kecamatan: {len(pts)} kecamatan.")

    all_rows = []
    years = range(2000, 2027) # 2000 s/d 2026

    for yr in years:
        t0 = time.time()
        col_yr = ee.ImageCollection('UCSB-CHC/CHIRPS/V3/DAILY_RNL').filterDate(f'{yr}-01-01', f'{yr+1}-01-01').select('precipitation')
        
        max_m = 10 if yr == 2026 else 12 # 2026 s.d. Oktober (Dasarian 30)
        
        band_images = []
        band_names = []
        
        for m in range(1, max_m + 1):
            _, days_in_month = calendar.monthrange(yr, m)
            
            # D1
            t1_start = f"{yr}-{m:02d}-01"
            t1_end = f"{yr}-{m:02d}-11"
            d1 = col_yr.filterDate(t1_start, t1_end).sum().rename(f"m{m:02d}_d1")
            band_images.append(d1)
            band_names.append(f"m{m:02d}_d1")
            
            # D2
            t2_start = f"{yr}-{m:02d}-11"
            t2_end = f"{yr}-{m:02d}-21"
            d2 = col_yr.filterDate(t2_start, t2_end).sum().rename(f"m{m:02d}_d2")
            band_images.append(d2)
            band_names.append(f"m{m:02d}_d2")
            
            # D3
            t3_start = f"{yr}-{m:02d}-21"
            if m == 12:
                t3_end = f"{yr+1}-01-01"
            else:
                t3_end = f"{yr}-{m+1:02d}-01"
            d3 = col_yr.filterDate(t3_start, t3_end).sum().rename(f"m{m:02d}_d3")
            band_images.append(d3)
            band_names.append(f"m{m:02d}_d3")
            
        stacked = ee.Image(band_images)
        res = stacked.reduceRegions(collection=fc_pts, reducer=ee.Reducer.first(), scale=5566).getInfo()
        
        for f in res['features']:
            props = f['properties']
            kec = props['kecamatan']
            for m in range(1, max_m + 1):
                base_d = (m - 1) * 3
                for d_in_m in [1, 2, 3]:
                    d_global = base_d + d_in_m
                    val = props.get(f"m{m:02d}_d{d_in_m}")
                    ch = round(float(val), 2) if val is not None else 0.0
                    all_rows.append({
                        'tahun': yr,
                        'bulan': m,
                        'dasarian': d_global,
                        'dasarian_dalam_bulan': d_in_m,
                        'kecamatan': kec,
                        'curah_hujan': max(0.0, ch)
                    })
        t1 = time.time()
        print(f"  Tahun {yr}: {max_m * 3} dasarian x {len(pts)} kecamatan diekstrak dalam {t1 - t0:.2f} detik.")

    df_das = pd.DataFrame(all_rows)
    df_das = df_das.sort_values(['tahun', 'dasarian', 'kecamatan']).reset_index(drop=True)
    df_das.to_csv(CSV_DASARIAN, index=False)
    print(f"\n[SUKSES] Dataset dasarian curah hujan Purworejo tersimpan di: {CSV_DASARIAN}")
    print(f"Total baris: {len(df_das)}")
    return df_das

def calculate_climatology_and_historical(df_das):
    years = range(2000, 2026) # 2000..2025
    kecamatan_list = sorted(df_das['kecamatan'].unique())

    print("\n[INFO] Menghitung AMH & AMK historis Purworejo (2000-2025)...")
    records_amh = []
    records_amk = []

    for yr in years:
        for kec in kecamatan_list:
            sub = df_das[(df_das['tahun'] == yr) & (df_das['kecamatan'] == kec)].sort_values('dasarian').reset_index(drop=True)
            
            # --- AMK (Kemarau) ---
            # Jendela pencarian: Dasarian 10 (Apr I) s.d. 24 (Agu III)
            sub_search_k = sub[(sub['dasarian'] >= 10) & (sub['dasarian'] <= 24)].reset_index(drop=True)
            amk_d = 14.0 # default Mei II
            for i in range(len(sub_search_k) - 2):
                p0 = sub_search_k.loc[i, 'curah_hujan']
                p1 = sub_search_k.loc[i+1, 'curah_hujan']
                p2 = sub_search_k.loc[i+2, 'curah_hujan']
                d0 = sub_search_k.loc[i, 'dasarian']
                if p0 < 50.0 and (p1 < 50.0 or p2 < 50.0):
                    amk_d = d0
                    break
                    
            sub_mk = sub[(sub['dasarian'] >= 13) & (sub['dasarian'] <= 27)]
            tot_mk = sub_mk['curah_hujan'].sum()
            
            sub_pmk = sub[(sub['dasarian'] >= 16) & (sub['dasarian'] <= 27)]
            if not sub_pmk.empty:
                pmk_d = sub_pmk.loc[sub_pmk['curah_hujan'].idxmin(), 'dasarian']
            else:
                pmk_d = 22 # Agu I
                
            records_amk.append({
                'tahun': yr,
                'kecamatan': kec,
                'amk_dasarian': amk_d,
                'amk_label': get_dasarian_label(amk_d),
                'total_hujan_mk_mm': round(tot_mk, 1),
                'pmk_dasarian': pmk_d,
                'pmk_label': get_dasarian_label(pmk_d)
            })

            # --- AMH (Hujan) ---
            # Jendela pencarian: Dasarian 22 (Agu I) s.d. 36 (Des III)
            sub_search_h = sub[(sub['dasarian'] >= 22) & (sub['dasarian'] <= 36)].reset_index(drop=True)
            amh_d = 29.0 # default Okt II
            for i in range(len(sub_search_h) - 2):
                p0 = sub_search_h.loc[i, 'curah_hujan']
                p1 = sub_search_h.loc[i+1, 'curah_hujan']
                p2 = sub_search_h.loc[i+2, 'curah_hujan']
                d0 = sub_search_h.loc[i, 'dasarian']
                if p0 >= 50.0 and (p1 >= 50.0 or p2 >= 50.0):
                    amh_d = d0
                    break
                    
            # Total hujan MH: Dasarian 28 (Okt I) s/d Dasarian 36 tahun ini + Dasarian 1 s/d 12 tahun berikutnya jika ada
            sub_h_curr = sub[(sub['dasarian'] >= 28) & (sub['dasarian'] <= 36)]
            sub_next = df_das[(df_das['tahun'] == yr + 1) & (df_das['kecamatan'] == kec) & (df_das['dasarian'] <= 12)]
            tot_mh = sub_h_curr['curah_hujan'].sum() + (sub_next['curah_hujan'].sum() if not sub_next.empty else 1500.0)
            
            # Puncak Musim Hujan: dasarian basah tertinggi antara Nov s.d. Feb
            sub_pmh = pd.concat([
                sub[(sub['dasarian'] >= 31) & (sub['dasarian'] <= 36)],
                sub_next[sub_next['dasarian'] <= 6] if not sub_next.empty else pd.DataFrame()
            ])
            if not sub_pmh.empty:
                pmh_d = sub_pmh.loc[sub_pmh['curah_hujan'].idxmax(), 'dasarian']
            else:
                pmh_d = 1 # Jan I
                
            records_amh.append({
                'tahun_musim': f"{yr}/{yr+1}",
                'tahun_awal': yr,
                'kecamatan': kec,
                'amh_dasarian': amh_d,
                'amh_label': get_dasarian_label(amh_d),
                'puncak_dasarian': pmh_d,
                'puncak_label': get_dasarian_label(pmh_d),
                'total_hujan_mh_mm': round(tot_mh, 1)
            })

    # Save AMK Historis & Normal
    df_amk = pd.DataFrame(records_amk)
    out_hist_k = DATA_DIR / "amk_historis_purworejo_2000_2025.csv"
    df_amk.to_csv(out_hist_k, index=False)
    
    df_norm_k = df_amk.groupby('kecamatan').agg(
        normal_amk_dasarian=('amk_dasarian', 'median'),
        mean_amk_dasarian=('amk_dasarian', 'mean'),
        std_amk_dasarian=('amk_dasarian', 'std'),
        terawal_amk_dasarian=('amk_dasarian', 'min'),
        terlambat_amk_dasarian=('amk_dasarian', 'max'),
        normal_total_hujan_mk_mm=('total_hujan_mk_mm', 'mean'),
        normal_pmk_dasarian=('pmk_dasarian', 'median')
    ).reset_index()
    df_norm_k['normal_amk_label'] = [get_dasarian_label(d) for d in df_norm_k['normal_amk_dasarian']]
    df_norm_k['terawal_amk_label'] = [get_dasarian_label(d) for d in df_norm_k['terawal_amk_dasarian']]
    df_norm_k['terlambat_amk_label'] = [get_dasarian_label(d) for d in df_norm_k['terlambat_amk_dasarian']]
    df_norm_k['normal_pmk_label'] = [get_dasarian_label(d) for d in df_norm_k['normal_pmk_dasarian']]
    df_norm_k['normal_pmk_bulan'] = [get_month_name(d) for d in df_norm_k['normal_pmk_dasarian']]
    out_norm_k = DATA_DIR / "normal_klimatologi_amk_purworejo.csv"
    df_norm_k.to_csv(out_norm_k, index=False)

    # Save AMH Historis & Normal
    df_amh = pd.DataFrame(records_amh)
    out_hist_h = DATA_DIR / "amh_historis_purworejo_2000_2025.csv"
    df_amh.to_csv(out_hist_h, index=False)
    
    df_norm_h = df_amh.groupby('kecamatan').agg(
        normal_amh_dasarian=('amh_dasarian', 'median'),
        mean_amh_dasarian=('amh_dasarian', 'mean'),
        std_amh_dasarian=('amh_dasarian', 'std'),
        terawal_amh_dasarian=('amh_dasarian', 'min'),
        terlambat_amh_dasarian=('amh_dasarian', 'max'),
        normal_total_hujan_mh_mm=('total_hujan_mh_mm', 'mean'),
        normal_puncak_dasarian=('puncak_dasarian', 'median')
    ).reset_index()
    df_norm_h['normal_amh_label'] = [get_dasarian_label(d) for d in df_norm_h['normal_amh_dasarian']]
    df_norm_h['terawal_amh_label'] = [get_dasarian_label(d) for d in df_norm_h['terawal_amh_dasarian']]
    df_norm_h['terlambat_amh_label'] = [get_dasarian_label(d) for d in df_norm_h['terlambat_amh_dasarian']]
    df_norm_h['normal_puncak_bulan'] = [get_month_name(d) for d in df_norm_h['normal_puncak_dasarian']]
    out_norm_h = DATA_DIR / "normal_klimatologi_amh_purworejo.csv"
    df_norm_h.to_csv(out_norm_h, index=False)

    print(f"[SUKSES] Baseline Klimatologi Purworejo berhasil dihitung:")
    print(f"  - AMK Normal: {out_norm_k}")
    print(f"  - AMH Normal: {out_norm_h}")
    return df_norm_h, df_norm_k

if __name__ == "__main__":
    df_das = extract_all_dasarians()
    calculate_climatology_and_historical(df_das)
