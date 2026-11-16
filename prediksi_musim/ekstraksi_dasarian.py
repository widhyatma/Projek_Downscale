#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Ekstraksi Dasarian & Baseline Klimatologi GEE Multi-Kabupaten (2000-2026)
Stasiun Meteorologi Kelas III Jerukagung

Mengekstrak 36 dasarian per tahun per kecamatan dari GEE CHIRPS v3,
menghitung AMH, AMK, Puncak Musim, dan Normal Klimatologi sesuai standar BMKG/WMO.

Usage:
  python prediksi_musim/ekstraksi_dasarian.py --kabupaten purworejo
  python prediksi_musim/ekstraksi_dasarian.py --kabupaten kebumen
"""

import os
import sys
import calendar
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import force_2d
import ee

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from gee_kabupaten_config import resolve_kabupaten_paths, KABUPATEN_REGISTRY

DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

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

def extract_all_dasarians(kabupaten_name: str = "purworejo", start_year: int = 2000, end_year: int = 2026):
    kab_key = str(kabupaten_name).lower().strip()
    geojson_path, _, kab_cfg = resolve_kabupaten_paths("chirps", kabupaten_name=kab_key)
    csv_out = DATA_DIR / f"dasarian_curah_hujan_{kab_key}_{start_year}_{end_year}.csv"

    print("=" * 75)
    print(f"EKSTRAKSI DASARIAN GEE CHIRPS UNTUK {kab_cfg['name'].upper()}")
    print(f"Periode : {start_year} s.d. {end_year}")
    print(f"GeoJSON : {geojson_path}")
    print(f"Output  : {csv_out}")
    print("=" * 75)

    print(f"[GEE] Inisialisasi Google Earth Engine dengan project staklimjerukagung...")
    try:
        ee.Initialize(project='staklimjerukagung')
    except Exception as e:
        print(f"[GEE] Mencoba autentikasi default: {e}")
        ee.Authenticate()
        ee.Initialize(project='staklimjerukagung')

    gdf = gpd.read_file(geojson_path)
    gdf['geometry'] = force_2d(gdf['geometry'])
    gdf_utm = gdf.to_crs(epsg=32749)
    cents_utm = gdf_utm.geometry.centroid
    cents_geo = gpd.GeoSeries(cents_utm, crs="EPSG:32749").to_crs("EPSG:4326")

    name_col = "nm_kecamatan" if "nm_kecamatan" in gdf.columns else ("name" if "name" in gdf.columns else "kecamatan")
    code_col = "code" if "code" in gdf.columns else ("kd_kecamatan" if "kd_kecamatan" in gdf.columns else name_col)

    pts = []
    for idx, row in gdf.iterrows():
        pt = cents_geo.iloc[idx]
        pts.append(ee.Feature(ee.Geometry.Point([pt.x, pt.y]), {'kecamatan': str(row[name_col]).strip(), 'code': str(row[code_col]).strip()}))
    fc_pts = ee.FeatureCollection(pts)
    print(f"[GEE] Titik kontrol kecamatan: {len(pts)} kecamatan.")

    all_rows = []
    years = range(start_year, end_year + 1)

    for yr in years:
        t0 = time.time()
        col_yr = ee.ImageCollection('UCSB-CHC/CHIRPS/V3/DAILY_RNL').filterDate(f'{yr}-01-01', f'{yr+1}-01-01').select('precipitation')
        max_m = 10 if yr == 2026 else 12

        band_images = []
        for m in range(1, max_m + 1):
            # D1
            d1 = col_yr.filterDate(f"{yr}-{m:02d}-01", f"{yr}-{m:02d}-11").sum().rename(f"m{m:02d}_d1")
            band_images.append(d1)
            # D2
            d2 = col_yr.filterDate(f"{yr}-{m:02d}-11", f"{yr}-{m:02d}-21").sum().rename(f"m{m:02d}_d2")
            band_images.append(d2)
            # D3
            t3_end = f"{yr+1}-01-01" if m == 12 else f"{yr}-{m+1:02d}-01"
            d3 = col_yr.filterDate(f"{yr}-{m:02d}-21", t3_end).sum().rename(f"m{m:02d}_d3")
            band_images.append(d3)

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
    df_das.to_csv(csv_out, index=False)
    print(f"\n[SUKSES] Dataset dasarian curah hujan {kab_cfg['name']} tersimpan di: {csv_out}")
    print(f"Total baris: {len(df_das)}")
    return df_das

def main():
    parser = argparse.ArgumentParser(description="Ekstraksi Dasarian GEE Multi-Kabupaten")
    parser.add_argument("--kabupaten", type=str, default="purworejo", choices=["kebumen", "purworejo"], help="Nama kabupaten target")
    parser.add_argument("--start-year", type=int, default=2000, help="Tahun awal (default: 2000)")
    parser.add_argument("--end-year", type=int, default=2026, help="Tahun akhir (default: 2026)")
    args = parser.parse_args()

    extract_all_dasarians(kabupaten_name=args.kabupaten, start_year=args.start_year, end_year=args.end_year)

if __name__ == "__main__":
    main()
