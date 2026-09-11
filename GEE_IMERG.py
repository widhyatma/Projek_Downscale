#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEE NASA GPM IMERG Downloader (Standalone CLI)
Mengunduh citra presipitasi NASA GPM IMERG v07 pada resolusi waktu asli terkecil:
SETENGAH JAM (HALF-HOURLY / 30-MENIT)

Koleksi: NASA/GPM_L3/IMERG_V07 (band: precipitation dalam mm/jam)
Mematuhi Rule 5: 1488 band/bulan dibagi menjadi 2 sub-stack 15-harian (<= 768 band <= 1024),
lalu digabungkan menjadi NetCDF bulanan utuh beresolusi 30-menit.

Usage:
  python GEE_IMERG.py --start-year 2024 --end-year 2026 --mode half-hourly
"""

import os
import sys
import glob
import json
import argparse
import calendar
from datetime import datetime, timedelta

import ee
import geopandas as gpd
import geemap
import rioxarray as rxr
import xarray as xr
import pandas as pd
from shapely.validation import make_valid
from shapely.ops import transform, unary_union
from google.oauth2.service_account import Credentials

def init_earth_engine(sa_path=None, project_id='staklimjerukagung'):
    """Inisialisasi GEE otomatis (Lokal SA JSON, Kaggle Secrets, atau OAuth)."""
    candidates = []
    if sa_path:
        candidates.append(sa_path)
    candidates.extend([
        os.path.join(os.getcwd(), "staklimjerukagung-b852a12a367e.json"),
        os.path.join(os.path.dirname(os.getcwd()), "staklimjerukagung-b852a12a367e.json"),
        "staklimjerukagung-b852a12a367e.json"
    ])
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    sa_info = json.load(f)
                SCOPES = ['https://www.googleapis.com/auth/earthengine']
                creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
                ee.Initialize(credentials=creds, project=sa_info.get('project_id', project_id))
                print(f"[GEE] Berhasil inisialisasi via Service Account: {os.path.basename(p)}")
                return
            except Exception as e:
                print(f"[GEE] Gagal auth via SA file {p}: {e}")

    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        sa_info = json.loads(user_secrets.get_secret("GEE_KEY"))
        SCOPES = ['https://www.googleapis.com/auth/earthengine']
        creds = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
        ee.Initialize(credentials=creds, project=sa_info.get('project_id', project_id))
        print("[GEE] Berhasil inisialisasi via Kaggle Secret (GEE_KEY)")
        return
    except Exception as e:
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            print(f"[GEE] Info Kaggle Secrets: {e}")
            print("[GEE] Tips di Kaggle: Pastikan menu Add-ons -> Secrets -> centang checkbox 'GEE_KEY' pada notebook ini!")

    try:
        ee.Initialize(project=project_id)
        print("[GEE] Berhasil inisialisasi GEE default project")
    except Exception:
        print("[GEE] Membuka otentikasi browser...")
        ee.Authenticate()
        ee.Initialize(project=project_id)

def prepare_geometry(geojson_path, buffer_deg=0.02):
    """Mempersiapkan geometri batas wilayah dan bounding box dengan buffer pengaman."""
    if not os.path.exists(geojson_path):
        candidates = glob.glob(f"**/{os.path.basename(geojson_path)}", recursive=True)
        if candidates:
            geojson_path = candidates[0]
        else:
            raise FileNotFoundError(f"File GeoJSON tidak ditemukan: {geojson_path}")

    gdf = gpd.read_file(geojson_path)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf[gdf.geometry.notna()].copy()

    def _to_2d(geom):
        if geom is None or geom.is_empty: return None
        return transform(lambda x, y, z=None: (x, y), geom)

    def _extract_polygonal(geom):
        if geom is None or geom.is_empty: return None
        if geom.geom_type in ("Polygon", "MultiPolygon"): return geom
        if geom.geom_type == "GeometryCollection":
            polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            if polys: return unary_union(polys)
        return None

    def _clean_geom(geom):
        if geom is None or geom.is_empty: return None
        geom = _to_2d(geom)
        geom = make_valid(geom)
        geom = _extract_polygonal(geom)
        if geom is None or geom.is_empty: return None
        geom = geom.buffer(0)
        return geom

    gdf["geometry"] = gdf["geometry"].apply(_clean_geom)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf.reset_index(drop=True, inplace=True)

    fc = ee.FeatureCollection(gdf.__geo_interface__["features"])
    bounds = gdf.total_bounds
    ee_bbox = ee.Geometry.BBox(
        bounds[0] - buffer_deg,
        bounds[1] - buffer_deg,
        bounds[2] + buffer_deg,
        bounds[3] + buffer_deg
    )
    return fc, ee_bbox, bounds

def unduh_imerg_halfhourly_bulanan(tahun, bulan, batas_ee, ee_bbox, output_base_dir):
    """
    Mengunduh presipitasi NASA GPM IMERG v07 setengah jam (30-menit / half-hourly)
    selama 1 bulan penuh (1344 - 1488 citra).
    Mematuhi Rule 5: dibagi menjadi 2 sub-stack (Part 1: tgl 1-15, Part 2: tgl 16-akhir),
    masing-masing <= 768 band (<= 1024 band limit GEE), lalu digabungkan dengan xarray.
    """
    folder_tahun = os.path.join(output_base_dir, str(tahun))
    os.makedirs(folder_tahun, exist_ok=True)

    nc_path = os.path.join(folder_tahun, f"imerg_{tahun}_{bulan:02d}.nc")
    if os.path.exists(nc_path):
        print(f"[{tahun}-{bulan:02d}] File sudah ada: {nc_path} (Dilewati)")
        return nc_path

    num_days = calendar.monthrange(tahun, bulan)[1]
    sub_ranges = [
        ("part1", f"{tahun}-{bulan:02d}-01", f"{tahun}-{bulan:02d}-16"),
        ("part2", f"{tahun}-{bulan:02d}-16", f"{tahun+1}-01-01" if bulan == 12 else f"{tahun}-{bulan+1:02d}-01")
    ]

    parts_da = []
    temp_files = []

    print(f"[{tahun}-{bulan:02d}] Mengambil koleksi IMERG V07 30-menit (Native Finest Resolution)...")
    try:
        for p_name, t_start, t_end in sub_ranges:
            tif_part = os.path.join(folder_tahun, f"temp_imerg_{tahun}_{bulan:02d}_{p_name}.tif")
            temp_files.append(tif_part)

            col_sub = (ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
                       .filterBounds(ee_bbox)
                       .filterDate(t_start, t_end)
                       .select('precipitation'))

            n_scenes = col_sub.size().getInfo()
            if n_scenes == 0:
                print(f"[{tahun}-{bulan:02d}] {p_name}: Tidak ada scene di GEE.")
                continue

            timestamps = col_sub.aggregate_array('system:time_start').getInfo()
            part_time_idx = pd.to_datetime(timestamps, unit='ms')

            stacked_part = col_sub.toBands().clip(ee_bbox)

            print(f"[{tahun}-{bulan:02d}] Ekspor {p_name} ({n_scenes} bands <= 1024) ke GeoTIFF...")
            geemap.ee_export_image(
                stacked_part,
                filename=tif_part,
                region=ee_bbox,
                scale=11132,  # 0.10 derajat (~11 km)
                file_per_band=False
            )

            with rxr.open_rasterio(tif_part, masked=True) as da:
                da = da.rename({'band': 'time'})
                da['time'] = part_time_idx[:len(da.time)]
                da.name = "precipitation"
                parts_da.append(da.load())

        if not parts_da:
            print(f"[{tahun}-{bulan:02d}] ⚠️ Gagal mengumpulkan data IMERG.")
            return None

        # Gabungkan Part 1 dan Part 2 sepanjang dimensi waktu (time)
        if len(parts_da) == 1:
            ds_full = parts_da[0].to_dataset(name="precipitation")
        else:
            merged_da = xr.concat(parts_da, dim="time")
            ds_full = merged_da.to_dataset(name="precipitation")

        ds_full.attrs["source"] = "NASA/GPM_L3/IMERG_V07"
        ds_full.attrs["temporal_resolution"] = "Half-Hourly (30-Minute)"
        ds_full.attrs["units"] = "mm/hr"
        ds_full.attrs["long_name"] = "Half-hourly Precipitation (GPM IMERG v07 Final Run)"
        ds_full.to_netcdf(nc_path)
        ds_full.close()

        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

        print(f"[{tahun}-{bulan:02d}] Sukses! Tersimpan NetCDF 30-menit: {nc_path}\n")
        return nc_path

    except Exception as e:
        print(f"[{tahun}-{bulan:02d}] Error: {e}")
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        return None

def unduh_imerg_monthly_tahunan(tahun, batas_ee, ee_bbox, output_base_dir):
    """
    Mode alternatif: Mengunduh 12 bulan akumulasi IMERG Monthly V07 dalam 1 file per tahun.
    """
    folder_tahun = os.path.join(output_base_dir, str(tahun))
    os.makedirs(folder_tahun, exist_ok=True)
    nc_path = os.path.join(folder_tahun, f"imerg_monthly_{tahun}.nc")
    tif_temp = os.path.join(folder_tahun, f"temp_imerg_monthly_{tahun}.tif")

    if os.path.exists(nc_path):
        print(f"[{tahun}] File bulanan sudah ada: {nc_path} (Dilewati)")
        return nc_path

    tgl_awal = f"{tahun}-01-01"
    tgl_akhir = f"{tahun+1}-01-01"

    col = (ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07")
           .filterBounds(ee_bbox)
           .filterDate(tgl_awal, tgl_akhir))

    n_img = col.size().getInfo()
    if n_img == 0:
        print(f"[{tahun}] Data IMERG Monthly tidak tersedia di GEE.")
        return None

    times_ms = col.aggregate_array('system:time_start').getInfo()
    time_index = pd.to_datetime(times_ms, unit='ms')

    def convert_to_monthly_accum(img):
        date = ee.Date(img.get('system:time_start'))
        days = date.advance(1, 'month').difference(date, 'day')
        rate_mm_day = img.select('precipitation')
        monthly_mm = rate_mm_day.multiply(days).rename('precipitation')
        return monthly_mm.copyProperties(img, ['system:time_start'])

    monthly_col = col.map(convert_to_monthly_accum)
    stacked = monthly_col.toBands().clip(ee_bbox)

    geemap.ee_export_image(
        stacked,
        filename=tif_temp,
        region=ee_bbox,
        scale=11132,
        file_per_band=False
    )

    with rxr.open_rasterio(tif_temp, masked=True) as da:
        da = da.rename({'band': 'time'})
        da['time'] = time_index[:len(da.time)]
        da.name = "precipitation"
        da.attrs["units"] = "mm/month"
        da.attrs["source"] = "NASA/GPM_L3/IMERG_MONTHLY_V07"
        da.to_netcdf(nc_path)

    if os.path.exists(tif_temp):
        os.remove(tif_temp)
    print(f"[{tahun}] Sukses! Tersimpan: {nc_path}\n")
    return nc_path

def main():
    parser = argparse.ArgumentParser(description="GEE NASA GPM IMERG Downloader CLI")
    parser.add_argument("--start-year", type=int, default=2024, help="Tahun awal pengunduhan (default: 2024)")
    parser.add_argument("--end-year", type=int, default=2026, help="Tahun akhir pengunduhan (default: 2026)")
    parser.add_argument("--start-month", type=int, default=1, help="Bulan awal (default: 1)")
    parser.add_argument("--end-month", type=int, default=12, help="Bulan akhir (default: 12)")
    parser.add_argument("--mode", choices=['half-hourly', 'monthly'], default='half-hourly',
                        help="Mode temporal: 'half-hourly' (resolusi waktu asli 30-menit) atau 'monthly' (akumulasi bulanan)")
    parser.add_argument("--output-dir", type=str, default=None, help="Folder basis output")
    parser.add_argument("--geojson", type=str, default="33.05_kecamatan.geojson", help="Path batas GeoJSON Kebumen")
    parser.add_argument("--service-account", type=str, default=None, help="Path file Service Account JSON GEE")
    args = parser.parse_args()

    default_out = os.path.join(os.getcwd(), "data", "imerg" if args.mode == "half-hourly" else "imerg_monthly")
    output_dir = args.output_dir if args.output_dir else default_out

    print("=" * 75)
    print(f"GEE NASA GPM IMERG v07 DOWNLOADER CLI (MODE: {args.mode.upper()})")
    print(f"Periode: {args.start_year} s.d. {args.end_year}")
    print(f"Output : {output_dir}")
    print("=" * 75)

    init_earth_engine(sa_path=args.service_account)
    batas_ee, ee_bbox, bounds = prepare_geometry(args.geojson)
    print(f"[Geometri] Extent Kebumen: Lon [{bounds[0]:.4f}, {bounds[2]:.4f}], Lat [{bounds[1]:.4f}, {bounds[3]:.4f}]\n")

    if args.mode == 'monthly':
        for tahun in range(args.start_year, args.end_year + 1):
            unduh_imerg_monthly_tahunan(tahun, batas_ee, ee_bbox, output_dir)
    else:
        for tahun in range(args.start_year, args.end_year + 1):
            m_start = args.start_month if tahun == args.start_year else 1
            m_end = args.end_month if tahun == args.end_year else 12
            for bulan in range(m_start, m_end + 1):
                unduh_imerg_halfhourly_bulanan(tahun, bulan, batas_ee, ee_bbox, output_dir)

    print("=" * 75)
    print("SELESAI PIPELINE UNDUH NASA GPM IMERG")
    print("=" * 75)

if __name__ == "__main__":
    main()
