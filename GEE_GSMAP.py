#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEE GSMaP Downloader (Standalone CLI)
Mengunduh citra presipitasi GSMaP pada resolusi waktu asli terkecil: PER JAM (HOURLY)
Koleksi: JAXA/GPM_L3/GSMaP/v8/operational (band: hourlyPrecipRateGC)

Usage:
  python GEE_GSMAP.py --start-year 2000 --end-year 2026
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
    except Exception:
        pass

    try:
        ee.Initialize(project=project_id)
        print("[GEE] Berhasil inisialisasi GEE default project")
    except Exception:
        print("[GEE] Membuka otentikasi browser...")
        ee.Authenticate()
        ee.Initialize(project=project_id)

def prepare_geometry(geojson_path, buffer_deg=0.01):
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

def unduh_gsmap_bulanan(tahun, bulan, batas_ee, ee_bbox, output_base_dir):
    """
    Mengunduh presipitasi GSMaP per jam (hourly) selama 1 bulan penuh
    menggunakan multi-band stacking (672-744 band) dan menyimpannya ke NetCDF.
    """
    folder_tahun = os.path.join(output_base_dir, str(tahun))
    os.makedirs(folder_tahun, exist_ok=True)

    nc_path = os.path.join(folder_tahun, f"gsmap_{tahun}_{bulan:02d}.nc")
    tif_path = os.path.join(folder_tahun, f"temp_gsmap_{tahun}_{bulan:02d}.tif")

    if os.path.exists(nc_path):
        print(f"[{tahun}-{bulan:02d}] File sudah ada: {nc_path} (Dilewati)")
        return nc_path

    tgl_mulai = f"{tahun}-{bulan:02d}-01"
    if bulan == 12:
        tgl_akhir = f"{tahun+1}-01-01"
    else:
        tgl_akhir = f"{tahun}-{bulan+1:02d}-01"

    print(f"[{tahun}-{bulan:02d}] Mengambil koleksi GSMaP v8 Hourly dari GEE...")
    try:
        col = (ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational')
               .filterBounds(ee_bbox)
               .filter(ee.Filter.date(tgl_mulai, tgl_akhir))
               .select('hourlyPrecipRateGC'))

        count = col.size().getInfo()
        if count == 0:
            print(f"[{tahun}-{bulan:02d}] Tidak ada data di GEE untuk periode ini.")
            return None

        timestamps_ms = col.aggregate_array('system:time_start').getInfo()
        time_index = pd.to_datetime(timestamps_ms, unit='ms')

        stacked = col.toBands().clip(ee_bbox)

        print(f"[{tahun}-{bulan:02d}] Mengunduh {count} jam sekaligus ke GeoTIFF...")
        geemap.ee_export_image(
            stacked,
            filename=tif_path,
            region=ee_bbox,
            scale=11132,  # 0.10 derajat (~11 km)
            file_per_band=False
        )

        with rxr.open_rasterio(tif_path, masked=True) as da:
            da = da.rename({'band': 'time'})
            da['time'] = time_index[:len(da.time)]
            da.name = "precipitation"
            da.attrs["long_name"] = "GSMaP v8 Operational Hourly Precipitation Rate (Gauge-Calibrated)"
            da.attrs["units"] = "mm/hr"
            da.attrs["temporal_resolution"] = "Hourly"
            da.to_netcdf(nc_path)

        if os.path.exists(tif_path):
            os.remove(tif_path)
        print(f"[{tahun}-{bulan:02d}] Sukses! Tersimpan: {nc_path}\n")
        return nc_path

    except Exception as e:
        print(f"[{tahun}-{bulan:02d}] Error: {e}")
        if os.path.exists(tif_path):
            os.remove(tif_path)
        return None

def main():
    parser = argparse.ArgumentParser(description="GEE GSMaP Hourly Downloader CLI")
    parser.add_argument("--start-year", type=int, default=2000, help="Tahun awal pengunduhan (default: 2000)")
    parser.add_argument("--end-year", type=int, default=2026, help="Tahun akhir pengunduhan (default: 2026)")
    parser.add_argument("--start-month", type=int, default=1, help="Bulan awal (default: 1)")
    parser.add_argument("--end-month", type=int, default=12, help="Bulan akhir (default: 12)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.getcwd(), "data", "gsmap"), help="Folder basis output GSMaP")
    parser.add_argument("--geojson", type=str, default="33.05_kecamatan.geojson", help="Path batas GeoJSON Kebumen")
    parser.add_argument("--service-account", type=str, default=None, help="Path file Service Account JSON GEE")
    args = parser.parse_args()

    print("=" * 75)
    print("GEE GSMaP HOURLY (PER JAM) NATIVE RESOLUTION DOWNLOADER")
    print(f"Periode: {args.start_year} s.d. {args.end_year}")
    print(f"Output : {args.output_dir}")
    print("=" * 75)

    init_earth_engine(sa_path=args.service_account)
    batas_ee, ee_bbox, bounds = prepare_geometry(args.geojson)
    print(f"[Geometri] Extent Kebumen: Lon [{bounds[0]:.4f}, {bounds[2]:.4f}], Lat [{bounds[1]:.4f}, {bounds[3]:.4f}]\n")

    for tahun in range(args.start_year, args.end_year + 1):
        m_start = args.start_month if tahun == args.start_year else 1
        m_end = args.end_month if tahun == args.end_year else 12

        for bulan in range(m_start, m_end + 1):
            unduh_gsmap_bulanan(tahun, bulan, batas_ee, ee_bbox, args.output_dir)

    print("=" * 75)
    print("SELESAI PIPELINE UNDUH GSMaP HOURLY")
    print("=" * 75)

if __name__ == "__main__":
    main()
