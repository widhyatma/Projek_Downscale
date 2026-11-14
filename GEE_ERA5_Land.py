#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEE ERA5-Land Downloader (Standalone CLI)
Mengunduh 6 variabel atmosferik ERA5-Land pada resolusi waktu asli terkecil: PER JAM (HOURLY)
Koleksi: ECMWF/ERA5_LAND/HOURLY

Usage:
  python GEE_ERA5_Land.py --start-year 2000 --end-year 2026
"""

import os
import sys
import glob
import json
import shutil
import argparse
import calendar
from datetime import datetime, timedelta

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

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

ERA5_EXPECTED_VARS = [
    'precipitation', 'temperature_2m', 'dewpoint_temperature_2m',
    'u_wind_10m', 'v_wind_10m', 'surface_pressure'
]

def check_file_integrity(nc_path, year, month, expected_type='era5_land'):
    """
    Tier 1: Validasi Integritas File NetCDF ERA5-Land.
    Memeriksa eksistensi, ukuran > 0, kecukupan 6 variabel atmosferik, dan total time steps (days * 24).
    """
    if not os.path.exists(nc_path):
        return False, "File belum ada", 0
    if os.path.getsize(nc_path) == 0:
        return False, "File 0 byte (kosong/korup)", 0

    _, days_in_month = calendar.monthrange(year, month)
    expected_steps = days_in_month * 24

    try:
        with xr.open_dataset(nc_path) as ds:
            if 'time' not in ds.dims and 'time' not in ds.coords:
                return False, "Dimensi 'time' tidak ditemukan", 0
            actual_steps = ds.sizes.get('time', len(ds['time']))
            missing_vars = [v for v in ERA5_EXPECTED_VARS if v not in ds.data_vars]
            if missing_vars:
                return False, f"Variabel hilang: {missing_vars}", actual_steps
            if actual_steps < expected_steps:
                return False, f"Time steps terpotong ({actual_steps}/{expected_steps} jam)", actual_steps
            return True, f"Lengkap ({actual_steps}/{expected_steps} jam, 6 variabel)", actual_steps
    except Exception as e:
        return False, f"File rusak / korup ({e})", 0

def check_disk_space(target_dir, min_free_gb=1.5):
    """
    Tier 3: Disk Space Guardrail.
    Memeriksa sisa kapasitas penyimpanan sebelum ekspor citra.
    """
    try:
        _, _, free = shutil.disk_usage(target_dir)
        free_gb = free / (1024 ** 3)
        if free_gb < min_free_gb:
            print(f"  [DISK GUARD] Peringatan: Sisa ruang disk tinggal {free_gb:.2f} GB (< {min_free_gb} GB) di {target_dir}")
            return False, free_gb
        return True, free_gb
    except Exception:
        return True, 999.0

def cari_file_input_era5(tahun, bulan):
    """Mengecek apakah file NetCDF ERA5-Land (lengkap maupun parsial) tersedia di Input Dataset Kaggle."""
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        input_base = "/kaggle/input/datasets/jerismeteo/gee-era5-land-kebumen"
        possible_paths = [
            f"{input_base}/data/era5_land/{tahun}/era5_land_{tahun}_{bulan:02d}.nc",
            f"{input_base}/{tahun}/era5_land_{tahun}_{bulan:02d}.nc",
            f"{input_base}/{tahun}/{tahun}_{bulan:02d}/era5_land_{tahun}_{bulan:02d}.nc",
            f"{input_base}/era5_land_{tahun}_{bulan:02d}.nc",
            f"/kaggle/input/gee-era5-land-kebumen/data/era5_land/{tahun}/era5_land_{tahun}_{bulan:02d}.nc",
            f"/kaggle/input/gee-era5-land-kebumen/{tahun}/era5_land_{tahun}_{bulan:02d}.nc"
        ]
        best_path = None
        max_steps = 0
        for p in possible_paths:
            if os.path.exists(p):
                valid, msg, steps = check_file_integrity(p, tahun, bulan, 'era5_land')
                if valid:
                    return p, True, steps
                elif steps > max_steps:
                    max_steps = steps
                    best_path = p
        if best_path:
            return best_path, False, max_steps
    return None, False, 0

def unduh_era5_land_bulanan(tahun, bulan, batas_ee, ee_bbox, output_base_dir, force=False):
    """
    Mengunduh 6 variabel atmosferik ERA5-Land per jam (hourly) selama 1 bulan penuh.
    Mematuhi Rule 5: per-variable export (maksimum 744 band <= 1024) lalu digabung ke NetCDF tunggal.
    Dilengkapi 4-tier Pre-Flight Guardrails & Atomic Writing.
    """
    folder_tahun = os.path.join(output_base_dir, str(tahun))
    os.makedirs(folder_tahun, exist_ok=True)

    nc_path = os.path.join(folder_tahun, f"era5_land_{tahun}_{bulan:02d}.nc")
    nc_path_tmp = f"{nc_path}.tmp"

    # Tier 1: Local Integrity Validation
    is_valid, msg, actual_steps = check_file_integrity(nc_path, tahun, bulan, 'era5_land')
    if not force and is_valid:
        print(f"[{tahun}-{bulan:02d}] [OK] File sudah lengkap ({actual_steps} jam, 6 vars): {nc_path} (Dilewati)")
        return nc_path

    # Tier 1.5: Kaggle Input Cache Check (Smart Partial & Full Cache)
    if not force:
        input_file, is_cache_valid, cache_steps = cari_file_input_era5(tahun, bulan)
        if input_file and os.path.exists(input_file):
            if is_cache_valid:
                print(f"[{tahun}-{bulan:02d}] [KAGGLE CACHE 100%] File lengkap ditemukan di {input_file}. Menyalin...")
                shutil.copy2(input_file, nc_path)
                return nc_path
            elif cache_steps > actual_steps:
                print(f"[{tahun}-{bulan:02d}] [KAGGLE CACHE PARTIAL] Mengadopsi cache parsial ({cache_steps} jam) dari {input_file}...")
                shutil.copy2(input_file, nc_path)
                actual_steps = cache_steps

    if actual_steps > 0 and not is_valid:
        print(f"[{tahun}-{bulan:02d}] [PERIKSA GEE] File lokal terdeteksi tidak lengkap: {msg}. Memeriksa ketersediaan data di GEE...")

    tgl_mulai = f"{tahun}-{bulan:02d}-01"
    if bulan == 12:
        tgl_akhir = f"{tahun+1}-01-01"
    else:
        tgl_akhir = f"{tahun}-{bulan+1:02d}-01"

    print(f"[{tahun}-{bulan:02d}] Mengambil koleksi ERA5-Land Hourly dari GEE...")
    temp_tifs = []
    try:
        col = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
               .filterBounds(ee_bbox)
               .filterDate(tgl_mulai, tgl_akhir))

        # Tier 2: GEE Pre-Flight Metadata Check
        count = col.size().getInfo()
        if count == 0:
            print(f"[{tahun}-{bulan:02d}] [INFO GEE] Tidak ada data di GEE untuk periode ini.")
            return None

        if not force and actual_steps > 0 and count <= actual_steps:
            print(f"[{tahun}-{bulan:02d}] [INFO GEE] File lokal sudah up-to-date dengan data GEE saat ini ({actual_steps}/{count} jam). Dilewati.")
            return nc_path

        # Tier 3: Disk Space Check
        check_disk_space(folder_tahun)

        timestamps = col.aggregate_array("system:time_start").getInfo()
        dates = pd.to_datetime(timestamps, unit='ms')

        # Konfigurasi 6 Variabel & Transformasi Satuan
        vars_config = [
            ('precipitation', 'total_precipitation_hourly', lambda img: img.multiply(1000)),
            ('temperature_2m', 'temperature_2m', lambda img: img.subtract(273.15)),
            ('dewpoint_temperature_2m', 'dewpoint_temperature_2m', lambda img: img.subtract(273.15)),
            ('u_wind_10m', 'u_component_of_wind_10m', lambda img: img),
            ('v_wind_10m', 'v_component_of_wind_10m', lambda img: img),
            ('surface_pressure', 'surface_pressure', lambda img: img.divide(100))
        ]

        ds_out = xr.Dataset()

        for var_out_name, gee_band_name, transform_fn in vars_config:
            tif_var_temp = os.path.join(folder_tahun, f"temp_era5_{var_out_name}_{tahun}_{bulan:02d}.tif")
            temp_tifs.append(tif_var_temp)

            col_var = col.select(gee_band_name).map(lambda img: transform_fn(img).rename(var_out_name))
            stacked_var_img = col_var.toBands().clip(ee_bbox)

            print(f"[{tahun}-{bulan:02d}] Mengekspor {var_out_name} ({count} jam <= 1024) ke GeoTIFF temporer...")
            geemap.ee_export_image(
                stacked_var_img,
                filename=tif_var_temp,
                region=ee_bbox,
                scale=11132,
                file_per_band=False
            )

            with rxr.open_rasterio(tif_var_temp, masked=True) as da_var:
                da_var = da_var.rename({'band': 'time'})
                da_var['time'] = dates[:len(da_var.time)]
                da_var.name = var_out_name
                ds_out[var_out_name] = da_var.load()

        # Tier 4: Atomic NetCDF Staging & Verification
        ds_out.attrs["source"] = "ECMWF/ERA5_LAND/HOURLY"
        ds_out.attrs["temporal_resolution"] = "Hourly"
        ds_out.to_netcdf(nc_path_tmp)
        ds_out.close()

        for temp_tif in temp_tifs:
            if os.path.exists(temp_tif):
                os.remove(temp_tif)

        tmp_valid, tmp_msg, tmp_steps = check_file_integrity(nc_path_tmp, tahun, bulan, 'era5_land')
        if tmp_valid or (tmp_steps == count):
            if os.path.exists(nc_path):
                os.remove(nc_path)
            os.replace(nc_path_tmp, nc_path)
            print(f"[{tahun}-{bulan:02d}] [SUKSES] Tersimpan secara atomic: {nc_path} ({tmp_steps} jam, 6 vars)\n")
            return nc_path
        else:
            print(f"[{tahun}-{bulan:02d}] [ERROR] Validasi atomic gagal ({tmp_msg}). File sementara dibuang.")
            if os.path.exists(nc_path_tmp):
                os.remove(nc_path_tmp)
            return None

    except Exception as e:
        print(f"[{tahun}-{bulan:02d}] Error: {e}")
        for temp_tif in temp_tifs:
            if os.path.exists(temp_tif):
                os.remove(temp_tif)
        if os.path.exists(nc_path_tmp):
            os.remove(nc_path_tmp)
        return None


def init_kaggle_auth():
    """Menginisialisasi kredensial Kaggle untuk upload versi baru dataset."""
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            k_user = None
            k_key = None
            for u_label in ["KAGGLE_USERNAME", "kaggle_username", "USERNAME"]:
                try:
                    k_user = secrets.get_secret(u_label)
                    if k_user: break
                except Exception:
                    pass
            for k_label in ["KAGGLE_KEY", "kaggle_key", "KAGGLE_API_KEY", "KAGGLE_TOKEN"]:
                try:
                    k_key = secrets.get_secret(k_label)
                    if k_key: break
                except Exception:
                    pass
            
            if not k_user:
                k_user = "jerismeteo"
            
            if k_key:
                os.environ["KAGGLE_USERNAME"] = k_user
                os.environ["KAGGLE_KEY"] = k_key
                import kagglehub
                kagglehub.auth.set_kaggle_credentials(k_user, k_key)
                print(f"[AUTH] Kredensial Kaggle aktif via Secrets untuk user: {k_user}")
                return True
        except Exception as e:
            print(f"[AUTH] Kaggle Secrets tidak tersedia: {e}")

    candidates = [
        os.path.join(os.getcwd(), "kaggle.json"),
        os.path.join(os.path.dirname(os.getcwd()), "kaggle.json"),
        os.path.expanduser("~/.kaggle/kaggle.json")
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    c = json.load(f)
                os.environ["KAGGLE_USERNAME"] = c["username"]
                os.environ["KAGGLE_KEY"] = c["key"]
                import kagglehub
                kagglehub.auth.set_kaggle_credentials(c["username"], c["key"])
                print(f"[AUTH] Kredensial Kaggle aktif via file: {p}")
                return True
            except Exception:
                pass

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    print("[AUTH] Peringatan: Kredensial Kaggle (KAGGLE_KEY) belum terkonfigurasi.")
    return False

def upload_ke_kaggle(dataset_handle, upload_root, subfolder_data, nama_dataset):
    """
    Mengunggah dataset NetCDF terbaru ke Kaggle Dataset menggunakan kagglehub.
    Menghitung jumlah file .nc valid dan mengabaikan file temporer/cache.
    """
    import kagglehub
    if not init_kaggle_auth():
        print(f"⚠️ [UPLOAD] Dilewati: Kredensial Kaggle tidak ditemukan. Tambahkan Secret 'KAGGLE_KEY' di Kaggle Secrets.")
        return False
        
    target_nc_dir = os.path.join(upload_root, subfolder_data)
    list_files = glob.glob(os.path.join(target_nc_dir, "**", "*.nc"), recursive=True)
    if not list_files:
        print(f"⚠️ [UPLOAD] Tidak ditemukan file NetCDF di: {target_nc_dir}. Upload dibatalkan.")
        return False

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M WIB")
    version_notes = f"Pembaruan Otomatis Data GEE ({nama_dataset}): {len(list_files)} file NetCDF per {now_str}"

    print(f"\n{'='*75}")
    print(f"🚀 MENGUNGGAH VERSI BARU KE KAGGLE DATASET")
    print(f"🏷️  Handle Dataset : {dataset_handle}")
    print(f"📁 Folder Basis   : {upload_root}")
    print(f"📊 Total File     : {len(list_files)} file NetCDF ({subfolder_data})")
    print(f"📝 Catatan Versi  : {version_notes}")
    print(f"{'='*75}")

    try:
        kagglehub.dataset_upload(
            handle=dataset_handle,
            local_dataset_dir=upload_root,
            version_notes=version_notes,
            ignore_patterns=[
                "*.tmp", "*.tif", "*.temp", "*.geojson", "*.json",
                "*.ipynb", "__pycache__/", ".ipynb_checkpoints/"
            ]
        )
        print(f"✅ [SUKSES] Dataset {dataset_handle} berhasil diperbarui di Kaggle!")
        return True
    except Exception as e:
        print(f"❌ [GAGAL] Terjadi kesalahan saat mengunggah dataset: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="GEE ERA5-Land Hourly Downloader CLI with Pre-Flight Guardrails")
    parser.add_argument("--start-year", type=int, default=2000, help="Tahun awal pengunduhan (default: 2000)")
    parser.add_argument("--end-year", type=int, default=2026, help="Tahun akhir pengunduhan (default: 2026)")
    parser.add_argument("--start-month", type=int, default=1, help="Bulan awal (default: 1)")
    parser.add_argument("--end-month", type=int, default=12, help="Bulan akhir (default: 12)")
    parser.add_argument("--force", action="store_true", help="Paksa unduh ulang meskipun file lokal sudah lengkap")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.getcwd(), "data", "era5_land"), help="Folder basis output ERA5-Land")
    parser.add_argument("--geojson", type=str, default="kebumen_33.05_batas_kecamatan.geojson", help="Path batas GeoJSON Kebumen")
    parser.add_argument("--service-account", type=str, default=None, help="Path file Service Account JSON GEE")
    parser.add_argument("--upload-kaggle", action="store_true", help="Unggah otomatis versi baru ke Kaggle Dataset via kagglehub")
    args = parser.parse_args()

    print("=" * 75)
    print("GEE ERA5-LAND HOURLY (PER JAM) MULTI-VARIABEL DOWNLOADER")
    print(f"Periode: {args.start_year} s.d. {args.end_year}")
    print(f"Output : {args.output_dir}")
    print(f"Force  : {args.force}")
    print("=" * 75)

    init_earth_engine(sa_path=args.service_account)
    batas_ee, ee_bbox, bounds = prepare_geometry(args.geojson)
    print(f"[Geometri] Extent Kebumen: Lon [{bounds[0]:.4f}, {bounds[2]:.4f}], Lat [{bounds[1]:.4f}, {bounds[3]:.4f}]\n")

    for tahun in range(args.start_year, args.end_year + 1):
        m_start = args.start_month if tahun == args.start_year else 1
        m_end = args.end_month if tahun == args.end_year else 12

        for bulan in range(m_start, m_end + 1):
            unduh_era5_land_bulanan(tahun, bulan, batas_ee, ee_bbox, args.output_dir, force=args.force)

    print("=" * 75)
    print("SELESAI PIPELINE UNDUH ERA5-LAND HOURLY")
    print("=" * 75)


    if args.upload_kaggle or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        upload_ke_kaggle(
            dataset_handle="jerismeteo/gee-era5-land-kebumen",
            upload_root=os.getcwd(),
            subfolder_data='data/era5_land',
            nama_dataset="ECMWF ERA5-Land Kebumen"
        )

if __name__ == "__main__":
    main()
    main()
