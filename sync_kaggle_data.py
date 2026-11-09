#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
KAGGLE DATASET AUTOMATED INCREMENTAL SYNCHRONIZER
Projek Downscale Kebumen - BMKG / JerisMeteo
=============================================================================
Secara otomatis menyinkronkan data presipitasi & atmosferik resolusi tinggi
dari 4 dataset Kaggle:
  1. jerismeteo/gee-chirps-kebumen   -> data/chirps/
  2. jerismeteo/gee-gsmap-kebumen    -> data/gsmap/
  3. jerismeteo/gee-era5-land-kebumen-> data/era5_land/
  4. jerismeteo/gee-imerg-kebumen    -> data/imerg/

Fitur Kritis:
  - Drive C Safety Guard: Mengalihkan cache kagglehub ke D:\\cache_kagglehub.
  - Multi-Auth Support: Mendukung kaggle.json di folder induk atau .kaggle_token.
  - Lightning-Fast Incremental NetCDF Validation (< 5ms per file via netCDF4).
  - Atomic File Writing & Staging Cleanup untuk menghemat kapasitas hard disk.
=============================================================================
"""

import os
import sys
import re
import json
import shutil
import logging
import argparse
import calendar
from datetime import datetime
from pathlib import Path

# 1. Konfigurasi encoding stdout untuk kompatibilitas Windows Console
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 2. Drive C Protection: Wajib mengarahkan cache kagglehub ke Drive D
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = Path("D:/cache_kagglehub")
os.environ["KAGGLEHUB_CACHE"] = str(DEFAULT_CACHE_DIR)
os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

# 3. Import modul setelah konfigurasi cache
import netCDF4

# Konfigurasi Dataset Kaggle Target
DATASET_REGISTRY = {
    "chirps": {
        "handle": "jerismeteo/gee-chirps-kebumen",
        "description": "CHIRPS v3 Daily Precipitation (SAT & RNL)",
        "dest_base": BASE_DIR / "data" / "chirps"
    },
    "gsmap": {
        "handle": "jerismeteo/gee-gsmap-kebumen",
        "description": "JAXA GSMaP v8 Hourly Precipitation",
        "dest_base": BASE_DIR / "data" / "gsmap"
    },
    "era5_land": {
        "handle": "jerismeteo/gee-era5-land-kebumen",
        "description": "ECMWF ERA5-Land Hourly Multi-Variable",
        "dest_base": BASE_DIR / "data" / "era5_land"
    },
    "imerg": {
        "handle": "jerismeteo/gee-imerg-kebumen",
        "description": "NASA GPM IMERG v07 Half-Hourly (30-Minute)",
        "dest_base": BASE_DIR / "data" / "imerg"
    }
}

ERA5_EXPECTED_VARS = [
    'precipitation', 'temperature_2m', 'dewpoint_temperature_2m',
    'u_wind_10m', 'v_wind_10m', 'surface_pressure'
]

def setup_logger(log_file=None):
    """Mengatur logging console dan file opsional."""
    logger = logging.getLogger("KaggleSync")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        f_handler = logging.FileHandler(log_file, encoding='utf-8')
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger

logger = setup_logger()

def authenticate_kaggle():
    """
    Memuat kredensial Kaggle secara otomatis dengan urutan prioritas:
    1. kaggle.json di folder induk repositori (BASE_DIR)
    2. .kaggle_token di folder induk (format KGAT_...)
    3. Variabel lingkungan sistem (KAGGLE_USERNAME, KAGGLE_KEY, KAGGLE_API_TOKEN)
    4. ~/.kaggle/kaggle.json standar
    """
    import kagglehub
    import kagglehub.auth

    # 1. Cek kaggle.json di root project
    local_kaggle_json = BASE_DIR / "kaggle.json"
    if local_kaggle_json.exists():
        try:
            with open(local_kaggle_json, "r", encoding="utf-8") as f:
                creds = json.load(f)
            if "username" in creds and "key" in creds:
                os.environ["KAGGLE_USERNAME"] = creds["username"]
                os.environ["KAGGLE_KEY"] = creds["key"]
                logger.info(f"Kredensial Kaggle dimuat dari {local_kaggle_json.name} (User: {creds['username']})")
                return True
            elif "api_token" in creds:
                kagglehub.auth.set_kaggle_api_token(creds["api_token"])
                logger.info(f"Kaggle API Token dimuat dari {local_kaggle_json.name}")
                return True
        except Exception as e:
            logger.warning(f"Gagal membaca {local_kaggle_json}: {e}")

    # 2. Cek .kaggle_token di root project
    local_token_file = BASE_DIR / ".kaggle_token"
    if local_token_file.exists():
        try:
            token = local_token_file.read_text(encoding="utf-8").strip()
            if token.startswith("KGAT_") or len(token) > 20:
                kagglehub.auth.set_kaggle_api_token(token)
                logger.info(f"Kaggle API Token dimuat dari {local_token_file.name}")
                return True
        except Exception as e:
            logger.warning(f"Gagal membaca {local_token_file}: {e}")

    # 3. Cek variabel lingkungan
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        logger.info(f"Menggunakan kredensial Kaggle dari variabel lingkungan (User: {os.environ.get('KAGGLE_USERNAME')})")
        return True

    # 4. Fallback ~/.kaggle/kaggle.json
    home_kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if home_kaggle_json.exists():
        logger.info("Menggunakan kredensial Kaggle default di ~/.kaggle/kaggle.json")
        return True

    logger.warning("Tidak ditemukan file kredensial Kaggle (kaggle.json / .kaggle_token). Mencoba akses anonim...")
    return False

def fast_check_integrity(nc_path, year, month, dataset_type):
    """
    Tier 1 Ultra-Cepat: Validasi Integritas File NetCDF (< 5ms per file).
    Memeriksa eksistensi, ukuran > 0, dimensi waktu, variabel wajib, dan kelengkapan time-step.
    """
    p = Path(nc_path)
    if not p.exists():
        return False, "File belum ada", 0
    if p.stat().st_size == 0:
        return False, "File 0 byte (kosong/korup)", 0

    _, days_in_month = calendar.monthrange(year, month)
    if dataset_type == "chirps":
        expected_steps = days_in_month
    elif dataset_type in ("gsmap", "era5_land"):
        expected_steps = days_in_month * 24
    elif dataset_type == "imerg":
        expected_steps = days_in_month * 48
    else:
        expected_steps = 1

    try:
        with netCDF4.Dataset(str(p), 'r') as nc:
            if 'time' not in nc.dimensions:
                return False, "Dimensi 'time' tidak ditemukan", 0
            actual_steps = len(nc.dimensions['time'])

            if dataset_type == "era5_land":
                missing = [v for v in ERA5_EXPECTED_VARS if v not in nc.variables]
                if missing:
                    return False, f"Variabel hilang: {missing}", actual_steps
            else:
                if 'precipitation' not in nc.variables:
                    return False, "Variabel 'precipitation' tidak ditemukan", actual_steps

            if actual_steps < expected_steps:
                return False, f"Terpotong ({actual_steps}/{expected_steps} steps)", actual_steps
            return True, f"Lengkap ({actual_steps}/{expected_steps} steps)", actual_steps
    except Exception as e:
        return False, f"Gagal dibuka / korup: {e}", 0

def resolve_target_destination(file_path):
    """
    Menentukan folder target lokal dan metadata (tahun, bulan, dataset_type)
    berdasarkan nama file NetCDF.
    """
    fname = Path(file_path).name

    # 1. CHIRPS SAT & RNL
    m = re.match(r"^chirps_(sat|rnl)_(\d{4})_(\d{2})\.nc$", fname)
    if m:
        varian, yr, mo = m.group(1), int(m.group(2)), int(m.group(3))
        dest = BASE_DIR / "data" / "chirps" / f"chirps_{varian}" / str(yr) / fname
        return "chirps", dest, yr, mo

    # 2. GSMaP
    m = re.match(r"^gsmap_(\d{4})_(\d{2})\.nc$", fname)
    if m:
        yr, mo = int(m.group(1)), int(m.group(2))
        dest = BASE_DIR / "data" / "gsmap" / str(yr) / fname
        return "gsmap", dest, yr, mo

    # 3. ERA5-Land
    m = re.match(r"^era5_land_(\d{4})_(\d{2})\.nc$", fname)
    if m:
        yr, mo = int(m.group(1)), int(m.group(2))
        dest = BASE_DIR / "data" / "era5_land" / str(yr) / fname
        return "era5_land", dest, yr, mo

    # 4. IMERG 30-menit
    m = re.match(r"^imerg_(\d{4})_(\d{2})\.nc$", fname)
    if m:
        yr, mo = int(m.group(1)), int(m.group(2))
        dest = BASE_DIR / "data" / "imerg" / str(yr) / fname
        return "imerg", dest, yr, mo

    return None, None, None, None

def sync_dataset(dataset_key, dry_run=False, force=False):
    """
    Menyinkronkan satu dataset Kaggle ke folder lokal secara inkremental berkecepatan tinggi.
    """
    import kagglehub

    cfg = DATASET_REGISTRY.get(dataset_key)
    if not cfg:
        logger.error(f"Dataset '{dataset_key}' tidak terdaftar dalam registry.")
        return False

    handle = cfg["handle"]
    desc = cfg["description"]
    logger.info(f"{'='*70}")
    logger.info(f"SINKRONISASI DATASET: {desc} ({handle})")
    logger.info(f"{'='*70}")

    try:
        logger.info(f"Memeriksa versi terbaru dataset di Kaggle via kagglehub...")
        download_path = kagglehub.dataset_download(handle)
        logger.info(f"Direktori staging lokal: {download_path}")
    except Exception as e:
        logger.error(f"Gagal mengunduh dataset {handle} dari Kaggle: {e}")
        return False

    download_dir = Path(download_path)
    if not download_dir.exists():
        logger.error(f"Path staging tidak ditemukan: {download_dir}")
        return False

    # Cari semua file NetCDF (.nc) di dalam unduhan Kaggle
    nc_files = list(download_dir.rglob("*.nc"))
    logger.info(f"Ditemukan {len(nc_files)} file NetCDF dalam rilis Kaggle.")

    stats = {"skipped": 0, "updated": 0, "added": 0, "failed": 0}

    for src_nc in sorted(nc_files):
        dtype, dest_nc, yr, mo = resolve_target_destination(src_nc)
        if not dtype or not dest_nc:
            continue

        dest_nc.parent.mkdir(parents=True, exist_ok=True)
        local_exists = dest_nc.exists()

        # 1. Fast-Skip: Jika ukuran file identik dan bukan force, langsung lewati tanpa I/O berat
        if not force and local_exists:
            dest_size = dest_nc.stat().st_size
            src_size = src_nc.stat().st_size
            if dest_size == src_size and dest_size > 0:
                stats["skipped"] += 1
                continue

            # Jika ukuran beda, cek apakah file lokal sudah lengkap 100%
            local_valid, local_msg, local_steps = fast_check_integrity(dest_nc, yr, mo, dtype)
            if local_valid:
                stats["skipped"] += 1
                continue
        else:
            local_valid, local_msg, local_steps = False, "Belum ada", 0

        # 2. Evaluasi File Unduhan Kaggle Baru
        src_valid, src_msg, src_steps = fast_check_integrity(src_nc, yr, mo, dtype)

        # Jika file Kaggle sendiri rusak/0 byte, jangan timpa file lokal
        if not src_valid and src_steps == 0:
            logger.warning(f"  [KORUP DI KAGGLE] {src_nc.name} -> {src_msg}. Dilewati.")
            stats["failed"] += 1
            continue

        if not force and local_exists and (local_steps >= src_steps and local_steps > 0):
            stats["skipped"] += 1
            continue

        # 3. Eksekusi Pembaruan / Penambahan
        action_label = "BARU" if not local_exists else ("REVISI" if src_steps > local_steps else "UPDATE")
        logger.info(f"  [{action_label}] {dest_nc.name} -> Kaggle: {src_steps} steps vs Lokal: {local_steps} steps ({src_msg})")

        if dry_run:
            logger.info(f"    [DRY-RUN] Target salin: {dest_nc}")
            if local_exists:
                stats["updated"] += 1
            else:
                stats["added"] += 1
            continue

        # Atomic Stage: Salin dulu ke .tmp lalu replace
        tmp_target = dest_nc.with_suffix(".nc.tmp")
        try:
            shutil.copy2(src_nc, tmp_target)
            tmp_valid, tmp_msg, tmp_steps = fast_check_integrity(tmp_target, yr, mo, dtype)
            if tmp_valid or (tmp_steps == src_steps):
                if dest_nc.exists():
                    dest_nc.unlink()
                tmp_target.rename(dest_nc)
                logger.info(f"    [OK] Sukses disimpan: {dest_nc.name}")
                if local_exists:
                    stats["updated"] += 1
                else:
                    stats["added"] += 1
            else:
                logger.error(f"    [FAIL] Validasi integritas gagal pada {tmp_target.name}: {tmp_msg}")
                if tmp_target.exists():
                    tmp_target.unlink()
                stats["failed"] += 1
        except Exception as e:
            logger.error(f"    [ERROR] Gagal memindahkan file {src_nc.name}: {e}")
            if tmp_target.exists():
                tmp_target.unlink()
            stats["failed"] += 1

    logger.info(f"Ringkasan Sinkronisasi {dataset_key.upper()}:")
    logger.info(f"  - Dilewati (Lengkap & Up-to-Date): {stats['skipped']}")
    logger.info(f"  - File Baru Ditambahkan         : {stats['added']}")
    logger.info(f"  - File Diperbarui (Lebih Lengkap): {stats['updated']}")
    logger.info(f"  - Gagal / Tidak Valid           : {stats['failed']}")

    # 4. Zero Waste Prune: Hapus folder staging setelah selesai agar Drive D tidak bengkak
    if not dry_run and download_dir.exists():
        try:
            dataset_root_cache = download_dir.parent.parent
            logger.info(f"Membersihkan cache staging di {dataset_root_cache}...")
            shutil.rmtree(dataset_root_cache, ignore_errors=True)
            logger.info("Pembersihan cache staging selesai.")
        except Exception as e:
            logger.warning(f"Gagal menghapus cache staging {download_dir}: {e}")

    logger.info(f"Selesai sinkronisasi {dataset_key.upper()}.\n")
    return True

def sync_all_datasets(dry_run=False, force=False):
    """Menyinkronkan seluruh 4 dataset terdaftar."""
    logger.info(f"\n{'#'*75}")
    logger.info(f"# MEMULAI SINKRONISASI LENGKAP DARI KAGGLE ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    logger.info(f"# Mode Dry-Run: {dry_run} | Mode Force: {force}")
    logger.info(f"{'#'*75}\n")

    auth_ok = authenticate_kaggle()
    if not auth_ok:
        logger.warning("Perhatian: Autentikasi Kaggle tidak ditemukan. Jika dataset private, proses unduh akan gagal.")

    results = {}
    for key in DATASET_REGISTRY.keys():
        results[key] = sync_dataset(key, dry_run=dry_run, force=force)

    logger.info(f"{'#'*75}")
    logger.info(f"# HASIL AKHIR SINKRONISASI SELURUH DATASET:")
    for k, ok in results.items():
        status = "SUKSES" if ok else "GAGAL"
        logger.info(f"#   - {k.upper():<12}: {status}")
    logger.info(f"{'#'*75}\n")

def main():
    parser = argparse.ArgumentParser(description="Kaggle Dataset Incremental Synchronizer CLI")
    parser.add_argument("--dataset", choices=["all", "chirps", "gsmap", "era5_land", "imerg"], default="all",
                        help="Pilih dataset yang akan disinkronkan (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulasikan proses tanpa memodifikasi file lokal")
    parser.add_argument("--force", action="store_true",
                        help="Paksa unduh dan timpa seluruh file lokal")
    parser.add_argument("--log-file", type=str, default=str(BASE_DIR / "logs" / "sync_kaggle.log"),
                        help="Lokasi file output log (default: logs/sync_kaggle.log)")
    args = parser.parse_args()

    global logger
    logger = setup_logger(log_file=args.log_file)

    if args.dataset == "all":
        sync_all_datasets(dry_run=args.dry_run, force=args.force)
    else:
        authenticate_kaggle()
        sync_dataset(args.dataset, dry_run=args.dry_run, force=args.force)

if __name__ == "__main__":
    main()
