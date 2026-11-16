#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard Helper Modul Konfigurasi Wilayah & Struktur Folder GEE Multi-Kabupaten
Stasiun Meteorologi Kelas III Jerukagung

Mendukung organisasi folder 5-tier hierarkis standar:
  data/<jenis_data>/<kabupaten>/<tahun>/<file_atau_bulan>.nc
Contoh:
  data/chirps/purworejo/chirps_sat/2026/2026_01/chirps_sat_2026_01_01.nc
  data/gsmap/purworejo/2026/gsmap_2026_01.nc
  data/era5_land/purworejo/2026/era5_land_2026_01.nc
  data/imerg/purworejo/2026/imerg_2026_01.nc

Mempertahankan backward compatibility 100% untuk data historis Kebumen yang sudah ada.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

KABUPATEN_REGISTRY = {
    "kebumen": {
        "name": "Kabupaten Kebumen",
        "kode": "33.05",
        "geojson": BASE_DIR / "kebumen_33.05_batas_kecamatan.geojson",
        "geojson_desa": BASE_DIR / "kebumen_33.05_batas_desa.geojson",
        "dem": BASE_DIR / "data" / "DEM_Topografi_Kebumen_30m.tif",
        "subfolder": "kebumen",
    },
    "purworejo": {
        "name": "Kabupaten Purworejo",
        "kode": "33.06",
        "geojson": BASE_DIR / "purworejo_33.06_batas_kecamatan.geojson",
        "geojson_desa": BASE_DIR / "purworejo_33.06_batas_desa.geojson",
        "dem": BASE_DIR / "data" / "DEM_Topografi_Purworejo_30m.tif",
        "subfolder": "purworejo",
    },
}

def resolve_kabupaten_paths(dataset_name: str, kabupaten_name: str = "kebumen", custom_geojson: str = None, custom_output_dir: str = None):
    """
    Menentukan path GeoJSON dan folder output hierarkis untuk download data GEE.
    Format standar: data/<jenis_data>/<kabupaten>/
    Backward compatibility: Jika kebumen dan data/<jenis_data>/ sudah berisi data tanpa subfolder 'kebumen',
    tetap menggunakan folder tersebut agar tidak mengulang download.
    """
    kab_key = str(kabupaten_name).lower().strip()
    config = KABUPATEN_REGISTRY.get(kab_key, {
        "name": f"Kabupaten {kabupaten_name.capitalize()}",
        "kode": "custom",
        "geojson": BASE_DIR / f"{kab_key}_batas_kecamatan.geojson",
        "geojson_desa": BASE_DIR / f"{kab_key}_batas_desa.geojson",
        "dem": BASE_DIR / "data" / f"DEM_Topografi_{kabupaten_name.capitalize()}_30m.tif",
        "subfolder": kab_key,
    })

    # 1. Resolve GeoJSON
    if custom_geojson and custom_geojson != "kebumen_33.05_batas_kecamatan.geojson":
        geojson_path = Path(custom_geojson)
    else:
        geojson_path = config["geojson"]

    # 2. Resolve Output Directory
    if custom_output_dir:
        output_dir = Path(custom_output_dir)
    else:
        base_data_dir = BASE_DIR / "data" / dataset_name
        # Cek backward compatibility untuk kebumen:
        # Jika folder kebumen belum ada tapi sudah ada folder tahun (2000..2026), gunakan base_data_dir langsung
        if kab_key == "kebumen" and not (base_data_dir / "kebumen").exists() and any(base_data_dir.glob("20*")):
            output_dir = base_data_dir
        else:
            output_dir = base_data_dir / config["subfolder"]

    output_dir.mkdir(parents=True, exist_ok=True)
    return str(geojson_path), str(output_dir), config
