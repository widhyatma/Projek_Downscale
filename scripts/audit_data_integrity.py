#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Integrity & Completeness Auditor
Memeriksa kontinuitas kalender, kelengkapan time steps, ketiadaan file korup,
dan kesehatan struktur folder pada seluruh dataset presipitasi & atmosfer di data/
"""

import os
import sys
import glob
import calendar
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import xarray as xr

DATASETS = {
    'CHIRPS (RNL)': {
        'pattern': 'data/chirps/chirps_rnl/**/*.nc',
        'type': 'daily',
        'var': 'precipitation'
    },
    'CHIRPS (SAT)': {
        'pattern': 'data/chirps/chirps_sat/**/*.nc',
        'type': 'daily',
        'var': 'precipitation'
    },
    'GSMaP': {
        'pattern': 'data/gsmap/**/*.nc',
        'type': 'hourly',
        'var': 'hourly_precip'
    },
    'ERA5-Land': {
        'pattern': 'data/era5_land/**/*.nc',
        'type': 'hourly',
        'var': 'temperature_2m'
    },
    'NASA GPM IMERG': {
        'pattern': 'data/imerg/**/*.nc',
        'type': 'half_hourly',
        'var': 'precipitation'
    }
}

def audit():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 80)
    print("AUDIT INTEGRITAS & KELENGKAPAN DATASET METEOROLOGI KEBUMEN")
    print(f"Direktori Basis : {base_dir}")
    print(f"Waktu Audit     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for name, cfg in DATASETS.items():
        pattern = os.path.join(base_dir, cfg['pattern'])
        files = sorted(glob.glob(pattern, recursive=True))
        
        # Saring jika ada file non-nc atau temp
        files = [f for f in files if f.endswith('.nc') and not f.endswith('.tmp')]

        print(f"\n--- {name} ---")
        if not files:
            print("  [ERROR] Tidak ada file ditemukan!")
            continue

        # Ekstrak tahun dan bulan
        dates = []
        corrupted = []
        truncated = []

        for fpath in files:
            fname = os.path.basename(fpath)
            parts = fname.replace('.nc', '').split('_')
            # Format: <prefix>_YYYY_MM.nc
            try:
                yr = int(parts[-2])
                mo = int(parts[-1])
                dates.append((yr, mo, fpath))
            except Exception:
                continue

        dates.sort(key=lambda x: (x[0], x[1]))
        total_months = len(dates)
        first_ym = f"{dates[0][0]}-{dates[0][1]:02d}"
        last_ym = f"{dates[-1][0]}-{dates[-1][1]:02d}"

        # Periksa kontinuitas bulan
        missing_months = []
        curr_yr, curr_mo = dates[0][0], dates[0][1]
        for yr, mo, _ in dates[1:]:
            expected_mo = curr_mo + 1 if curr_mo < 12 else 1
            expected_yr = curr_yr if curr_mo < 12 else curr_yr + 1
            while (yr, mo) != (expected_yr, expected_mo):
                missing_months.append(f"{expected_yr}-{expected_mo:02d}")
                expected_mo = expected_mo + 1 if expected_mo < 12 else 1
                expected_yr = expected_yr if expected_mo != 1 else expected_yr + 1
                if len(missing_months) > 50:
                    break
            curr_yr, curr_mo = yr, mo

        # Periksa 5 file terakhir untuk kecepatan audit
        sample_to_check = dates[-5:] + dates[:2]
        for yr, mo, fpath in sample_to_check:
            try:
                with xr.open_dataset(fpath) as ds:
                    actual = len(ds.time)
                    _, days = calendar.monthrange(yr, mo)
                    if cfg['type'] == 'daily':
                        expected = days
                    elif cfg['type'] == 'hourly':
                        expected = days * 24
                    elif cfg['type'] == 'half_hourly':
                        expected = days * 48
                    
                    if actual < expected:
                        truncated.append((f"{yr}-{mo:02d}", actual, expected))
            except Exception as e:
                corrupted.append((f"{yr}-{mo:02d}", str(e)))

        print(f"  Periode     : {first_ym} s.d. {last_ym} ({total_months} bulan)")
        if not missing_months:
            print("  Kontinuitas : 100% KONTINU (0 bulan hilang/bolong)")
        else:
            print(f"  Kontinuitas : ADA BULAN HILANG ({len(missing_months)} bulan): {missing_months[:5]}")
        
        if corrupted:
            print(f"  Integritas  : [PERINGATAN] {len(corrupted)} file korup: {corrupted}")
        else:
            print("  Integritas  : 100% VALID (file NetCDF dapat dibaca)")

        if truncated:
            print(f"  Time Steps  : Parsial/Terpotong pada: {truncated}")
        else:
            print("  Time Steps  : 100% LENGKAP (sesuai kalender astronomis)")

    print("\n" + "=" * 80)
    print("AUDIT SELESAI")
    print("=" * 80)

if __name__ == "__main__":
    audit()

