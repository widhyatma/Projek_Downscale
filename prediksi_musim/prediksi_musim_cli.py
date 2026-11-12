#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production CLI Runner: Sistem Prediksi Musim Hujan Kebumen (Multi-Sensor & Reanalisis)
Menyediakan antarmuka baris perintah (CLI) untuk memprediksi Awal Musim Hujan (AMH),
Sifat Hujan, Puncak Musim Hujan, dan Rekomendasi Kalender Tanam 26 Kecamatan di Kabupaten Kebumen.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Path root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from prediksi_musim.lstm_engine import predict_future_rainy_season_2026_lstm, run_loyo_benchmark

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistem Prediksi Awal Musim Hujan (AMH) & Sifat Hujan Tingkat Kecamatan di Kabupaten Kebumen."
    )
    parser.add_argument("--target-year", type=int, default=2026, help="Tahun target awal musim hujan (default: 2026).")
    parser.add_argument("--oni-jas", type=float, default=2.16, help="Nilai Indeks Oceanic Niño Index (ONI) JAS (default: 2.16).")
    parser.add_argument("--oni-jja", type=float, default=1.80, help="Nilai Indeks ONI JJA (default: 1.80).")
    parser.add_argument("--dmi-agu", type=float, default=0.35, help="Nilai Indeks Dipole Mode Index (DMI) Agustus (default: 0.35).")
    parser.add_argument("--dmi-sep", type=float, default=0.42, help="Nilai Indeks DMI September (default: 0.42).")
    parser.add_argument("--run-loyo", action="store_true", help="Jalankan validasi Leave-One-Year-Out (LOYO) pada data historis 2000-2025.")
    parser.add_argument("--compare-climatology", action="store_true", help="Jalankan evaluasi komparatif klimatologi multi-sensor (CHIRPS, GSMaP, ERA5, IMERG).")
    parser.add_argument("--build-atlas-250m", action="store_true", help="Bangun 36 raster GeoTIFF Atlas Normal Dasarian 250m sepanjang tahun.")
    parser.add_argument("--downscale-250m", action="store_true", help="Jalankan spatial downscaling topografi resolusi 250m (GeoTIFF).")
    parser.add_argument("--export-plots", action="store_true", help="Otomatis hasilkan 10 peta tematik resolusi tinggi 16:9 (300 DPI).")
    parser.add_argument("--sync-dashboard", action="store_true", help="Sinkronkan GeoJSON & JSON hasil ke repositori web Dashboard_Jerukagung-Meteorologi.")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "prediksi_musim" / "data"), help="Direktori output CSV hasil.")
    return parser.parse_args()

def run_prediction_pipeline(args):
    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"🌦️ SISTEM PREDIKSI MUSIM HUJAN RESOLUSI TINGGI KABUPATEN KEBUMEN")
    print(f"Target Musim: {args.target_year}/{args.target_year + 1}")
    print(f"Parameter Global: ONI JAS = {args.oni_jas:+.2f} | DMI Sep = {args.dmi_sep:+.2f}")
    print("="*80)
    
    if args.run_loyo:
        print("\n[STEP 1/3] Menjalankan Benchmark Model Leave-One-Year-Out (LOYO 2000-2025)...")
        run_loyo_benchmark()
    else:
        print("\n[STEP 1/3] Menggunakan arsitektur model LSTM Multi-Head...")
        
    print("\n[STEP 2/3] Menginferensi Prakiraan AMH, Sifat Hujan, & Kalender Tanam 26 Kecamatan berbasis LSTM...")
    df_out = predict_future_rainy_season_2026_lstm(
        oni_jas=args.oni_jas,
        oni_jja=args.oni_jja,
        dmi_agu=args.dmi_agu,
        dmi_sep=args.dmi_sep
    )
    
    print("\n" + "="*80)
    print(f"HASIL PREDIKSI AWAL MUSIM HUJAN KEBUMEN {args.target_year}/{args.target_year + 1}")
    print("="*80)
    print(df_out[['kecamatan', 'normal_amh_label', 'prakiraan_amh_label', 'perbandingan_terhadap_normal', 'prakiraan_sifat_hujan', 'rekomendasi_tanam_padi_mt1']].to_string(index=False))
    
    if args.compare_climatology:
        print("\n[STEP KHUSUS] Menjalankan Evaluasi Komparatif & Fusi Klimatologi Multi-Sensor...")
        from prediksi_musim.multi_sensor_climatology import build_multi_sensor_comparison
        build_multi_sensor_comparison()

    if args.build_atlas_250m:
        print("\n[STEP KHUSUS] Menghasilkan 36 GeoTIFF Atlas Normal Dasarian 250m Sepanjang Tahun...")
        from prediksi_musim.atlas_klimatologi_250m import generate_atlas_36_dasarians
        generate_atlas_36_dasarians()

    if args.downscale_250m:
        print("\n[STEP 3/4] Menjalankan Topographic Spatial Downscaling 250m...")
        from prediksi_musim.topographic_downscaling import run_spatial_downscaling_2026
        run_spatial_downscaling_2026()

    if args.export_plots:
        print("\n[STEP 4/4] Menghasilkan 10 Peta Tematik Standar Publikasi 16:9 (termasuk Raster 250m & Komparasi Sensor)...")
        from prediksi_musim.visualisasi_peta import generate_all_maps
        generate_all_maps()
    else:
        print("\n[STEP 4/4] Pembuatan peta dilewati (Gunakan flag --export-plots untuk membuat peta).")

    if args.sync_dashboard:
        print("\n[STEP 5] Menyinkronkan data & peta interaktif ke Dashboard_Jerukagung-Meteorologi...")
        from prediksi_musim.sync_to_dashboard import sync_to_dashboard
        sync_to_dashboard()
        
    print("\n[SELESAI] Pipeline Prediksi Musim Hujan berhasil dituntaskan!\n")

if __name__ == "__main__":
    args = parse_args()
    run_prediction_pipeline(args)

