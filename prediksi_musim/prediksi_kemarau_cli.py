#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone CLI Runner: Sistem Operasional Prediksi Musim Kemarau Kabupaten Kebumen
Mendukung inferensi operasional, benchmark LOYO, downscaling topografi 250m, dan ekspor 10 peta tematik 16:9.

Penggunaan:
    python prediksi_musim/prediksi_kemarau_cli.py --target-year 2026 --oni-djf 1.85 --oni-jfm 1.45 --dmi-mar 0.30 --downscale-250m --export-plots
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from prediksi_musim.lstm_kemarau_engine import (
    run_loyo_benchmark_kemarau,
    predict_future_dry_season_2026
)
from prediksi_musim.downscaling_kemarau_250m import run_spatial_downscaling_kemarau_2026
from prediksi_musim.visualisasi_kemarau import generate_all_kemarau_maps

def main():
    parser = argparse.ArgumentParser(
        description="Sistem Prediksi Musim Kemarau Kabupaten Kebumen (LSTM Multi-Head & Topographic Downscaling 250m)"
    )
    parser.add_argument("--target-year", type=int, default=2026, help="Tahun target prakiraan musim kemarau (default: 2026)")
    parser.add_argument("--oni-djf", type=float, default=1.85, help="Indeks ONI DJF (default: 1.85)")
    parser.add_argument("--oni-jfm", type=float, default=1.45, help="Indeks ONI JFM (default: 1.45)")
    parser.add_argument("--dmi-feb", type=float, default=0.25, help="Indeks DMI Februari (default: 0.25)")
    parser.add_argument("--dmi-mar", type=float, default=0.30, help="Indeks DMI Maret (default: 0.30)")
    parser.add_argument("--run-loyo", action="store_true", help="Jalankan evaluasi komparatif Leave-One-Year-Out (LOYO 2000-2025)")
    parser.add_argument("--downscale-250m", action="store_true", help="Jalankan downscaling spasial topografi 250m dan konservasi massa")
    parser.add_argument("--export-plots", action="store_true", help="Ekspor 10 peta tematik standar publikasi 16:9 (300 DPI)")
    parser.add_argument("--sync-dashboard", action="store_true", help="Sinkronkan GeoJSON & JSON hasil ke repositori web Dashboard_Jerukagung-Meteorologi")
    parser.add_argument("--all", action="store_true", help="Jalankan seluruh rangkaian pipeline secara berurutan")

    args = parser.parse_args()

    print("=" * 85)
    print("SISTEM OPERASIONAL PREDIKSI MUSIM KEMARAU KABUPATEN KEBUMEN")
    print("=" * 85)

    if args.all or args.run_loyo:
        print("\n>>> Langkah 1: Menjalankan Benchmark Komparatif Multi-Algoritma LOYO 2000-2025...")
        run_loyo_benchmark_kemarau()

    print(f"\n>>> Menjalankan Inferensi Prediksi Musim Kemarau Tahun {args.target_year}...")
    predict_future_dry_season_2026(
        oni_djf=args.oni_djf,
        oni_jfm=args.oni_jfm,
        dmi_feb=args.dmi_feb,
        dmi_mar=args.dmi_mar
    )

    if args.all or args.downscale_250m:
        print("\n>>> Menjalankan Spatial Downscaling Topografi 250m & Konservasi Massa Regional...")
        run_spatial_downscaling_kemarau_2026()

    if args.all or args.export_plots:
        print("\n>>> Mengekspor 10 Peta Tematik Standar 16:9 (300 DPI) ke 'prediksi_musim/prediksi_musim_kemarau/'...")
        generate_all_kemarau_maps()

    if args.all or args.sync_dashboard:
        print("\n>>> Menyinkronkan data & peta interaktif ke Dashboard_Jerukagung-Meteorologi...")
        from prediksi_musim.sync_to_dashboard import sync_to_dashboard
        sync_to_dashboard()

    print("\n" + "=" * 85)
    print("PIPELINE PREDIKSI MUSIM KEMARAU SELESAI DENGAN SUKSES!")
    print("=" * 85)

if __name__ == "__main__":
    main()

