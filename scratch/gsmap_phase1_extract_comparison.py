"""
GSMaP vs CHIRPS Comparative Downscaling - Phase 1: Feature Extraction & Alignment (2001 - 2025)
==============================================================================================
Extracts, aligns, and calculates comparative metrics between GSMaP v8 (Passive Microwave)
and CHIRPS v2.0 (Thermal Infrared + Station) across 300 months (2001 - 2025).
Prepares seasonal bins (DJF, MAM, JJA, SON) and rain gauge benchmarks.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

def run():
    t0 = time.time()
    print("=" * 85)
    print("🚀 FASE 1: EKSTRAKSI & PENYELARASAN KOMPARATIF GSMaP vs CHIRPS (2001 - 2025)")
    print("   Wilayah: Kabupaten Kebumen | Horizon: 300 Bulan | Sensor: PMW vs TIR")
    print("=" * 85)

    out_dir = Path("data/processed_gsmap_vs_chirps")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Muat dataset 25 tahun yang sudah tervalidasi
    src_csv = Path("data/processed_25yr/multivariate_25yr_timeseries_monthly.csv")
    df_ts = pd.read_csv(src_csv)

    # Tambahkan metrik komparasi
    df_ts['diff_mm'] = df_ts['gsmap_raw_mm'] - df_ts['chirps_raw_mm']
    df_ts['ratio_gsmap_chirps'] = df_ts['gsmap_raw_mm'] / np.where(df_ts['chirps_raw_mm'] > 0, df_ts['chirps_raw_mm'], 1.0)

    # Klasifikasi Musim Tropis Indonesia
    # DJF: Des, Jan, Feb (Musim Hujan)
    # MAM: Mar, Apr, Mei (Peralihan 1)
    # JJA: Jun, Jul, Agu (Musim Kemarau)
    # SON: Sep, Okt, Nov (Peralihan 2)
    def get_season(m):
        if m in [12, 1, 2]:
            return 'DJF (Hujan)'
        elif m in [3, 4, 5]:
            return 'MAM (Peralihan 1)'
        elif m in [6, 7, 8]:
            return 'JJA (Kemarau)'
        else:
            return 'SON (Peralihan 2)'

    df_ts['season'] = df_ts['month'].apply(get_season)

    # Ringkasan Statistik Komparasi
    c_mean = df_ts['chirps_raw_mm'].mean()
    g_mean = df_ts['gsmap_raw_mm'].mean()
    diff_mean = df_ts['diff_mm'].mean()
    corr = df_ts['chirps_raw_mm'].corr(df_ts['gsmap_raw_mm'])

    print(f"\n✓ Rata-rata CHIRPS 25 Tahun : {c_mean:.2f} mm/bulan")
    print(f"✓ Rata-rata GSMaP 25 Tahun  : {g_mean:.2f} mm/bulan")
    print(f"✓ Selisih Rata-rata (GSMaP - CHIRPS): {diff_mean:+.2f} mm/bulan")
    print(f"✓ Korelasi Pearson (r) Antar-Satelit: {corr:.4f}")

    print("\n[Rangkuman Karakteristik Berdasarkan Musim (25 Tahun)]")
    season_grp = df_ts.groupby('season').agg({
        'chirps_raw_mm': 'mean',
        'gsmap_raw_mm': 'mean',
        'diff_mm': 'mean',
        'ratio_gsmap_chirps': 'mean'
    }).reset_index()
    print(season_grp.to_string(index=False))

    out_csv = out_dir / "comparative_timeseries_25yr.csv"
    df_ts.to_csv(out_csv, index=False)
    print(f"\n✓ Dataset komparatif tersimpan di: {out_csv}")

    # Ekstraksi 30 Stasiun Observasi Lapangan
    df_gauges = pd.read_csv("downscale_curah_hujan/data/raw/rain_gauges_daily.csv")
    stations = df_gauges.groupby('station_id').first().reset_index()
    stations.to_csv(out_dir / "station_metadata_30.csv", index=False)
    print(f"✓ Metadata 30 stasiun tersimpan di: {out_dir / 'station_metadata_30.csv'}")

    print("\n" + "=" * 85)
    print(f"🎉 FASE 1 SELESAI DALAM {time.time() - t0:.2f} DETIK!")
    print("=" * 85)

if __name__ == "__main__":
    run()
