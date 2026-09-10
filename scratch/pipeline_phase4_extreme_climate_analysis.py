"""
Phase 4: Climate Extremes & ENSO Impact Analysis (2001-2025)
============================================================
Quantifies spatial precipitation impacts of historical extreme climate events:
1. Super El Niño: 2015 and 2023 (Severe agricultural drought & water deficits)
2. Super La Niña: 2010 and 2022 (Severe rainfall surplus, flood & landslide triggers)
3. Computes absolute (mm) and percentage (%) spatial anomalies against 25-year baseline
4. Calculates zonal vulnerability for all 26 kecamatan in Kebumen
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from rasterio import features
from rasterio.transform import Affine

def run():
    t0 = time.time()
    print("=" * 85)
    print("🚀 FASE 4: ANALISIS IKLIM EKSTREM ENSO & ANOMALI SPASIAL KABUPATEN KEBUMEN")
    print("   Target: El Niño (2015, 2023) vs La Niña (2010, 2022) vs Baseline 25 Tahun")
    print("=" * 85)

    out_dir = Path("data/processed_25yr")
    clim_npz = np.load(out_dir / "reconstructed_climatology_250m.npz")
    terrain_npz = np.load(out_dir / "terrain_covariates_250m.npz")

    mean_annual = clim_npz['mean_annual_grid']
    annual_grids = clim_npz['annual_grids']
    years = list(clim_npz['years'])
    mask_keb = clim_npz['mask_kebumen']
    trans = Affine(*terrain_npz['transform_meta'])
    h, w = mean_annual.shape

    # Indeks tahun ekstrem
    y_2010_idx = years.index(2010)
    y_2015_idx = years.index(2015)
    y_2022_idx = years.index(2022)
    y_2023_idx = years.index(2023)

    p_2010 = annual_grids[y_2010_idx]
    p_2015 = annual_grids[y_2015_idx]
    p_2022 = annual_grids[y_2022_idx]
    p_2023 = annual_grids[y_2023_idx]

    # Kalkulasi Anomali Spasial
    print("\n[Langkah 1/3] Mengkalkulasi Anomali Spasial Presipitasi...")
    ano_2010_mm = p_2010 - mean_annual
    ano_2015_mm = p_2015 - mean_annual
    ano_2022_mm = p_2022 - mean_annual
    ano_2023_mm = p_2023 - mean_annual

    ano_2010_pct = np.zeros_like(ano_2010_mm)
    ano_2015_pct = np.zeros_like(ano_2015_mm)
    ano_2022_pct = np.zeros_like(ano_2022_mm)
    ano_2023_pct = np.zeros_like(ano_2023_mm)

    valid_mask = mask_keb & (mean_annual > 0)
    ano_2010_pct[valid_mask] = (ano_2010_mm[valid_mask] / mean_annual[valid_mask]) * 100.0
    ano_2015_pct[valid_mask] = (ano_2015_mm[valid_mask] / mean_annual[valid_mask]) * 100.0
    ano_2022_pct[valid_mask] = (ano_2022_mm[valid_mask] / mean_annual[valid_mask]) * 100.0
    ano_2023_pct[valid_mask] = (ano_2023_mm[valid_mask] / mean_annual[valid_mask]) * 100.0

    print(f"✓ 2010 (Super La Niña): Rata-rata Anomali Kebumen = {np.mean(ano_2010_mm[valid_mask]):+.1f} mm ({np.mean(ano_2010_pct[valid_mask]):+.1f}%) [SURPLUS]")
    print(f"✓ 2015 (Super El Niño): Rata-rata Anomali Kebumen = {np.mean(ano_2015_mm[valid_mask]):+.1f} mm ({np.mean(ano_2015_pct[valid_mask]):+.1f}%) [DEFISIT]")
    print(f"✓ 2022 (Triple-dip La Niña): Rata-rata Anomali Kebumen = {np.mean(ano_2022_mm[valid_mask]):+.1f} mm ({np.mean(ano_2022_pct[valid_mask]):+.1f}%) [SURPLUS]")
    print(f"✓ 2023 (Strong El Niño): Rata-rata Anomali Kebumen = {np.mean(ano_2023_mm[valid_mask]):+.1f} mm ({np.mean(ano_2023_pct[valid_mask]):+.1f}%) [DEFISIT]")

    # 2. Zonal Anomaly untuk 26 Kecamatan
    print("\n[Langkah 2/3] Mengekstraksi Anomali Zonal untuk 26 Kecamatan...")
    gdf_kec = gpd.read_file("33.05_kecamatan.geojson")
    gdf_kec['geometry'] = gdf_kec['geometry'].apply(shapely.force_2d)

    zonal_enso_rows = []
    for _, kec in gdf_kec.iterrows():
        kec_name = kec['nm_kecamatan']
        kec_mask = features.rasterize(
            [(kec['geometry'], 1)],
            out_shape=(h, w),
            transform=trans,
            fill=0
        ).astype(bool) & mask_keb

        if kec_mask.sum() == 0:
            continue

        base_val = float(np.mean(mean_annual[kec_mask]))
        val_2010 = float(np.mean(p_2010[kec_mask]))
        val_2015 = float(np.mean(p_2015[kec_mask]))
        val_2022 = float(np.mean(p_2022[kec_mask]))
        val_2023 = float(np.mean(p_2023[kec_mask]))

        ano_2010_val = val_2010 - base_val
        ano_2015_val = val_2015 - base_val
        ano_2022_val = val_2022 - base_val
        ano_2023_val = val_2023 - base_val

        # Rentang kontras ekstrem (La Niña 2010 vs El Niño 2015)
        contrast_mm = val_2010 - val_2015

        zonal_enso_rows.append({
            'Kecamatan': kec_name,
            'Baseline_25yr_mm': round(base_val, 1),
            'LaNina_2010_mm': round(val_2010, 1),
            'Ano_2010_Pct': round(ano_2010_val / base_val * 100.0, 1),
            'ElNino_2015_mm': round(val_2015, 1),
            'Ano_2015_Pct': round(ano_2015_val / base_val * 100.0, 1),
            'LaNina_2022_mm': round(val_2022, 1),
            'Ano_2022_Pct': round(ano_2022_val / base_val * 100.0, 1),
            'ElNino_2023_mm': round(val_2023, 1),
            'Ano_2023_Pct': round(ano_2023_val / base_val * 100.0, 1),
            'Contrast_Range_mm': round(contrast_mm, 1)
        })

    df_enso_zonal = pd.DataFrame(zonal_enso_rows)
    df_enso_zonal = df_enso_zonal.sort_values(by='Contrast_Range_mm', ascending=False).reset_index(drop=True)
    print("\n" + "=" * 85)
    print("🌪️ DAMPAK IKLIM EKSTREM ENSO PADA 26 KECAMATAN DI KABUPATEN KEBUMEN:")
    print("=" * 85)
    print(df_enso_zonal.to_string(index=False))

    out_csv = out_dir / "extreme_enso_zonal_anomalies.csv"
    df_enso_zonal.to_csv(out_csv, index=False)
    print(f"\n✓ Tabel anomali ekstrem ENSO disimpan di {out_csv}")

    # 3. Simpan Grid Anomali untuk Visualisasi
    print("\n[Langkah 3/3] Menyimpan Grid Anomali Ekstrem...")
    np.savez_compressed(
        out_dir / "extreme_enso_anomaly_grids.npz",
        p_2010=p_2010,
        p_2015=p_2015,
        p_2022=p_2022,
        p_2023=p_2023,
        ano_2010_mm=ano_2010_mm,
        ano_2015_mm=ano_2015_mm,
        ano_2022_mm=ano_2022_mm,
        ano_2023_mm=ano_2023_mm,
        ano_2010_pct=ano_2010_pct,
        ano_2015_pct=ano_2015_pct,
        ano_2022_pct=ano_2022_pct,
        ano_2023_pct=ano_2023_pct
    )
    print(f"✓ Grid anomali ekstrem ENSO disimpan di {out_dir / 'extreme_enso_anomaly_grids.npz'}")

    print("\n" + "=" * 85)
    print(f"🎉 FASE 4 SELESAI DALAM {time.time() - t0:.2f} DETIK!")
    print("=" * 85)

if __name__ == "__main__":
    run()
