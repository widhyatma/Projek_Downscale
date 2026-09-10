"""
GSMaP vs CHIRPS Comparative Downscaling - Phase 4: Extreme ENSO & Zonal Administrative Analysis
================================================================================================
Evaluates differential sensor behavior during extreme ENSO events:
- 2010 Super La Niña
- 2015 Super El Niño
- 2022 Triple-dip La Niña
- 2023 Strong El Niño
Computes zonal statistics and differences across all 26 kecamatan in Kebumen.
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
    print("🚀 FASE 4: ANALISIS IKLIM EKSTREM ENSO & STATISTIK ZONAL 26 KECAMATAN (GSMaP vs CHIRPS)")
    print("=" * 85)

    out_dir = Path("data/processed_gsmap_vs_chirps")
    npz_data = np.load(out_dir / "reconstructed_gsmap_vs_chirps_250m.npz")
    npz_terrain = np.load("data/processed_25yr/terrain_covariates_250m.npz")

    mean_ann_g = npz_data['mean_ann_g']
    mean_ann_c = npz_data['mean_ann_c']
    mean_ann_f = npz_data['mean_ann_f']
    diff_ann_mm = npz_data['diff_ann_mm']
    diff_ann_pct = npz_data['diff_ann_pct']
    annual_g = npz_data['annual_g']
    annual_c = npz_data['annual_c']
    annual_f = npz_data['annual_f']
    years = list(npz_data['years'])
    mask_keb = npz_data['mask_kebumen']
    trans = Affine(*npz_terrain['transform_meta'])
    h, w = mean_ann_g.shape

    # 1. Analisis ENSO Ekstrem
    print("\n[Langkah 1/2] Analisis Respon Satelit terhadap Kejadian Ekstrem ENSO...")
    y_2010_idx = years.index(2010)
    y_2015_idx = years.index(2015)
    y_2022_idx = years.index(2022)
    y_2023_idx = years.index(2023)

    valid_mask = mask_keb & (mean_ann_c > 0) & (mean_ann_g > 0)

    # GSMaP Anomalies
    ano_g_2010 = float(np.mean(annual_g[y_2010_idx][valid_mask]) - np.mean(mean_ann_g[valid_mask]))
    ano_g_2010_pct = float(ano_g_2010 / np.mean(mean_ann_g[valid_mask]) * 100.0)

    ano_g_2015 = float(np.mean(annual_g[y_2015_idx][valid_mask]) - np.mean(mean_ann_g[valid_mask]))
    ano_g_2015_pct = float(ano_g_2015 / np.mean(mean_ann_g[valid_mask]) * 100.0)

    ano_g_2023 = float(np.mean(annual_g[y_2023_idx][valid_mask]) - np.mean(mean_ann_g[valid_mask]))
    ano_g_2023_pct = float(ano_g_2023 / np.mean(mean_ann_g[valid_mask]) * 100.0)

    # CHIRPS Anomalies
    ano_c_2010 = float(np.mean(annual_c[y_2010_idx][valid_mask]) - np.mean(mean_ann_c[valid_mask]))
    ano_c_2010_pct = float(ano_c_2010 / np.mean(mean_ann_c[valid_mask]) * 100.0)

    ano_c_2015 = float(np.mean(annual_c[y_2015_idx][valid_mask]) - np.mean(mean_ann_c[valid_mask]))
    ano_c_2015_pct = float(ano_c_2015 / np.mean(mean_ann_c[valid_mask]) * 100.0)

    ano_c_2023 = float(np.mean(annual_c[y_2023_idx][valid_mask]) - np.mean(mean_ann_c[valid_mask]))
    ano_c_2023_pct = float(ano_c_2023 / np.mean(mean_ann_c[valid_mask]) * 100.0)

    enso_records = [
        {'Event': 'Super La Niña 2010', 'GSMaP_Total_mm': round(float(np.mean(annual_g[y_2010_idx][valid_mask])), 1), 'GSMaP_Ano_mm': round(ano_g_2010, 1), 'GSMaP_Ano_Pct': round(ano_g_2010_pct, 1), 'CHIRPS_Total_mm': round(float(np.mean(annual_c[y_2010_idx][valid_mask])), 1), 'CHIRPS_Ano_mm': round(ano_c_2010, 1), 'CHIRPS_Ano_Pct': round(ano_c_2010_pct, 1)},
        {'Event': 'Super El Niño 2015', 'GSMaP_Total_mm': round(float(np.mean(annual_g[y_2015_idx][valid_mask])), 1), 'GSMaP_Ano_mm': round(ano_g_2015, 1), 'GSMaP_Ano_Pct': round(ano_g_2015_pct, 1), 'CHIRPS_Total_mm': round(float(np.mean(annual_c[y_2015_idx][valid_mask])), 1), 'CHIRPS_Ano_mm': round(ano_c_2015, 1), 'CHIRPS_Ano_Pct': round(ano_c_2015_pct, 1)},
        {'Event': 'Strong El Niño 2023', 'GSMaP_Total_mm': round(float(np.mean(annual_g[y_2023_idx][valid_mask])), 1), 'GSMaP_Ano_mm': round(ano_g_2023, 1), 'GSMaP_Ano_Pct': round(ano_g_2023_pct, 1), 'CHIRPS_Total_mm': round(float(np.mean(annual_c[y_2023_idx][valid_mask])), 1), 'CHIRPS_Ano_mm': round(ano_c_2023, 1), 'CHIRPS_Ano_Pct': round(ano_c_2023_pct, 1)}
    ]
    df_enso_comp = pd.DataFrame(enso_records)
    print(df_enso_comp.to_string(index=False))
    df_enso_comp.to_csv(out_dir / "extreme_enso_comparative_anomalies.csv", index=False)

    # 2. Statistik Zonal 26 Kecamatan (GSMaP vs CHIRPS vs Fused)
    print("\n[Langkah 2/2] Mengekstraksi Statistik Zonal 26 Kecamatan...")
    gdf_kec = gpd.read_file("33.05_kecamatan.geojson")
    gdf_kec['geometry'] = gdf_kec['geometry'].apply(shapely.force_2d)

    zonal_list = []
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

        g_mean_k = float(np.mean(mean_ann_g[kec_mask]))
        c_mean_k = float(np.mean(mean_ann_c[kec_mask]))
        f_mean_k = float(np.mean(mean_ann_f[kec_mask]))
        diff_k = g_mean_k - c_mean_k
        diff_pct_k = (diff_k / c_mean_k) * 100.0

        # Kontras Ekstrem La Nina vs El Nino pada GSMaP vs CHIRPS
        contrast_g = float(np.mean(annual_g[y_2010_idx][kec_mask]) - np.mean(annual_g[y_2015_idx][kec_mask]))
        contrast_c = float(np.mean(annual_c[y_2010_idx][kec_mask]) - np.mean(annual_c[y_2015_idx][kec_mask]))

        zonal_list.append({
            'Kecamatan': kec_name,
            'GSMaP_Mean_mm': round(g_mean_k, 1),
            'CHIRPS_Mean_mm': round(c_mean_k, 1),
            'Fused_Mean_mm': round(f_mean_k, 1),
            'Diff_GSMaP_CHIRPS_mm': round(diff_k, 1),
            'Diff_Pct': round(diff_pct_k, 1),
            'Contrast_GSMaP_mm': round(contrast_g, 1),
            'Contrast_CHIRPS_mm': round(contrast_c, 1)
        })

    df_zonal_comp = pd.DataFrame(zonal_list).sort_values(by='Fused_Mean_mm', ascending=False).reset_index(drop=True)
    print("\n" + "=" * 95)
    print("📋 TABEL STATISTIK ZONAL KOMPARASI 26 KECAMATAN (GSMaP vs CHIRPS vs FUSED):")
    print("=" * 95)
    print(df_zonal_comp.head(10).to_string(index=False))
    print("... (dan 16 kecamatan lainnya)")

    df_zonal_comp.to_csv(out_dir / "zonal_stats_gsmap_vs_chirps_26kec.csv", index=False)
    print(f"\n✓ Tabel statistik zonal tersimpan di: {out_dir / 'zonal_stats_gsmap_vs_chirps_26kec.csv'}")

    print("\n" + "=" * 85)
    print(f"🎉 FASE 4 SELESAI DALAM {time.time() - t0:.2f} DETIK!")
    print("=" * 85)

if __name__ == "__main__":
    run()
