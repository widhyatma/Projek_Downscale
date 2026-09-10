"""
GSMaP vs CHIRPS Comparative Downscaling - Phase 3: 25-Year Reconstruction & Spatial Difference
=============================================================================================
Reconstructs 300 months of high-resolution (250m) precipitation for:
1. Downscaled GSMaP (Standalone)
2. Downscaled CHIRPS (Standalone)
3. Downscaled Fused Multi-Sensor
Computes:
- Mean annual precipitation maps (2001 - 2025)
- Spatial Difference Maps (GSMaP - CHIRPS in mm/year and %)
- North-South Orographic Transect Profile (Mountain to Coast)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

def run():
    t0 = time.time()
    print("=" * 85)
    print("🚀 FASE 3: REKONSTRUKSI 25 TAHUN (2001-2025) & ANALISIS SELISIH SPASIAL GSMaP vs CHIRPS")
    print("   Resolusi: 250m | Target: GSMaP vs CHIRPS vs Fused | Domain: 21.704 Sel Daratan Kebumen")
    print("=" * 85)

    out_dir = Path("data/processed_gsmap_vs_chirps")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Muat kovariat dan model
    df_ts = pd.read_csv(out_dir / "comparative_timeseries_25yr.csv")
    npz_terrain = np.load("data/processed_25yr/terrain_covariates_250m.npz")
    elev = npz_terrain['elev']
    slope = npz_terrain['slope']
    aspect = npz_terrain['aspect']
    coast = npz_terrain['coast']
    mask_keb = npz_terrain['mask']
    xs = npz_terrain['xs']
    ys = npz_terrain['ys']
    h, w = elev.shape

    sin_aspect = np.sin(np.radians(aspect))
    cos_aspect = np.cos(np.radians(aspect))

    with open(out_dir / "trained_gsmap_chirps_models.pkl", "rb") as f:
        models = pickle.load(f)
    xgb_g = models['xgb_gsmap']
    xgb_c = models['xgb_chirps']
    xgb_f = models['xgb_fused']
    topo_cols = models['topo_cols']
    fused_cols = models['fused_cols']

    # 2. Muat NDVI spasial
    print("\n[Memuat 300 Band NDVI Spasial 250m...]")
    src_ndvi = rasterio.open("data/NDVI_Spasial_Bulanan_Kebumen.tif")
    ndvi_dict = {}
    for b_idx in range(1, src_ndvi.count + 1):
        desc = src_ndvi.descriptions[b_idx - 1]
        if desc and 'NDVI_' in desc:
            parts = desc.split('_')
            ndvi_dict[f"{parts[1]}_{parts[2]}"] = src_ndvi.read(b_idx, out_shape=(h, w), resampling=Resampling.bilinear)
    src_ndvi.close()

    idx_y, idx_x = np.where(mask_keb)
    n_pix = len(idx_y)
    elev_sub = elev[idx_y, idx_x]
    slope_sub = slope[idx_y, idx_x]
    sin_sub = sin_aspect[idx_y, idx_x]
    cos_sub = cos_aspect[idx_y, idx_x]
    coast_sub = coast[idx_y, idx_x]

    years = list(range(2001, 2026))
    annual_g = np.zeros((len(years), h, w), dtype=np.float32)
    annual_c = np.zeros((len(years), h, w), dtype=np.float32)
    annual_f = np.zeros((len(years), h, w), dtype=np.float32)

    monthly_g = np.zeros((12, h, w), dtype=np.float32)
    monthly_c = np.zeros((12, h, w), dtype=np.float32)
    monthly_f = np.zeros((12, h, w), dtype=np.float32)
    m_counts = np.zeros(12, dtype=np.int32)

    print("\n[Melakukan Inferensi Spasial 250m untuk 300 Bulan pada 3 Trek Model...]")
    for y_idx, yr in enumerate(years):
        yr_g = np.zeros((h, w), dtype=np.float32)
        yr_c = np.zeros((h, w), dtype=np.float32)
        yr_f = np.zeros((h, w), dtype=np.float32)

        for m in range(1, 13):
            m_idx = m - 1
            ym = f"{yr}_{m:02d}"
            row = df_ts[df_ts['year_month'] == ym].iloc[0]

            c_raw = float(row['chirps_raw_mm'])
            g_raw = float(row['gsmap_raw_mm'])
            t2m = float(row['era5_t2m_c'])
            rh = float(row['era5_rh_pct'])
            ws = float(row['era5_ws_ms'])
            sp = float(row['era5_sp_hpa'])

            ndvi_arr = ndvi_dict.get(ym, np.full((h, w), 0.65, dtype=np.float32))[idx_y, idx_x]
            ndvi_arr = np.nan_to_num(ndvi_arr, nan=0.65)

            # Trek GSMaP
            X_g = pd.DataFrame({
                'gsmap_raw': np.full(n_pix, g_raw, dtype=np.float32),
                'elev': elev_sub, 'slope': slope_sub, 'sin_aspect': sin_sub, 'cos_aspect': cos_sub, 'coast': coast_sub
            })[['gsmap_raw'] + topo_cols]
            pred_g = np.clip(xgb_g.predict(X_g), 0.0, None)
            if np.mean(pred_g) > 0:
                pred_g *= (g_raw / np.mean(pred_g))  # Mass conservation

            # Trek CHIRPS
            X_c = pd.DataFrame({
                'chirps_raw': np.full(n_pix, c_raw, dtype=np.float32),
                'elev': elev_sub, 'slope': slope_sub, 'sin_aspect': sin_sub, 'cos_aspect': cos_sub, 'coast': coast_sub
            })[['chirps_raw'] + topo_cols]
            pred_c = np.clip(xgb_c.predict(X_c), 0.0, None)
            if np.mean(pred_c) > 0:
                pred_c *= (c_raw / np.mean(pred_c))

            # Trek Fused
            X_f = pd.DataFrame({
                'chirps_raw': np.full(n_pix, c_raw, dtype=np.float32),
                'gsmap_raw': np.full(n_pix, g_raw, dtype=np.float32),
                'elev': elev_sub, 'slope': slope_sub, 'sin_aspect': sin_sub, 'cos_aspect': cos_sub, 'coast': coast_sub,
                't2m': np.full(n_pix, t2m, dtype=np.float32),
                'rh': np.full(n_pix, rh, dtype=np.float32),
                'ws': np.full(n_pix, ws, dtype=np.float32),
                'sp': np.full(n_pix, sp, dtype=np.float32),
                'ndvi': ndvi_arr
            })[fused_cols]
            pred_f = np.clip(xgb_f.predict(X_f), 0.0, None)
            targ_f = 0.55 * c_raw + 0.45 * g_raw
            if np.mean(pred_f) > 0:
                pred_f *= (targ_f / np.mean(pred_f))

            # Masukkan ke grid
            g_grid = np.zeros((h, w), dtype=np.float32)
            c_grid = np.zeros((h, w), dtype=np.float32)
            f_grid = np.zeros((h, w), dtype=np.float32)

            g_grid[idx_y, idx_x] = pred_g
            c_grid[idx_y, idx_x] = pred_c
            f_grid[idx_y, idx_x] = pred_f

            yr_g += g_grid
            yr_c += c_grid
            yr_f += f_grid

            monthly_g[m_idx] += g_grid
            monthly_c[m_idx] += c_grid
            monthly_f[m_idx] += f_grid
            m_counts[m_idx] += 1

        annual_g[y_idx] = yr_g
        annual_c[y_idx] = yr_c
        annual_f[y_idx] = yr_f

    # Rata-rata 25 tahun
    mean_ann_g = np.mean(annual_g, axis=0)
    mean_ann_c = np.mean(annual_c, axis=0)
    mean_ann_f = np.mean(annual_f, axis=0)

    for m_idx in range(12):
        if m_counts[m_idx] > 0:
            monthly_g[m_idx] /= m_counts[m_idx]
            monthly_c[m_idx] /= m_counts[m_idx]
            monthly_f[m_idx] /= m_counts[m_idx]

    # Peta Selisih Spasial (GSMaP - CHIRPS)
    diff_ann_mm = np.where(mask_keb, mean_ann_g - mean_ann_c, 0.0)
    diff_ann_pct = np.where(mask_keb & (mean_ann_c > 0), (diff_ann_mm / mean_ann_c) * 100.0, 0.0)

    print(f"\n✓ Rata-rata Tahunan GSMaP Downscaled : {np.mean(mean_ann_g[idx_y, idx_x]):.1f} mm/tahun")
    print(f"✓ Rata-rata Tahunan CHIRPS Downscaled: {np.mean(mean_ann_c[idx_y, idx_x]):.1f} mm/tahun")
    print(f"✓ Rata-rata Tahunan Fused Multi-Sensor: {np.mean(mean_ann_f[idx_y, idx_x]):.1f} mm/tahun")
    print(f"✓ Selisih Rata-rata (GSMaP - CHIRPS) : {np.mean(diff_ann_mm[idx_y, idx_x]):+.1f} mm/tahun ({np.mean(diff_ann_pct[idx_y, idx_x]):+.1f}%)")

    # 3. Profil Transekt Orografis Utara-Selatan
    # Garis transekt bujur tengah (sekitar lon 109.65°E dari lat -7.50°S ke -7.80°S)
    mid_col = int(w * 0.55)
    transect_rows = []
    for r in range(h):
        if mask_keb[r, mid_col]:
            transect_rows.append({
                'latitude': float(ys[r, mid_col]),
                'longitude': float(xs[r, mid_col]),
                'elevation_m': float(elev[r, mid_col]),
                'gsmap_mm': float(mean_ann_g[r, mid_col]),
                'chirps_mm': float(mean_ann_c[r, mid_col]),
                'fused_mm': float(mean_ann_f[r, mid_col]),
                'diff_mm': float(diff_ann_mm[r, mid_col])
            })
    df_transect = pd.DataFrame(transect_rows).sort_values(by='latitude', ascending=False).reset_index(drop=True)
    df_transect.to_csv(out_dir / "orographic_transect_profile.csv", index=False)
    print(f"✓ Profil transekt orografis Utara-Selatan diekstrak ({len(df_transect)} titik).")

    # Simpan hasil rekonstruksi
    np.savez_compressed(
        out_dir / "reconstructed_gsmap_vs_chirps_250m.npz",
        mean_ann_g=mean_ann_g,
        mean_ann_c=mean_ann_c,
        mean_ann_f=mean_ann_f,
        diff_ann_mm=diff_ann_mm,
        diff_ann_pct=diff_ann_pct,
        annual_g=annual_g,
        annual_c=annual_c,
        annual_f=annual_f,
        monthly_g=monthly_g,
        monthly_c=monthly_c,
        monthly_f=monthly_f,
        mask_kebumen=mask_keb,
        years=years
    )
    print(f"✓ File rekonstruksi tersimpan di: {out_dir / 'reconstructed_gsmap_vs_chirps_250m.npz'}")

    print("\n" + "=" * 85)
    print(f"🎉 FASE 3 SELESAI DALAM {time.time() - t0:.2f} DETIK!")
    print("=" * 85)

if __name__ == "__main__":
    run()
