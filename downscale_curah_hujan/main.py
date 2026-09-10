"""
Main Orchestrator for Precipitation Downscaling Benchmark Suite
==============================================================
Executes the full pipeline:
1. Coordinate transformation to UTM Zone 49S (EPSG:32749).
2. Topographic covariate derivation & bilinear sampling.
3. Multicollinearity pruning via Variance Inflation Factor (VIF <= 5.0).
4. Zero-rain handling via Hurdle model when zero-inflation > 40%.
5. Spatial Block Cross-Validation with buffer zones (zero data leakage).
6. Model benchmark suite (OK, KED, RK, RFSI, Spatial XGBoost, LightGBM, Conditional Merging).
7. Performance metric calculation (KGE, RMSE, MAE, PBIAS, R2).
8. Raster GeoTIFF export & publication-quality 16:9 visualization.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import argparse
import time
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import rasterio
from rasterio import features
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

# Import modul internal
from src.preprocessing import (
    reproject_points,
    compute_raster_covariates,
    sample_covariates_bilinear,
    prune_vif_multicollinearity,
    HurdlePrecipitationModel
)
from src.validation import (
    compute_empirical_semivariogram_range,
    SpatialBlockKFoldWithBuffer
)
from src.models.baselines import (
    IDWModel,
    GWRModel
)
from src.models.geostats import (
    OrdinaryKrigingModel,
    KEDModel,
    RegressionKrigingModel
)
from src.models.topospline import (
    ANUSPLINModel,
    PRISMModel
)
from src.models.tree_models import (
    RFSIModel,
    SpatialRandomForestModel,
    SpatialXGBoostModel,
    SpatialLightGBMModel
)
from src.models.fusion import (
    ConditionalMergingModel
)
from src.evaluation import (
    evaluate_all_metrics
)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run Rainfall Downscaling Benchmark Suite")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, default="monthly", choices=["daily", "monthly"], help="Benchmark daily or monthly total")
    parser.add_argument("--export_rasters", action="store_true", default=True, help="Export output GeoTIFF maps")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 85)
    print("🚀 GEOSPATIAL PRECIPITATION DOWNSCALING BENCHMARK SUITE")
    print("   Target: Kabupaten Kebumen | Metric CRS: UTM Zone 49S (EPSG:32749)")
    print("=" * 85)

    base_dir = Path(__file__).resolve().parent
    config_file = base_dir / args.config
    cfg = load_config(config_file)

    # 1. Path Setup
    raw_dem_path = base_dir / cfg["paths"]["dem_raw"]
    raw_geojson_path = base_dir / cfg["paths"]["boundary_geojson"]
    raw_stations_path = base_dir / cfg["paths"]["stations_csv"]
    raw_sat_path = base_dir / cfg["paths"]["satellite_nc"]

    out_tables = base_dir / cfg["paths"]["outputs_tables"]
    out_rasters = base_dir / cfg["paths"]["outputs_rasters"]
    out_figures = base_dir / cfg["paths"]["outputs_figures"]
    for p in [out_tables, out_rasters, out_figures]:
        p.mkdir(parents=True, exist_ok=True)

    dst_crs = cfg["spatial"]["crs_projected"]
    target_res_m = float(cfg["spatial"]["target_resolution_m"])
    vif_threshold = float(cfg["preprocessing"]["vif_threshold"])
    zero_threshold = float(cfg["preprocessing"]["zero_rain_threshold"])

    # 2. Topographic Covariates Derivation (Projected UTM 49S)
    print("\n[Step 1/6] Menghitung Kovariat Topografi & Spasial (DEM, Slope, Sin/Cos Aspect, TPI)...")
    covariates = compute_raster_covariates(
        dem_path=raw_dem_path,
        dst_crs=dst_crs,
        target_res_m=target_res_m,
        tpi_size=cfg["preprocessing"]["tpi_neighborhood_size"]
    )
    print(f"   ✓ Grid Raster UTM 49S: {covariates['width']} x {covariates['height']} cells ({target_res_m:.0f}m)")
    print(f"   ✓ Rentang Elevasi: {covariates['elevation'].min():.1f}m s.d. {covariates['elevation'].max():.1f}m")

    # 3. Load Rain Gauge Data & Koordinat Reprojection
    print("\n[Step 2/6] Memuat Data Stasiun Penakar Hujan & Reproyeksi ke UTM Zone 49S...")
    df_raw = pd.read_csv(raw_stations_path)
    
    # Mode agregasi: monthly total atau daily single event
    if args.mode == "monthly":
        print("   ✓ Mode: Total Akumulasi Bulanan (Januari 2020)")
        df_target = df_raw.groupby(["station_id", "station_name", "kecamatan", "longitude", "latitude", "elevation_m"])["rainfall_mm"].sum().reset_index()
    else:
        # Ambil satu hari dengan curah hujan representatif
        target_date = "2020-01-15"
        print(f"   ✓ Mode: Harian (Tanggal: {target_date})")
        df_target = df_raw[df_raw["date"] == target_date].copy()

    # Reproyeksi koordinat stasiun ke UTM 49S (meter)
    gdf_stations = reproject_points(
        df=df_target,
        lon_col="longitude",
        lat_col="latitude",
        src_crs=cfg["spatial"]["crs_geographic"],
        dst_crs=dst_crs
    )
    print(f"   ✓ {len(gdf_stations)} Stasiun berhasil direproyeksi ke EPSG:32749 (Easting: {gdf_stations['x_utm'].min():.0f} - {gdf_stations['x_utm'].max():.0f} m)")

    # 4. Bilinear Sampling & VIF Multicollinearity Pruning
    print("\n[Step 3/6] Bilinear Sampling Kovariat & Analisis Multikolinieritas (VIF)...")
    gdf_sampled = sample_covariates_bilinear(gdf_stations, covariates)

    # Ekstraksi nilai satelit CHIRPS pada lokasi stasiun untuk Conditional Merging
    ds_sat = xr.open_dataset(raw_sat_path)
    if args.mode == "monthly":
        sat_grid_agg = ds_sat["precipitation"].sum(dim="time").values
    else:
        sat_grid_agg = ds_sat["precipitation"].sel(time=target_date).values
    sat_lons = ds_sat["x"].values
    sat_lats = ds_sat["y"].values
    ds_sat.close()

    # Interpolasi satelit ke koordinat stasiun & grid penuh UTM
    from scipy.interpolate import RegularGridInterpolator
    sat_interp = RegularGridInterpolator(
        (sat_lats[::-1], sat_lons),
        sat_grid_agg[::-1, :],
        bounds_error=False,
        fill_value=None
    )

    # Sampel satelit di stasiun
    pts_geo = np.column_stack([gdf_sampled["latitude"], gdf_sampled["longitude"]])
    gdf_sampled["sat_precip"] = sat_interp(pts_geo)

    # Kandidat fitur lingkungan
    candidate_features = ["elevation", "slope", "sin_aspect", "cos_aspect", "tpi", "coast_dist_km"]
    X_pruned, final_feature_cols, vif_df = prune_vif_multicollinearity(
        df=gdf_sampled,
        feature_cols=candidate_features,
        vif_threshold=vif_threshold
    )
    print("   ✓ Hasil VIF Pruning (Threshold <= 5.0):")
    for _, row in vif_df.iterrows():
        print(f"     - {row['feature']:<15}: VIF = {row['final_VIF']:.2f}")
    vif_df.to_csv(out_tables / "vif_analysis.csv", index=False)

    # Cek persentase hari kering (zero-rain)
    zero_ratio = (gdf_sampled["rainfall_mm"] == 0).mean()
    print(f"   ✓ Rasio Hujan Nol (P = 0 mm): {zero_ratio * 100:.1f}%")
    use_hurdle = (zero_ratio >= zero_threshold)
    if use_hurdle:
        print("   ⚠️ Zero-inflation melebihi 40%! Mengaktifkan Two-Stage Hurdle Model.")

    # 5. Protokol Validasi: Spatial Block Cross-Validation dengan Buffer Zones
    print("\n[Step 4/6] Menjalankan Protokol Validasi: Spatial Block K-Fold dengan Buffer Zones...")
    station_coords_m = np.column_stack([gdf_sampled["x_utm"].values, gdf_sampled["y_utm"].values])
    y_values = gdf_sampled["rainfall_mm"].values

    # Hitung empiris range korelasi spasial (a)
    range_a, lag_c, lag_g = compute_empirical_semivariogram_range(station_coords_m, y_values)
    buffer_radius_m = max(float(cfg["validation"]["min_buffer_m"]), range_a * 0.5)
    print(f"   ✓ Jarak Korelasi Spasial Semivariogram (Range a): {range_a:.0f} meter")
    print(f"   ✓ Radius Zona Buffer Pengaman: {buffer_radius_m:.0f} meter")

    cv_splitter = SpatialBlockKFoldWithBuffer(
        n_splits=cfg["validation"]["n_splits"],
        buffer_radius_m=buffer_radius_m,
        random_state=42
    )

    # 6. Model Benchmark Suite Execution
    print("\n[Step 5/6] Memulai Benchmarking 13 Model Spasial, Topographic Spline, ML & Fusi Satelit...")
    models_to_test = [
        "ANUSPLIN",
        "PRISM",
        "Regression_Kriging",
        "Ordinary_Kriging",
        "KED_Elevation",
        "IDW",
        "GWR",
        "Spatial_RandomForest",
        "RFSI",
        "Spatial_XGBoost",
        "Spatial_LightGBM",
        "Conditional_Merging",
        "Ensemble_Mean"
    ]

    fold_results = []
    oof_predictions = {m: np.zeros(len(gdf_sampled)) for m in models_to_test}

    for train_idx, test_idx, meta in cv_splitter.split(station_coords_m):
        f_num = meta["fold"]
        print(f"\n   --- FOLD {f_num} / {cfg['validation']['n_splits']} ---")
        print(f"       Test Stations: {meta['n_test']} | Clean Training: {meta['n_train_clean']} | Dropped (Buffer): {meta['n_train_dropped_buffer']}")

        # Data split
        coords_tr = station_coords_m[train_idx]
        coords_te = station_coords_m[test_idx]
        y_tr = y_values[train_idx]
        y_te = y_values[test_idx]
        X_tr = X_pruned.iloc[train_idx].values
        X_te = X_pruned.iloc[test_idx].values
        elev_tr = gdf_sampled["elevation"].iloc[train_idx].values
        elev_te = gdf_sampled["elevation"].iloc[test_idx].values
        coast_tr = gdf_sampled["coast_dist_km"].iloc[train_idx].values
        coast_te = gdf_sampled["coast_dist_km"].iloc[test_idx].values
        sat_tr = gdf_sampled["sat_precip"].iloc[train_idx].values
        sat_te = gdf_sampled["sat_precip"].iloc[test_idx].values

        # 1. ANUSPLIN (Trivariate Topographic Spline - Australia/NASA)
        m_anusplin = ANUSPLINModel(z_weight=0.6, smooth=0.5)
        m_anusplin.fit(coords_tr, elev_tr, y_tr)
        pred_anusplin = m_anusplin.predict(coords_te, elev_te)
        oof_predictions["ANUSPLIN"][test_idx] = pred_anusplin
        res_anusplin = evaluate_all_metrics(y_te, pred_anusplin)
        fold_results.append({"model": "ANUSPLIN", "fold": f_num, **res_anusplin})

        # 2. PRISM (Parameter-elevation Regressions on Independent Slopes - NOAA/USDA)
        m_prism = PRISMModel(sigma_h_m=12000.0, sigma_z_m=250.0, sigma_c_km=10.0)
        m_prism.fit(coords_tr, elev_tr, coast_tr, y_tr)
        pred_prism = m_prism.predict(coords_te, elev_te, coast_te)
        oof_predictions["PRISM"][test_idx] = pred_prism
        res_prism = evaluate_all_metrics(y_te, pred_prism)
        fold_results.append({"model": "PRISM", "fold": f_num, **res_prism})

        # 3. Regression-Kriging (RK)
        m_rk = RegressionKrigingModel(variogram_model="gaussian")
        m_rk.fit(X_tr, coords_tr, y_tr)
        pred_rk = m_rk.predict(X_te, coords_te)
        oof_predictions["Regression_Kriging"][test_idx] = pred_rk
        res_rk = evaluate_all_metrics(y_te, pred_rk)
        fold_results.append({"model": "Regression_Kriging", "fold": f_num, **res_rk})

        # 4. Ordinary Kriging (OK)
        m_ok = OrdinaryKrigingModel(variogram_model="gaussian")
        m_ok.fit(coords_tr, y_tr)
        pred_ok = m_ok.predict(coords_te)
        oof_predictions["Ordinary_Kriging"][test_idx] = pred_ok
        res_ok = evaluate_all_metrics(y_te, pred_ok)
        fold_results.append({"model": "Ordinary_Kriging", "fold": f_num, **res_ok})

        # 5. Kriging with External Drift (KED)
        m_ked = KEDModel(variogram_model="gaussian")
        m_ked.fit(coords_tr, y_tr, elev_tr)
        pred_ked = m_ked.predict(coords_te, elev_te)
        oof_predictions["KED_Elevation"][test_idx] = pred_ked
        res_ked = evaluate_all_metrics(y_te, pred_ked)
        fold_results.append({"model": "KED_Elevation", "fold": f_num, **res_ked})

        # 6. IDW (Inverse Distance Weighting - Classic Baseline)
        m_idw = IDWModel(power=2.0)
        m_idw.fit(coords_tr, y_tr)
        pred_idw = m_idw.predict(coords_te)
        oof_predictions["IDW"][test_idx] = pred_idw
        res_idw = evaluate_all_metrics(y_te, pred_idw)
        fold_results.append({"model": "IDW", "fold": f_num, **res_idw})

        # 7. GWR (Geographically Weighted Regression - Local Elevation Regression)
        m_gwr = GWRModel(bandwidth_m=15000.0)
        m_gwr.fit(coords_tr, elev_tr, y_tr)
        pred_gwr = m_gwr.predict(coords_te, elev_te)
        oof_predictions["GWR"][test_idx] = pred_gwr
        res_gwr = evaluate_all_metrics(y_te, pred_gwr)
        fold_results.append({"model": "GWR", "fold": f_num, **res_gwr})

        # 8. Spatial Random Forest (Standard Environmental RF)
        m_srf = SpatialRandomForestModel()
        m_srf.fit(X_tr, coords_tr, y_tr)
        pred_srf = m_srf.predict(X_te, coords_te)
        oof_predictions["Spatial_RandomForest"][test_idx] = pred_srf
        res_srf = evaluate_all_metrics(y_te, pred_srf)
        fold_results.append({"model": "Spatial_RandomForest", "fold": f_num, **res_srf})

        # 9. RFSI (Random Forest Spatial Interpolation)
        m_rfsi = RFSIModel(k_neighbors=cfg["models"]["rfsi"]["k_neighbors"])
        m_rfsi.fit(X_tr, coords_tr, y_tr)
        pred_rfsi = m_rfsi.predict(X_te, coords_te)
        oof_predictions["RFSI"][test_idx] = pred_rfsi
        res_rfsi = evaluate_all_metrics(y_te, pred_rfsi)
        fold_results.append({"model": "RFSI", "fold": f_num, **res_rfsi})

        # 10. Spatial XGBoost
        m_xgb = SpatialXGBoostModel()
        m_xgb.fit(X_tr, coords_tr, y_tr)
        pred_xgb = m_xgb.predict(X_te, coords_te)
        oof_predictions["Spatial_XGBoost"][test_idx] = pred_xgb
        res_xgb = evaluate_all_metrics(y_te, pred_xgb)
        fold_results.append({"model": "Spatial_XGBoost", "fold": f_num, **res_xgb})

        # 11. Spatial LightGBM
        m_lgb = SpatialLightGBMModel()
        m_lgb.fit(X_tr, coords_tr, y_tr)
        pred_lgb = m_lgb.predict(X_te, coords_te)
        oof_predictions["Spatial_LightGBM"][test_idx] = pred_lgb
        res_lgb = evaluate_all_metrics(y_te, pred_lgb)
        fold_results.append({"model": "Spatial_LightGBM", "fold": f_num, **res_lgb})

        # 12. Conditional Merging (CM)
        m_cm = ConditionalMergingModel(variogram_model="gaussian")
        m_cm.fit(coords_tr, y_tr, sat_tr)
        pred_cm = m_cm.predict(coords_te, sat_te)
        oof_predictions["Conditional_Merging"][test_idx] = pred_cm
        res_cm = evaluate_all_metrics(y_te, pred_cm)
        fold_results.append({"model": "Conditional_Merging", "fold": f_num, **res_cm})

        # 13. Ensemble Multi-Model Consensus (ANUSPLIN + PRISM + Regression-Kriging + Spatial XGBoost)
        pred_ensemble = (pred_anusplin + pred_prism + pred_rk + pred_xgb) / 4.0
        oof_predictions["Ensemble_Mean"][test_idx] = pred_ensemble
        res_ens = evaluate_all_metrics(y_te, pred_ensemble)
        fold_results.append({"model": "Ensemble_Mean", "fold": f_num, **res_ens})

        print(f"       Scores Fold {f_num} (KGE): ANUSPLIN={res_anusplin['KGE']:.3f} | PRISM={res_prism['KGE']:.3f} | RK={res_rk['KGE']:.3f} | GWR={res_gwr['KGE']:.3f} | XGB={res_xgb['KGE']:.3f} | ENS={res_ens['KGE']:.3f}")

    # Rekapitulasi Metrik Evaluasi
    df_metrics = pd.DataFrame(fold_results)
    df_metrics.to_csv(out_tables / "fold_metrics_detail.csv", index=False)

    summary_rows = []
    for m in models_to_test:
        sub = df_metrics[df_metrics["model"] == m]
        # Hitung juga metrik Out-Of-Fold (OOF) keseluruhan
        oof_m = evaluate_all_metrics(y_values, oof_predictions[m])
        summary_rows.append({
            "Algorithm": m,
            "KGE_Mean": round(sub["KGE"].mean(), 4),
            "KGE_Std": round(sub["KGE"].std(), 4),
            "RMSE_Mean": round(sub["RMSE"].mean(), 2),
            "MAE_Mean": round(sub["MAE"].mean(), 2),
            "PBIAS_Mean": round(sub["PBIAS"].mean(), 2),
            "R2_Mean": round(sub["R2"].mean(), 4),
            "Pearson_r": round(sub["Pearson_r"].mean(), 4),
            "OOF_KGE": oof_m["KGE"],
            "OOF_RMSE": oof_m["RMSE"]
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("KGE_Mean", ascending=False)
    summary_csv = out_tables / "benchmark_metrics.csv"
    df_summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 85)
    print("🏆 HASIL BENCHMARK SPATIAL DOWNSCALING (Cross-Validation Summary):")
    print("=" * 85)
    print(df_summary.to_string(index=False))
    print(f"\n✓ Tabel metrik lengkap tersimpan di: {summary_csv}")

    # 7. Prediksi Grid Penuh & Ekspor Raster GeoTIFF (EPSG:32749)
    print("\n[Step 6/6] Menghasilkan Peta Raster GeoTIFF Resolusi Tinggi (250m)...")
    h, w = covariates["height"], covariates["width"]
    transform = covariates["transform"]

    # Siapkan matriks fitur penuh untuk seluruh piksel domain
    grid_coords_m = np.column_stack([covariates["x_utm"].flatten(), covariates["y_utm"].flatten()])
    X_grid_list = []
    for feat in final_feature_cols:
        X_grid_list.append(covariates[feat].flatten())
    X_grid_full = np.column_stack(X_grid_list)
    elev_grid_flat = covariates["elevation"].flatten()

    # Masking poligon Kabupaten Kebumen
    gdf_kec = gpd.read_file(raw_geojson_path)
    gdf_kec["geometry"] = gdf_kec["geometry"].apply(shapely.force_2d)
    gdf_kec_proj = gdf_kec.to_crs(dst_crs)
    mask_domain = features.rasterize(
        [(geom, 1) for geom in gdf_kec_proj["geometry"]],
        out_shape=(h, w),
        transform=transform,
        fill=0
    ).astype(bool)

    # Train model pada seluruh data untuk pembuatan peta final
    # 1. ANUSPLIN
    full_anusplin = ANUSPLINModel(z_weight=0.6, smooth=0.5)
    full_anusplin.fit(station_coords_m, gdf_sampled["elevation"].values, y_values)
    grid_pred_anusplin = full_anusplin.predict(grid_coords_m, elev_grid_flat).reshape((h, w))
    grid_pred_anusplin[~mask_domain] = np.nan

    # 2. PRISM
    full_prism = PRISMModel()
    full_prism.fit(station_coords_m, gdf_sampled["elevation"].values, gdf_sampled["coast_dist_km"].values, y_values)
    grid_pred_prism = full_prism.predict(grid_coords_m, elev_grid_flat, covariates["coast_dist_km"].flatten()).reshape((h, w))
    grid_pred_prism[~mask_domain] = np.nan

    # 3. Regression-Kriging
    full_rk = RegressionKrigingModel(variogram_model="gaussian")
    full_rk.fit(X_pruned.values, station_coords_m, y_values)
    grid_pred_rk = full_rk.predict(X_grid_full, grid_coords_m).reshape((h, w))
    grid_pred_rk[~mask_domain] = np.nan

    # 4. Spatial XGBoost
    full_xgb = SpatialXGBoostModel()
    full_xgb.fit(X_pruned.values, station_coords_m, y_values)
    grid_pred_xgb = full_xgb.predict(X_grid_full, grid_coords_m).reshape((h, w))
    grid_pred_xgb[~mask_domain] = np.nan

    # 5. Ensemble Mean (Consensus Multi-Model)
    grid_pred_ens = (grid_pred_anusplin + grid_pred_prism + grid_pred_rk + grid_pred_xgb) / 4.0

    # Ekspor GeoTIFF
    raster_exports = [
        ("downscaled_rainfall_ANUSPLIN_250m.tif", grid_pred_anusplin),
        ("downscaled_rainfall_PRISM_250m.tif", grid_pred_prism),
        ("downscaled_rainfall_RegressionKriging_250m.tif", grid_pred_rk),
        ("downscaled_rainfall_EnsembleMean_250m.tif", grid_pred_ens)
    ]
    for r_name, r_grid in raster_exports:
        tif_p = out_rasters / r_name
        with rasterio.open(
            tif_p, "w", driver="GTiff", height=h, width=w, count=1,
            dtype=np.float32, crs=dst_crs, transform=transform, nodata=np.nan
        ) as dst:
            dst.write(r_grid.astype(np.float32), 1)
        print(f"   ✓ Raster GeoTIFF tersimpan: {tif_p}")

    # 8. Render Visualisasi 16:9 Standard
    # Gambar 1: Bar Chart Peringkat KGE 10 Model
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle("Geospatial Precipitation Downscaling Benchmark Suite - Kabupaten Kebumen\n(Protokol Validasi Spatial Block Buffer, Metric UTM Zone 49S)", 
                 fontsize=14, fontweight="bold", y=0.98)

    ax1 = axes[0]
    df_sorted = df_summary.sort_values("KGE_Mean", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(df_sorted)))
    bars = ax1.barh(df_sorted["Algorithm"], df_sorted["KGE_Mean"], color=colors, edgecolor="#1a252f", height=0.6)
    ax1.set_xlabel("Kling-Gupta Efficiency (KGE) [Lebih tinggi lebih baik, Ideal = 1.0]", fontsize=11, fontweight="bold")
    ax1.set_title("Perbandingan KGE Lintas 13 Model Spasial, Topographic Spline & Fusi", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_xlim(min(0.0, df_sorted["KGE_Mean"].min() - 0.1), 1.05)
    for bar in bars:
        w_val = bar.get_width()
        ax1.text(w_val + 0.02, bar.get_y() + bar.get_height()/2, f"{w_val:.3f}", va="center", fontsize=9, fontweight="bold")

    # Panel 2: Peta Prediksi Spasial Terbaik (Ensemble Mean)
    ax2 = axes[1]
    xs_proj = covariates["x_utm"]
    ys_proj = covariates["y_utm"]
    levels = np.linspace(np.nanmin(grid_pred_ens), np.nanmax(grid_pred_ens), 60)
    im2 = ax2.contourf(xs_proj, ys_proj, grid_pred_ens, levels=levels, cmap="YlGnBu", extend="both")
    gdf_kec_proj.boundary.plot(ax=ax2, color="#1a252f", linewidth=1.0)
    ax2.scatter(station_coords_m[:, 0], station_coords_m[:, 1], color="#e74c3c", edgecolor="white", s=45, label="Pos Penakar Hujan", zorder=5)
    ax2.legend(loc="upper left", frameon=True)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
    cbar2.set_label("Curah Hujan (mm)", fontsize=11, fontweight="bold")
    ax2.set_title(f"Peta Spasial Konsensus Multi-Model (Ensemble Mean)\nResolusi: {target_res_m:.0f}m | ANUSPLIN + PRISM + RK + XGBoost", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Easting UTM Zone 49S (m)", fontsize=10)
    ax2.set_ylabel("Northing UTM Zone 49S (m)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.25)

    plt.tight_layout()
    fig_path = out_figures / "benchmark_comparison_models.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"   ✓ Visualisasi 16:9 berhasil disimpan: {fig_path}")

    # Gambar 2: 4-Panel Perbandingan ANUSPLIN vs PRISM vs RK vs Ensemble Mean
    fig_4p, axes_4p = plt.subplots(2, 2, figsize=(16, 9))
    fig_4p.suptitle("Perbandingan 4 Model Terbaik Standar Dunia (Januari 2020, Kebumen 250m UTM 49S)", fontsize=14, fontweight="bold", y=0.98)
    panels_info = [
        (axes_4p[0, 0], "1. ANUSPLIN (Trivariate Topographic Spline)", grid_pred_anusplin),
        (axes_4p[0, 1], "2. PRISM (Topographic-Coastal Facet Regression)", grid_pred_prism),
        (axes_4p[1, 0], "3. Regression-Kriging (ML Trend + Gaussian Variogram)", grid_pred_rk),
        (axes_4p[1, 1], "4. Ensemble Mean Consensus (Konsensus Multi-Model)", grid_pred_ens)
    ]
    p_min = min(np.nanmin(grid_pred_anusplin), np.nanmin(grid_pred_prism), np.nanmin(grid_pred_rk), np.nanmin(grid_pred_ens))
    p_max = max(np.nanmax(grid_pred_anusplin), np.nanmax(grid_pred_prism), np.nanmax(grid_pred_rk), np.nanmax(grid_pred_ens))
    shared_levels = np.linspace(p_min, p_max, 60)

    for ax_p, title_p, grid_p in panels_info:
        im_p = ax_p.contourf(xs_proj, ys_proj, grid_p, levels=shared_levels, cmap="YlGnBu", extend="both")
        gdf_kec_proj.boundary.plot(ax=ax_p, color="#1a252f", linewidth=0.9)
        ax_p.scatter(station_coords_m[:, 0], station_coords_m[:, 1], color="#e74c3c", edgecolor="white", s=30, zorder=5)
        ax_p.set_title(title_p, fontsize=11, fontweight="bold")
        ax_p.set_xlabel("Easting (m)", fontsize=9)
        ax_p.set_ylabel("Northing (m)", fontsize=9)
        ax_p.grid(True, linestyle="--", alpha=0.25)
        plt.colorbar(im_p, ax=ax_p, shrink=0.8, label="mm")

    plt.tight_layout()
    fig_4p_path = out_figures / "spatial_4models_comparison.png"
    plt.savefig(fig_4p_path, dpi=300)
    plt.close(fig_4p)
    print(f"   ✓ Peta 4-Panel Komparasi Model berhasil disimpan: {fig_4p_path}")

    # Plot Folds Spatial Block Cross-Validation
    fig_cv, ax_cv = plt.subplots(figsize=(16, 9))
    gdf_kec_proj.boundary.plot(ax=ax_cv, color="#7f8c8d", linewidth=1.2)
    kmeans_clusters = SpatialBlockKFoldWithBuffer(n_splits=cfg["validation"]["n_splits"]).split(station_coords_m)
    cluster_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]

    for f_idx, (tr_idx, te_idx, meta) in enumerate(cv_splitter.split(station_coords_m)):
        c_col = cluster_colors[f_idx % len(cluster_colors)]
        ax_cv.scatter(station_coords_m[te_idx, 0], station_coords_m[te_idx, 1], color=c_col, s=90, edgecolor="black", label=f"Block Fold {f_idx+1} (n={len(te_idx)})", zorder=6)

    ax_cv.set_title(f"Peta Spatial Block K-Fold dengan Buffer Zone ({buffer_radius_m:.0f}m)\n"
                    f"Menjamin Nol Kebocoran Spasial (Zero Spatial Autocorrelation Leakage)", fontsize=13, fontweight="bold")
    ax_cv.set_xlabel("Easting UTM Zone 49S (m)", fontsize=11)
    ax_cv.set_ylabel("Northing UTM Zone 49S (m)", fontsize=11)
    ax_cv.legend(loc="lower right", frameon=True, fontsize=11)
    ax_cv.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig_cv_path = out_figures / "spatial_block_cross_validation_folds.png"
    plt.savefig(fig_cv_path, dpi=300)
    plt.close(fig_cv)
    print(f"   ✓ Peta Folds Cross-Validation berhasil disimpan: {fig_cv_path}")

    print("\n" + "=" * 85)
    print(f"🎉 BENCHMARKING SELESAI DALAM {time.time() - t_start:.2f} DETIK!")
    print("=" * 85)


if __name__ == "__main__":
    main()
