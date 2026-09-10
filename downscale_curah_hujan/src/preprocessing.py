"""
Preprocessing & Feature Engineering Module
==========================================
1. Coordinate Reference System (CRS) transformation to metric UTM Zone 49S (EPSG:32749).
2. Topographic covariate derivation: Elevation, Slope, Sin/Cos Aspect, TPI, and Coastal Distance.
3. Bilinear sampling to station locations.
4. Multicollinearity pruning via Variance Inflation Factor (VIF <= 5.0).
5. Zero-rain handling via two-stage classification-regression hurdle model when zero-inflation > 40%.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import uniform_filter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin


def reproject_points(
    df: pd.DataFrame,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:32749"
) -> gpd.GeoDataFrame:
    """
    Transforms station coordinates from geographic (EPSG:4326) to projected (e.g. UTM 49S EPSG:32749)
    to enable true Euclidean distance calculations in meters.
    """
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=src_crs)
    gdf_proj = gdf.to_crs(dst_crs)
    gdf_proj["x_utm"] = gdf_proj.geometry.x
    gdf_proj["y_utm"] = gdf_proj.geometry.y
    return gdf_proj


def compute_raster_covariates(
    dem_path: Path,
    dst_crs: str = "EPSG:32749",
    target_res_m: float = 250.0,
    tpi_size: int = 5
) -> Dict[str, Any]:
    """
    Reads DEM, warps to projected CRS at target resolution in meters, and computes:
    - Elevation (m)
    - Slope (degrees)
    - Aspect decomposed into sin(aspect) and cos(aspect)
    - Topographic Position Index (TPI)
    - Distance to Coast (km)
    - Spatial coordinate grids (x_utm, y_utm)
    """
    with rasterio.open(dem_path) as src:
        src_crs = src.crs
        src_trans = src.transform
        
        # Hitung transform baru untuk proyeksi tujuan (UTM)
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=(target_res_m, target_res_m)
        )
        
        elev = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=elev,
            src_transform=src_trans,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

    # Tangani NoData atau nilai negatif tidak valid
    elev[np.isnan(elev)] = 0.0
    elev = np.clip(elev, 0.0, None)

    # Hitung Gradient spasial untuk Slope & Aspect
    dy, dx = np.gradient(elev, target_res_m, target_res_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    aspect_rad = np.arctan2(-dx, dy)
    aspect_rad = np.where(aspect_rad < 0, aspect_rad + 2 * np.pi, aspect_rad)
    sin_aspect = np.sin(aspect_rad)
    cos_aspect = np.cos(aspect_rad)

    # Topographic Position Index (TPI = DEM - Mean_neighborhood)
    mean_elev = uniform_filter(elev, size=tpi_size, mode='nearest')
    tpi = elev - mean_elev

    # Grid koordinat spasial UTM
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs_utm, ys_utm = rasterio.transform.xy(transform, rows, cols)
    xs_utm = np.array(xs_utm).reshape((height, width))
    ys_utm = np.array(ys_utm).reshape((height, width))

    # Jarak ke Pantai Selatan (garis pantai selatan Kebumen kira-kira pada y_utm_min)
    y_coast_utm = np.min(ys_utm)
    coast_dist_km = np.clip((ys_utm - y_coast_utm) / 1000.0, a_min=0.0, a_max=None)

    return {
        "transform": transform,
        "crs": dst_crs,
        "width": width,
        "height": height,
        "elevation": elev,
        "slope": slope_deg,
        "sin_aspect": sin_aspect,
        "cos_aspect": cos_aspect,
        "tpi": tpi,
        "coast_dist_km": coast_dist_km,
        "x_utm": xs_utm,
        "y_utm": ys_utm
    }


def sample_covariates_bilinear(
    stations_gdf: gpd.GeoDataFrame,
    covariates: Dict[str, Any]
) -> pd.DataFrame:
    """
    Samples continuous raster covariates at station point locations using bilinear interpolation.
    """
    df_out = stations_gdf.copy()
    transform = covariates["transform"]
    inv_trans = ~transform
    h, w = covariates["height"], covariates["width"]

    feature_names = ["elevation", "slope", "sin_aspect", "cos_aspect", "tpi", "coast_dist_km"]
    sampled_data = {feat: [] for feat in feature_names}

    for pt in stations_gdf.geometry:
        col_f, row_f = inv_trans * (pt.x, pt.y)
        
        # Bilinear interpolation
        c0 = int(np.floor(col_f))
        r0 = int(np.floor(row_f))
        c1 = min(c0 + 1, w - 1)
        r1 = min(r0 + 1, h - 1)
        c0 = max(0, min(c0, w - 1))
        r0 = max(0, min(r0, h - 1))

        dc = col_f - c0
        dr = row_f - r0

        for feat in feature_names:
            grid = covariates[feat]
            val = (
                (1 - dr) * (1 - dc) * grid[r0, c0] +
                (1 - dr) * dc * grid[r0, c1] +
                dr * (1 - dc) * grid[r1, c0] +
                dr * dc * grid[r1, c1]
            )
            sampled_data[feat].append(val)

    for feat in feature_names:
        df_out[feat] = sampled_data[feat]

    return df_out


def prune_vif_multicollinearity(
    df: pd.DataFrame,
    feature_cols: List[str],
    vif_threshold: float = 5.0
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Iteratively calculates Variance Inflation Factor (VIF) and drops any covariate with VIF > threshold.
    Ensures that remaining features have low collinearity.
    """
    remaining_features = list(feature_cols)
    iteration = 0
    vif_history = []

    while len(remaining_features) > 1:
        X = df[remaining_features].dropna()
        # Tambah konstanta untuk VIF calculation
        X_const = X.copy()
        X_const["_intercept"] = 1.0

        vif_data = pd.DataFrame()
        vif_data["feature"] = remaining_features
        vif_data["VIF"] = [
            variance_inflation_factor(X_const.values, i)
            for i in range(len(remaining_features))
        ]

        max_vif = vif_data["VIF"].max()
        if max_vif > vif_threshold:
            drop_feature = vif_data.sort_values("VIF", ascending=False).iloc[0]["feature"]
            # Simpan histori
            vif_history.append({"iteration": iteration, "dropped": drop_feature, "vif": max_vif})
            remaining_features.remove(drop_feature)
            iteration += 1
        else:
            break

    # Final VIF table
    X_final = df[remaining_features].dropna().copy()
    X_final["_intercept"] = 1.0
    final_vif_df = pd.DataFrame({
        "feature": remaining_features,
        "final_VIF": [
            variance_inflation_factor(X_final.values, i)
            for i in range(len(remaining_features))
        ]
    })

    return df[remaining_features], remaining_features, final_vif_df


class HurdlePrecipitationModel(BaseEstimator, RegressorMixin):
    """
    Two-Stage Hurdle Model for Zero-Inflated Rainfall:
    - Stage 1: Classifier for Rain Occurrence (P > 0.1 mm)
    - Stage 2: Regressor for Precipitation Amount conditioned on Rain Occurrence (P | P > 0.1 mm)
    Final prediction = P(occurrence) * E(amount | occurrence)
    """
    def __init__(
        self,
        classifier=None,
        regressor=None,
        rain_threshold: float = 0.1
    ):
        self.classifier = classifier if classifier is not None else LogisticRegression(max_iter=1000)
        self.regressor = regressor if regressor is not None else RandomForestRegressor(n_estimators=100, random_state=42)
        self.rain_threshold = rain_threshold

    def fit(self, X, y):
        y_arr = np.array(y)
        # Stage 1: Binary occurrence
        y_binary = (y_arr > self.rain_threshold).astype(int)
        self.classifier.fit(X, y_binary)

        # Stage 2: Amount on wet observations
        wet_mask = y_binary == 1
        if np.sum(wet_mask) >= 5:
            self.regressor.fit(X[wet_mask], y_arr[wet_mask])
        else:
            # Fallback jika sangat sedikit hari basah
            self.regressor.fit(X, y_arr)
        return self

    def predict(self, X):
        # Probabilitas hujan
        if hasattr(self.classifier, "predict_proba"):
            p_occur = self.classifier.predict_proba(X)[:, 1]
        else:
            p_occur = self.classifier.predict(X).astype(float)

        # Estimasi jumlah curah hujan
        pred_amount = np.clip(self.regressor.predict(X), 0.0, None)

        # Kombinasi hurdle expectation
        y_final = p_occur * pred_amount
        return np.where(p_occur < 0.2, 0.0, y_final)
