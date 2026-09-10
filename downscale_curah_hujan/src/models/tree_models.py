"""
Tree-Based Spatial Machine Learning Models Module
=================================================
1. RFSI (Random Forest Spatial Interpolation):
   Dynamic spatial features consisting of distances and observation values
   of the nearest k training stations + environmental covariates.
2. Spatial XGBoost:
   Gradient boosted regression with coordinates (X, Y), coastal distance, and terrain indices.
3. Spatial LightGBM:
   Fast histogram-based gradient boosting on spatial and morphological covariates.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial.distance import cdist
import xgboost as xgb
import lightgbm as lgb


class RFSIModel:
    """
    Random Forest Spatial Interpolation (RFSI) - Sekulić et al. (2020), Hengl et al. (2018):
    Incorporates distances (d_1, ..., d_k) and values (p_1, ..., p_k) of the nearest k
    observation stations as dynamic features alongside environmental covariates.
    """
    def __init__(
        self,
        k_neighbors: int = 5,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.k_neighbors = k_neighbors
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        self.coords_train = None
        self.values_train = None

    def _extract_nearest_station_features(
        self,
        query_coords: np.ndarray,
        is_training: bool = False
    ) -> np.ndarray:
        """
        Calculates distances (d_1, ..., d_k) and observed rainfall values (p_1, ..., p_k)
        to the nearest k training stations.
        If is_training=True, distance to self (d=0) is excluded to avoid identity leakage.
        """
        dists = cdist(query_coords, self.coords_train)
        n_query = len(query_coords)
        features = []

        k = min(self.k_neighbors, len(self.coords_train) - (1 if is_training else 0))

        for i in range(n_query):
            row_dists = dists[i].copy()
            if is_training:
                # Mask out self-point
                row_dists[i] = np.inf

            nearest_idx = np.argsort(row_dists)[:k]
            near_dists = row_dists[nearest_idx]
            near_vals = self.values_train[nearest_idx]

            # Jika k kurang dari self.k_neighbors (karena sedikit stasiun), pad dengan nilai mean
            if len(near_dists) < self.k_neighbors:
                pad_len = self.k_neighbors - len(near_dists)
                near_dists = np.pad(near_dists, (0, pad_len), constant_values=10000.0)
                near_vals = np.pad(near_vals, (0, pad_len), constant_values=np.mean(self.values_train))

            # Gabungkan jarak dan nilai curah hujan terdekat
            features.append(np.concatenate([near_dists, near_vals]))

        return np.array(features)

    def fit(self, X_covariates: np.ndarray, coords_m: np.ndarray, values: np.ndarray):
        self.coords_train = coords_m
        self.values_train = values.astype(float)

        # Ekstraksi fitur spasial nearest stations (dengan mask self)
        spatial_feats = self._extract_nearest_station_features(coords_m, is_training=True)
        
        # Gabungkan environmental covariates + nearest dynamic spatial features
        X_full = np.column_stack([X_covariates, spatial_feats])
        self.rf.fit(X_full, self.values_train)
        return self

    def predict(self, X_covariates_pred: np.ndarray, coords_pred_m: np.ndarray) -> np.ndarray:
        spatial_feats_pred = self._extract_nearest_station_features(coords_pred_m, is_training=False)
        X_full_pred = np.column_stack([X_covariates_pred, spatial_feats_pred])
        preds = self.rf.predict(X_full_pred)
        return np.clip(preds, 0.0, None)


class SpatialXGBoostModel:
    """
    Spatial XGBoost Regressor:
    Uses environmental covariates + projected spatial coordinates (x_utm, y_utm) + distance to coast.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        random_state: int = 42
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_covariates: np.ndarray, coords_m: np.ndarray, values: np.ndarray):
        # Normalisasi koordinat untuk XGBoost
        x_norm = (coords_m[:, 0] - np.mean(coords_m[:, 0])) / np.std(coords_m[:, 0])
        y_norm = (coords_m[:, 1] - np.mean(coords_m[:, 1])) / np.std(coords_m[:, 1])
        X_full = np.column_stack([X_covariates, x_norm, y_norm])
        self.mean_x = np.mean(coords_m[:, 0])
        self.std_x = np.std(coords_m[:, 0]) if np.std(coords_m[:, 0]) > 1e-4 else 1.0
        self.mean_y = np.mean(coords_m[:, 1])
        self.std_y = np.std(coords_m[:, 1]) if np.std(coords_m[:, 1]) > 1e-4 else 1.0

        self.model.fit(X_full, values.astype(float))
        return self

    def predict(self, X_covariates_pred: np.ndarray, coords_pred_m: np.ndarray) -> np.ndarray:
        x_norm = (coords_pred_m[:, 0] - self.mean_x) / self.std_x
        y_norm = (coords_pred_m[:, 1] - self.mean_y) / self.std_y
        X_full_pred = np.column_stack([X_covariates_pred, x_norm, y_norm])
        preds = self.model.predict(X_full_pred)
        return np.clip(preds, 0.0, None)


class SpatialLightGBMModel:
    """
    Spatial LightGBM Regressor:
    Fast histogram-based gradient boosting on spatial and morphological covariates.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        random_state: int = 42
    ):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
            verbosity=-1
        )

    def fit(self, X_covariates: np.ndarray, coords_m: np.ndarray, values: np.ndarray):
        x_norm = (coords_m[:, 0] - np.mean(coords_m[:, 0])) / (np.std(coords_m[:, 0]) if np.std(coords_m[:, 0]) > 1e-4 else 1.0)
        y_norm = (coords_m[:, 1] - np.mean(coords_m[:, 1])) / (np.std(coords_m[:, 1]) if np.std(coords_m[:, 1]) > 1e-4 else 1.0)
        X_full = np.column_stack([X_covariates, x_norm, y_norm])
        self.mean_x = np.mean(coords_m[:, 0])
        self.std_x = np.std(coords_m[:, 0]) if np.std(coords_m[:, 0]) > 1e-4 else 1.0
        self.mean_y = np.mean(coords_m[:, 1])
        self.std_y = np.std(coords_m[:, 1]) if np.std(coords_m[:, 1]) > 1e-4 else 1.0

        self.model.fit(X_full, values.astype(float))
        return self

    def predict(self, X_covariates_pred: np.ndarray, coords_pred_m: np.ndarray) -> np.ndarray:
        x_norm = (coords_pred_m[:, 0] - self.mean_x) / self.std_x
        y_norm = (coords_pred_m[:, 1] - self.mean_y) / self.std_y
        X_full_pred = np.column_stack([X_covariates_pred, x_norm, y_norm])
        preds = self.model.predict(X_full_pred)
        return np.clip(preds, 0.0, None)


class SpatialRandomForestModel:
    """
    Spatial Random Forest Regressor (Standard Spatial RF baseline):
    Random Forest fitted on environmental covariates combined with projected spatial coordinates.
    """
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.mean_x = None
        self.std_x = None
        self.mean_y = None
        self.std_y = None

    def fit(self, X_covariates: np.ndarray, coords_m: np.ndarray, values: np.ndarray):
        self.mean_x = np.mean(coords_m[:, 0])
        self.std_x = np.std(coords_m[:, 0]) if np.std(coords_m[:, 0]) > 1e-4 else 1.0
        self.mean_y = np.mean(coords_m[:, 1])
        self.std_y = np.std(coords_m[:, 1]) if np.std(coords_m[:, 1]) > 1e-4 else 1.0

        x_norm = (coords_m[:, 0] - self.mean_x) / self.std_x
        y_norm = (coords_m[:, 1] - self.mean_y) / self.std_y
        X_full = np.column_stack([X_covariates, x_norm, y_norm])
        self.rf.fit(X_full, values.astype(float))
        return self

    def predict(self, X_covariates_pred: np.ndarray, coords_pred_m: np.ndarray) -> np.ndarray:
        x_norm = (coords_pred_m[:, 0] - self.mean_x) / self.std_x
        y_norm = (coords_pred_m[:, 1] - self.mean_y) / self.std_y
        X_full_pred = np.column_stack([X_covariates_pred, x_norm, y_norm])
        preds = self.rf.predict(X_full_pred)
        return np.clip(preds, 0.0, None)
