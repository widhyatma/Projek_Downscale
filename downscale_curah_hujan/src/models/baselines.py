"""
Spatial Baseline & Geographically Weighted Regression Models: IDW & GWR
=======================================================================
1. IDW (Inverse Distance Weighting): Classic deterministic benchmark (Shepard, 1968).
2. GWR (Geographically Weighted Regression): Local spatial regression (Brunsdon et al., 1996).
"""

from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist


class IDWModel:
    """
    Inverse Distance Weighting (IDW) Interpolator:
    Deterministic spatial baseline weighting observations by 1 / (distance^power).
    """
    def __init__(self, power: float = 2.0):
        self.power = power
        self.coords_train = None
        self.values_train = None
        self.train_max = None

    def fit(self, coords_m: np.ndarray, values: np.ndarray):
        self.coords_train = coords_m
        self.values_train = values.astype(float)
        self.train_max = float(np.max(self.values_train))
        return self

    def predict(self, coords_pred_m: np.ndarray) -> np.ndarray:
        dists = cdist(coords_pred_m, self.coords_train)
        # Tangani jarak mendekati nol (titik yang sama)
        dists = np.maximum(dists, 1.0)
        weights = 1.0 / (dists ** self.power)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        norm_weights = weights / weights_sum
        preds = np.sum(norm_weights * self.values_train, axis=1)
        return np.clip(preds, 0.0, self.train_max * 1.5)


class GWRModel:
    """
    Geographically Weighted Regression (GWR):
    Fits local linear regression against elevation weighted by a Gaussian spatial distance kernel.
    """
    def __init__(self, bandwidth_m: float = 15000.0):
        self.bandwidth_m = bandwidth_m
        self.coords_train = None
        self.elev_train = None
        self.values_train = None
        self.train_max = None

    def fit(self, coords_m: np.ndarray, elevation_m: np.ndarray, values: np.ndarray):
        self.coords_train = coords_m
        self.elev_train = elevation_m.astype(float)
        self.values_train = values.astype(float)
        self.train_max = float(np.max(self.values_train))
        return self

    def predict(self, coords_pred_m: np.ndarray, elevation_pred_m: np.ndarray) -> np.ndarray:
        n_pred = len(coords_pred_m)
        dists = cdist(coords_pred_m, self.coords_train)
        preds = np.zeros(n_pred)
        X_tr = np.column_stack([np.ones_like(self.elev_train), self.elev_train])

        for i in range(n_pred):
            w = np.exp(-0.5 * (dists[i] / self.bandwidth_m) ** 2)
            w_sum = np.sum(w)
            if w_sum < 1e-8:
                preds[i] = np.mean(self.values_train)
                continue
            w_norm = w / w_sum

            # Weighted Least Squares (WLS)
            XtW = X_tr.T * w_norm
            try:
                beta = np.linalg.solve(XtW @ X_tr + 1e-4 * np.eye(2), XtW @ self.values_train)
                p = beta[0] + beta[1] * elevation_pred_m[i]
            except Exception:
                p = np.sum(w_norm * self.values_train)
            preds[i] = p

        return np.clip(preds, 0.0, self.train_max * 1.5)
