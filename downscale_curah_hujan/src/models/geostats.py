"""
Geostatistical Models Module: OK, KED, and Regression-Kriging
============================================================
1. Ordinary Kriging (OK): Variogram fitting (Spherical/Exponential/Gaussian).
2. Kriging with External Drift (KED): Continuous elevational trend from DEM.
3. Regression-Kriging (RK): Machine Learning deterministic trend (Random Forest / XGBoost)
   combined with Ordinary Kriging on residual errors.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
import warnings

try:
    from pykrige.ok import OrdinaryKriging
    from pykrige.uk import UniversalKriging
    HAS_PYKRIGE = True
except ImportError:
    HAS_PYKRIGE = False


# Semivariogram theoretical models
def variogram_gaussian(h, nugget, sill, range_a):
    return nugget + sill * (1.0 - np.exp(- (h / (range_a / 2.0)) ** 2))

def variogram_exponential(h, nugget, sill, range_a):
    return nugget + sill * (1.0 - np.exp(- (h / (range_a / 3.0))))

def variogram_spherical(h, nugget, sill, range_a):
    hr = np.clip(h / range_a, 0.0, 1.0)
    return nugget + sill * (1.5 * hr - 0.5 * hr ** 3)


class OrdinaryKrigingModel:
    """
    Ordinary Kriging with automatic semivariogram fitting.
    """
    def __init__(self, variogram_model: str = "gaussian"):
        self.variogram_model = variogram_model
        self.x_train = None
        self.y_train = None
        self.z_train = None
        self.ok_instance = None
        self.nugget = 0.0
        self.sill = 1.0
        self.range_a = 10000.0

    def fit(self, coords_m: np.ndarray, values: np.ndarray):
        self.x_train = coords_m[:, 0]
        self.y_train = coords_m[:, 1]
        self.z_train = values.astype(float)

        if HAS_PYKRIGE:
            try:
                # pykrige OrdinaryKriging
                self.ok_instance = OrdinaryKriging(
                    self.x_train,
                    self.y_train,
                    self.z_train,
                    variogram_model=self.variogram_model,
                    verbose=False,
                    enable_plotting=False
                )
                return self
            except Exception as e:
                warnings.warn(f"PyKrige fitting failed ({e}), falling back to direct geostatistical solver.")
                self.ok_instance = None

        # Direct Geostatistical Matrix Solver (Fallback)
        var_data = np.var(self.z_train)
        self.sill = var_data * 0.85 if var_data > 1e-4 else 1.0
        self.nugget = var_data * 0.15
        self.range_a = 10000.0 # 10 km default
        return self

    def predict(self, coords_pred_m: np.ndarray) -> np.ndarray:
        if self.ok_instance is not None:
            try:
                xp = coords_pred_m[:, 0]
                yp = coords_pred_m[:, 1]
                z_pred, _ = self.ok_instance.execute("points", xp, yp)
                return np.clip(np.nan_to_num(z_pred, nan=float(np.mean(self.z_train))), 0.0, float(np.max(self.z_train) * 1.5))
            except Exception:
                pass

        # Matrix Kriging direct solver
        n = len(self.z_train)
        dists_train = cdist(np.column_stack([self.x_train, self.y_train]), np.column_stack([self.x_train, self.y_train]))
        gamma_matrix = variogram_gaussian(dists_train, self.nugget, self.sill, self.range_a)
        
        # Bentuk matriks Kriging berdimensi (N+1) x (N+1)
        K = np.zeros((n + 1, n + 1))
        K[:n, :n] = gamma_matrix
        K[:n, n] = 1.0
        K[n, :n] = 1.0
        K[n, n] = 0.0

        # Regularisasi diagonal untuk stabilitas numerik
        np.fill_diagonal(K[:n, :n], self.nugget + 1e-5)

        dists_pred = cdist(coords_pred_m, np.column_stack([self.x_train, self.y_train]))
        gamma_pred = variogram_gaussian(dists_pred, self.nugget, self.sill, self.range_a)

        preds = []
        try:
            K_inv = np.linalg.pinv(K)
            for m in range(len(coords_pred_m)):
                k0 = np.append(gamma_pred[m], 1.0)
                weights = np.dot(K_inv, k0)[:n]
                pred_val = np.dot(weights, self.z_train)
                preds.append(pred_val)
        except Exception:
            # Simple IDW fallback if singular
            inv_d = 1.0 / np.maximum(dists_pred, 1.0)
            weights = inv_d / np.sum(inv_d, axis=1, keepdims=True)
        preds = np.clip(np.array(preds), 0.0, float(np.max(self.z_train) * 1.5))
        return preds


class KEDModel:
    """
    Kriging with External Drift (KED):
    Incorporates continuous elevational drift (DEM) into the kriging system.
    """
    def __init__(self, variogram_model: str = "gaussian"):
        self.variogram_model = variogram_model
        self.coords_train = None
        self.values_train = None
        self.elev_train = None
        self.beta = None
        self.res_kriging = None

    def fit(self, coords_m: np.ndarray, values: np.ndarray, elevation: np.ndarray):
        self.coords_train = coords_m
        self.values_train = values.astype(float)
        self.elev_train = elevation.astype(float)

        # Regresi linier terhadap drift elevasi: P = beta_0 + beta_1 * Z
        X_drift = np.column_stack([np.ones_like(self.elev_train), self.elev_train])
        self.beta = np.linalg.lstsq(X_drift, self.values_train, rcond=None)[0]

        # Residuals
        drift_trend = self.beta[0] + self.beta[1] * self.elev_train
        residuals = self.values_train - drift_trend

        # Fit Kriging pada residual
        self.res_kriging = OrdinaryKrigingModel(variogram_model=self.variogram_model)
        self.res_kriging.fit(self.coords_train, residuals)
        return self

    def predict(self, coords_pred_m: np.ndarray, elevation_pred: np.ndarray) -> np.ndarray:
        # 1. Deterministic elevational drift
        drift_pred = self.beta[0] + self.beta[1] * elevation_pred.astype(float)
        # 2. Residual kriging
        res_pred = self.res_kriging.predict(coords_pred_m)
        # 3. Final prediction
        return np.clip(drift_pred + res_pred, 0.0, None)


class RegressionKrigingModel:
    """
    Regression-Kriging (RK):
    Deterministic machine learning trend on multi-variable environmental covariates
    combined with Ordinary Kriging on spatial residuals.
    """
    def __init__(
        self,
        base_estimator=None,
        variogram_model: str = "gaussian"
    ):
        self.base_estimator = base_estimator if base_estimator is not None else RandomForestRegressor(n_estimators=100, random_state=42)
        self.variogram_model = variogram_model
        self.res_kriging = None
        self.coords_train = None

    def fit(self, X_covariates: np.ndarray, coords_m: np.ndarray, values: np.ndarray):
        self.coords_train = coords_m
        values = values.astype(float)

        # 1. Fit ML trend pada kovariat topografi & spasial
        self.base_estimator.fit(X_covariates, values)
        trend_pred = self.base_estimator.predict(X_covariates)

        # 2. Hitung residual spasial: e = y - f(X)
        residuals = values - trend_pred

        # 3. Fit Kriging pada residual spasial
        self.res_kriging = OrdinaryKrigingModel(variogram_model=self.variogram_model)
        self.res_kriging.fit(self.coords_train, residuals)
        self.max_val = float(np.max(values))
        return self

    def predict(self, X_covariates_pred: np.ndarray, coords_pred_m: np.ndarray) -> np.ndarray:
        # 1. Trend pred
        trend_pred = self.base_estimator.predict(X_covariates_pred)
        # 2. Residual kriging pred
        res_pred = self.res_kriging.predict(coords_pred_m)
        # 3. Gabungkan dan pastikan batas fisik non-negatif & rasional
        upper_limit = self.max_val * 1.5 if hasattr(self, "max_val") else None
        return np.clip(trend_pred + res_pred, 0.0, upper_limit)
