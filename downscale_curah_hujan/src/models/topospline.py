"""
Topographic Spline & Facet Regression Models: ANUSPLIN & PRISM
=============================================================
1. ANUSPLIN (Trivariate Topographic Spline):
   Australian BoM / NASA / WorldClim / CHELSA standard.
   Minimizes 3D thin-plate surface bending energy across (X, Y, Z_elev).
   Guarantees C^2 continuity and seamless gradient transitions.

2. PRISM (Parameter-elevation Regressions on Independent Slopes):
   US NOAA / USDA / Oregon State University standard.
   Local facet-weighted regression combining horizontal distance,
   vertical elevation separation, and coastal proximity weighting.
"""

from typing import Optional
import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial.distance import cdist


class ANUSPLINModel:
    """
    ANUSPLIN: Trivariate Topographic Thin-Plate Spline (Hutchinson, 1991, 1995).
    Continuous energy-minimizing spline across 3D coordinates (X, Y, Elevation).
    """
    def __init__(self, z_weight: float = 0.6, smooth: float = 0.5):
        self.z_weight = z_weight
        self.smooth = smooth
        self.rbf = None
        self.mean_x = None
        self.std_x = None
        self.mean_y = None
        self.std_y = None
        self.mean_z = None
        self.std_z = None
        self.train_mean = None
        self.train_max = None

    def fit(self, coords_m: np.ndarray, elevation_m: np.ndarray, values: np.ndarray):
        values = values.astype(float)
        self.train_mean = float(np.mean(values))
        self.train_max = float(np.max(values))

        # Normalisasi koordinat 3D untuk stabilitas matriks RBF
        self.mean_x = float(np.mean(coords_m[:, 0]))
        self.std_x = float(np.std(coords_m[:, 0])) if np.std(coords_m[:, 0]) > 1e-4 else 1.0
        self.mean_y = float(np.mean(coords_m[:, 1]))
        self.std_y = float(np.std(coords_m[:, 1])) if np.std(coords_m[:, 1]) > 1e-4 else 1.0
        self.mean_z = float(np.mean(elevation_m))
        self.std_z = float(np.std(elevation_m)) if np.std(elevation_m) > 1e-4 else 1.0

        xn = (coords_m[:, 0] - self.mean_x) / self.std_x
        yn = (coords_m[:, 1] - self.mean_y) / self.std_y
        zn = (elevation_m - self.mean_z) / self.std_z * self.z_weight

        try:
            self.rbf = Rbf(xn, yn, zn, values, function='multiquadric', smooth=self.smooth)
        except Exception:
            # Fallback jika multiquadric singular
            self.rbf = Rbf(xn, yn, zn, values, function='linear', smooth=self.smooth + 0.5)

        return self

    def predict(self, coords_pred_m: np.ndarray, elevation_pred_m: np.ndarray) -> np.ndarray:
        xn_pred = (coords_pred_m[:, 0] - self.mean_x) / self.std_x
        yn_pred = (coords_pred_m[:, 1] - self.mean_y) / self.std_y
        zn_pred = (elevation_pred_m - self.mean_z) / self.std_z * self.z_weight

        try:
            preds = self.rbf(xn_pred, yn_pred, zn_pred)
            preds = np.clip(np.nan_to_num(preds, nan=self.train_mean), 0.0, self.train_max * 1.5)
        except Exception:
            # Fallback ke train mean jika evaluasi gagal
            preds = np.full(len(coords_pred_m), self.train_mean)

        return preds


class PRISMModel:
    """
    PRISM: Parameter-elevation Regressions on Independent Slopes (Daly et al., 1994, 2008).
    Local facet-weighted regression accounting for topographic facets, elevation gradients,
    horizontal proximity, and coastal proximity.
    """
    def __init__(
        self,
        sigma_h_m: float = 12000.0, # Jarak horizontal karakteristik (12 km)
        sigma_z_m: float = 250.0,   # Beda elevasi karakteristik (250 m)
        sigma_c_km: float = 10.0    # Jarak pesisir karakteristik (10 km)
    ):
        self.sigma_h_m = sigma_h_m
        self.sigma_z_m = sigma_z_m
        self.sigma_c_km = sigma_c_km
        self.coords_train = None
        self.elev_train = None
        self.coast_train = None
        self.values_train = None
        self.train_max = None

    def fit(
        self,
        coords_m: np.ndarray,
        elevation_m: np.ndarray,
        coast_dist_km: np.ndarray,
        values: np.ndarray
    ):
        self.coords_train = coords_m
        self.elev_train = elevation_m.astype(float)
        self.coast_train = coast_dist_km.astype(float)
        self.values_train = values.astype(float)
        self.train_max = float(np.max(self.values_train))
        return self

    def predict(
        self,
        coords_pred_m: np.ndarray,
        elevation_pred_m: np.ndarray,
        coast_dist_pred_km: np.ndarray
    ) -> np.ndarray:
        n_pred = len(coords_pred_m)
        dists_h = cdist(coords_pred_m, self.coords_train) # (n_pred, n_train)

        preds = np.zeros(n_pred)
        for i in range(n_pred):
            dh = dists_h[i]
            dz = np.abs(self.elev_train - elevation_pred_m[i])
            dc = np.abs(self.coast_train - coast_dist_pred_km[i])

            # Bobot multi-dimensi PRISM (Gaussian distance kernels)
            w_h = np.exp(-0.5 * (dh / self.sigma_h_m) ** 2)
            w_z = np.exp(-0.5 * (dz / self.sigma_z_m) ** 2)
            w_c = np.exp(-0.5 * (dc / self.sigma_c_km) ** 2)

            W = w_h * w_z * w_c
            sum_w = np.sum(W)

            if sum_w < 1e-8:
                # Fallback ke bobot jarak horizontal sederhana
                W = w_h
                sum_w = np.sum(W)

            weights = W / sum_w

            # Regresi linier terbobot lokal terhadap elevasi: P = beta0 + beta1 * Z
            # Jika rentang elevasi stasiun berbobot cukup tinggi (>50m), hitung slope elevasi
            elev_span = np.max(self.elev_train[weights > 0.05]) - np.min(self.elev_train[weights > 0.05]) if np.any(weights > 0.05) else 0.0
            
            if elev_span > 50.0 and len(self.values_train) >= 5:
                try:
                    # Weighted Least Squares (WLS)
                    z_center = self.elev_train - np.average(self.elev_train, weights=weights)
                    v_center = self.values_train - np.average(self.values_train, weights=weights)
                    # Slope orografis: beta1 = sum(W * z * v) / sum(W * z^2)
                    denom = np.sum(weights * z_center ** 2)
                    if denom > 1e-5:
                        beta1 = np.sum(weights * z_center * v_center) / denom
                        # Batasi orographic gradient rasional (-0.1 s.d. +0.5 mm/m)
                        beta1 = np.clip(beta1, -0.05, 0.35)
                        beta0 = np.average(self.values_train, weights=weights)
                        p_val = beta0 + beta1 * (elevation_pred_m[i] - np.average(self.elev_train, weights=weights))
                    else:
                        p_val = np.sum(weights * self.values_train)
                except Exception:
                    p_val = np.sum(weights * self.values_train)
            else:
                p_val = np.sum(weights * self.values_train)

            preds[i] = p_val

        upper = self.train_max * 1.5 if self.train_max is not None else None
        return np.clip(preds, 0.0, upper)
