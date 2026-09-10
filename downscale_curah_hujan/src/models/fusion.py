"""
Satellite-Gauge Fusion Module: Conditional Merging (CM)
======================================================
Implements Conditional Merging (Pegram & Sinclair, 2004; Sinclair & Pegram, 2005):
Calibrates gridded satellite baseline (e.g. CHIRPS / GSMaP) with ground rain gauge observations:
1. Sample satellite field at gauge locations: S(s_i).
2. Kriging interpolation of ground observations: G_hat_OK(s).
3. Kriging interpolation of sampled satellite values: S_hat_OK(s).
4. Satellite spatial perturbation/error field: Delta S(s) = S(s) - S_hat_OK(s).
5. Final conditionally merged precipitation: P_CM(s) = G_hat_OK(s) + Delta S(s).
"""

from typing import Optional
import numpy as np
from .geostats import OrdinaryKrigingModel


class ConditionalMergingModel:
    """
    Conditional Merging (CM) for Satellite-Gauge Fusion.
    Preserves ground observation truth at gauge locations while retaining
    spatial rainfall patterns and texture from gridded satellite data.
    """
    def __init__(self, variogram_model: str = "gaussian"):
        self.variogram_model = variogram_model
        self.krig_gauge = None
        self.krig_sat = None
        self.coords_train = None
        self.gauge_train = None
        self.sat_train = None

    def fit(
        self,
        coords_m: np.ndarray,
        gauge_values: np.ndarray,
        sat_at_gauges: np.ndarray
    ):
        """
        coords_m: (N, 2) UTM coordinates of rain gauges
        gauge_values: (N,) Observed rainfall at rain gauges
        sat_at_gauges: (N,) Gridded satellite precipitation sampled at rain gauges
        """
        self.coords_train = coords_m
        self.gauge_train = gauge_values.astype(float)
        self.sat_train = sat_at_gauges.astype(float)

        # 1. Kriging pada observasi penakar hujan
        self.krig_gauge = OrdinaryKrigingModel(variogram_model=self.variogram_model)
        self.krig_gauge.fit(self.coords_train, self.gauge_train)

        # 2. Kriging pada nilai satelit di stasiun
        self.krig_sat = OrdinaryKrigingModel(variogram_model=self.variogram_model)
        self.krig_sat.fit(self.coords_train, self.sat_train)

        return self

    def predict(
        self,
        coords_pred_m: np.ndarray,
        sat_pred_field: np.ndarray
    ) -> np.ndarray:
        """
        coords_pred_m: (M, 2) Prediction target coordinates (e.g. test stations or full raster grid)
        sat_pred_field: (M,) Raw satellite precipitation values at prediction target coordinates
        """
        # 1. G_hat_OK: Kriged ground observations
        g_hat = self.krig_gauge.predict(coords_pred_m)

        # 2. S_hat_OK: Kriged satellite observations from stations
        s_hat = self.krig_sat.predict(coords_pred_m)

        # 3. Delta S: Satellite spatial perturbation field
        delta_s = sat_pred_field.astype(float) - s_hat

        # 4. Conditionally merged precipitation
        p_cm = g_hat + delta_s
        upper = float(np.max(self.gauge_train) * 1.5)
        return np.clip(p_cm, 0.0, upper)
