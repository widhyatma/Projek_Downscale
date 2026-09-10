"""
Evaluation Metrics Module
=========================
Calculates standard meteorological and hydrological performance metrics:
- KGE (Kling-Gupta Efficiency - Gupta et al., 2009): Primary optimization metric.
- RMSE (Root Mean Square Error): Quadratic misprediction penalty.
- MAE (Mean Absolute Error): Linear deviation metric.
- PBIAS (Percent Bias): Water volume conservation indicator.
- R2 (Coefficient of Determination)
- Pearson correlation coefficient (r)
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
from scipy.stats import pearsonr

Tuple_KGE = Tuple[float, float, float, float]

def calculate_kge(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple_KGE:
    """
    Computes Kling-Gupta Efficiency (KGE):
    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    where:
      r = Pearson correlation coefficient
      alpha = sigma_pred / sigma_true (relative variability)
      beta = mu_pred / mu_true (bias ratio)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Tangani kasus flat/zero variance
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    if std_true < 1e-6 or std_pred < 1e-6:
        r = 1.0 if np.allclose(y_true, y_pred) else 0.0
        alpha = 1.0 if np.isclose(std_true, std_pred) else 0.0
    else:
        r_val, _ = pearsonr(y_true, y_pred)
        r = float(r_val) if not np.isnan(r_val) else 0.0
        alpha = float(std_pred / std_true)

    if abs(mean_true) < 1e-6:
        beta = 1.0 if abs(mean_pred) < 1e-6 else float(mean_pred + 1.0)
    else:
        beta = float(mean_pred / mean_true)

    kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    return float(kge), float(r), float(alpha), float(beta)


# Helper type
Tuple_KGE = tuple[float, float, float, float]


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def calculate_pbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percent Bias (PBIAS): Volume conservation indicator.
    PBIAS = 100 * sum(y_pred - y_true) / sum(y_true)
    Ideal value = 0.0%. Positive means overestimation, negative means underestimation.
    """
    sum_true = np.sum(y_true)
    if abs(sum_true) < 1e-6:
        return 0.0
    return float(100.0 * np.sum(np.asarray(y_pred) - np.asarray(y_true)) / sum_true)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R2)."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot < 1e-6:
        return 1.0 if ss_res < 1e-6 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def evaluate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes the full suite of evaluation metrics for a model fold.
    """
    kge, r, alpha, beta = calculate_kge(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    pbias = calculate_pbias(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)

    return {
        "KGE": round(kge, 4),
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "PBIAS": round(pbias, 2),
        "R2": round(r2, 4),
        "Pearson_r": round(r, 4),
        "KGE_alpha": round(alpha, 4),
        "KGE_beta": round(beta, 4)
    }
