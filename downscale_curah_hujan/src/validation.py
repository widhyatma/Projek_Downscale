"""
Validation Protocol Module (NON-NEGOTIABLE)
===========================================
Strict enforcement of Spatial Block Cross-Validation with Buffer Zones:
1. No random train-test splits.
2. Stations grouped into geographic spatial blocks.
3. Test stations within a spatial block NEVER share training weights or residuals
   from neighboring stations within the spatial correlation range (a).
4. Any training station within buffer distance (d < a) from test stations is discarded.
"""

from typing import List, Tuple, Dict, Any, Generator, Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def compute_empirical_semivariogram_range(
    coords_m: np.ndarray,
    values: np.ndarray,
    n_lags: int = 15
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes empirical semivariogram and estimates the spatial correlation length (range 'a' in meters).
    Semivariance: gamma(h) = (1 / (2 * N(h))) * sum((z_i - z_j)^2)
    """
    dists = cdist(coords_m, coords_m)
    n = len(values)
    
    pairs_h = []
    pairs_gamma = []
    for i in range(n):
        for j in range(i + 1, n):
            h = dists[i, j]
            gamma = 0.5 * (values[i] - values[j]) ** 2
            pairs_h.append(h)
            pairs_gamma.append(gamma)
            
    pairs_h = np.array(pairs_h)
    pairs_gamma = np.array(pairs_gamma)

    # Binning ke lag distances
    max_h = np.percentile(pairs_h, 80) # Gunakan 80% dari max distance
    bins = np.linspace(0, max_h, n_lags + 1)
    
    lag_centers = []
    lag_gamma = []
    for k in range(n_lags):
        mask = (pairs_h >= bins[k]) & (pairs_h < bins[k + 1])
        if np.sum(mask) >= 3:
            lag_centers.append(np.mean(pairs_h[mask]))
            lag_gamma.append(np.mean(pairs_gamma[mask]))

    lag_centers = np.array(lag_centers)
    lag_gamma = np.array(lag_gamma)

    # Estimasi range 'a': titik di mana gamma mencapai 95% dari sill (varians data)
    data_sill = np.var(values)
    range_idx = np.where(lag_gamma >= 0.90 * data_sill)[0]
    if len(range_idx) > 0:
        range_a = float(lag_centers[range_idx[0]])
    else:
        # Fallback ke 40% dari max domain distance
        range_a = float(max_h * 0.4)

    # Batasi range rasional untuk Kebumen (misal 4 km - 15 km)
    range_a = np.clip(range_a, 4000.0, 15000.0)
    return range_a, lag_centers, lag_gamma


class SpatialBlockKFoldWithBuffer:
    """
    Spatial Block Cross-Validation Generator with Buffer Zone Guardrail:
    - Partitions stations into spatial clusters using KMeans on UTM projected coordinates.
    - Eliminates spatial autocorrelation leakage by removing any training station located
      within the buffer distance (a) of the test block stations.
    """
    def __init__(
        self,
        n_splits: int = 4,
        buffer_radius_m: float = 5000.0,
        random_state: int = 42
    ):
        self.n_splits = n_splits
        self.buffer_radius_m = buffer_radius_m
        self.random_state = random_state

    def split(
        self,
        coords_m: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray, Dict[str, Any]], None, None]:
        """
        Yields (train_indices, test_indices, fold_metadata) for each spatial block fold.
        """
        n_stations = len(coords_m)
        kmeans = KMeans(n_clusters=self.n_splits, random_state=self.random_state, n_init=10)
        block_labels = kmeans.fit_predict(coords_m)

        # Matriks jarak antar semua stasiun (meter)
        dist_matrix = cdist(coords_m, coords_m)

        for fold in range(self.n_splits):
            test_mask = (block_labels == fold)
            test_idx = np.where(test_mask)[0]
            
            # Calon training stasiun (semua stasiun di luar blok fold)
            candidate_train_idx = np.where(~test_mask)[0]

            # Terapkan Buffer Zone: Hapus stasiun training yang berjarak < buffer_radius_m dari ANY stasiun test
            clean_train_idx = []
            dropped_buffer_idx = []

            for tr_i in candidate_train_idx:
                min_dist_to_test = np.min(dist_matrix[tr_i, test_idx])
                if min_dist_to_test >= self.buffer_radius_m:
                    clean_train_idx.append(tr_i)
                else:
                    dropped_buffer_idx.append(tr_i)

            clean_train_idx = np.array(clean_train_idx)
            dropped_buffer_idx = np.array(dropped_buffer_idx)

            fold_meta = {
                "fold": fold + 1,
                "n_test": len(test_idx),
                "n_train_clean": len(clean_train_idx),
                "n_train_dropped_buffer": len(dropped_buffer_idx),
                "buffer_radius_m": self.buffer_radius_m,
                "test_stations": test_idx.tolist(),
                "dropped_stations": dropped_buffer_idx.tolist()
            }

            yield clean_train_idx, test_idx, fold_meta
