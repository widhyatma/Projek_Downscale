"""
Organize 2025 13-Algorithm Downscaling Benchmark into documents/penelitian_downscaling_2025/
============================================================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import shutil
from pathlib import Path

def run():
    base_doc = Path("documents")
    target_2025 = base_doc / "penelitian_downscaling_2025"
    target_fig = target_2025 / "figures"
    target_tab = target_2025 / "tables"

    target_fig.mkdir(parents=True, exist_ok=True)
    target_tab.mkdir(parents=True, exist_ok=True)

    # Move/copy 2025 files
    for p in base_doc.glob("laporan_perbandingan_downscaling_2025.*"):
        shutil.copy2(p, target_2025 / p.name)
        print(f"  ✓ Tersalin: {p.name}")

    # Copy 2025 figures
    fig_2025 = [
        "benchmark_13models_evaluation.png",
        "model_uncertainty_spread_2025.png",
        "monthly_12m_ANUSPLIN_2025.png",
        "monthly_12m_EnsembleMean_2025.png",
        "monthly_12m_PRISM_2025.png",
        "monthly_12m_RegressionKriging_2025.png",
        "monthly_spatial_downscaled_2025.png",
        "monthly_trajectory_algorithms_comparison_2025.png",
        "seasonal_matrix_4algorithms_comparison_2025.png",
        "spatial_4models_comparison_2025.png",
        "spatial_block_cv_folds.png",
        "zonal_kecamatan_hyetograph_2025.png"
    ]
    for fname in fig_2025:
        src = base_doc / "figures" / fname
        if src.exists():
            shutil.copy2(src, target_fig / fname)
            print(f"  ✓ Tersalin 2025 fig: {fname}")

    print("✓ Penelitian 2025 berhasil diarsipkan di documents/penelitian_downscaling_2025/!")

if __name__ == "__main__":
    run()
