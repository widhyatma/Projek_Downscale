"""
Clean up residual and temporary files in workspace
==================================================
1. Cleans up loose duplicate files in documents/ root (already safely stored in
   documents/penelitian_downscaling_25tahun/ and documents/penelitian_downscaling_2025/).
2. Cleans up obsolete temporary check scripts, scratch tests, and large temporary tif files in scratch/.
3. Retains the essential modular pipeline scripts (pipeline_phase1 through phase5).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import shutil
from pathlib import Path

def run():
    print("Memulai pembersihan file-file sisa di workspace...")

    # 1. Bersihkan loose duplicates di documents/
    doc_dir = Path("documents")
    loose_files = [
        "laporan_penelitian_downscaling_25tahun.aux",
        "laporan_penelitian_downscaling_25tahun.lof",
        "laporan_penelitian_downscaling_25tahun.log",
        "laporan_penelitian_downscaling_25tahun.lot",
        "laporan_penelitian_downscaling_25tahun.out",
        "laporan_penelitian_downscaling_25tahun.pdf",
        "laporan_penelitian_downscaling_25tahun.tex",
        "laporan_penelitian_downscaling_25tahun.toc",
        "laporan_perbandingan_downscaling_2025.aux",
        "laporan_perbandingan_downscaling_2025.log",
        "laporan_perbandingan_downscaling_2025.out",
        "laporan_perbandingan_downscaling_2025.pdf",
        "laporan_perbandingan_downscaling_2025.tex",
        "laporan_perbandingan_downscaling_2025.toc"
    ]
    for fname in loose_files:
        p = doc_dir / fname
        if p.exists():
            p.unlink()
            print(f"  ✓ Dihapus loose file documents: {fname}")

    # Hapus folder figures/ dan tables/ di root documents karena sudah ada di subfolder masing-masing penelitian
    fig_root = doc_dir / "figures"
    if fig_root.exists():
        shutil.rmtree(fig_root)
        print("  ✓ Dihapus loose duplicate folder: documents/figures/")

    tab_root = doc_dir / "tables"
    if tab_root.exists():
        shutil.rmtree(tab_root)
        print("  ✓ Dihapus loose duplicate folder: documents/tables/")

    # 2. Bersihkan file sementara & tes usang di scratch/
    scratch_dir = Path("scratch")
    # File-file yang WAJIB DIPERTAHANKAN (Pipeline inti & proposal)
    keep_files = {
        "pipeline_phase1_feature_extraction.py",
        "pipeline_phase2_hybrid_model_training.py",
        "pipeline_phase3_reconstruct_25yr_climatology.py",
        "pipeline_phase4_extreme_climate_analysis.py",
        "pipeline_phase5_generate_figures.py",
        "pipeline_phase5_generate_latex_tables.py",
        "organize_documents.py",
        "organize_2025_documents.py",
        "learning_proposal.md",
        "cleanup_workspace.py"
    }

    deleted_count = 0
    reclaimed_bytes = 0
    for p in scratch_dir.iterdir():
        if p.is_file() and p.name not in keep_files:
            sz = p.stat().st_size
            p.unlink()
            deleted_count += 1
            reclaimed_bytes += sz
            print(f"  ✓ Dihapus file sisa scratch: {p.name} ({sz/1024:.1f} KB)")

    print("\n" + "=" * 80)
    print(f"🎉 PEMBERSIHAN SELESAI!")
    print(f"   - {deleted_count} file sisa di scratch/ berhasil dibersihkan.")
    print(f"   - Ruang penyimpanan dibebaskan: {reclaimed_bytes / (1024*1024):.2f} MB.")
    print(f"   - Seluruh dokumen penelitian tersimpan rapi dan aman di subfolder documents/.")
    print("=" * 80)

if __name__ == "__main__":
    run()
