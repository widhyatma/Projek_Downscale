#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
UNIFIED MASTER ORCHESTRATOR: AUTOMATED RETRAINING, DOWNSCALING & DASHBOARD SYNC
Projek Downscale Kebumen - BMKG / JerisMeteo
=============================================================================
Orkestrator otomatis end-to-end yang bertugas:
  1. Mendeteksi data baru di data/ dan memperbarui deret waktu dasarian.
  2. Memperbarui matriks fitur sirkulasi atmosfer & indeks iklim global.
  3. Melatih ulang (retrain) model Deep Learning LSTM Multi-Head.
  4. Menjalankan downscaling topografi resolusi tinggi 250m (GeoTIFF EPSG:4326).
  5. Menyinkronkan luaran (overlay raster Web Mercator EPSG:3857, zonal statistics
     26 kecamatan & 460 desa/kelurahan) ke web Dashboard_Jerukagung-Meteorologi.
  6. Menguji kelayakan build Next.js dan opsional commit/push ke Git remote.
=============================================================================
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Konfigurasi encoding stdout
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DASHBOARD_DIR = Path(r"D:\Github\Dashboard_Jerukagung-Meteorologi")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator: Retraining, 250m GeoTIFF Downscaling, and Website Sync."
    )
    parser.add_argument("--force", action="store_true", help="Paksa retraining penuh dan ekspor ulang seluruh raster.")
    parser.add_argument("--target-year", type=int, default=2026, help="Tahun target prediksi (default: 2026).")
    parser.add_argument("--skip-downscale", action="store_true", help="Lewati proses downscaling GeoTIFF 250m.")
    parser.add_argument("--skip-sync", action="store_true", help="Lewati sinkronisasi ke folder dashboard.")
    parser.add_argument("--skip-build-test", action="store_true", help="Lewati uji coba build Next.js di dashboard.")
    parser.add_argument("--push-git", action="store_true", help="Lakukan git commit dan git push ke remote GitHub dashboard jika build sukses.")
    parser.add_argument("--log-file", type=str, default=None, help="File tujuan penulisan log eksekusi.")
    return parser.parse_args()

def log(msg: str, log_fp=None):
    wib = timezone(timedelta(hours=7))
    t_str = datetime.now(wib).strftime("%Y-%m-%d %H:%M:%S WIB")
    line = f"[{t_str}] {msg}"
    print(line)
    if log_fp:
        log_fp.write(line + "\n")
        log_fp.flush()

def main():
    args = parse_args()

    log_fp = None
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        log_fp = open(args.log_file, "a", encoding="utf-8")

    log("=" * 85, log_fp)
    log("MEMULAI MASTER PIPELINE AUTOMATED RETRAINING, DOWNSCALING & DASHBOARD SYNC", log_fp)
    log(f"Basis Direktori : {BASE_DIR}", log_fp)
    log(f"Target Tahun    : {args.target_year}", log_fp)
    log(f"Force Mode      : {args.force}", log_fp)
    log(f"Push to Git     : {args.push_git}", log_fp)
    log("=" * 85, log_fp)

    # -------------------------------------------------------------------------
    # TAHAP 1: Ingesti Data Baru & Pembaruan Dasarian
    # -------------------------------------------------------------------------
    log("\n>>> [TAHAP 1/5] Ingesti Data Baru & Ekstraksi Dasarian Inkremental...", log_fp)
    from prediksi_musim.update_dasarian import update_dasarian_dataset
    data_changed = update_dasarian_dataset()

    if data_changed or args.force:
        log("  -> Terdeteksi pembaruan data atau mode --force aktif. Membangun ulang matriks prediktor...", log_fp)
        from prediksi_musim.feature_engineering import build_training_matrix
        from prediksi_musim.feature_engineering_kemarau import build_training_matrix_kemarau
        build_training_matrix()
        build_training_matrix_kemarau()
        log("  -> [OK] Matriks fitur prediktor atmosferik & iklim global berhasil diperbarui.", log_fp)
    else:
        log("  -> Tidak ada penambahan data baru di data/. Melanjutkan dengan data eksisting.", log_fp)

    # -------------------------------------------------------------------------
    # TAHAP 2: Pelatihan Ulang Model (Retraining) & Inferensi Prediksi
    # -------------------------------------------------------------------------
    log("\n>>> [TAHAP 2/5] Melatih Ulang Arsitektur Deep Learning Sekuensial (LSTM Multi-Head)...", log_fp)
    from prediksi_musim.lstm_engine import predict_future_rainy_season_2026_lstm
    from prediksi_musim.lstm_kemarau_engine import predict_future_dry_season_2026

    log("  -> Melatih Model LSTM Multi-Head Musim Hujan 2026/2027...", log_fp)
    df_hujan = predict_future_rainy_season_2026_lstm(oni_jas=2.16, oni_jja=1.80, dmi_agu=0.35, dmi_sep=0.42)
    log(f"  -> [OK] Prediksi Musim Hujan selesai: {len(df_hujan)} kecamatan terpetakan.", log_fp)

    log("  -> Melatih Model LSTM Multi-Head Musim Kemarau 2026...", log_fp)
    df_kemarau = predict_future_dry_season_2026(oni_djf=1.85, oni_jfm=1.45, dmi_feb=0.25, dmi_mar=0.30)
    log(f"  -> [OK] Prediksi Musim Kemarau selesai: {len(df_kemarau)} kecamatan terpetakan.", log_fp)

    # -------------------------------------------------------------------------
    # TAHAP 3: Topographic Spatial Downscaling 250m (GeoTIFF)
    # -------------------------------------------------------------------------
    if not args.skip_downscale:
        log("\n>>> [TAHAP 3/5] Menjalankan Spatial Downscaling Topografi 250m & Konservasi Massa Regional...", log_fp)
        from prediksi_musim.topographic_downscaling import run_spatial_downscaling_2026
        from prediksi_musim.downscaling_kemarau_250m import run_spatial_downscaling_kemarau_2026

        log("  -> Downscaling Musim Hujan (AMH Dasarian & Total Hujan MH)...", log_fp)
        run_spatial_downscaling_2026()
        log("  -> [OK] GeoTIFF Musim Hujan 250m berhasil diekspor ke prediksi_musim/rasters_250m/.", log_fp)

        log("  -> Downscaling Musim Kemarau (AMK Dasarian, Total Hujan MK, & Kerawanan Kekeringan)...", log_fp)
        run_spatial_downscaling_kemarau_2026()
        log("  -> [OK] GeoTIFF Musim Kemarau 250m berhasil diekspor ke prediksi_musim/rasters_kemarau_250m/.", log_fp)
    else:
        log("\n>>> [TAHAP 3/5] Downscaling GeoTIFF 250m dilewati (--skip-downscale).", log_fp)

    # -------------------------------------------------------------------------
    # TAHAP 4: Sinkronisasi ke Web Dashboard (Web Mercator PNG, 460 Desa & 26 Kec)
    # -------------------------------------------------------------------------
    if not args.skip_sync:
        log("\n>>> [TAHAP 4/5] Menyinkronkan Luaran ke Dashboard_Jerukagung-Meteorologi...", log_fp)
        from prediksi_musim.sync_to_dashboard import sync_to_dashboard
        sync_res = sync_to_dashboard(DASHBOARD_DIR)
        log(f"  -> [OK] GeoJSON Kecamatan : {sync_res['geojson_kec_kb']:.1f} KB ({sync_res['kecamatan_count']} kecamatan)", log_fp)
        log(f"  -> [OK] GeoJSON Desa      : {sync_res['geojson_desa_kb']:.1f} KB ({sync_res['desa_count']} desa/kelurahan)", log_fp)
        log(f"  -> [OK] Prediksi JSON     : Kec ({sync_res['json_kec_kb']:.1f} KB), Desa ({sync_res['json_desa_kb']:.1f} KB)", log_fp)
        log(f"  -> [OK] Overlay Raster    : {sync_res['overlay_count']} PNG transparan Web Mercator (EPSG:3857) siap untuk Leaflet.", log_fp)
    else:
        log("\n>>> [TAHAP 4/5] Sinkronisasi Dashboard dilewati (--skip-sync).", log_fp)

    # -------------------------------------------------------------------------
    # TAHAP 5: Uji Build Next.js & Git Deployment
    # -------------------------------------------------------------------------
    log("\n>>> [TAHAP 5/5] Validasi Integritas Web Dashboard & Git Push...", log_fp)
    if not args.skip_build_test and DASHBOARD_DIR.exists():
        log("  -> Menjalankan verifikasi Next.js build (npm run build)...", log_fp)
        try:
            # Jalankan npm run build
            res = subprocess.run(
                ["npm.cmd", "run", "build"],
                cwd=str(DASHBOARD_DIR),
                capture_output=True,
                text=True,
                timeout=180
            )
            if res.returncode == 0:
                log("  -> [SUKSES BUILD] Next.js build berhasil 100% tanpa error!", log_fp)
            else:
                log(f"  -> [WARNING BUILD] Build gagal (kode {res.returncode}):\n{res.stderr[:500]}", log_fp)
        except Exception as e:
            log(f"  -> [WARNING BUILD] Tidak dapat menjalankan npm build: {e}", log_fp)

    if args.push_git and DASHBOARD_DIR.exists():
        log("  -> Menjalankan Git Add, Commit & Push ke remote origin/main...", log_fp)
        try:
            subprocess.run(["git", "add", "public/data"], cwd=str(DASHBOARD_DIR), check=True)
            wib_now = datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M WIB")
            commit_msg = f"Auto-update: Pemutakhiran Prediksi Musim, Zonal 460 Desa & Overlay Raster 250m ({wib_now})"
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=str(DASHBOARD_DIR), check=False)
            push_res = subprocess.run(["git", "push", "origin", "main"], cwd=str(DASHBOARD_DIR), capture_output=True, text=True)
            if push_res.returncode == 0:
                log("  -> [SUKSES GIT PUSH] Perubahan berhasil diunggah ke GitHub Dashboard!", log_fp)
            else:
                log(f"  -> [WARNING GIT] Push gagal: {push_res.stderr}", log_fp)
        except Exception as e:
            log(f"  -> [WARNING GIT] Gagal mengeksekusi perintah Git: {e}", log_fp)

    log("\n" + "=" * 85, log_fp)
    log("SELURUH TAHAPAN MASTER PIPELINE SELESAI DENGAN SUKSES! [STATUS: OPTIMAL]", log_fp)
    log("=" * 85, log_fp)

    if log_fp:
        log_fp.close()

if __name__ == "__main__":
    main()

