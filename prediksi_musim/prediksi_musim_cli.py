#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production CLI Runner: Sistem Prediksi Musim Hujan Kebumen (Multi-Sensor & Reanalisis)
Menyediakan antarmuka baris perintah (CLI) untuk memprediksi Awal Musim Hujan (AMH),
Sifat Hujan, Puncak Musim Hujan, dan Rekomendasi Kalender Tanam 26 Kecamatan di Kabupaten Kebumen.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Path root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from prediksi_musim.model_engine import train_and_evaluate_loyo, predict_future_season_2026, FEATURE_COLS, KATEGORI_MAP, INV_KATEGORI_MAP, SIFAT_MAP, INV_SIFAT_MAP, get_dasarian_label
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistem Prediksi Awal Musim Hujan (AMH) & Sifat Hujan Tingkat Kecamatan di Kabupaten Kebumen."
    )
    parser.add_argument("--target-year", type=int, default=2026, help="Tahun target awal musim hujan (default: 2026).")
    parser.add_argument("--oni-jas", type=float, default=2.16, help="Nilai Indeks Oceanic Niño Index (ONI) JAS (default: 2.16).")
    parser.add_argument("--oni-jja", type=float, default=1.80, help="Nilai Indeks ONI JJA (default: 1.80).")
    parser.add_argument("--dmi-agu", type=float, default=0.35, help="Nilai Indeks Dipole Mode Index (DMI) Agustus (default: 0.35).")
    parser.add_argument("--dmi-sep", type=float, default=0.42, help="Nilai Indeks DMI September (default: 0.42).")
    parser.add_argument("--run-loyo", action="store_true", help="Jalankan validasi Leave-One-Year-Out (LOYO) pada data historis 2000-2025.")
    parser.add_argument("--export-plots", action="store_true", help="Otomatis hasilkan 6 peta tematik resolusi tinggi 16:9 (300 DPI).")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR / "prediksi_musim" / "data"), help="Direktori output CSV hasil.")
    return parser.parse_args()

def run_prediction_pipeline(args):
    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"🌦️ SISTEM PREDIKSI MUSIM HUJAN RESOLUSI TINGGI KABUPATEN KEBUMEN")
    print(f"Target Musim: {args.target_year}/{args.target_year + 1}")
    print(f"Parameter Global: ONI JAS = {args.oni_jas:+.2f} | DMI Sep = {args.dmi_sep:+.2f}")
    print("="*80)
    
    if args.run_loyo:
        print("\n[STEP 1/3] Menjalankan Validasi Model Leave-One-Year-Out (LOYO)...")
        train_and_evaluate_loyo()
    else:
        print("\n[STEP 1/3] Menggunakan bobot model berbasis data training 2000-2025...")
        
    print("\n[STEP 2/3] Menginferensi Prakiraan AMH, Sifat Hujan, & Kalender Tanam 26 Kecamatan...")
    train_file = data_dir / "dataset_training_prediksi_musim.csv"
    norm_file = data_dir / "normal_klimatologi_amh_kebumen.csv"
    
    if not train_file.exists() or not norm_file.exists():
        raise FileNotFoundError(f"File data prasyarat tidak ditemukan di {data_dir}. Jalankan ekstraksi data terlebih dahulu.")
        
    df_train = pd.read_csv(train_file)
    df_norm = pd.read_csv(norm_file)
    
    X_train = df_train[FEATURE_COLS]
    y_kat = df_train['kategori_amh'].map(KATEGORI_MAP)
    y_das = df_train['amh_dasarian']
    y_sifat = df_train['sifat_hujan'].map(SIFAT_MAP)
    y_puncak = df_train['puncak_dasarian']
    
    clf_kat = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
    clf_kat.fit(X_train, y_kat)
    
    reg_das = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42)
    reg_das.fit(X_train, y_das)
    
    clf_sifat = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
    clf_sifat.fit(X_train, y_sifat)
    
    reg_puncak = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    reg_puncak.fit(X_train, y_puncak)
    
    results = []
    for _, norm_row in df_norm.iterrows():
        kec = norm_row['kecamatan']
        feat = {
            'oni_jas': args.oni_jas,
            'oni_jja': args.oni_jja,
            'dmi_agu': args.dmi_agu,
            'dmi_sep': args.dmi_sep,
            'u10_m7': -1.5,
            'rh_m7': 78.0,
            't2m_m7': 24.5,
            'u10_m8': -1.2,
            'rh_m8': 76.5,
            't2m_m8': 24.8,
            'u10_m9': -0.8,
            'rh_m9': 75.0,
            't2m_m9': 25.2,
            'normal_amh_dasarian': norm_row['normal_amh_dasarian'],
            'normal_total_hujan_mh_mm': norm_row['normal_total_hujan_mh_mm']
        }
        X_pred = pd.DataFrame([feat])[FEATURE_COLS]
        
        kat_prob = clf_kat.predict_proba(X_pred)[0]
        kat_pred = INV_KATEGORI_MAP[int(np.argmax(kat_prob))]
        das_pred = float(reg_das.predict(X_pred)[0])
        sifat_pred = INV_SIFAT_MAP[int(clf_sifat.predict(X_pred)[0])]
        puncak_pred = float(reg_puncak.predict(X_pred)[0])
        
        olah_tanah_das = int(round(das_pred)) + 1
        tanam_das = olah_tanah_das + 1
        
        results.append({
            'kecamatan': kec,
            'normal_amh_dasarian': norm_row['normal_amh_dasarian'],
            'normal_amh_label': norm_row['normal_amh_label'],
            'prakiraan_amh_dasarian': round(das_pred, 1),
            'prakiraan_amh_label': get_dasarian_label(das_pred),
            'perbandingan_terhadap_normal': kat_pred,
            'probabilitas_maju_pct': round(kat_prob[0] * 100, 1),
            'probabilitas_sama_pct': round(kat_prob[1] * 100, 1),
            'probabilitas_mundur_pct': round(kat_prob[2] * 100, 1),
            'prakiraan_sifat_hujan': sifat_pred,
            'prakiraan_puncak_dasarian': round(puncak_pred, 1),
            'prakiraan_puncak_label': get_dasarian_label(puncak_pred),
            'rekomendasi_olah_tanah': get_dasarian_label(olah_tanah_das),
            'rekomendasi_tanam_padi_mt1': get_dasarian_label(tanam_das)
        })
        
    df_out = pd.DataFrame(results)
    out_filename = data_dir / f"prakiraan_musim_{args.target_year}_{args.target_year + 1}_kebumen.csv"
    df_out.to_csv(out_filename, index=False)
    print(f"[OK] File prediksi tersimpan di: {out_filename}")
    
    print("\n" + "="*80)
    print(f"HASIL PREDIKSI AWAL MUSIM HUJAN KEBUMEN {args.target_year}/{args.target_year + 1}")
    print("="*80)
    print(df_out[['kecamatan', 'normal_amh_label', 'prakiraan_amh_label', 'perbandingan_terhadap_normal', 'prakiraan_sifat_hujan', 'rekomendasi_tanam_padi_mt1']].to_string(index=False))
    
    if args.export_plots:
        print("\n[STEP 3/3] Menghasilkan 6 Peta Tematik Standar Publikasi 16:9...")
        from prediksi_musim.visualisasi_peta import generate_all_maps
        generate_all_maps()
    else:
        print("\n[STEP 3/3] Pembuatan peta dilewati (Gunakan flag --export-plots untuk membuat peta).")
        
    print("\n[SELESAI] Pipeline Prediksi Musim Hujan berhasil dituntaskan!\n")

if __name__ == "__main__":
    args = parse_args()
    run_prediction_pipeline(args)
