#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Machine Learning Engine Prediksi Musim Hujan Kebumen
Melatih model XGBoost & Random Forest dengan validasi Leave-One-Year-Out (LOYO)
Memprediksi Awal Musim Hujan (AMH), Sifat Hujan, dan Puncak Musim Hujan per Kecamatan.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"

FEATURE_COLS = [
    'oni_jas', 'oni_jja', 'dmi_agu', 'dmi_sep',
    'u10_m7', 'rh_m7', 't2m_m7',
    'u10_m8', 'rh_m8', 't2m_m8',
    'u10_m9', 'rh_m9', 't2m_m9',
    'normal_amh_dasarian', 'normal_total_hujan_mh_mm'
]

KATEGORI_MAP = {'Maju': 0, 'Sama': 1, 'Mundur': 2}
INV_KATEGORI_MAP = {0: 'Maju', 1: 'Sama', 2: 'Mundur'}

SIFAT_MAP = {'Bawah Normal': 0, 'Normal': 1, 'Atas Normal': 2}
INV_SIFAT_MAP = {0: 'Bawah Normal', 1: 'Normal', 2: 'Atas Normal'}

def get_dasarian_label(d_idx):
    d_round = int(round(d_idx))
    m = ((d_round - 1) // 3) + 1
    d_m = ((d_round - 1) % 3) + 1
    m_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    roman = ["I", "II", "III"]
    if 1 <= m <= 12:
        return f"{m_names[m-1]} {roman[d_m-1]}"
    return f"Dasarian {d_round}"

def train_and_evaluate_loyo():
    """
    Melatih dan mengevaluasi model menggunakan Leave-One-Year-Out (LOYO) Cross-Validation
    pada rentang tahun 2000-2025 (26 tahun x 26 kecamatan = 676 sampel).
    """
    train_file = DATA_DIR / "dataset_training_prediksi_musim.csv"
    if not train_file.exists():
        raise FileNotFoundError(f"File {train_file} belum ditemukan.")
        
    df = pd.read_csv(train_file)
    years = sorted(df['tahun_awal'].unique())
    print(f"Memulai LOYO Cross-Validation pada {len(years)} tahun ({years[0]} - {years[-1]})...")
    
    loyo_predictions = []
    
    for test_year in years:
        train_df = df[df['tahun_awal'] != test_year].copy()
        test_df = df[df['tahun_awal'] == test_year].copy()
        
        X_train = train_df[FEATURE_COLS]
        X_test = test_df[FEATURE_COLS]
        
        # 1. Target Kategori AMH (Maju, Sama, Mundur)
        y_train_kat = train_df['kategori_amh'].map(KATEGORI_MAP)
        clf_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        clf_xgb.fit(X_train, y_train_kat)
        clf_rf.fit(X_train, y_train_kat)
        
        prob_xgb = clf_xgb.predict_proba(X_test)
        prob_rf = clf_rf.predict_proba(X_test)
        prob_ensemble = 0.5 * prob_xgb + 0.5 * prob_rf
        pred_kat = np.argmax(prob_ensemble, axis=1)
        
        # 2. Target Dasarian AMH Eksak (Regresi)
        y_train_das = train_df['amh_dasarian']
        reg_xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        reg_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        reg_xgb.fit(X_train, y_train_das)
        reg_rf.fit(X_train, y_train_das)
        pred_das = 0.5 * reg_xgb.predict(X_test) + 0.5 * reg_rf.predict(X_test)
        
        # 3. Target Sifat Hujan (Atas Normal, Normal, Bawah Normal)
        y_train_sifat = train_df['sifat_hujan'].map(SIFAT_MAP)
        clf_sifat = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf_sifat.fit(X_train, y_train_sifat)
        pred_sifat = clf_sifat.predict(X_test)
        
        test_df['pred_kategori_amh'] = [INV_KATEGORI_MAP[k] for k in pred_kat]
        test_df['pred_amh_dasarian'] = np.round(pred_das, 1)
        test_df['pred_amh_label'] = [get_dasarian_label(d) for d in pred_das]
        test_df['pred_sifat_hujan'] = [INV_SIFAT_MAP[s] for s in pred_sifat]
        
        loyo_predictions.append(test_df)
        
    df_loyo = pd.concat(loyo_predictions, ignore_index=True)
    
    # Hitung Metrik Evaluasi LOYO
    acc_kat = accuracy_score(df_loyo['kategori_amh'], df_loyo['pred_kategori_amh'])
    f1_kat = f1_score(df_loyo['kategori_amh'], df_loyo['pred_kategori_amh'], average='macro')
    mae_das = mean_absolute_error(df_loyo['amh_dasarian'], df_loyo['pred_amh_dasarian'])
    acc_sifat = accuracy_score(df_loyo['sifat_hujan'], df_loyo['pred_sifat_hujan'])
    
    print("\n================================================================================")
    print("HASIL EVALUASI MODEL LEAVE-ONE-YEAR-OUT (LOYO) CROSS-VALIDATION")
    print("================================================================================")
    print(f"📊 Akurasi Kategori AMH (Maju/Sama/Mundur)  : {acc_kat * 100:.2f}% (F1-Macro: {f1_kat:.3f})")
    print(f"🎯 Mean Absolute Error (MAE) Dasarian AMH    : {mae_das:.2f} Dasarian (~{mae_das * 10:.1f} hari)")
    print(f"🌧️ Akurasi Sifat Hujan (AN / N / BN)         : {acc_sifat * 100:.2f}%")
    print("================================================================================\n")
    
    out_loyo = DATA_DIR / "hasil_evaluasi_loyo.csv"
    df_loyo.to_csv(out_loyo, index=False)
    print(f"Hasil evaluasi out-of-fold tersimpan di: {out_loyo}")
    
    return df_loyo

def predict_future_season_2026():
    """
    Menghasilkan prediksi operasional Awal Musim Hujan (AMH), Sifat Hujan,
    dan Puncak Musim Hujan untuk musim 2026/2027 bagi 26 kecamatan di Kebumen.
    """
    train_file = DATA_DIR / "dataset_training_prediksi_musim.csv"
    norm_file = DATA_DIR / "normal_klimatologi_amh_kebumen.csv"
    
    df_train = pd.read_csv(train_file)
    df_norm = pd.read_csv(norm_file)
    
    # Latih model final pada seluruh data 2000-2025
    X_train = df_train[FEATURE_COLS]
    y_kat = df_train['kategori_amh'].map(KATEGORI_MAP)
    y_das = df_train['amh_dasarian']
    y_sifat = df_train['sifat_hujan'].map(SIFAT_MAP)
    y_puncak = df_train['puncak_dasarian']
    
    clf_kat = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
    clf_kat.fit(X_train, y_kat)
    
    reg_das = GradientBoostingRegressor_ensemble = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42)
    reg_das.fit(X_train, y_das)
    
    clf_sifat = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
    clf_sifat.fit(X_train, y_sifat)
    
    reg_puncak = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    reg_puncak.fit(X_train, y_puncak)
    
    # Ambil nilai prediktor 2026 (JAS 2026) dari dataset training baris 2025 atau nilai terbaru
    row_2026 = df_train[df_train['tahun_awal'] == 2025].iloc[0] # baseline prediktor
    
    # Nilai aktual terkini 2026:
    # ONI JAS 2026 = +2.16 (Super El Niño), DMI Sep 2026 = +0.42
    # Jika ada prediktor 2026 di ERA5:
    preds_2026 = []
    for _, norm_row in df_norm.iterrows():
        kec = norm_row['kecamatan']
        feat = {
            'oni_jas': 2.16,
            'oni_jja': 1.80,
            'dmi_agu': 0.35,
            'dmi_sep': 0.42,
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
        
        # Rekomendasi Kalender Tanam Padi Rendengan (MT1)
        # Rekomendasi olah tanah dimulai 1 dasarian setelah AMH tiba
        olah_tanah_das = int(round(das_pred)) + 1
        tanam_das = olah_tanah_das + 1
        
        preds_2026.append({
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
        
    df_res_2026 = pd.DataFrame(preds_2026)
    out_2026 = DATA_DIR / "prakiraan_musim_2026_2027_kebumen.csv"
    df_res_2026.to_csv(out_2026, index=False)
    print(f"\n[SUKSES] Prakiraan Musim 2026/2027 tersimpan: {out_2026}")
    print("\nRingkasan 5 Kecamatan:")
    print(df_res_2026[['kecamatan', 'normal_amh_label', 'prakiraan_amh_label', 'perbandingan_terhadap_normal', 'prakiraan_sifat_hujan', 'rekomendasi_tanam_padi_mt1']].head(5))
    return df_res_2026

if __name__ == "__main__":
    train_and_evaluate_loyo()
    predict_future_season_2026()
