#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Pelatihan Model & Evaluasi Leave-One-Year-Out (LOYO) Kabupaten Purworejo (2000-2025)
Stasiun Meteorologi Kelas III Jerukagung
Membangun dataset training prediktor iklim (ENSO, IOD, ERA5-Land, Normal Lokal)
dan mengeksekusi LOYO Cross-Validation untuk Musim Hujan dan Musim Kemarau.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"

KATEGORI_MAP = {'Maju': 0, 'Sama': 1, 'Mundur': 2}
INV_KATEGORI_MAP = {0: 'Maju', 1: 'Sama', 2: 'Mundur'}
SIFAT_MAP = {'Bawah Normal': 0, 'Normal': 1, 'Atas Normal': 2}
INV_SIFAT_MAP = {0: 'Bawah Normal', 1: 'Normal', 2: 'Atas Normal'}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
FULL_MONTH_NAMES = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
ROMAN = ["I", "II", "III"]

def get_dasarian_label(d_idx):
    d_round = int(round(d_idx))
    m = ((d_round - 1) // 3) + 1
    d_m = ((d_round - 1) % 3) + 1
    if 1 <= m <= 12:
        return f"{MONTH_NAMES[m-1]} {ROMAN[d_m-1]}"
    return f"Dasarian {d_round}"

def get_month_name(d_idx):
    d_round = int(round(d_idx))
    m = ((d_round - 1) // 3) + 1
    if 1 <= m <= 12:
        return FULL_MONTH_NAMES[m-1]
    return "Januari"

def build_training_and_loyo_hujan():
    print("\n" + "="*80)
    print("MEMBANGUN DATASET & EVALUASI LOYO MUSIM HUJAN PURWOREJO (2000-2025)")
    print("="*80)

    df_amh = pd.read_csv(DATA_DIR / "amh_historis_purworejo_2000_2025.csv")
    df_norm = pd.read_csv(DATA_DIR / "normal_klimatologi_amh_purworejo.csv")
    df_train_kebumen = pd.read_csv(DATA_DIR / "dataset_training_prediksi_musim.csv")

    # Ambil prediktor iklim (ENSO, IOD, ERA5) unik per tahun dari dataset Kebumen
    climate_cols = [
        'tahun_awal', 'oni_jas', 'oni_jja', 'dmi_agu', 'dmi_sep',
        'u10_m7', 'rh_m7', 't2m_m7',
        'u10_m8', 'rh_m8', 't2m_m8',
        'u10_m9', 'rh_m9', 't2m_m9'
    ]
    df_climate = df_train_kebumen[climate_cols].drop_duplicates(subset=['tahun_awal'])

    # Gabung AMH historis dengan Normal
    df_merged = df_amh.merge(
        df_norm[['kecamatan', 'normal_amh_dasarian', 'normal_total_hujan_mh_mm']],
        on='kecamatan',
        how='left'
    )

    df_merged['selisih_dasarian'] = df_merged['amh_dasarian'] - df_merged['normal_amh_dasarian']
    
    def cat_amh(diff):
        if diff <= -1.0:
            return "Maju"
        elif diff >= 1.0:
            return "Mundur"
        return "Sama"
    df_merged['kategori_amh'] = df_merged['selisih_dasarian'].apply(cat_amh)

    df_merged['persentase_normal_hujan'] = (df_merged['total_hujan_mh_mm'] / df_merged['normal_total_hujan_mh_mm']) * 100.0
    def cat_sifat(pct):
        if pct < 85.0:
            return "Bawah Normal"
        elif pct > 115.0:
            return "Atas Normal"
        return "Normal"
    df_merged['sifat_hujan'] = df_merged['persentase_normal_hujan'].apply(cat_sifat)

    # Gabung dengan prediktor iklim
    df_merged = df_merged.merge(df_climate, on='tahun_awal', how='left')

    out_train = DATA_DIR / "dataset_training_prediksi_musim_purworejo.csv"
    df_merged.to_csv(out_train, index=False)
    print(f"Dataset training tersimpan di: {out_train} ({len(df_merged)} baris).")

    # Jalankan LOYO Cross-Validation
    feature_cols = [
        'oni_jas', 'oni_jja', 'dmi_agu', 'dmi_sep',
        'u10_m7', 'rh_m7', 't2m_m7',
        'u10_m8', 'rh_m8', 't2m_m8',
        'u10_m9', 'rh_m9', 't2m_m9',
        'normal_amh_dasarian', 'normal_total_hujan_mh_mm'
    ]

    years = sorted(df_merged['tahun_awal'].unique())
    loyo_rows = []

    for test_year in years:
        train_df = df_merged[df_merged['tahun_awal'] != test_year].copy()
        test_df = df_merged[df_merged['tahun_awal'] == test_year].copy()

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        # 1. Kategori AMH
        y_train_kat = train_df['kategori_amh'].map(KATEGORI_MAP)
        clf_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf_xgb.fit(X_train, y_train_kat)
        clf_rf.fit(X_train, y_train_kat)
        prob = 0.5 * clf_xgb.predict_proba(X_test) + 0.5 * clf_rf.predict_proba(X_test)
        pred_kat = np.argmax(prob, axis=1)

        # 2. Dasarian AMH
        y_train_das = train_df['amh_dasarian']
        reg_xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        reg_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        reg_xgb.fit(X_train, y_train_das)
        reg_rf.fit(X_train, y_train_das)
        pred_das = 0.5 * reg_xgb.predict(X_test) + 0.5 * reg_rf.predict(X_test)

        # 3. Sifat Hujan
        y_train_sifat = train_df['sifat_hujan'].map(SIFAT_MAP)
        clf_sifat = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf_sifat.fit(X_train, y_train_sifat)
        pred_sifat = clf_sifat.predict(X_test)

        test_df['pred_kategori_amh'] = [INV_KATEGORI_MAP[k] for k in pred_kat]
        test_df['pred_amh_dasarian'] = np.round(pred_das, 1)
        test_df['pred_amh_label'] = [get_dasarian_label(d) for d in pred_das]
        test_df['pred_sifat_hujan'] = [INV_SIFAT_MAP[s] for s in pred_sifat]

        loyo_rows.append(test_df)

    df_loyo_h = pd.concat(loyo_rows, ignore_index=True)
    out_loyo_h = DATA_DIR / "hasil_evaluasi_loyo_purworejo.csv"
    df_loyo_h.to_csv(out_loyo_h, index=False)

    acc_kat = accuracy_score(df_loyo_h['kategori_amh'], df_loyo_h['pred_kategori_amh'])
    f1_kat = f1_score(df_loyo_h['kategori_amh'], df_loyo_h['pred_kategori_amh'], average='macro')
    mae_das = mean_absolute_error(df_loyo_h['amh_dasarian'], df_loyo_h['pred_amh_dasarian'])
    acc_sifat = accuracy_score(df_loyo_h['sifat_hujan'], df_loyo_h['pred_sifat_hujan'])

    print(f"📊 Akurasi Kategori AMH (Maju/Sama/Mundur)  : {acc_kat * 100:.2f}% (F1-Macro: {f1_kat:.3f})")
    print(f"🎯 Mean Absolute Error (MAE) Dasarian AMH    : {mae_das:.2f} Dasarian (~{mae_das * 10:.1f} hari)")
    print(f"🌧️ Akurasi Sifat Hujan (AN / N / BN)         : {acc_sifat * 100:.2f}%")
    print(f"Tersimpan di: {out_loyo_h}")
    return df_loyo_h

def build_training_and_loyo_kemarau():
    print("\n" + "="*80)
    print("MEMBANGUN DATASET & EVALUASI LOYO MUSIM KEMARAU PURWOREJO (2000-2025)")
    print("="*80)

    df_amk = pd.read_csv(DATA_DIR / "amk_historis_purworejo_2000_2025.csv")
    df_norm = pd.read_csv(DATA_DIR / "normal_klimatologi_amk_purworejo.csv")
    df_train_kebumen_k = pd.read_csv(DATA_DIR / "dataset_training_prediksi_kemarau.csv")

    climate_cols_k = [
        'tahun', 'oni_djf', 'oni_jfm', 'dmi_feb', 'dmi_mar',
        'u10_m1', 'rh_m1', 't2m_m1',
        'u10_m2', 'rh_m2', 't2m_m2',
        'u10_m3', 'rh_m3', 't2m_m3'
    ]
    df_climate_k = df_train_kebumen_k[climate_cols_k].drop_duplicates(subset=['tahun'])

    df_merged = df_amk.merge(
        df_norm[['kecamatan', 'normal_amk_dasarian', 'normal_total_hujan_mk_mm']],
        on='kecamatan',
        how='left'
    )

    df_merged['selisih_dasarian'] = df_merged['amk_dasarian'] - df_merged['normal_amk_dasarian']
    def cat_amk(diff):
        if diff <= -1.0:
            return "Maju"
        elif diff >= 1.0:
            return "Mundur"
        return "Sama"
    df_merged['kategori_amk'] = df_merged['selisih_dasarian'].apply(cat_amk)

    df_merged['persentase_normal_hujan'] = (df_merged['total_hujan_mk_mm'] / df_merged['normal_total_hujan_mk_mm']) * 100.0
    def cat_sifat(pct):
        if pct < 85.0:
            return "Bawah Normal"
        elif pct > 115.0:
            return "Atas Normal"
        return "Normal"
    df_merged['sifat_hujan_mk'] = df_merged['persentase_normal_hujan'].apply(cat_sifat)

    df_merged = df_merged.merge(df_climate_k, on='tahun', how='left')

    out_train_k = DATA_DIR / "dataset_training_prediksi_kemarau_purworejo.csv"
    df_merged.to_csv(out_train_k, index=False)
    print(f"Dataset training kemarau tersimpan di: {out_train_k} ({len(df_merged)} baris).")

    feature_cols = [
        'oni_djf', 'oni_jfm', 'dmi_feb', 'dmi_mar',
        'u10_m1', 'rh_m1', 't2m_m1',
        'u10_m2', 'rh_m2', 't2m_m2',
        'u10_m3', 'rh_m3', 't2m_m3',
        'normal_amk_dasarian', 'normal_total_hujan_mk_mm'
    ]

    years = sorted(df_merged['tahun'].unique())
    loyo_rows = []

    for test_year in years:
        train_df = df_merged[df_merged['tahun'] != test_year].copy()
        test_df = df_merged[df_merged['tahun'] == test_year].copy()

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        # 1. Kategori AMK
        y_train_kat = train_df['kategori_amk'].map(KATEGORI_MAP)
        clf_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf_xgb.fit(X_train, y_train_kat)
        clf_rf.fit(X_train, y_train_kat)
        prob = 0.5 * clf_xgb.predict_proba(X_test) + 0.5 * clf_rf.predict_proba(X_test)
        pred_kat = np.argmax(prob, axis=1)

        # 2. Dasarian AMK
        y_train_das = train_df['amk_dasarian']
        reg_xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        reg_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        reg_xgb.fit(X_train, y_train_das)
        reg_rf.fit(X_train, y_train_das)
        pred_das = 0.5 * reg_xgb.predict(X_test) + 0.5 * reg_rf.predict(X_test)

        # 3. Sifat Hujan MK
        y_train_sifat = train_df['sifat_hujan_mk'].map(SIFAT_MAP)
        clf_sifat = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf_sifat.fit(X_train, y_train_sifat)
        pred_sifat = clf_sifat.predict(X_test)

        # 4. Puncak Kemarau (PMK)
        y_train_pmk = train_df['pmk_dasarian']
        reg_pmk = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
        reg_pmk.fit(X_train, y_train_pmk)
        pred_pmk = reg_pmk.predict(X_test)

        res_df = pd.DataFrame({
            'tahun': test_df['tahun'].values,
            'kecamatan': test_df['kecamatan'].values,
            'kategori_amk': test_df['kategori_amk'].values,
            'amk_dasarian': test_df['amk_dasarian'].values,
            'sifat_hujan_mk': test_df['sifat_hujan_mk'].values,
            'pmk_dasarian': test_df['pmk_dasarian'].values,
            'normal_amk_dasarian': test_df['normal_amk_dasarian'].values,
            'normal_total_hujan_mk_mm': test_df['normal_total_hujan_mk_mm'].values,
            'pred_kat_lstm': [INV_KATEGORI_MAP[k] for k in pred_kat],
            'pred_amk_das_lstm': np.round(pred_das, 1),
            'pred_sifat_lstm': [INV_SIFAT_MAP[s] for s in pred_sifat],
            'pred_pmk_das_lstm': np.round(pred_pmk, 1)
        })

        loyo_rows.append(res_df)

    df_loyo_k = pd.concat(loyo_rows, ignore_index=True)
    out_loyo_k = DATA_DIR / "hasil_evaluasi_loyo_kemarau_purworejo.csv"
    df_loyo_k.to_csv(out_loyo_k, index=False)

    acc_kat = accuracy_score(df_loyo_k['kategori_amk'], df_loyo_k['pred_kat_lstm'])
    f1_kat = f1_score(df_loyo_k['kategori_amk'], df_loyo_k['pred_kat_lstm'], average='macro')
    mae_das = mean_absolute_error(df_loyo_k['amk_dasarian'], df_loyo_k['pred_amk_das_lstm'])
    acc_sifat = accuracy_score(df_loyo_k['sifat_hujan_mk'], df_loyo_k['pred_sifat_lstm'])

    print(f"📊 Akurasi Kategori AMK (Maju/Sama/Mundur)  : {acc_kat * 100:.2f}% (F1-Macro: {f1_kat:.3f})")
    print(f"🎯 Mean Absolute Error (MAE) Dasarian AMK    : {mae_das:.2f} Dasarian (~{mae_das * 10:.1f} hari)")
    print(f"🌧️ Akurasi Sifat Hujan MK (AN / N / BN)      : {acc_sifat * 100:.2f}%")
    print(f"Tersimpan di: {out_loyo_k}")
    return df_loyo_k

if __name__ == "__main__":
    build_training_and_loyo_hujan()
    build_training_and_loyo_kemarau()
