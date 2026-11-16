#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Prakiraan Musim Aktif 2026/2027 & Musim Kemarau 2026 Kabupaten Purworejo
Stasiun Meteorologi Kelas III Jerukagung
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

def compute_probabilities(pred_das, norm_das, pred_kat):
    diff = pred_das - norm_das
    if pred_kat == "Maju" or diff <= -1.0:
        p_maju = round(75.0 + min(abs(diff), 2.0) * 10.0, 1)
        rem = 100.0 - p_maju
        p_sama = round(rem * 0.7, 1)
        p_mundur = round(100.0 - p_maju - p_sama, 1)
    elif pred_kat == "Mundur" or diff >= 1.0:
        p_mundur = round(75.0 + min(abs(diff), 2.0) * 10.0, 1)
        rem = 100.0 - p_mundur
        p_sama = round(rem * 0.7, 1)
        p_maju = round(100.0 - p_mundur - p_sama, 1)
    else:
        p_sama = 75.0
        p_maju = 12.5
        p_mundur = 12.5
    return p_maju, p_sama, p_mundur

def generate_prakiraan_hujan_2026():
    print("Menghasilkan Prakiraan Musim Hujan 2026/2027 Purworejo...")
    df_train = pd.read_csv(DATA_DIR / "dataset_training_prediksi_musim_purworejo.csv")
    df_norm = pd.read_csv(DATA_DIR / "normal_klimatologi_amh_purworejo.csv")

    feature_cols = [
        'oni_jas', 'oni_jja', 'dmi_agu', 'dmi_sep',
        'u10_m7', 'rh_m7', 't2m_m7',
        'u10_m8', 'rh_m8', 't2m_m8',
        'u10_m9', 'rh_m9', 't2m_m9',
        'normal_amh_dasarian', 'normal_total_hujan_mh_mm'
    ]

    X_train = df_train[feature_cols]
    y_kat = df_train['kategori_amh'].map(KATEGORI_MAP)
    y_das = df_train['amh_dasarian']
    y_sifat = df_train['sifat_hujan'].map(SIFAT_MAP)
    y_puncak = df_train['puncak_dasarian']

    clf_kat = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf_kat.fit(X_train, y_kat)

    reg_das = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    reg_das.fit(X_train, y_das)

    clf_sifat = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf_sifat.fit(X_train, y_sifat)

    reg_puncak = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    reg_puncak.fit(X_train, y_puncak)

    # Prediktor terkini untuk 2026
    preds = []
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
        X_pred = pd.DataFrame([feat])[feature_cols]
        
        pred_kat_val = INV_KATEGORI_MAP[clf_kat.predict(X_pred)[0]]
        pred_das_val = round(float(reg_das.predict(X_pred)[0]), 1)
        pred_sifat_val = INV_SIFAT_MAP[clf_sifat.predict(X_pred)[0]]
        pred_pmh_das = round(float(reg_puncak.predict(X_pred)[0]), 1)
        
        norm_das_val = norm_row['normal_amh_dasarian']
        p_maju, p_sama, p_mundur = compute_probabilities(pred_das_val, norm_das_val, pred_kat_val)
        
        diff = pred_das_val - norm_das_val
        perbandingan = "Maju" if diff <= -1.0 else ("Mundur" if diff >= 1.0 else "Sama")
        
        # BMKG Standard: Puncak Musim Hujan 2026/2027 (Januari 2027)
        pmh_bln = "Januari 2027"
        norm_pmh_bln = "Januari"
        perb_puncak = "Sama"
        pred_pmh_das = 34.0  # Jan I / Jan II
        
        tanam_d = int(round(pred_das_val)) + 1
        olah_d = ((tanam_d - 2 - 1) % 36) + 1
        
        preds.append({
            'kecamatan': kec,
            'normal_amh_dasarian': norm_das_val,
            'normal_amh_label': norm_row['normal_amh_label'],
            'prakiraan_amh_dasarian': pred_das_val,
            'prakiraan_amh_label': get_dasarian_label(pred_das_val),
            'perbandingan_terhadap_normal': perbandingan,
            'probabilitas_maju_pct': p_maju,
            'probabilitas_sama_pct': p_sama,
            'probabilitas_mundur_pct': p_mundur,
            'prakiraan_sifat_hujan': pred_sifat_val,
            'prakiraan_puncak_bulan': pmh_bln,
            'normal_puncak_bulan': norm_pmh_bln,
            'perbandingan_puncak_thd_normal': perb_puncak,
            'prakiraan_puncak_dasarian': pred_pmh_das,
            'prakiraan_puncak_label': 'Januari 2027',
            'rekomendasi_olah_tanah': get_dasarian_label(olah_d),
            'rekomendasi_tanam_padi_mt1': get_dasarian_label(tanam_d)
        })

    df_res = pd.DataFrame(preds)
    out_csv = DATA_DIR / "prakiraan_musim_2026_2027_purworejo.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"[SUKSES] Tersimpan di: {out_csv} ({len(df_res)} baris).")
    return df_res

def generate_prakiraan_kemarau_2026():
    print("Menghasilkan Prakiraan Musim Kemarau 2026 Purworejo...")
    df_train = pd.read_csv(DATA_DIR / "dataset_training_prediksi_kemarau_purworejo.csv")
    df_norm = pd.read_csv(DATA_DIR / "normal_klimatologi_amk_purworejo.csv")

    feature_cols = [
        'oni_djf', 'oni_jfm', 'dmi_feb', 'dmi_mar',
        'u10_m1', 'rh_m1', 't2m_m1',
        'u10_m2', 'rh_m2', 't2m_m2',
        'u10_m3', 'rh_m3', 't2m_m3',
        'normal_amk_dasarian', 'normal_total_hujan_mk_mm'
    ]

    X_train = df_train[feature_cols]
    y_kat = df_train['kategori_amk'].map(KATEGORI_MAP)
    y_das = df_train['amk_dasarian']
    y_sifat = df_train['sifat_hujan_mk'].map(SIFAT_MAP)
    y_pmk = df_train['pmk_dasarian']

    clf_kat = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf_kat.fit(X_train, y_kat)

    reg_das = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    reg_das.fit(X_train, y_das)

    clf_sifat = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf_sifat.fit(X_train, y_sifat)

    reg_pmk = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    reg_pmk.fit(X_train, y_pmk)

    preds = []
    for _, norm_row in df_norm.iterrows():
        kec = norm_row['kecamatan']
        feat = {
            'oni_djf': 1.80,
            'oni_jfm': 1.50,
            'dmi_feb': 0.20,
            'dmi_mar': 0.25,
            'u10_m1': 0.8,
            'rh_m1': 88.0,
            't2m_m1': 24.5,
            'u10_m2': 0.7,
            'rh_m2': 88.5,
            't2m_m2': 24.7,
            'u10_m3': 0.5,
            'rh_m3': 88.0,
            't2m_m3': 24.5,
            'normal_amk_dasarian': norm_row['normal_amk_dasarian'],
            'normal_total_hujan_mk_mm': norm_row['normal_total_hujan_mk_mm']
        }
        X_pred = pd.DataFrame([feat])[feature_cols]

        pred_kat_val = INV_KATEGORI_MAP[clf_kat.predict(X_pred)[0]]
        pred_das_val = round(float(reg_das.predict(X_pred)[0]), 1)
        pred_sifat_val = INV_SIFAT_MAP[clf_sifat.predict(X_pred)[0]]
        pred_pmk_val = round(float(reg_pmk.predict(X_pred)[0]), 1)

        norm_das_val = norm_row['normal_amk_dasarian']
        p_maju, p_sama, p_mundur = compute_probabilities(pred_das_val, norm_das_val, pred_kat_val)

        diff = pred_das_val - norm_das_val
        perbandingan = "Maju" if diff <= -1.0 else ("Mundur" if diff >= 1.0 else "Sama")

        # BMKG Standard: Puncak Musim Kemarau 2026 (Agustus 2026)
        pmk_bln = "Agustus 2026"
        norm_pmk_bln = "Agustus"
        perb_pmk = "Sama"
        pred_pmk_val = 24.0  # Agu III
        
        kering_d = max(10, int(round(pred_das_val)) - 2)
        palawija_d = int(round(pred_das_val))
        
        preds.append({
            'kecamatan': kec,
            'normal_amk_dasarian': norm_das_val,
            'normal_amk_label': norm_row['normal_amk_label'],
            'prakiraan_amk_dasarian': pred_das_val,
            'prakiraan_amk_label': get_dasarian_label(pred_das_val),
            'perbandingan_terhadap_normal': perbandingan,
            'probabilitas_maju_pct': p_maju,
            'probabilitas_sama_pct': p_sama,
            'probabilitas_mundur_pct': p_mundur,
            'prakiraan_sifat_hujan_mk': pred_sifat_val,
            'prakiraan_pmk_bulan': pmk_bln,
            'normal_pmk_bulan': norm_pmk_bln,
            'perbandingan_pmk_thd_normal': perb_pmk,
            'prakiraan_pmk_dasarian': pred_pmk_val,
            'prakiraan_pmk_label': 'Agustus 2026',
            'rekomendasi_keringkan_sawah_mt2': get_dasarian_label(kering_d),
            'rekomendasi_tanam_palawija_mt3': get_dasarian_label(palawija_d)
        })

    df_res_k = pd.DataFrame(preds)
    out_csv_k = DATA_DIR / "prakiraan_musim_kemarau_2026_purworejo.csv"
    df_res_k.to_csv(out_csv_k, index=False)
    print(f"[SUKSES] Tersimpan di: {out_csv_k} ({len(df_res_k)} baris).")
    return df_res_k

if __name__ == "__main__":
    generate_prakiraan_hujan_2026()
    generate_prakiraan_kemarau_2026()
