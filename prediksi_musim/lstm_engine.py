#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Deep Learning Sekuensial (LSTM / Bi-LSTM / GRU) & Benchmark Prediksi Musim Hujan Kebumen
Memodelkan dinamika temporal deret waktu pra-musim 27 Dasarian (Januari - September)
Menggunakan protokol Leave-One-Year-Out (LOYO) 2000-2025 (676 sampel out-of-fold).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Matikan pesan info berlebih TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42)
np.random.seed(42)

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"

KATEGORI_MAP = {'Maju': 0, 'Sama': 1, 'Mundur': 2}
INV_KATEGORI_MAP = {0: 'Maju', 1: 'Sama', 2: 'Mundur'}
SIFAT_MAP = {'Bawah Normal': 0, 'Normal': 1, 'Atas Normal': 2}
INV_SIFAT_MAP = {0: 'Bawah Normal', 1: 'Normal', 2: 'Atas Normal'}

def build_sequential_dataset():
    """
    Membangun tensor 3-dimensi (samples, 27 dasarians, n_features)
    menggabungkan CHIRPS dasarian harian, indeks ENSO/IOD bulanan, dan atmosfer ERA5-Land.
    """
    df_das = pd.read_csv(DATA_DIR / "dasarian_curah_hujan_kebumen_2000_2026.csv")
    df_train = pd.read_csv(DATA_DIR / "dataset_training_prediksi_musim.csv")
    df_norm = pd.read_csv(DATA_DIR / "normal_klimatologi_amh_kebumen.csv")
    
    # Hitung rata-rata normal curah hujan per dasarian (1..36) per kecamatan
    norm_das = df_das[df_das['tahun'] <= 2025].groupby(['kecamatan', 'dasarian'])['curah_hujan'].mean().rename('normal_hujan_dasarian').reset_index()
    df_das = df_das.merge(norm_das, on=['kecamatan', 'dasarian'], how='left')
    
    # Ambil 27 dasarian pertama (Januari I s/d September III)
    df_pre = df_das[df_das['dasarian'] <= 27].copy()
    
    samples_X = []
    samples_meta = []
    
    years = sorted(df_train['tahun_awal'].unique()) # 2000..2025
    kecamatan_list = sorted(df_train['kecamatan'].unique()) # 26 kecamatan
    
    print(f"Membangun dataset sekuensial untuk {len(years)} tahun x {len(kecamatan_list)} kecamatan = {len(years)*len(kecamatan_list)} sampel...")
    
    for yr in years:
        for kec in kecamatan_list:
            # Ambil target dari df_train
            row_target = df_train[(df_train['tahun_awal'] == yr) & (df_train['kecamatan'] == kec)].iloc[0]
            
            # Ambil 27 dasarian tahun yr untuk kecamatan kec
            sub = df_pre[(df_pre['tahun'] == yr) & (df_pre['kecamatan'] == kec)].sort_values('dasarian')
            
            if len(sub) < 27:
                continue
                
            # Fitur sekuensial:
            # 1. curah hujan aktual
            # 2. anomali curah hujan terhadap normal dasarian
            # 3. persentase terhadap normal dasarian
            # 4. indeks ONI (Nino 3.4) yang dipetakan per bulan dasarian
            # 5. indeks DMI (IOD) yang dipetakan per bulan dasarian
            # 6. normal amh dasarian kecamatan
            # 7. normal total curah hujan musim hujan
            
            features_seq = []
            for _, r in sub.iterrows():
                m = int(r['bulan'])
                # Pemetaan aproksimasi ONI & DMI
                oni_val = row_target['oni_jas'] if m >= 7 else (row_target['oni_jja'] if m >= 5 else 0.0)
                dmi_val = row_target['dmi_sep'] if m >= 8 else (row_target['dmi_agu'] if m >= 6 else 0.0)
                
                features_seq.append([
                    r['curah_hujan'],
                    r['curah_hujan'] - r['normal_hujan_dasarian'],
                    (r['curah_hujan'] + 1.0) / (r['normal_hujan_dasarian'] + 1.0),
                    oni_val,
                    dmi_val,
                    row_target['normal_amh_dasarian'],
                    row_target['normal_total_hujan_mh_mm']
                ])
                
            samples_X.append(features_seq)
            samples_meta.append({
                'tahun_awal': yr,
                'kecamatan': kec,
                'kategori_amh': row_target['kategori_amh'],
                'amh_dasarian': row_target['amh_dasarian'],
                'sifat_hujan': row_target['sifat_hujan'],
                'puncak_dasarian': row_target['puncak_dasarian']
            })
            
    X = np.array(samples_X, dtype=np.float32) # (676, 27, 7)
    df_meta = pd.DataFrame(samples_meta)
    print(f"Tensor Sekuensial terbentuk: Bentuk X = {X.shape}, Metadata = {len(df_meta)} baris.")
    return X, df_meta

def create_lstm_model(input_shape):
    """Membangun arsitektur LSTM Multi-Head."""
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(48, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(24)(x)
    shared = layers.Dense(24, activation='relu')(x)
    
    out_kat = layers.Dense(3, activation='softmax', name='kategori')(shared)
    out_das = layers.Dense(1, activation='linear', name='dasarian')(shared)
    out_sifat = layers.Dense(3, activation='softmax', name='sifat')(shared)
    
    model = models.Model(inputs=inp, outputs=[out_kat, out_das, out_sifat])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss={
            'kategori': 'sparse_categorical_crossentropy',
            'dasarian': 'huber',
            'sifat': 'sparse_categorical_crossentropy'
        },
        loss_weights={'kategori': 1.0, 'dasarian': 0.1, 'sifat': 1.0}
    )
    return model

def create_bilstm_model(input_shape):
    """Membangun arsitektur Bi-LSTM Multi-Head."""
    inp = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(16))(x)
    shared = layers.Dense(24, activation='relu')(x)
    
    out_kat = layers.Dense(3, activation='softmax', name='kategori')(shared)
    out_das = layers.Dense(1, activation='linear', name='dasarian')(shared)
    out_sifat = layers.Dense(3, activation='softmax', name='sifat')(shared)
    
    model = models.Model(inputs=inp, outputs=[out_kat, out_das, out_sifat])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss={
            'kategori': 'sparse_categorical_crossentropy',
            'dasarian': 'huber',
            'sifat': 'sparse_categorical_crossentropy'
        },
        loss_weights={'kategori': 1.0, 'dasarian': 0.1, 'sifat': 1.0}
    )
    return model

def create_gru_model(input_shape):
    """Membangun arsitektur GRU Multi-Head."""
    inp = layers.Input(shape=input_shape)
    x = layers.GRU(48, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.GRU(24)(x)
    shared = layers.Dense(24, activation='relu')(x)
    
    out_kat = layers.Dense(3, activation='softmax', name='kategori')(shared)
    out_das = layers.Dense(1, activation='linear', name='dasarian')(shared)
    out_sifat = layers.Dense(3, activation='softmax', name='sifat')(shared)
    
    model = models.Model(inputs=inp, outputs=[out_kat, out_das, out_sifat])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss={
            'kategori': 'sparse_categorical_crossentropy',
            'dasarian': 'huber',
            'sifat': 'sparse_categorical_crossentropy'
        },
        loss_weights={'kategori': 1.0, 'dasarian': 0.1, 'sifat': 1.0}
    )
    return model

def run_loyo_benchmark():
    """
    Mengevaluasi dan membandingkan 5 algoritma secara komparatif:
    1. Baseline Random Forest & XGBoost (Tabular)
    2. LightGBM (Gradient Boosting)
    3. LSTM (Deep Learning)
    4. Bi-LSTM (Deep Learning)
    5. GRU (Deep Learning)
    """
    X_seq, df_meta = build_sequential_dataset()
    years = sorted(df_meta['tahun_awal'].unique())
    n_samples, time_steps, n_feats = X_seq.shape
    
    # Flatten X untuk LightGBM
    X_flat = X_seq.reshape(n_samples, time_steps * n_feats)
    
    # Matriks Prediksi OOF untuk tiap model
    preds_lgb = {'kat': [], 'das': [], 'sifat': []}
    preds_gru = {'kat': [], 'das': [], 'sifat': []}
    preds_lstm = {'kat': [], 'das': [], 'sifat': []}
    preds_bilstm = {'kat': [], 'das': [], 'sifat': []}
    
    print(f"\nMemulai LOYO Benchmark pada {len(years)} Tahun...")
    
    for test_year in years:
        idx_train = df_meta[df_meta['tahun_awal'] != test_year].index
        idx_test = df_meta[df_meta['tahun_awal'] == test_year].index
        
        # Standarisasi fitur per fold (Rule 12: Zero Data Leakage)
        scaler = StandardScaler()
        X_train_seq = scaler.fit_transform(X_seq[idx_train].reshape(-1, n_feats)).reshape(-1, time_steps, n_feats)
        X_test_seq = scaler.transform(X_seq[idx_test].reshape(-1, n_feats)).reshape(-1, time_steps, n_feats)
        
        X_train_flat = X_flat[idx_train]
        X_test_flat = X_flat[idx_test]
        
        y_train_kat = df_meta.loc[idx_train, 'kategori_amh'].map(KATEGORI_MAP).values
        y_train_das = df_meta.loc[idx_train, 'amh_dasarian'].values
        y_train_sifat = df_meta.loc[idx_train, 'sifat_hujan'].map(SIFAT_MAP).values
        
        # 1. Model LightGBM
        clf_lgb_kat = lgb.LGBMClassifier(n_estimators=60, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
        clf_lgb_kat.fit(X_train_flat, y_train_kat)
        p_lgb_kat = clf_lgb_kat.predict(X_test_flat)
        
        reg_lgb_das = lgb.LGBMRegressor(n_estimators=60, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
        reg_lgb_das.fit(X_train_flat, y_train_das)
        p_lgb_das = reg_lgb_das.predict(X_test_flat)
        
        clf_lgb_sifat = lgb.LGBMClassifier(n_estimators=60, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
        clf_lgb_sifat.fit(X_train_flat, y_train_sifat)
        p_lgb_sifat = clf_lgb_sifat.predict(X_test_flat)
        
        preds_lgb['kat'].extend(p_lgb_kat)
        preds_lgb['das'].extend(p_lgb_das)
        preds_lgb['sifat'].extend(p_lgb_sifat)
        
        # 2. Model GRU
        model_gru = create_gru_model((time_steps, n_feats))
        model_gru.fit(
            X_train_seq,
            {'kategori': y_train_kat, 'dasarian': y_train_das, 'sifat': y_train_sifat},
            epochs=25, batch_size=32, verbose=0
        )
        p_gru = model_gru.predict(X_test_seq, verbose=0)
        preds_gru['kat'].extend(np.argmax(p_gru[0], axis=1))
        preds_gru['das'].extend(p_gru[1].flatten())
        preds_gru['sifat'].extend(np.argmax(p_gru[2], axis=1))
        
        # 3. Model LSTM
        model_lstm = create_lstm_model((time_steps, n_feats))
        model_lstm.fit(
            X_train_seq,
            {'kategori': y_train_kat, 'dasarian': y_train_das, 'sifat': y_train_sifat},
            epochs=25, batch_size=32, verbose=0
        )
        p_lstm = model_lstm.predict(X_test_seq, verbose=0)
        preds_lstm['kat'].extend(np.argmax(p_lstm[0], axis=1))
        preds_lstm['das'].extend(p_lstm[1].flatten())
        preds_lstm['sifat'].extend(np.argmax(p_lstm[2], axis=1))
        
        # 4. Model Bi-LSTM
        model_bilstm = create_bilstm_model((time_steps, n_feats))
        model_bilstm.fit(
            X_train_seq,
            {'kategori': y_train_kat, 'dasarian': y_train_das, 'sifat': y_train_sifat},
            epochs=25, batch_size=32, verbose=0
        )
        p_bilstm = model_bilstm.predict(X_test_seq, verbose=0)
        preds_bilstm['kat'].extend(np.argmax(p_bilstm[0], axis=1))
        preds_bilstm['das'].extend(p_bilstm[1].flatten())
        preds_bilstm['sifat'].extend(np.argmax(p_bilstm[2], axis=1))
        
    y_true_kat = df_meta['kategori_amh'].map(KATEGORI_MAP).values
    y_true_das = df_meta['amh_dasarian'].values
    y_true_sifat = df_meta['sifat_hujan'].map(SIFAT_MAP).values
    
    # Ambil hasil baseline RF+XGBoost yang sudah dievaluasi sebelumnya
    df_loyo_base = pd.read_csv(DATA_DIR / "hasil_evaluasi_loyo.csv")
    acc_base_kat = accuracy_score(df_loyo_base['kategori_amh'], df_loyo_base['pred_kategori_amh'])
    f1_base_kat = f1_score(df_loyo_base['kategori_amh'], df_loyo_base['pred_kategori_amh'], average='macro')
    mae_base_das = mean_absolute_error(df_loyo_base['amh_dasarian'], df_loyo_base['pred_amh_dasarian'])
    acc_base_sifat = accuracy_score(df_loyo_base['sifat_hujan'], df_loyo_base['pred_sifat_hujan'])
    
    # Hitung metrik model baru
    models_dict = {
        'Baseline RF + XGBoost (Tabular)': (acc_base_kat, f1_base_kat, mae_base_das, acc_base_sifat),
        'LightGBM Regressor / Classifier': (
            accuracy_score(y_true_kat, preds_lgb['kat']),
            f1_score(y_true_kat, preds_lgb['kat'], average='macro'),
            mean_absolute_error(y_true_das, preds_lgb['das']),
            accuracy_score(y_true_sifat, preds_lgb['sifat'])
        ),
        'LSTM Multi-Head (Sequential Deep Learning)': (
            accuracy_score(y_true_kat, preds_lstm['kat']),
            f1_score(y_true_kat, preds_lstm['kat'], average='macro'),
            mean_absolute_error(y_true_das, preds_lstm['das']),
            accuracy_score(y_true_sifat, preds_lstm['sifat'])
        ),
        'Bi-LSTM Multi-Head (Bidirectional RNN)': (
            accuracy_score(y_true_kat, preds_bilstm['kat']),
            f1_score(y_true_kat, preds_bilstm['kat'], average='macro'),
            mean_absolute_error(y_true_das, preds_bilstm['das']),
            accuracy_score(y_true_sifat, preds_bilstm['sifat'])
        ),
        'GRU Multi-Head (Gated Recurrent Unit)': (
            accuracy_score(y_true_kat, preds_gru['kat']),
            f1_score(y_true_kat, preds_gru['kat'], average='macro'),
            mean_absolute_error(y_true_das, preds_gru['das']),
            accuracy_score(y_true_sifat, preds_gru['sifat'])
        )
    }
    
    res_rows = []
    for name, (acc_k, f1_k, mae_d, acc_s) in models_dict.items():
        res_rows.append({
            'Algoritma': name,
            'Akurasi_Kategori_AMH_Pct': round(acc_k * 100, 2),
            'F1_Macro_Kategori': round(f1_k, 3),
            'MAE_Dasarian_AMH': round(mae_d, 2),
            'MAE_Hari_AMH': round(mae_d * 10, 1),
            'Akurasi_Sifat_Hujan_Pct': round(acc_s * 100, 2)
        })
        
    df_compare = pd.DataFrame(res_rows)
    out_csv = DATA_DIR / "perbandingan_algoritma_musim.csv"
    df_compare.to_csv(out_csv, index=False)
    
    print("\n" + "="*95)
    print("HASIL BENCHMARK KOMPARATIF ALGORITMA PREDIKSI MUSIM HUJAN KEBUMEN (LOYO 2000-2025)")
    print("="*95)
    print(df_compare.to_string(index=False))
    print("="*95)
    print(f"\nTabel perbandingan tersimpan di: {out_csv}")
    
    return df_compare

if __name__ == "__main__":
    run_loyo_benchmark()

