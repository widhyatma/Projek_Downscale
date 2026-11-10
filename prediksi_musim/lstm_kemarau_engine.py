#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Deep Learning Sekuensial LSTM & Benchmark Prediksi Musim Kemarau Kabupaten Kebumen
Memodelkan dinamika temporal deret waktu pra-kemarau 12 Dasarian (Januari I - April III)
Menggunakan protokol Leave-One-Year-Out (LOYO) 2000-2025 (676 sampel out-of-fold).
Menghasilkan prediksi Multi-Head: Kategori AMK, Dasarian AMK, Sifat Hujan MK, dan Puncak Kemarau (PMK).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Matikan log berlebih TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(42)
np.random.seed(42)

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"

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

def build_sequential_dataset_kemarau():
    """
    Membangun tensor 3-dimensi (676 sampel, 12 dasarian, 7 fitur)
    mencakup periode prakondisi Januari I s/d April III (Dasarian 1..12).
    """
    df_das = pd.read_csv(DATA_DIR / "dasarian_curah_hujan_kebumen_2000_2026.csv")
    df_train = pd.read_csv(DATA_DIR / "dataset_training_prediksi_kemarau.csv")
    
    # Hitung rata-rata normal curah hujan per dasarian (1..36) per kecamatan
    norm_das = df_das[df_das['tahun'] <= 2025].groupby(['kecamatan', 'dasarian'])['curah_hujan'].mean().rename('normal_hujan_dasarian').reset_index()
    df_das = df_das.merge(norm_das, on=['kecamatan', 'dasarian'], how='left')
    
    # Ambil 12 dasarian pertama (Januari I s/d April III)
    df_pre = df_das[df_das['dasarian'] <= 12].copy()
    
    samples_X = []
    samples_meta = []
    
    years = sorted(df_train['tahun'].unique()) # 2000..2025
    kecamatan_list = sorted(df_train['kecamatan'].unique()) # 26 kecamatan
    
    print(f"Membangun dataset sekuensial pra-kemarau untuk {len(years)} tahun x {len(kecamatan_list)} kecamatan = {len(years)*len(kecamatan_list)} sampel...")
    
    for yr in years:
        for kec in kecamatan_list:
            row_target = df_train[(df_train['tahun'] == yr) & (df_train['kecamatan'] == kec)].iloc[0]
            sub = df_pre[(df_pre['tahun'] == yr) & (df_pre['kecamatan'] == kec)].sort_values('dasarian')
            
            if len(sub) < 12:
                continue
                
            features_seq = []
            for _, r in sub.iterrows():
                m = int(r['bulan'])
                # Pemetaan indeks iklim global pada bulan berjalan
                oni_val = row_target['oni_jfm'] if m >= 3 else row_target['oni_djf']
                dmi_val = row_target['dmi_mar'] if m >= 3 else row_target['dmi_feb']
                
                features_seq.append([
                    r['curah_hujan'],
                    r['curah_hujan'] - r['normal_hujan_dasarian'],
                    (r['curah_hujan'] + 1.0) / (r['normal_hujan_dasarian'] + 1.0),
                    oni_val,
                    dmi_val,
                    row_target['normal_amk_dasarian'],
                    row_target['normal_total_hujan_mk_mm']
                ])
                
            samples_X.append(features_seq)
            samples_meta.append({
                'tahun': yr,
                'kecamatan': kec,
                'kategori_amk': row_target['kategori_amk'],
                'amk_dasarian': row_target['amk_dasarian'],
                'sifat_hujan_mk': row_target['sifat_hujan_mk'],
                'pmk_dasarian': row_target['pmk_dasarian'],
                'normal_amk_dasarian': row_target['normal_amk_dasarian'],
                'normal_total_hujan_mk_mm': row_target['normal_total_hujan_mk_mm']
            })
            
    X = np.array(samples_X, dtype=np.float32) # (676, 12, 7)
    df_meta = pd.DataFrame(samples_meta)
    print(f"Tensor Sekuensial Kemarau terbentuk: Bentuk X = {X.shape}, Metadata = {len(df_meta)} baris.")
    return X, df_meta

def create_lstm_kemarau_model(input_shape):
    """Membangun arsitektur LSTM Multi-Head untuk Prediksi Musim Kemarau."""
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(48, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(24)(x)
    shared = layers.Dense(24, activation='relu')(x)
    
    out_kat = layers.Dense(3, activation='softmax', name='kategori')(shared)
    out_das = layers.Dense(1, activation='linear', name='dasarian')(shared)
    out_sifat = layers.Dense(3, activation='softmax', name='sifat')(shared)
    out_pmk = layers.Dense(1, activation='linear', name='pmk')(shared)
    
    model = models.Model(inputs=inp, outputs=[out_kat, out_das, out_sifat, out_pmk])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss={
            'kategori': 'sparse_categorical_crossentropy',
            'dasarian': 'huber',
            'sifat': 'sparse_categorical_crossentropy',
            'pmk': 'huber'
        },
        loss_weights={'kategori': 1.0, 'dasarian': 0.1, 'sifat': 1.0, 'pmk': 0.1}
    )
    return model

def create_bilstm_kemarau_model(input_shape):
    """Membangun arsitektur Bi-LSTM Multi-Head."""
    inp = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(16))(x)
    shared = layers.Dense(24, activation='relu')(x)
    
    out_kat = layers.Dense(3, activation='softmax', name='kategori')(shared)
    out_das = layers.Dense(1, activation='linear', name='dasarian')(shared)
    out_sifat = layers.Dense(3, activation='softmax', name='sifat')(shared)
    out_pmk = layers.Dense(1, activation='linear', name='pmk')(shared)
    
    model = models.Model(inputs=inp, outputs=[out_kat, out_das, out_sifat, out_pmk])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss={
            'kategori': 'sparse_categorical_crossentropy',
            'dasarian': 'huber',
            'sifat': 'sparse_categorical_crossentropy',
            'pmk': 'huber'
        },
        loss_weights={'kategori': 1.0, 'dasarian': 0.1, 'sifat': 1.0, 'pmk': 0.1}
    )
    return model

def create_gru_kemarau_model(input_shape):
    """Membangun arsitektur GRU Multi-Head."""
    inp = layers.Input(shape=input_shape)
    x = layers.GRU(48, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.GRU(24)(x)
    shared = layers.Dense(24, activation='relu')(x)
    
    out_kat = layers.Dense(3, activation='softmax', name='kategori')(shared)
    out_das = layers.Dense(1, activation='linear', name='dasarian')(shared)
    out_sifat = layers.Dense(3, activation='softmax', name='sifat')(shared)
    out_pmk = layers.Dense(1, activation='linear', name='pmk')(shared)
    
    model = models.Model(inputs=inp, outputs=[out_kat, out_das, out_sifat, out_pmk])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss={
            'kategori': 'sparse_categorical_crossentropy',
            'dasarian': 'huber',
            'sifat': 'sparse_categorical_crossentropy',
            'pmk': 'huber'
        },
        loss_weights={'kategori': 1.0, 'dasarian': 0.1, 'sifat': 1.0, 'pmk': 0.1}
    )
    return model

def run_loyo_benchmark_kemarau():
    """
    Mengevaluasi dan membandingkan 5 algoritma secara komparatif untuk Prediksi Musim Kemarau:
    1. Baseline Random Forest (Tabular)
    2. LightGBM (Gradient Boosting)
    3. LSTM (Sequential Deep Learning)
    4. Bi-LSTM (Bidirectional Deep Learning)
    5. GRU (Gated Recurrent Unit)
    """
    X_seq, df_meta = build_sequential_dataset_kemarau()
    years = sorted(df_meta['tahun'].unique())
    n_samples, time_steps, n_feats = X_seq.shape
    
    X_flat = X_seq.reshape(n_samples, time_steps * n_feats)
    
    # Inisialisasi wadah prediksi OOF
    preds_rf = {'kat': [], 'das': [], 'sifat': [], 'pmk': []}
    preds_lgb = {'kat': [], 'das': [], 'sifat': [], 'pmk': []}
    preds_lstm = {'kat': [], 'das': [], 'sifat': [], 'pmk': []}
    preds_bilstm = {'kat': [], 'das': [], 'sifat': [], 'pmk': []}
    preds_gru = {'kat': [], 'das': [], 'sifat': [], 'pmk': []}
    
    y_true_kat = []
    y_true_das = []
    y_true_sifat = []
    y_true_pmk = []
    
    print("\n" + "="*85)
    print("MEMULAI BENCHMARK KOMPARATIF LEAVE-ONE-YEAR-OUT (LOYO 2000-2025: 26 FOLDS)...")
    print("="*85)
    
    for fold_idx, test_year in enumerate(years):
        test_mask = (df_meta['tahun'] == test_year).values
        train_mask = ~test_mask
        
        # Scaling fitur
        scaler = StandardScaler()
        X_train_seq = scaler.fit_transform(X_seq[train_mask].reshape(-1, n_feats)).reshape(-1, time_steps, n_feats)
        X_test_seq = scaler.transform(X_seq[test_mask].reshape(-1, n_feats)).reshape(-1, time_steps, n_feats)
        
        scaler_flat = StandardScaler()
        X_train_flat = scaler_flat.fit_transform(X_flat[train_mask])
        X_test_flat = scaler_flat.transform(X_flat[test_mask])
        
        # Target
        y_train_k = df_meta.loc[train_mask, 'kategori_amk'].map(KATEGORI_MAP).values
        y_train_d = df_meta.loc[train_mask, 'amk_dasarian'].values
        y_train_s = df_meta.loc[train_mask, 'sifat_hujan_mk'].map(SIFAT_MAP).values
        y_train_p = df_meta.loc[train_mask, 'pmk_dasarian'].values
        
        y_test_k = df_meta.loc[test_mask, 'kategori_amk'].map(KATEGORI_MAP).values
        y_test_d = df_meta.loc[test_mask, 'amk_dasarian'].values
        y_test_s = df_meta.loc[test_mask, 'sifat_hujan_mk'].map(SIFAT_MAP).values
        y_test_p = df_meta.loc[test_mask, 'pmk_dasarian'].values
        
        y_true_kat.extend(y_test_k)
        y_true_das.extend(y_test_d)
        y_true_sifat.extend(y_test_s)
        y_true_pmk.extend(y_test_p)
        
        # 1. Baseline Random Forest
        rf_k = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf_k.fit(X_train_flat, y_train_k)
        preds_rf['kat'].extend(rf_k.predict(X_test_flat))
        
        rf_d = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf_d.fit(X_train_flat, y_train_d)
        preds_rf['das'].extend(rf_d.predict(X_test_flat))
        
        rf_s = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf_s.fit(X_train_flat, y_train_s)
        preds_rf['sifat'].extend(rf_s.predict(X_test_flat))
        
        rf_p = RandomForestRegressor(n_estimators=80, max_depth=4, random_state=42)
        rf_p.fit(X_train_flat, y_train_p)
        preds_rf['pmk'].extend(rf_p.predict(X_test_flat))
        
        # 2. LightGBM
        clf_lgb_k = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
        clf_lgb_k.fit(X_train_flat, y_train_k)
        preds_lgb['kat'].extend(clf_lgb_k.predict(X_test_flat))
        
        reg_lgb_d = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
        reg_lgb_d.fit(X_train_flat, y_train_d)
        preds_lgb['das'].extend(reg_lgb_d.predict(X_test_flat))
        
        clf_lgb_s = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
        clf_lgb_s.fit(X_train_flat, y_train_s)
        preds_lgb['sifat'].extend(clf_lgb_s.predict(X_test_flat))
        
        reg_lgb_p = lgb.LGBMRegressor(n_estimators=40, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
        reg_lgb_p.fit(X_train_flat, y_train_p)
        preds_lgb['pmk'].extend(reg_lgb_p.predict(X_test_flat))
        
        # 3. LSTM Multi-Head
        model_lstm = create_lstm_kemarau_model((time_steps, n_feats))
        model_lstm.fit(
            X_train_seq,
            {'kategori': y_train_k, 'dasarian': y_train_d, 'sifat': y_train_s, 'pmk': y_train_p},
            epochs=25, batch_size=32, verbose=0
        )
        p_k, p_d, p_s, p_pmk = model_lstm.predict(X_test_seq, verbose=0)
        preds_lstm['kat'].extend(np.argmax(p_k, axis=1))
        preds_lstm['das'].extend(p_d.flatten())
        preds_lstm['sifat'].extend(np.argmax(p_s, axis=1))
        preds_lstm['pmk'].extend(p_pmk.flatten())
        
        # 4. Bi-LSTM Multi-Head
        model_bilstm = create_bilstm_kemarau_model((time_steps, n_feats))
        model_bilstm.fit(
            X_train_seq,
            {'kategori': y_train_k, 'dasarian': y_train_d, 'sifat': y_train_s, 'pmk': y_train_p},
            epochs=25, batch_size=32, verbose=0
        )
        p_k, p_d, p_s, p_pmk = model_bilstm.predict(X_test_seq, verbose=0)
        preds_bilstm['kat'].extend(np.argmax(p_k, axis=1))
        preds_bilstm['das'].extend(p_d.flatten())
        preds_bilstm['sifat'].extend(np.argmax(p_s, axis=1))
        preds_bilstm['pmk'].extend(p_pmk.flatten())
        
        # 5. GRU Multi-Head
        model_gru = create_gru_kemarau_model((time_steps, n_feats))
        model_gru.fit(
            X_train_seq,
            {'kategori': y_train_k, 'dasarian': y_train_d, 'sifat': y_train_s, 'pmk': y_train_p},
            epochs=25, batch_size=32, verbose=0
        )
        p_k, p_d, p_s, p_pmk = model_gru.predict(X_test_seq, verbose=0)
        preds_gru['kat'].extend(np.argmax(p_k, axis=1))
        preds_gru['das'].extend(p_d.flatten())
        preds_gru['sifat'].extend(np.argmax(p_s, axis=1))
        preds_gru['pmk'].extend(p_pmk.flatten())
        
        if (fold_idx + 1) % 5 == 0 or (fold_idx + 1) == len(years):
            print(f"  -> Fold LOYO {fold_idx + 1:2d}/{len(years)} (Tahun Evaluasi {test_year}) selesai.")
            
    # Evaluasi Metrik Komparatif
    models_dict = {
        'Baseline RF (Tabular)': preds_rf,
        'LightGBM (GBDT)': preds_lgb,
        'Bi-LSTM Multi-Head': preds_bilstm,
        'LSTM Multi-Head': preds_lstm,
        'GRU Multi-Head': preds_gru
    }
    
    benchmark_results = []
    for m_name, p in models_dict.items():
        acc_k = accuracy_score(y_true_kat, p['kat']) * 100.0
        f1_k = f1_score(y_true_kat, p['kat'], average='macro')
        mae_d = mean_absolute_error(y_true_das, p['das'])
        mae_hari = mae_d * 10.0
        acc_s = accuracy_score(y_true_sifat, p['sifat']) * 100.0
        mae_pmk = mean_absolute_error(y_true_pmk, p['pmk'])
        
        benchmark_results.append({
            'Model': m_name,
            'Akurasi_Kategori_AMK_%': round(acc_k, 2),
            'F1_Macro_Kategori': round(f1_k, 3),
            'MAE_Dasarian_AMK': round(mae_d, 2),
            'MAE_Hari_AMK': round(mae_hari, 1),
            'Akurasi_Sifat_Hujan_%': round(acc_s, 2),
            'MAE_Dasarian_PMK': round(mae_pmk, 2)
        })
        
    df_res = pd.DataFrame(benchmark_results)
    out_bench = DATA_DIR / "perbandingan_algoritma_kemarau.csv"
    df_res.to_csv(out_bench, index=False)
    
    print("\n" + "="*95)
    print("HASIL BENCHMARK KOMPARATIF MULTI-ALGORITMA PREDIKSI MUSIM KEMARAU KEBUMEN (LOYO 2000-2025)")
    print("="*95)
    print(df_res.to_string(index=False))
    print("="*95)
    print(f"\n[SUKSES] Tabel hasil perbandingan tersimpan di: {out_bench}")
    
    # Simpan hasil prediksi OOF model terbaik LSTM
    df_meta['pred_kat_lstm'] = [INV_KATEGORI_MAP[k] for k in preds_lstm['kat']]
    df_meta['pred_amk_das_lstm'] = np.round(preds_lstm['das'], 2)
    df_meta['pred_sifat_lstm'] = [INV_SIFAT_MAP[s] for s in preds_lstm['sifat']]
    df_meta['pred_pmk_das_lstm'] = np.round(preds_lstm['pmk'], 2)
    
    out_loyo = DATA_DIR / "hasil_evaluasi_loyo_kemarau.csv"
    df_meta.to_csv(out_loyo, index=False)
    print(f"[SUKSES] Hasil evaluasi OOF LOYO tersimpan di: {out_loyo}")
    return df_res

def predict_future_dry_season_2026(oni_djf=1.85, oni_jfm=1.45, dmi_feb=0.25, dmi_mar=0.30):
    """
    Menginferensi Prakiraan Musim Kemarau 2026 untuk 26 Kecamatan di Kabupaten Kebumen
    menggunakan model LSTM Multi-Head yang dilatih pada seluruh data 2000-2025.
    """
    print("\n" + "="*85)
    print(f"INFERENSI PREDIKSI MUSIM KEMARAU KEBUMEN 2026 (MODEL LSTM MULTI-HEAD)")
    print(f"Kondisi Global: ONI DJF={oni_djf:+.2f}, ONI JFM={oni_jfm:+.2f}, DMI Mar={dmi_mar:+.2f}")
    print("="*85)
    
    X_seq, df_meta = build_sequential_dataset_kemarau()
    n_samples, time_steps, n_feats = X_seq.shape
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seq.reshape(-1, n_feats)).reshape(-1, time_steps, n_feats)
    
    y_k = df_meta['kategori_amk'].map(KATEGORI_MAP).values
    y_d = df_meta['amk_dasarian'].values
    y_s = df_meta['sifat_hujan_mk'].map(SIFAT_MAP).values
    y_p = df_meta['pmk_dasarian'].values
    
    # Latih model LSTM penuh pada 2000-2025
    model_lstm = create_lstm_kemarau_model((time_steps, n_feats))
    model_lstm.fit(
        X_scaled,
        {'kategori': y_k, 'dasarian': y_d, 'sifat': y_s, 'pmk': y_p},
        epochs=35, batch_size=32, verbose=0
    )
    
    df_norm = pd.read_csv(DATA_DIR / "normal_klimatologi_amk_kebumen.csv")
    df_das = pd.read_csv(DATA_DIR / "dasarian_curah_hujan_kebumen_2000_2026.csv")
    norm_das = df_das[df_das['tahun'] <= 2025].groupby(['kecamatan', 'dasarian'])['curah_hujan'].mean().rename('normal_hujan_dasarian').reset_index()
    
    results = []
    
    for _, norm_row in df_norm.iterrows():
        kec = norm_row['kecamatan']
        
        # Gunakan deret sekuensial awal tahun 2026 jika ada, atau pola normal teranomali
        sub_2026 = df_das[(df_das['tahun'] == 2026) & (df_das['kecamatan'] == kec)].sort_values('dasarian')
        
        features_seq = []
        for d in range(1, 13):
            sub_d = sub_2026[sub_2026['dasarian'] == d]
            norm_val = norm_das[(norm_das['kecamatan'] == kec) & (norm_das['dasarian'] == d)]['normal_hujan_dasarian'].values[0]
            
            if not sub_d.empty:
                cur_val = sub_d['curah_hujan'].values[0]
            else:
                cur_val = norm_val * 0.9 # proyeksi musim kemarau lebih kering
                
            m = ((d - 1) // 3) + 1
            oni_val = oni_jfm if m >= 3 else oni_djf
            dmi_val = dmi_mar if m >= 3 else dmi_feb
            
            features_seq.append([
                cur_val,
                cur_val - norm_val,
                (cur_val + 1.0) / (norm_val + 1.0),
                oni_val,
                dmi_val,
                norm_row['normal_amk_dasarian'],
                norm_row['normal_total_hujan_mk_mm']
            ])
            
        X_input = np.array([features_seq], dtype=np.float32)
        X_input_scaled = scaler.transform(X_input.reshape(-1, n_feats)).reshape(1, time_steps, n_feats)
        
        pred_k, pred_d, pred_s, pred_p = model_lstm.predict(X_input_scaled, verbose=0)
        
        kat_prob = pred_k[0]
        kat_pred = INV_KATEGORI_MAP[int(np.argmax(kat_prob))]
        das_pred = float(pred_d[0][0])
        sifat_pred = INV_SIFAT_MAP[int(np.argmax(pred_s[0]))]
        pmk_pred = float(pred_p[0][0])
        
        # Rekomendasi Pola Tanam & Mitigasi Kekeringan:
        # Waktu pengeringan sawah Padi MT2: 1 dasarian sebelum AMK
        # Masa tanam Palawija MT3: saat awal kemarau (AMK)
        keringkan_sawah_das = max(10, int(round(das_pred)) - 1)
        tanam_palawija_das = int(round(das_pred))
        
        results.append({
            'kecamatan': kec,
            'normal_amk_dasarian': norm_row['normal_amk_dasarian'],
            'normal_amk_label': norm_row['normal_amk_label'],
            'prakiraan_amk_dasarian': round(das_pred, 1),
            'prakiraan_amk_label': get_dasarian_label(das_pred),
            'perbandingan_terhadap_normal': kat_pred,
            'probabilitas_maju_pct': round(kat_prob[0] * 100, 1),
            'probabilitas_sama_pct': round(kat_prob[1] * 100, 1),
            'probabilitas_mundur_pct': round(kat_prob[2] * 100, 1),
            'prakiraan_sifat_hujan_mk': sifat_pred,
            'prakiraan_pmk_dasarian': round(pmk_pred, 1),
            'prakiraan_pmk_label': get_dasarian_label(pmk_pred),
            'rekomendasi_keringkan_sawah_mt2': get_dasarian_label(keringkan_sawah_das),
            'rekomendasi_tanam_palawija_mt3': get_dasarian_label(tanam_palawija_das)
        })
        
    df_out = pd.DataFrame(results)
    out_file = DATA_DIR / "prakiraan_musim_kemarau_2026_kebumen.csv"
    df_out.to_csv(out_file, index=False)
    print(f"\n[SUKSES] Prakiraan Musim Kemarau 2026 tersimpan di: {out_file}")
    
    print("\n" + "="*85)
    print("RINGKASAN PRAKIRAAN AWAL MUSIM KEMARAU KEBUMEN 2026 (MODEL LSTM)")
    print("="*85)
    print(df_out[['kecamatan', 'normal_amk_label', 'prakiraan_amk_label', 'perbandingan_terhadap_normal', 'prakiraan_sifat_hujan_mk', 'rekomendasi_tanam_palawija_mt3']].to_string(index=False))
    print("="*85)
    return df_out

if __name__ == "__main__":
    run_loyo_benchmark_kemarau()
    predict_future_dry_season_2026()

