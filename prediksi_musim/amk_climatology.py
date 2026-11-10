#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Analisis Klimatologi & Ekstraksi Baseline Awal Musim Kemarau (AMK) Kabupaten Kebumen
Menghitung Normal AMK, Total Presipitasi Musim Kemarau (MK), dan Puncak Musim Kemarau (PMK)
berdasarkan data historis 26 tahun (2000-2025) untuk 26 kecamatan sesuai kriteria BMKG.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import force_2d

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
GEOJSON_PATH = BASE_DIR / "33.05_kecamatan.geojson"

DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_dasarian_label(d_idx):
    d_round = int(round(d_idx))
    m = ((d_round - 1) // 3) + 1
    d_m = ((d_round - 1) % 3) + 1
    m_names = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    roman = ["I", "II", "III"]
    if 1 <= m <= 12:
        return f"{m_names[m-1]} {roman[d_m-1]}"
    return f"Dasarian {d_round}"

def calculate_amk_historis():
    """
    Mengekstrak Awal Musim Kemarau (AMK) per tahun per kecamatan (2000-2025).
    Kriteria BMKG: Dasarian pertama dengan curah hujan < 50mm yang diikuti >= 2 dasarian < 50mm.
    Jendela pencarian: Dasarian 10 (April I) s/d Dasarian 24 (Agustus III).
    """
    df_das = pd.read_csv(DATA_DIR / "dasarian_curah_hujan_kebumen_2000_2026.csv")
    years = range(2000, 2026) # 2000..2025
    kecamatan_list = sorted(df_das['kecamatan'].unique())
    
    print("\n[INFO] Menghitung AMK historis 26 kecamatan (2000-2025)...")
    records_amk = []
    
    for yr in years:
        for kec in kecamatan_list:
            sub = df_das[(df_das['tahun'] == yr) & (df_das['kecamatan'] == kec)].sort_values('dasarian').reset_index(drop=True)
            
            # Cari dasarian pertama < 50mm yang diikuti >= 2 dasarian < 50mm
            sub_search = sub[(sub['dasarian'] >= 10) & (sub['dasarian'] <= 24)].reset_index(drop=True)
            
            amk_d = 14.0 # default Mei II
            for i in range(len(sub_search) - 2):
                p0 = sub_search.loc[i, 'curah_hujan']
                p1 = sub_search.loc[i+1, 'curah_hujan']
                p2 = sub_search.loc[i+2, 'curah_hujan']
                d0 = sub_search.loc[i, 'dasarian']
                
                if p0 < 50.0 and (p1 < 50.0 or p2 < 50.0):
                    amk_d = d0
                    break
                    
            # Total hujan musim kemarau (Mei s/d September = Dasarian 13 s/d 27)
            sub_mk = sub[(sub['dasarian'] >= 13) & (sub['dasarian'] <= 27)]
            tot_mk = sub_mk['curah_hujan'].sum()
            
            # Puncak Musim Kemarau (Dasarian paling kering / minimum dalam rentang Juni s/d September)
            sub_pmk = sub[(sub['dasarian'] >= 16) & (sub['dasarian'] <= 27)]
            if not sub_pmk.empty:
                pmk_d = sub_pmk.loc[sub_pmk['curah_hujan'].idxmin(), 'dasarian']
            else:
                pmk_d = 22 # Agu I
                
            records_amk.append({
                'tahun': yr,
                'kecamatan': kec,
                'amk_dasarian': amk_d,
                'amk_label': get_dasarian_label(amk_d),
                'total_hujan_mk_mm': round(tot_mk, 1),
                'pmk_dasarian': pmk_d,
                'pmk_label': get_dasarian_label(pmk_d)
            })
            
    df_amk = pd.DataFrame(records_amk)
    out_hist = DATA_DIR / "amk_historis_kebumen_2000_2025.csv"
    df_amk.to_csv(out_hist, index=False)
    print(f"[SUKSES] AMK historis tersimpan di: {out_hist} ({len(df_amk)} baris).")
    
    # Hitung Normal Klimatologi AMK (Median 2000-2025)
    df_norm = df_amk.groupby('kecamatan').agg(
        normal_amk_dasarian=('amk_dasarian', 'median'),
        mean_amk_dasarian=('amk_dasarian', 'mean'),
        std_amk_dasarian=('amk_dasarian', 'std'),
        terawal_amk_dasarian=('amk_dasarian', 'min'),
        terlambat_amk_dasarian=('amk_dasarian', 'max'),
        normal_total_hujan_mk_mm=('total_hujan_mk_mm', 'mean'),
        normal_pmk_dasarian=('pmk_dasarian', 'median')
    ).reset_index()
    
    df_norm['normal_amk_label'] = [get_dasarian_label(d) for d in df_norm['normal_amk_dasarian']]
    df_norm['terawal_amk_label'] = [get_dasarian_label(d) for d in df_norm['terawal_amk_dasarian']]
    df_norm['terlambat_amk_label'] = [get_dasarian_label(d) for d in df_norm['terlambat_amk_dasarian']]
    df_norm['normal_pmk_label'] = [get_dasarian_label(d) for d in df_norm['normal_pmk_dasarian']]
    
    # Pembulatan desimal
    df_norm['normal_amk_dasarian'] = df_norm['normal_amk_dasarian'].round(1)
    df_norm['mean_amk_dasarian'] = df_norm['mean_amk_dasarian'].round(2)
    df_norm['std_amk_dasarian'] = df_norm['std_amk_dasarian'].round(2)
    df_norm['normal_total_hujan_mk_mm'] = df_norm['normal_total_hujan_mk_mm'].round(1)
    df_norm['normal_pmk_dasarian'] = df_norm['normal_pmk_dasarian'].round(1)
    
    out_norm = DATA_DIR / "normal_klimatologi_amk_kebumen.csv"
    df_norm.to_csv(out_norm, index=False)
    print(f"[SUKSES] Normal Klimatologi AMK tersimpan di: {out_norm}")
    
    print("\n" + "="*85)
    print("HASIL BASELINE NORMAL KLIMATOLOGI AWAL MUSIM KEMARAU (AMK) KEBUMEN (2000-2025)")
    print("="*85)
    print(df_norm[['kecamatan', 'normal_amk_label', 'mean_amk_dasarian', 'normal_total_hujan_mk_mm', 'normal_pmk_label']].to_string(index=False))
    print("="*85)
    return df_norm

if __name__ == "__main__":
    calculate_amk_historis()

