#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul Visualisasi Peta Tematik Spasial Prediksi Musim Hujan Kabupaten Kebumen
Standar Rasio Aspek 16:9 (figsize=(16, 9), 300 DPI) sesuai Aturan 8 & 9.
Menghasilkan 6 Peta Tematik Standar Publikasi & Diseminasi Multi-Kanal.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import force_2d

BASE_DIR = Path(r"d:\Github\Projek_Downscale")
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
PLOTS_DIR = BASE_DIR / "prediksi_musim" / "plots_16_9"
GEOJSON_PATH = BASE_DIR / "33.05_kecamatan.geojson"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Palet Warna Standar BMKG / WMO
COLOR_PALETTE = {
    # Dasarian Awal Musim
    'Okt I': '#1b7837',    # Hijau tua
    'Okt II': '#7fbf7b',   # Hijau muda
    'Okt III': '#d9f0d3',  # Hijau sangat muda
    'Nov I': '#fee08b',    # Kuning
    'Nov II': '#fdae61',   # Jingga muda
    'Nov III': '#f46d43',  # Jingga kemerahan
    'Des I': '#d53e4f',    # Merah bata
    'Des II': '#9e0142',   # Merah marun
    
    # Perbandingan terhadap Normal
    'Maju': '#2b83ba',     # Biru
    'Sama': '#abdda4',     # Hijau pastel
    'Mundur': '#d7191c',   # Merah koral
    
    # Sifat Hujan
    'Atas Normal': '#2b83ba',   # Biru
    'Normal': '#abdda4',        # Hijau
    'Bawah Normal': '#fdae61',  # Oranye / Kuning tua
    
    # Kalender Tanam
    'Nov III': '#66c2a5',  # Hijau toska
    'Des I': '#fc8d62',    # Oranye terang
    'Des II': '#8da0cb'    # Biru keunguan
}

def load_geodata():
    """Memuat batas administrasi kecamatan dan menggabungkan dengan hasil prediksi."""
    gdf = gpd.read_file(GEOJSON_PATH)
    # Konversi 2D untuk keamanan
    gdf['geometry'] = force_2d(gdf['geometry'])
    
    df_norm = pd.read_csv(DATA_DIR / "normal_klimatologi_amh_kebumen.csv")
    df_pred = pd.read_csv(DATA_DIR / "prakiraan_musim_2026_2027_kebumen.csv")
    
    # Merge
    gdf_merged = gdf.merge(df_norm, left_on='nm_kecamatan', right_on='kecamatan')
    gdf_merged = gdf_merged.merge(df_pred, left_on='nm_kecamatan', right_on='kecamatan', suffixes=('', '_pred'))
    
    return gdf_merged

def add_decorations(ax, title, subtitle, source_text="Sumber: Analisis Model Prediksi Multi-Sensor & Reanalisis ERA5-Land (2000-2026)"):
    """Menambahkan dekorasi kartografi: Judul 16:9, Arah Utara, Skala, dan Anotasi Sumber."""
    # Judul dan Subjudul
    ax.set_title(f"{title}\n{subtitle}", fontsize=15, fontweight='bold', pad=15, color='#1a1a1a', loc='center')
    
    # Arah Mata Angin (North Arrow) di pojok kanan atas
    x_arrow, y_arrow = 0.96, 0.90
    ax.annotate('U', xy=(x_arrow, y_arrow), xytext=(x_arrow, y_arrow - 0.08),
                xycoords='axes fraction', ha='center', va='center',
                fontsize=13, fontweight='bold', color='#1a1a1a',
                arrowprops=dict(facecolor='#1a1a1a', edgecolor='black', width=3, headwidth=9))
    
    # Sumber data dan footer
    ax.text(0.01, 0.015, source_text, transform=ax.transAxes, fontsize=9.5,
            color='#444444', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
            
    # Format Grid & Axis
    ax.tick_params(labelsize=9, colors='#555555')
    ax.set_xlabel("Bujur Timur (°E)", fontsize=10, labelpad=5, color='#444444')
    ax.set_ylabel("Lintang Selatan (°S)", fontsize=10, labelpad=5, color='#444444')
    ax.grid(True, linestyle='--', alpha=0.4, color='#999999')

def annotate_districts(ax, gdf, value_col=None):
    """Menambahkan label nama kecamatan dan nilai pada titik representatif poligon."""
    for _, row in gdf.iterrows():
        pt = row.geometry.representative_point()
        kec_name = row['nm_kecamatan']
        if value_col and value_col in row:
            val = str(row[value_col])
            text = f"{kec_name}\n({val})"
        else:
            text = kec_name
            
        ax.text(pt.x, pt.y, text, fontsize=7.2, fontweight='bold', ha='center', va='center',
                color='#111111',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))

def plot_peta_1_normal_amh(gdf):
    """Peta 1: Normal Klimatologi Awal Musim Hujan Kebumen (2000-2025)."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    cat_colors = {cat: COLOR_PALETTE.get(cat, '#cccccc') for cat in ['Okt I', 'Okt II', 'Okt III']}
    gdf['color'] = gdf['normal_amh_label'].map(cat_colors)
    
    gdf.plot(color=gdf['color'], edgecolor='#222222', linewidth=0.8, ax=ax)
    annotate_districts(ax, gdf, 'normal_amh_label')
    
    # Legenda Kustom
    patches = [mpatches.Patch(color=color, label=f"Dasarian {cat} (Normal)") for cat, color in cat_colors.items()]
    ax.legend(handles=patches, title="Normal AMH (Klimatologi 2000–2025)", loc='lower right',
              fontsize=10.5, title_fontsize=11, framealpha=0.92, facecolor='white', edgecolor='#999999')
    
    add_decorations(ax, 
                    "PETA 1: NORMAL KLIMATOLOGI AWAL MUSIM HUJAN (AMH) KABUPATEN KEBUMEN", 
                    "Garis Pangkal Baseline Rata-rata Multi-Dekade Berdasarkan Kriteria Dasarian Curah Hujan BMKG (P ≥ 50 mm)")
    
    plt.tight_layout()
    out_file = PLOTS_DIR / "Peta_1_Normal_AMH_Kebumen.png"
    plt.savefig(out_file, dpi=300)
    plt.close('all')
    print(f"[SUKSES] Peta 1 tersimpan di: {out_file}")

def plot_peta_2_prakiraan_amh(gdf):
    """Peta 2: Prakiraan Awal Musim Hujan 2026/2027 Kebumen."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    order = ['Nov I', 'Nov II', 'Nov III']
    cat_colors = {cat: COLOR_PALETTE.get(cat, '#cccccc') for cat in order}
    gdf['color'] = gdf['prakiraan_amh_label'].map(cat_colors)
    
    gdf.plot(color=gdf['color'], edgecolor='#222222', linewidth=0.8, ax=ax)
    annotate_districts(ax, gdf, 'prakiraan_amh_label')
    
    patches = [mpatches.Patch(color=cat_colors[cat], label=f"Dasarian {cat}") for cat in order if cat in cat_colors]
    ax.legend(handles=patches, title="Prakiraan AMH Musim 2026/2027", loc='lower right',
              fontsize=10.5, title_fontsize=11, framealpha=0.92, facecolor='white', edgecolor='#999999')
    
    add_decorations(ax, 
                    "PETA 2: PRAKIRAAN AWAL MUSIM HUJAN (AMH) KABUPATEN KEBUMEN 2026/2027", 
                    "Prediksi Spasial Resolusi Tinggi Berbasis Machine Learning (XGBoost + Random Forest) di Bawah Kondisi ENSO Positif")
    
    plt.tight_layout()
    out_file = PLOTS_DIR / "Peta_2_Prakiraan_AMH_2026_2027.png"
    plt.savefig(out_file, dpi=300)
    plt.close('all')
    print(f"[SUKSES] Peta 2 tersimpan di: {out_file}")

def plot_peta_3_perbandingan_normal(gdf):
    """Peta 3: Perbandingan AMH 2026/2027 Terhadap Normal (Maju/Sama/Mundur)."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    cat_colors = {'Maju': COLOR_PALETTE['Maju'], 'Sama': COLOR_PALETTE['Sama'], 'Mundur': COLOR_PALETTE['Mundur']}
    gdf['color'] = gdf['perbandingan_terhadap_normal'].map(cat_colors)
    
    gdf.plot(color=gdf['color'], edgecolor='#222222', linewidth=0.8, ax=ax)
    
    # Anotasi nama kecamatan beserta probabilitas mundur
    for _, row in gdf.iterrows():
        pt = row.geometry.representative_point()
        kec_name = row['nm_kecamatan']
        prob = row['probabilitas_mundur_pct']
        text = f"{kec_name}\n(Mundur {prob:.0f}%)"
        ax.text(pt.x, pt.y, text, fontsize=7.2, fontweight='bold', ha='center', va='center',
                color='#111111',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))
                
    patches = [
        mpatches.Patch(color=cat_colors['Maju'], label="Maju (Lebih Awal dari Normal)"),
        mpatches.Patch(color=cat_colors['Sama'], label="Sama (Sesuai dengan Normal)"),
        mpatches.Patch(color=cat_colors['Mundur'], label="Mundur (Lebih Lambat dari Normal)")
    ]
    ax.legend(handles=patches, title="Perbandingan Terhadap Normal", loc='lower right',
              fontsize=10.5, title_fontsize=11, framealpha=0.92, facecolor='white', edgecolor='#999999')
    
    add_decorations(ax, 
                    "PETA 3: PERBANDINGAN PRAKIRAAN AMH 2026/2027 TERHADAP NORMAL KLIMATOLOGI", 
                    "Klasifikasi Anomali Waktu Masuk Musim Hujan per Kecamatan & Estimasi Probabilitas Terjadinya Keterlambatan")
    
    plt.tight_layout()
    out_file = PLOTS_DIR / "Peta_3_Perbandingan_Terhadap_Normal.png"
    plt.savefig(out_file, dpi=300)
    plt.close('all')
    print(f"[SUKSES] Peta 3 tersimpan di: {out_file}")

def plot_peta_4_sifat_hujan(gdf):
    """Peta 4: Prakiraan Sifat Hujan Musim Hujan 2026/2027 (Atas Normal, Normal, Bawah Normal)."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    cat_colors = {
        'Atas Normal': COLOR_PALETTE['Atas Normal'],
        'Normal': COLOR_PALETTE['Normal'],
        'Bawah Normal': COLOR_PALETTE['Bawah Normal']
    }
    gdf['color'] = gdf['prakiraan_sifat_hujan'].map(cat_colors)
    
    gdf.plot(color=gdf['color'], edgecolor='#222222', linewidth=0.8, ax=ax)
    annotate_districts(ax, gdf, 'prakiraan_sifat_hujan')
    
    patches = [
        mpatches.Patch(color=cat_colors['Atas Normal'], label="Atas Normal (> 115% terhadap Rerata Normal)"),
        mpatches.Patch(color=cat_colors['Normal'], label="Normal (85% – 115% terhadap Rerata Normal)"),
        mpatches.Patch(color=cat_colors['Bawah Normal'], label="Bawah Normal (< 85% terhadap Rerata Normal)")
    ]
    ax.legend(handles=patches, title="Prakiraan Sifat Hujan 2026/2027", loc='lower right',
              fontsize=10.5, title_fontsize=11, framealpha=0.92, facecolor='white', edgecolor='#999999')
    
    add_decorations(ax, 
                    "PETA 4: PRAKIRAAN SIFAT HUJAN KABUPATEN KEBUMEN MUSIM HUJAN 2026/2027", 
                    "Estimasi Akumulasi Curah Hujan Total Selama Musim Hujan Relatif Terhadap Rerata Klimatologi 25 Tahun")
    
    plt.tight_layout()
    out_file = PLOTS_DIR / "Peta_4_Prakiraan_Sifat_Hujan.png"
    plt.savefig(out_file, dpi=300)
    plt.close('all')
    print(f"[SUKSES] Peta 4 tersimpan di: {out_file}")

def plot_peta_5_puncak_musim(gdf):
    """Peta 5: Prakiraan Puncak Musim Hujan 2026/2027."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    order = ['Des I', 'Des II']
    cat_colors = {cat: COLOR_PALETTE.get(cat, '#cccccc') for cat in order}
    gdf['color'] = gdf['prakiraan_puncak_label'].map(cat_colors)
    
    gdf.plot(color=gdf['color'], edgecolor='#222222', linewidth=0.8, ax=ax)
    annotate_districts(ax, gdf, 'prakiraan_puncak_label')
    
    patches = [mpatches.Patch(color=cat_colors[cat], label=f"Dasarian {cat}") for cat in order if cat in cat_colors]
    ax.legend(handles=patches, title="Prakiraan Puncak Musim Hujan", loc='lower right',
              fontsize=10.5, title_fontsize=11, framealpha=0.92, facecolor='white', edgecolor='#999999')
    
    add_decorations(ax, 
                    "PETA 5: PRAKIRAAN PUNCAK MUSIM HUJAN KABUPATEN KEBUMEN 2026/2027", 
                    "Periode Dasarian dengan Intensitas Curah Hujan Tertinggi dan Potensi Maksimum Kejadian Hidrometeorologi Ekstrem")
    
    plt.tight_layout()
    out_file = PLOTS_DIR / "Peta_5_Prakiraan_Puncak_Musim_Hujan.png"
    plt.savefig(out_file, dpi=300)
    plt.close('all')
    print(f"[SUKSES] Peta 5 tersimpan di: {out_file}")

def plot_peta_6_kalender_tanam(gdf):
    """Peta 6: Zonasi Rekomendasi Kalender Tanam Padi MT1 2026/2027."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    order = ['Nov III', 'Des I', 'Des II']
    cat_colors = {cat: COLOR_PALETTE.get(cat, '#cccccc') for cat in order}
    gdf['color'] = gdf['rekomendasi_tanam_padi_mt1'].map(cat_colors)
    
    gdf.plot(color=gdf['color'], edgecolor='#222222', linewidth=0.8, ax=ax)
    annotate_districts(ax, gdf, 'rekomendasi_tanam_padi_mt1')
    
    patches = [
        mpatches.Patch(color=COLOR_PALETTE['Nov III'], label="Tahap 1: Tanam Nov III (Olah Tanah Nov II)"),
        mpatches.Patch(color=COLOR_PALETTE['Des I'], label="Tahap 2: Tanam Des I (Olah Tanah Nov III)"),
        mpatches.Patch(color=COLOR_PALETTE['Des II'], label="Tahap 3: Tanam Des II (Olah Tanah Des I)")
    ]
    ax.legend(handles=patches, title="Zonasi Waktu Tanam Padi MT1", loc='lower right',
              fontsize=10.5, title_fontsize=11, framealpha=0.92, facecolor='white', edgecolor='#999999')
    
    add_decorations(ax, 
                    "PETA 6: ZONASI REKOMENDASI KALENDER TANAM PADI SAWAH MT1 KEBUMEN 2026/2027", 
                    "Panduan Waktu Olah Tanah dan Tanam Padi Rendengan Terintegrasi untuk Mitigasi Risiko Kekeringan Awal Musim")
    
    plt.tight_layout()
    out_file = PLOTS_DIR / "Peta_6_Zonasi_Kalender_Tanam_Pertanian.png"
    plt.savefig(out_file, dpi=300)
    plt.close('all')
    print(f"[SUKSES] Peta 6 tersimpan di: {out_file}")

def generate_all_maps():
    """Menghasilkan seluruh 6 peta tematik 16:9 secara otomatis."""
    print("Memuat data geospasial dan tabel hasil prediksi...")
    gdf = load_geodata()
    print(f"Data termuat untuk {len(gdf)} kecamatan. Menghasilkan peta 16:9...")
    
    plot_peta_1_normal_amh(gdf)
    plot_peta_2_prakiraan_amh(gdf)
    plot_peta_3_perbandingan_normal(gdf)
    plot_peta_4_sifat_hujan(gdf)
    plot_peta_5_puncak_musim(gdf)
    plot_peta_6_kalender_tanam(gdf)
    
    print("\n[SELESAI] Semua 6 peta tematik 16:9 telah berhasil dibuat dengan kualitas 300 DPI!")

if __name__ == "__main__":
    generate_all_maps()
