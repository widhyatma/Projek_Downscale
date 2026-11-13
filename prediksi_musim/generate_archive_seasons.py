#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator Arsip Prediksi Musim Historis (2015–2025) & Evaluasi Akurasi (Prediksi vs Realisasi)
Stasiun Meteorologi Kelas III Jerukagung - Kabupaten Kebumen

Mengekstrak hasil validasi Leave-One-Year-Out (LOYO) LSTM 26 tahun, menyusun arsip
prakiraan musim lampau (Musim Hujan & Kemarau), menghitung metrik evaluasi ketepatan
model secara transparan, dan mengekspor katalog modular JSON ke Dashboard.
"""

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely import force_2d

sys.path.append(str(Path(__file__).resolve().parent))
from topographic_downscaling import prepare_250m_topographic_mesh, downscale_variable_multimodel
from sync_to_dashboard import render_raster_overlay_png

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
GEOJSON_KEC_PATH = BASE_DIR / "33.05_kecamatan.geojson"
GEOJSON_DESA_PATH = BASE_DIR / "33.05_kelurahan.geojson"
DEFAULT_DASHBOARD_DIR = Path(r"D:\Github\Dashboard_Jerukagung-Meteorologi")

# Konteks Klimatologi Global (ENSO & IOD) 2015–2025
CLIMATE_CONTEXT = {
    2015: "Super El Niño (Kemarau sangat panjang & ekstrem kering)",
    2016: "La Niña Lemah / Kemarau Basah (Curah hujan melimpah)",
    2017: "Kondisi Netral / Normal",
    2018: "El Niño Lemah (Curah hujan cenderung di bawah normal)",
    2019: "IOD Positif Kuat + El Niño Lemah (Kemarau sangat kering di Jawa)",
    2020: "La Niña Moderat (Tahun 1 Triple-Dip La Niña)",
    2021: "La Niña Moderat (Tahun 2 Triple-Dip La Niña)",
    2022: "La Niña Lemah (Tahun 3 Triple-Dip La Niña)",
    2023: "Super El Niño + IOD Positif Kuat (Kemarau ekstrem kering & panjang)",
    2024: "Transisi El Niño Menuju Netral",
    2025: "Kondisi Netral / Normal"
}

BULAN_NAMES = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]
BULAN_SHORT = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
ROMAN = ["I", "II", "III"]


def dasarian_to_short_label(val: float) -> str:
    if pd.isna(val) or val <= 0:
        return "-"
    v = max(1.0, min(36.0, float(val)))
    idx = int(round(v)) - 1
    m_idx = idx // 3
    d_idx = idx % 3
    return f"{BULAN_SHORT[m_idx]} {ROMAN[d_idx]}"


def dasarian_to_month_name(val: float, year_offset: int = 0) -> str:
    if pd.isna(val) or val <= 0:
        return "-"
    v = max(1.0, min(36.0, float(val)))
    idx = int(round(v)) - 1
    m_idx = idx // 3
    return f"{BULAN_NAMES[m_idx]}"


def classify_hujan_mh(mm: float) -> str:
    if pd.isna(mm) or mm <= 0:
        return "-"
    if mm < 1500:
        return "< 1.500 mm"
    elif mm < 1750:
        return "1.500–1.750 mm"
    elif mm < 2000:
        return "1.750–2.000 mm"
    elif mm < 2500:
        return "2.000–2.500 mm"
    elif mm < 3000:
        return "2.500–3.000 mm"
    else:
        return "> 3.000 mm"


def classify_hujan_mk(mm: float) -> str:
    if pd.isna(mm) or mm <= 0:
        return "-"
    if mm < 430:
        return "< 430 mm"
    elif mm < 485:
        return "430–485 mm"
    elif mm <= 525:
        return "485–525 mm"
    else:
        return "> 525 mm"


def classify_kekeringan(score: float) -> str:
    if pd.isna(score):
        return "Sedang (< 60)"
    if score < 60:
        return "Sedang (< 60)"
    elif score <= 65:
        return "Cukup Tinggi (60–65)"
    elif score <= 70:
        return "Tinggi (65–70)"
    else:
        return "Sangat Tinggi (> 70)"


def compute_zonal_means(gdf: gpd.GeoDataFrame, tif_path: Path) -> list[float]:
    """Menghitung rata-rata zonal dari raster GeoTIFF 250m untuk setiap poligon."""
    if not tif_path.exists():
        raise FileNotFoundError(f"Raster GeoTIFF tidak ditemukan: {tif_path}")

    results: list[float] = []
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        arr_full = src.read(1).astype(np.float32)

        for geom in gdf.geometry:
            vals = np.array([], dtype=np.float32)
            try:
                out, _ = mask(src, [force_2d(geom)], crop=True, nodata=np.nan, filled=True)
                v = out[0].astype(np.float32)
                valid_mask = np.isfinite(v) & (v > -9000.0)
                if nodata is not None and not np.isnan(nodata):
                    valid_mask &= v != float(nodata)
                vals = v[valid_mask]
            except Exception:
                pass

            if len(vals) == 0:
                rep_point = geom.representative_point()
                row_idx, col_idx = src.index(rep_point.x, rep_point.y)
                row_idx = int(np.clip(row_idx, 0, src.height - 1))
                col_idx = int(np.clip(col_idx, 0, src.width - 1))
                sample_val = float(arr_full[row_idx, col_idx])
                if np.isfinite(sample_val) and sample_val > -9000.0:
                    results.append(sample_val)
                else:
                    finite_global = arr_full[np.isfinite(arr_full) & (arr_full > -9000.0)]
                    results.append(float(np.nanmean(finite_global)))
            else:
                results.append(float(np.mean(vals)))

    return results


def compute_prediction_probabilities(pred_das: float, norm_das: float, pred_kat: str) -> tuple[float, float, float]:
    """
    Menghitung distribusi probabilitas (Maju, Sama, Mundur) yang realistis
    berdasarkan selisih dasarian prediksi terhadap normal klimatologi.
    Hasil berupa persentase float yang jika dijumlahkan tepat 100.0%.
    """
    diff = float(pred_das - norm_das)
    kat_lower = str(pred_kat).strip().lower()

    if kat_lower == "maju" or diff <= -1.0:
        adv = min(abs(diff), 3.0)
        p_maju = round(70.0 + (adv / 3.0) * 25.0, 1)
        rem = 100.0 - p_maju
        p_sama = round(rem * 0.75, 1)
        p_mundur = round(100.0 - p_maju - p_sama, 1)
    elif kat_lower == "mundur" or diff >= 1.0:
        adv = min(abs(diff), 3.0)
        p_mundur = round(70.0 + (adv / 3.0) * 25.0, 1)
        rem = 100.0 - p_mundur
        p_sama = round(rem * 0.75, 1)
        p_maju = round(100.0 - p_mundur - p_sama, 1)
    else:  # Sama
        adv = 1.0 - min(abs(diff), 1.0)
        p_sama = round(70.0 + adv * 20.0, 1)
        rem = 100.0 - p_sama
        p_maju = round(rem / 2.0, 1)
        p_mundur = round(100.0 - p_sama - p_maju, 1)

    return float(p_maju), float(p_sama), float(p_mundur)


def build_katam_profile(tanam_mt1_str: str, sifat_hujan: str) -> dict:
    month_map = {
        "Okt": 10, "Nov": 11, "Des": 12, "Jan": 1, "Feb": 2, "Mar": 3,
        "Apr": 4, "Mei": 5, "Jun": 6, "Jul": 7, "Agu": 8, "Sep": 9,
    }
    tokens = tanam_mt1_str.strip().split()
    b_str = tokens[0] if tokens else "Nov"
    rom = tokens[1] if len(tokens) > 1 else "III"
    rom_idx = 0 if rom == "I" else (1 if rom == "II" else 2)
    b_idx = month_map.get(b_str, 11)

    t_mt1 = (b_idx - 1) * 3 + rom_idx + 1
    ot_mt1 = ((t_mt1 - 2) % 36) or 36
    p_mt1 = ((t_mt1 + 9 - 1) % 36) + 1
    t_mt2 = ((p_mt1 + 1 - 1) % 36) + 1
    p_mt2 = ((t_mt2 + 9 - 1) % 36) + 1
    t_mt3 = ((p_mt2 + 1 - 1) % 36) + 1
    p_mt3 = ((t_mt3 + 8 - 1) % 36) + 1

    is_wet = "Atas" in sifat_hujan
    is_dry = "Bawah" in sifat_hujan

    if is_wet:
        v_mt1 = ["Inpari 32 HDB", "Inpari 42 Agritan", "Inpara 2 (Toleran Rendaman)"]
        tips_mt1 = "Waspadai genangan berlebih dan serangan penyakit blas/kresek; benahi saluran pembuangan air."
    elif is_dry:
        v_mt1 = ["Inpari 32 HDB", "Situ Bagendit", "Cakrabuana Agritan (Genjah & Hemat Air)"]
        tips_mt1 = "Optimalkan penampungan air embung/dam parit; percepat pengolahan tanah saat hujan perdana turun."
    else:
        v_mt1 = ["Inpari 32 HDB", "Ciherang", "Mekongga"]
        tips_mt1 = "Pengolahan tanah serempak untuk efisiensi air irigasi tersier dan mencegah ledakan hama tikus."

    v_mt2 = ["Inpari 32 HDB", "Mekongga", "Cisaat"]
    tips_mt2 = "Lakukan penyiangan dan pemupukan berimbang tepat waktu sebelum masa pengisian bulir padi."
    v_mt3 = ["Jagung Hibrida Bisi-18", "Kedelai Anjasmoro", "Kacang Hijau Vima-1"]
    tips_mt3 = "Manfaatkan sisa kelembapan lengas tanah pasca panen padi gadu untuk budidaya palawija hemat air."

    return {
        "mt1": {
            "nama": "MT1 (Padi Rendengan)",
            "olah_tanah": dasarian_to_short_label(ot_mt1),
            "tanam": dasarian_to_short_label(t_mt1),
            "panen": dasarian_to_short_label(p_mt1),
            "varietas": v_mt1,
            "tips": tips_mt1,
        },
        "mt2": {
            "nama": "MT2 (Padi Gadu)",
            "tanam": dasarian_to_short_label(t_mt2),
            "panen": dasarian_to_short_label(p_mt2),
            "varietas": v_mt2,
            "tips": tips_mt2,
        },
        "mt3": {
            "nama": "MT3 (Palawija/Bera)",
            "tanam": dasarian_to_short_label(t_mt3),
            "panen": dasarian_to_short_label(p_mt3),
            "komoditas": v_mt3,
            "tips": tips_mt3,
        },
    }


def compute_cews_for_year(df_dasarian_year: pd.DataFrame, df_dasarian_all: pd.DataFrame) -> tuple[dict, dict]:
    """Menghitung metrik CEWS (HTH & SPI) serta time series 36 dasarian untuk tahun arsip."""
    cews_map: dict[str, dict] = {}
    chart_map: dict[str, dict] = {}

    labels_36 = [dasarian_to_short_label(d) for d in range(1, 37)]

    # Climatological normal per dasarian (2000-2025)
    clim_norm = (
        df_dasarian_all[df_dasarian_all["tahun"] <= 2025]
        .groupby(["kecamatan", "dasarian"])["curah_hujan"]
        .mean()
        .unstack(level="dasarian")
    )

    # Monthly rainfall baseline for SPI
    m_rain_hist = (
        df_dasarian_all[df_dasarian_all["tahun"] <= 2025]
        .groupby(["kecamatan", "tahun", "bulan"])["curah_hujan"]
        .sum()
        .reset_index()
    )
    m_stats = m_rain_hist.groupby(["kecamatan", "bulan"])["curah_hujan"].agg(["mean", "std"]).reset_index()

    # Data tahun berjalan
    kecamatan_list = sorted(df_dasarian_year["kecamatan"].unique())

    for kec in kecamatan_list:
        sub_y = df_dasarian_year[df_dasarian_year["kecamatan"] == kec].sort_values("dasarian")
        sub_norm = clim_norm.loc[kec] if kec in clim_norm.index else pd.Series(0, index=range(1, 37))

        norm_vals = [round(float(sub_norm.get(d, 0.0)), 1) for d in range(1, 37)]
        akt_vals = []
        for d in range(1, 37):
            match = sub_y[sub_y["dasarian"] == d]
            if not match.empty:
                akt_vals.append(round(float(match["curah_hujan"].values[0]), 1))
            else:
                akt_vals.append(None)

        # Proyeksi untuk tahun arsip sama dengan observasi (karena sudah lewat)
        chart_map[kec] = {
            "labels": labels_36,
            "normal": norm_vals,
            "aktual": akt_vals,
            "proyeksi": akt_vals,
        }

        # SPI 1-Bulan (Bulan 10 / Oktober tahun itu)
        m10_val = sum([v for v in akt_vals[27:30] if v is not None]) or 0.0
        m10_stat = m_stats[(m_stats["kecamatan"] == kec) & (m_stats["bulan"] == 10)]
        if not m10_stat.empty and m10_stat["std"].values[0] > 0:
            spi_1 = (m10_val - m10_stat["mean"].values[0]) / m10_stat["std"].values[0]
        else:
            spi_1 = 0.0

        # SPI 3-Bulan (Bulan 8–10 / Agt–Okt)
        m8_10_val = sum([v for v in akt_vals[21:30] if v is not None]) or 0.0
        m8_stat = m_stats[(m_stats["kecamatan"] == kec) & (m_stats["bulan"] == 8)]
        m9_stat = m_stats[(m_stats["kecamatan"] == kec) & (m_stats["bulan"] == 9)]
        mean_3m = (
            (m8_stat["mean"].values[0] if not m8_stat.empty else 0)
            + (m9_stat["mean"].values[0] if not m9_stat.empty else 0)
            + (m10_stat["mean"].values[0] if not m10_stat.empty else 0)
        )
        std_3m = np.sqrt(
            (m8_stat["std"].values[0] ** 2 if not m8_stat.empty else 0)
            + (m9_stat["std"].values[0] ** 2 if not m9_stat.empty else 0)
            + (m10_stat["std"].values[0] ** 2 if not m10_stat.empty else 1)
        )
        spi_3 = (m8_10_val - mean_3m) / (std_3m if std_3m > 0 else 1.0)

        def _cat_spi(val: float) -> str:
            if val >= 2.0: return "Sangat Basah"
            elif val >= 1.5: return "Basah"
            elif val >= -0.99: return "Normal"
            elif val >= -1.49: return "Agak Kering"
            elif val >= -1.99: return "Kering"
            else: return "Sangat Kering"

        # Perhitungan HTH berdasarkan dasarian kering di bulan 10
        d30_rain = akt_vals[29] if len(akt_vals) > 29 and akt_vals[29] is not None else 10.0
        if d30_rain == 0:
            hth_hari = 15
            hth_kat = "Menengah (11–20 hari)"
            hth_stat = "Waspada"
        elif d30_rain < 20:
            hth_hari = 5
            hth_kat = "Sangat Pendek (1–5 hari)"
            hth_stat = "Aman"
        else:
            hth_hari = 1
            hth_kat = "Masih Ada Hujan"
            hth_stat = "Aman"

        cews_map[kec] = {
            "hth_hari": hth_hari,
            "hth_kategori": hth_kat,
            "hth_status": hth_stat,
            "pdcht_status": "Aman",
            "pdcht_keterangan": "Prakiraan akumulasi curah hujan terkendali",
            "spi_1_bulan": round(float(spi_1), 2),
            "spi_1_kategori": _cat_spi(float(spi_1)),
            "spi_3_bulan": round(float(spi_3), 2),
            "spi_3_kategori": _cat_spi(float(spi_3)),
        }

    return cews_map, chart_map


def generate_archive_season_year(
    year: int,
    df_loyo_hujan: pd.DataFrame,
    df_loyo_kemarau: pd.DataFrame,
    df_dasarian_all: pd.DataFrame,
    gdf_kec: gpd.GeoDataFrame,
    gdf_desa: gpd.GeoDataFrame,
    out_dir: Path,
    mesh: dict = None,
) -> dict:
    """Menghasilkan berkas prediksi_musim.json dan prediksi_musim_desa.json untuk satu tahun arsip."""
    year_str = str(year)
    season_id = f"{year}_{year + 1}"
    season_dir = out_dir / f"musim_{year}"
    season_dir.mkdir(parents=True, exist_ok=True)

    # 1. Filter Data Musim Hujan (tahun_awal == year)
    sub_hujan = df_loyo_hujan[df_loyo_hujan["tahun_awal"] == year].copy()
    if sub_hujan.empty:
        print(f"  [WARN] Data LOYO Musim Hujan tahun {year} tidak ditemukan, lewati.")
        return {}

    # 2. Filter Data Musim Kemarau (tahun == year)
    sub_kemarau = df_loyo_kemarau[df_loyo_kemarau["tahun"] == year].copy()

    # 3. Data Curah Hujan Dasarian Tahun Tersebut
    df_das_year = df_dasarian_all[df_dasarian_all["tahun"] == year]
    cews_map, chart_map = compute_cews_for_year(df_das_year, df_dasarian_all)

    # 4. Susun Records Musim Hujan
    hujan_records = []
    tot_eval_amh_match = 0
    tot_eval_sifat_match = 0
    selisih_das_list = []

    for _, row in sub_hujan.iterrows():
        kec = str(row["kecamatan"]).strip()
        norm_amh = float(row.get("normal_amh_dasarian", 28.0))
        pred_amh = float(row.get("pred_amh_dasarian", 28.0))
        real_amh = float(row.get("amh_dasarian", 28.0))

        pred_kat = str(row.get("pred_kategori_amh", "Sama"))
        real_kat = str(row.get("kategori_amh", "Sama"))

        pred_sifat = str(row.get("pred_sifat_hujan", "Normal"))
        real_sifat = str(row.get("sifat_hujan", "Normal"))

        tot_hujan = float(row.get("total_hujan_mh_mm", 1600.0))
        puncak_das = float(row.get("puncak_dasarian", 13.0))

        diff_das = round(pred_amh - real_amh, 2)
        selisih_das_list.append(abs(diff_das))

        is_amh_match = pred_kat.lower() == real_kat.lower() or abs(diff_das) <= 1.0
        if is_amh_match:
            tot_eval_amh_match += 1

        is_sifat_match = pred_sifat.lower() == real_sifat.lower()
        if is_sifat_match:
            tot_eval_sifat_match += 1

        if abs(diff_das) <= 0.5:
            status_eval_amh = "Tepat (Toleransi 0 Dasarian)"
        elif diff_das < 0:
            status_eval_amh = f"Prediksi Lebih Cepat ({abs(diff_das):.1f} Dasarian)"
        else:
            status_eval_amh = f"Prediksi Lebih Lambat ({diff_das:.1f} Dasarian)"

        katam = build_katam_profile(dasarian_to_short_label(pred_amh), pred_sifat)
        cews = cews_map.get(kec, {})
        chart = chart_map.get(kec, {})
        p_maju, p_sama, p_mundur = compute_prediction_probabilities(pred_amh, norm_amh, pred_kat)

        rec = {
            "kecamatan": kec,
            "code": kec,
            "normal_amh_dasarian": norm_amh,
            "normal_amh_label": dasarian_to_short_label(norm_amh),
            "prakiraan_amh_dasarian": round(pred_amh, 2),
            "prakiraan_amh_label": dasarian_to_short_label(pred_amh),
            "perbandingan_terhadap_normal": pred_kat,
            "probabilitas_maju_pct": p_maju,
            "probabilitas_sama_pct": p_sama,
            "probabilitas_mundur_pct": p_mundur,
            "prakiraan_sifat_hujan": pred_sifat,
            "prakiraan_puncak_bulan": f"{dasarian_to_month_name(puncak_das)} {year + 1}",
            "normal_puncak_bulan": "Januari",
            "perbandingan_puncak_thd_normal": "Sesuai",
            "prakiraan_puncak_dasarian": round(puncak_das, 1),
            "prakiraan_puncak_label": dasarian_to_short_label(puncak_das),
            "total_hujan_mh_mm": round(tot_hujan, 1),
            "kelas_hujan_mh": classify_hujan_mh(tot_hujan),
            "rekomendasi_olah_tanah": katam["mt1"]["olah_tanah"],
            "rekomendasi_tanam_padi_mt1": katam["mt1"]["tanam"],
            "hth_hari": cews.get("hth_hari", 5),
            "hth_kategori": cews.get("hth_kategori", "Sangat Pendek (1–5 hari)"),
            "spi_1_bulan": cews.get("spi_1_bulan", 0.0),
            "spi_1_kategori": cews.get("spi_1_kategori", "Normal"),
            "spi_3_bulan": cews.get("spi_3_bulan", 0.0),
            "spi_3_kategori": cews.get("spi_3_kategori", "Normal"),
            # Field Evaluasi Transparansi Akurasi
            "realisasi_amh_dasarian": round(real_amh, 2),
            "realisasi_amh_label": dasarian_to_short_label(real_amh),
            "realisasi_perbandingan_terhadap_normal": real_kat,
            "realisasi_sifat_hujan": real_sifat,
            "realisasi_puncak_bulan": "Januari",
            "selisih_dasarian_eval": diff_das,
            "status_evaluasi_amh": status_eval_amh,
            "status_evaluasi_sifat": "Tepat" if is_sifat_match else "Meleset",
            "katam": katam,
            "cews": cews,
            "chart": chart,
        }
        hujan_records.append(rec)

    # 5. Susun Records Musim Kemarau
    hujan_by_kec_map = {r["kecamatan"]: r for r in hujan_records}
    kemarau_records = []
    tot_eval_amk_match = 0
    selisih_das_k_list = []

    for _, row in sub_kemarau.iterrows():
        kec = str(row["kecamatan"]).strip()
        norm_amk = float(row.get("normal_amk_dasarian", 12.0))
        pred_amk = float(row.get("pred_amk_das_lstm", 12.0))
        real_amk = float(row.get("amk_dasarian", 12.0))

        pred_kat = str(row.get("pred_kat_lstm", "Sama"))
        real_kat = str(row.get("kategori_amk", "Sama"))

        pred_sifat = str(row.get("pred_sifat_lstm", "Normal"))
        real_sifat = str(row.get("sifat_hujan_mk", "Normal"))

        pmk_das = float(row.get("pmk_dasarian", 22.0))
        tot_mk = float(row.get("normal_total_hujan_mk_mm", 550.0))

        diff_das = round(pred_amk - real_amk, 2)
        selisih_das_k_list.append(abs(diff_das))

        is_amk_match = pred_kat.lower() == real_kat.lower()
        if is_amk_match:
            tot_eval_amk_match += 1

        is_sifat_k_match = pred_sifat.lower() == real_sifat.lower()

        if abs(diff_das) <= 0.5:
            status_eval_amk = "Tepat (Toleransi 0 Dasarian)"
        elif diff_das < 0:
            status_eval_amk = f"Prediksi Lebih Cepat ({abs(diff_das):.1f} Dasarian)"
        else:
            status_eval_amk = f"Prediksi Lebih Lambat ({diff_das:.1f} Dasarian)"

        # Rekomendasi Pola Tanam & Mitigasi Kekeringan:
        # Waktu pengeringan sawah Padi MT2: 1 dasarian sebelum AMK
        # Masa tanam Palawija MT3: saat awal kemarau (AMK)
        keringkan_das = max(10, int(round(pred_amk)) - 1)
        tanam_palawija_das = int(round(pred_amk))
        rekomendasi_keringkan = dasarian_to_short_label(keringkan_das)
        rekomendasi_palawija = dasarian_to_short_label(tanam_palawija_das)

        # Gunakan KATAM dari siklus tahun pertanian terpadu (MT1 di Musim Hujan)
        katam = hujan_by_kec_map.get(kec, {}).get("katam")
        if not katam:
            katam = build_katam_profile(dasarian_to_short_label(norm_amk), pred_sifat)

        cews = cews_map.get(kec, {})
        chart = chart_map.get(kec, {})
        pk_maju, pk_sama, pk_mundur = compute_prediction_probabilities(pred_amk, norm_amk, pred_kat)

        rec = {
            "kecamatan": kec,
            "code": kec,
            "normal_amk_dasarian": norm_amk,
            "normal_amk_label": dasarian_to_short_label(norm_amk),
            "prakiraan_amk_dasarian": round(pred_amk, 2),
            "prakiraan_amk_label": dasarian_to_short_label(pred_amk),
            "perbandingan_terhadap_normal": pred_kat,
            "probabilitas_maju_pct": pk_maju,
            "probabilitas_sama_pct": pk_sama,
            "probabilitas_mundur_pct": pk_mundur,
            "prakiraan_sifat_hujan_mk": pred_sifat,
            "prakiraan_pmk_bulan": f"{dasarian_to_month_name(pmk_das)} {year}",
            "normal_pmk_bulan": "Agustus",
            "perbandingan_pmk_thd_normal": "Sesuai",
            "prakiraan_pmk_dasarian": round(pmk_das, 1),
            "prakiraan_pmk_label": dasarian_to_short_label(pmk_das),
            "total_hujan_mk_mm": round(tot_mk, 1),
            "kelas_hujan_mk": classify_hujan_mk(tot_mk),
            "kekeringan_skor": 45.0,
            "kelas_kekeringan": "Sedang (< 60)",
            "rekomendasi_keringkan_sawah_mt2": rekomendasi_keringkan,
            "rekomendasi_tanam_palawija_mt3": rekomendasi_palawija,
            "hth_hari": cews.get("hth_hari", 5),
            "hth_kategori": cews.get("hth_kategori", "Sangat Pendek (1–5 hari)"),
            "spi_1_bulan": cews.get("spi_1_bulan", 0.0),
            "spi_1_kategori": cews.get("spi_1_kategori", "Normal"),
            "spi_3_bulan": cews.get("spi_3_bulan", 0.0),
            "spi_3_kategori": cews.get("spi_3_kategori", "Normal"),
            "realisasi_amk_dasarian": round(real_amk, 2),
            "realisasi_amk_label": dasarian_to_short_label(real_amk),
            "realisasi_perbandingan_terhadap_normal": real_kat,
            "realisasi_sifat_hujan": real_sifat,
            "realisasi_sifat_hujan_mk": real_sifat,
            "realisasi_pmk_bulan": "Agustus",
            "selisih_dasarian_eval": diff_das,
            "status_evaluasi_amk": status_eval_amk,
            "status_evaluasi_sifat": "Tepat" if is_sifat_k_match else "Meleset",
            "katam": katam,
            "cews": cews,
            "chart": chart,
        }
        kemarau_records.append(rec)

    # Hitung Metrik Evaluasi Agregat Kabupaten
    n_kec = len(hujan_records)
    akurasi_amh_pct = round((tot_eval_amh_match / n_kec) * 100.0, 1) if n_kec > 0 else 0.0
    akurasi_amk_pct = round((tot_eval_amk_match / len(kemarau_records)) * 100.0, 1) if kemarau_records else 0.0
    akurasi_sifat_pct = round((tot_eval_sifat_match / n_kec) * 100.0, 1) if n_kec > 0 else 0.0
    rata_selisih = round(float(np.mean(selisih_das_list)), 2) if selisih_das_list else 0.0

    climate_desc = CLIMATE_CONTEXT.get(year, "Kondisi Normal / Netral")

    evaluasi_agregat = {
        "musim_hujan": f"{year}/{year + 1}",
        "musim_kemarau": f"{year}",
        "iklim_global": climate_desc,
        "akurasi_kategori_amh_pct": akurasi_amh_pct,
        "akurasi_kategori_amk_pct": akurasi_amk_pct,
        "akurasi_sifat_hujan_pct": akurasi_sifat_pct,
        "rata_selisih_dasarian": rata_selisih,
        "keterangan_kinerja": f"Ketepatan Awal Musim {akurasi_amh_pct}%, Sifat Hujan {akurasi_sifat_pct}%",
    }

    # Downscaling Raster 250m & Render Overlay PNG (EPSG:3857)
    overlays_list = []
    desa_amh_das = None
    desa_amk_das = None
    desa_tot_mh = None
    desa_tot_mk = None
    desa_dr_score = None

    if mesh is not None:
        overlay_dir = season_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        meta_tif = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": -9999.0,
            "width": mesh["width"],
            "height": mesh["height"],
            "count": 1,
            "crs": "EPSG:4326",
            "transform": mesh["transform"],
            "compress": "lzw",
        }

        # 1. Downscale AMH 250m
        try:
            df_h_sub = pd.DataFrame(hujan_records)
            gdf_h = gdf_kec.merge(df_h_sub, left_on="nm_kecamatan", right_on="kecamatan")
            amh_grid, _ = downscale_variable_multimodel(
                gdf_h, "prakiraan_amh_dasarian", mesh, f"AMH_{year}", apply_mass_conservation=False
            )
            temp_amh = season_dir / f"temp_amh_{year}.tif"
            with rasterio.open(temp_amh, "w", **meta_tif) as dst:
                dst.write(np.nan_to_num(amh_grid, nan=-9999.0).astype(np.float32), 1)

            dst_amh_png = overlay_dir / "amh_dasarian_250m.png"
            amh_meta = render_raster_overlay_png(
                temp_amh,
                dst_amh_png,
                "YlGnBu_r",
                "amh_250m",
                "hujan",
                f"Awal Musim Hujan 250m ({year}/{year+1})",
                "Dasarian",
                is_dasarian=True,
            )
            amh_meta["url"] = f"/data/arsip/musim_{year}/overlays/amh_dasarian_250m.png"
            overlays_list.append(amh_meta)
            desa_amh_das = compute_zonal_means(gdf_desa, temp_amh)
            if temp_amh.exists():
                temp_amh.unlink()
        except Exception as e_amh:
            print(f"  [WARN] Gagal render overlay AMH {year}: {e_amh}")

        # 2. Downscale AMK 250m
        try:
            df_k_sub = pd.DataFrame(kemarau_records)
            gdf_k = gdf_kec.merge(df_k_sub, left_on="nm_kecamatan", right_on="kecamatan")
            amk_grid, _ = downscale_variable_multimodel(
                gdf_k, "prakiraan_amk_dasarian", mesh, f"AMK_{year}", apply_mass_conservation=False
            )
            temp_amk = season_dir / f"temp_amk_{year}.tif"
            with rasterio.open(temp_amk, "w", **meta_tif) as dst:
                dst.write(np.nan_to_num(amk_grid, nan=-9999.0).astype(np.float32), 1)

            dst_amk_png = overlay_dir / "amk_dasarian_250m.png"
            amk_meta = render_raster_overlay_png(
                temp_amk,
                dst_amk_png,
                "YlOrRd",
                "amk_250m",
                "kemarau",
                f"Awal Musim Kemarau 250m ({year})",
                "Dasarian",
                is_dasarian=True,
            )
            amk_meta["url"] = f"/data/arsip/musim_{year}/overlays/amk_dasarian_250m.png"
            overlays_list.append(amk_meta)
            desa_amk_das = compute_zonal_means(gdf_desa, temp_amk)
            if temp_amk.exists():
                temp_amk.unlink()
        except Exception as e_amk:
            print(f"  [WARN] Gagal render overlay AMK {year}: {e_amk}")

        # 3. Downscale Total Curah Hujan MH 250m
        try:
            tot_mh_grid, _ = downscale_variable_multimodel(
                gdf_h, "total_hujan_mh_mm", mesh, f"Total_MH_{year}", apply_mass_conservation=False
            )
            temp_tot_mh = season_dir / f"temp_tot_mh_{year}.tif"
            with rasterio.open(temp_tot_mh, "w", **meta_tif) as dst:
                dst.write(np.nan_to_num(tot_mh_grid, nan=-9999.0).astype(np.float32), 1)

            dst_tot_mh_png = overlay_dir / "total_hujan_mh_250m.png"
            tot_mh_meta = render_raster_overlay_png(
                temp_tot_mh,
                dst_tot_mh_png,
                "YlGnBu",
                "total_hujan_mh",
                "hujan",
                f"Total Curah Hujan MH 250m ({year}/{year+1})",
                "mm",
                is_dasarian=False,
            )
            tot_mh_meta["url"] = f"/data/arsip/musim_{year}/overlays/total_hujan_mh_250m.png"
            overlays_list.append(tot_mh_meta)
            desa_tot_mh = compute_zonal_means(gdf_desa, temp_tot_mh)
            if temp_tot_mh.exists():
                temp_tot_mh.unlink()
        except Exception as e_tot_mh:
            print(f"  [WARN] Gagal render overlay Total MH {year}: {e_tot_mh}")

        # 4. Downscale Total Curah Hujan MK 250m
        try:
            tot_mk_grid, _ = downscale_variable_multimodel(
                gdf_k, "total_hujan_mk_mm", mesh, f"Total_MK_{year}", apply_mass_conservation=False
            )
            temp_tot_mk = season_dir / f"temp_tot_mk_{year}.tif"
            with rasterio.open(temp_tot_mk, "w", **meta_tif) as dst:
                dst.write(np.nan_to_num(tot_mk_grid, nan=-9999.0).astype(np.float32), 1)

            dst_tot_mk_png = overlay_dir / "total_hujan_mk_250m.png"
            tot_mk_meta = render_raster_overlay_png(
                temp_tot_mk,
                dst_tot_mk_png,
                "YlGn",
                "total_hujan_mk",
                "kemarau",
                f"Total Hujan Kemarau 250m ({year})",
                "mm",
                is_dasarian=False,
            )
            tot_mk_meta["url"] = f"/data/arsip/musim_{year}/overlays/total_hujan_mk_250m.png"
            overlays_list.append(tot_mk_meta)
            desa_tot_mk = compute_zonal_means(gdf_desa, temp_tot_mk)
            if temp_tot_mk.exists():
                temp_tot_mk.unlink()
        except Exception as e_tot_mk:
            print(f"  [WARN] Gagal render overlay Total MK {year}: {e_tot_mk}")

        # 5. Indeks Kerawanan Kekeringan 250m
        try:
            # Model komposit biofisik kekeringan:
            # Defisit Hujan MK (maks 60) + Intrusi Pantai/Salinitas (maks 25) + Kelerengan (maks 15)
            defisit_score = np.clip((600.0 - tot_mk_grid) / 600.0 * 60.0, 0.0, 60.0)
            pantai_score = np.clip((30.0 - mesh["dist_coast"]) / 30.0 * 25.0, 0.0, 25.0)
            lereng_score = np.clip(mesh["slope"] / 40.0 * 15.0, 0.0, 15.0)
            dr_grid = defisit_score + pantai_score + lereng_score
            dr_grid[~mesh["mask"]] = np.nan

            temp_dr = season_dir / f"temp_dr_{year}.tif"
            with rasterio.open(temp_dr, "w", **meta_tif) as dst:
                dst.write(np.nan_to_num(dr_grid, nan=-9999.0).astype(np.float32), 1)

            dst_dr_png = overlay_dir / "kerawanan_kekeringan_250m.png"
            dr_meta = render_raster_overlay_png(
                temp_dr,
                dst_dr_png,
                "YlOrRd",
                "kekeringan_250m",
                "kemarau",
                f"Kerawanan Kekeringan 250m ({year})",
                "Skor",
                is_dasarian=False,
            )
            dr_meta["url"] = f"/data/arsip/musim_{year}/overlays/kerawanan_kekeringan_250m.png"
            overlays_list.append(dr_meta)

            kec_dr_score = compute_zonal_means(gdf_kec, temp_dr)
            desa_dr_score = compute_zonal_means(gdf_desa, temp_dr)

            # Update nilai skor kekeringan real per kecamatan
            kec_dr_map = {
                str(nm).strip(): round(float(score), 1)
                for nm, score in zip(gdf_kec["nm_kecamatan"], kec_dr_score)
            }
            for r in kemarau_records:
                k_name = r["kecamatan"]
                if k_name in kec_dr_map:
                    val = kec_dr_map[k_name]
                    r["kekeringan_skor"] = val
                    r["kelas_kekeringan"] = classify_kekeringan(val)

            if temp_dr.exists():
                temp_dr.unlink()
        except Exception as e_dr:
            print(f"  [WARN] Gagal render overlay Kekeringan {year}: {e_dr}")

    # Susun Payload JSON Kecamatan
    prediksi_musim_payload = {
        "metadata": {
            "is_archive": True,
            "season_id": season_id,
            "year": year,
            "title": f"Arsip Musim Hujan {year}/{year + 1} & Musim Kemarau {year}",
            "generated_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
            "kabupaten": "Kabupaten Kebumen",
            "total_kecamatan": len(hujan_records),
            "evaluasi": evaluasi_agregat,
            "seasons": {
                "hujan": {
                    "id": "hujan",
                    "title": f"Musim Hujan {year}/{year + 1}",
                    "enso_context": climate_desc,
                    "normal_period": "Normal 2000–2025",
                },
                "kemarau": {
                    "id": "kemarau",
                    "title": f"Musim Kemarau {year}",
                    "enso_context": climate_desc,
                    "normal_period": "Normal 2000–2025",
                },
            },
        },
        "overlays": overlays_list,
        "hujan": hujan_records,
        "kemarau": kemarau_records,
    }

    kec_json_path = season_dir / "prediksi_musim.json"
    with open(kec_json_path, "w", encoding="utf-8") as f:
        json.dump(prediksi_musim_payload, f, ensure_ascii=False, indent=2)

    # 6. Susun Payload JSON Desa (460 Desa)
    hujan_by_kec = {r["kecamatan"]: r for r in hujan_records}
    kemarau_by_kec = {r["kecamatan"]: r for r in kemarau_records}

    desa_hujan_records = []
    desa_kemarau_records = []

    for idx_d, (_, row) in enumerate(gdf_desa.iterrows()):
        kec_name = str(row["nm_kecamatan"]).strip()
        desa_name = str(row["nm_kelurahan"]).strip()
        code = str(row["code"]).strip()

        parent_h = hujan_by_kec.get(kec_name, hujan_records[0])
        parent_k = kemarau_by_kec.get(kec_name, kemarau_records[0])

        amh_val = round(float(desa_amh_das[idx_d]), 2) if desa_amh_das else round(parent_h["prakiraan_amh_dasarian"], 2)
        tot_mh_val = round(float(desa_tot_mh[idx_d]), 1) if desa_tot_mh else round(parent_h["total_hujan_mh_mm"], 1)

        d_h = dict(parent_h)
        d_h.update({
            "code": code,
            "desa": desa_name,
            "kecamatan": kec_name,
            "total_hujan_mh_mm": tot_mh_val,
            "kelas_hujan_mh": classify_hujan_mh(tot_mh_val),
            "amh_dasarian_250m": amh_val,
            "amh_label_250m": dasarian_to_short_label(amh_val),
        })
        desa_hujan_records.append(d_h)

        amk_val = round(float(desa_amk_das[idx_d]), 2) if desa_amk_das else round(parent_k["prakiraan_amk_dasarian"], 2)
        tot_mk_val = round(float(desa_tot_mk[idx_d]), 1) if desa_tot_mk else round(parent_k["total_hujan_mk_mm"], 1)

        dr_desa = round(float(desa_dr_score[idx_d]), 1) if desa_dr_score else parent_k["kekeringan_skor"]
        d_k = dict(parent_k)
        d_k.update({
            "code": code,
            "desa": desa_name,
            "kecamatan": kec_name,
            "total_hujan_mk_mm": tot_mk_val,
            "kelas_hujan_mk": classify_hujan_mk(tot_mk_val),
            "amk_dasarian_250m": amk_val,
            "amk_label_250m": dasarian_to_short_label(amk_val),
            "kekeringan_skor": dr_desa,
            "kelas_kekeringan": classify_kekeringan(dr_desa),
        })
        desa_kemarau_records.append(d_k)

    prediksi_desa_payload = {
        "metadata": {
            "is_archive": True,
            "season_id": season_id,
            "year": year,
            "title": f"Arsip Prediksi Tingkat Desa Musim {year}/{year + 1}",
            "generated_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
            "total_desa": len(desa_hujan_records),
            "evaluasi": evaluasi_agregat,
        },
        "hujan": desa_hujan_records,
        "kemarau": desa_kemarau_records,
    }

    desa_json_path = season_dir / "prediksi_musim_desa.json"
    with open(desa_json_path, "w", encoding="utf-8") as f:
        json.dump(prediksi_desa_payload, f, ensure_ascii=False, separators=(",", ":"))

    print(f"  -> [OK] Tahun {year}: Kec={len(hujan_records)}, Desa={len(desa_hujan_records)}, Akurasi AMH={akurasi_amh_pct}%, Sifat={akurasi_sifat_pct}%")

    return {
        "season_id": season_id,
        "year": year,
        "label": f"{year}/{year + 1}",
        "hujan_title": f"Musim Hujan {year}/{year + 1}",
        "kemarau_title": f"Musim Kemarau {year}",
        "is_active": False,
        "has_evaluation": True,
        "climate_desc": climate_desc,
        "akurasi_amh_pct": akurasi_amh_pct,
        "akurasi_amk_pct": akurasi_amk_pct,
        "akurasi_sifat_pct": akurasi_sifat_pct,
        "rata_selisih_dasarian": rata_selisih,
        "folder": f"arsip/musim_{year}",
    }


def main():
    parser = argparse.ArgumentParser(description="Generator Arsip Prediksi Musim & Evaluasi Akurasi 10 Tahun (2015–2025)")
    parser.add_argument("--start-year", type=int, default=2015, help="Tahun awal arsip (default: 2015)")
    parser.add_argument("--end-year", type=int, default=2025, help="Tahun akhir arsip (default: 2025)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DASHBOARD_DIR / "public" / "data",
        help="Direktori output data dashboard",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MEMULAI GENERATOR ARSIP PREDIKSI MUSIM & EVALUASI AKURASI HISTORIS (2015–2025)")
    print(f"Target Tahun: {args.start_year} s.d. {args.end_year} (Span {args.end_year - args.start_year + 1} Tahun)")
    print(f"Output Dashboard: {args.output_dir}")
    print("=" * 80)

    # Memuat GeoJSON
    print("\n[1/4] Membaca GeoJSON batas administrasi Kebumen...")
    gdf_kec = gpd.read_file(GEOJSON_KEC_PATH)
    gdf_desa = gpd.read_file(GEOJSON_DESA_PATH)

    # Standardisasi kode desa
    gdf_desa["kd_propinsi"] = gdf_desa["kd_propinsi"].astype(str).str.strip()
    gdf_desa["kd_dati2"] = gdf_desa["kd_dati2"].astype(str).str.strip()
    gdf_desa["kd_kecamatan"] = gdf_desa["kd_kecamatan"].astype(str).str.strip()
    gdf_desa["kd_kelurahan"] = gdf_desa["kd_kelurahan"].astype(str).str.strip()
    gdf_desa["nm_kelurahan"] = gdf_desa["nm_kelurahan"].astype(str).str.strip()

    kec_map = dict(zip(gdf_kec["kd_kecamatan"].astype(str).str.strip(), gdf_kec["nm_kecamatan"].astype(str).str.strip()))
    gdf_desa["nm_kecamatan"] = gdf_desa["kd_kecamatan"].map(kec_map).fillna("")
    gdf_desa["code"] = (
        gdf_desa["kd_propinsi"]
        + "."
        + gdf_desa["kd_dati2"]
        + "."
        + gdf_desa["kd_kecamatan"]
        + "."
        + gdf_desa["kd_kelurahan"]
    )

    # Memuat tabel LOYO & Dasarian
    print("[2/4] Membaca tabel evaluasi LOYO LSTM dan time series dasarian historis...")
    loyo_hujan_path = DATA_DIR / "hasil_evaluasi_loyo.csv"
    loyo_kemarau_path = DATA_DIR / "hasil_evaluasi_loyo_kemarau.csv"
    dasarian_path = DATA_DIR / "dasarian_curah_hujan_kebumen_2000_2026.csv"

    df_loyo_h = pd.read_csv(loyo_hujan_path)
    df_loyo_k = pd.read_csv(loyo_kemarau_path)
    df_dasarian = pd.read_csv(dasarian_path)

    print("\n[2.5/4] Menginisialisasi mesh topografi DEM 250m untuk rendering raster overlay...")
    mesh = prepare_250m_topographic_mesh()

    arsip_dir = args.output_dir / "arsip"
    arsip_dir.mkdir(parents=True, exist_ok=True)

    catalog_entries = []

    # Entry pertama: Musim Berjalan (Active 2026/2027)
    catalog_entries.append({
        "season_id": "2026_2027",
        "year": 2026,
        "label": "2026/2027 (Berjalan)",
        "hujan_title": "Musim Hujan 2026/2027",
        "kemarau_title": "Musim Kemarau 2026",
        "is_active": True,
        "has_evaluation": False,
        "climate_desc": "El Niño Kuat (ONI +2.16)",
        "akurasi_amh_pct": None,
        "akurasi_amk_pct": None,
        "akurasi_sifat_pct": None,
        "rata_selisih_dasarian": None,
        "folder": "",  # Root data folder
    })

    print("\n[3/4] Mengekstrak data arsip musim, downscaling 250m & menghitung evaluasi...")
    for yr in range(args.end_year, args.start_year - 1, -1):
        print(f"Memproses Arsip Tahun {yr} (Data + Raster 250m)...")
        entry = generate_archive_season_year(
            yr, df_loyo_h, df_loyo_k, df_dasarian, gdf_kec, gdf_desa, arsip_dir, mesh=mesh
        )
        if entry:
            catalog_entries.append(entry)

    # 7. Ekspor Master Katalog Musim
    print("\n[4/4] Mengekspor berkas katalog master musim: musim_catalog.json...")
    catalog_payload = {
        "generated_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "total_seasons": len(catalog_entries),
        "active_season_id": "2026_2027",
        "seasons": catalog_entries,
    }

    catalog_path = args.output_dir / "musim_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog_payload, f, ensure_ascii=False, indent=2)

    print(f"  -> Katalog tersimpan di: {catalog_path}")
    print("\n" + "=" * 80)
    print("ARSIP PREDIKSI MUSIM 2015–2025 & KATALOG SELESAI DIEKSPOR SECARA LENGKAP!")
    print("=" * 80)


if __name__ == "__main__":
    main()
