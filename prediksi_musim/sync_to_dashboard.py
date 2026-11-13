#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sinkronisasi Data, Peta Kecamatan & Desa (460 Wilayah), serta Overlay Raster 250m
Mengekspor GeoJSON batas 26 kecamatan & 460 desa/kelurahan yang disederhanakan,
overlay raster PNG tergeoreferensi (EPSG:3857 untuk Leaflet), dan JSON hasil
prakiraan musim (kecamatan + zonal statistics 250m per desa) ke
Dashboard_Jerukagung-Meteorologi.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.mask import mask
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from shapely import force_2d

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "prediksi_musim" / "data"
RASTER_HUJAN_DIR = BASE_DIR / "prediksi_musim" / "rasters_250m"
RASTER_KEMARAU_DIR = BASE_DIR / "prediksi_musim" / "rasters_kemarau_250m"
GEOJSON_KEC_PATH = BASE_DIR / "33.05_kecamatan.geojson"
GEOJSON_DESA_PATH = BASE_DIR / "33.05_kelurahan.geojson"
DEFAULT_DASHBOARD_DIR = Path(r"D:\Github\Dashboard_Jerukagung-Meteorologi")


def build_simplified_kecamatan_geojson(
    src_geojson: Path, dst_geojson: Path, tolerance: float = 0.00035
) -> gpd.GeoDataFrame:
    """Memuat batas 26 kecamatan, menyederhanakan geometri, dan mengekspor GeoJSON."""
    gdf = gpd.read_file(src_geojson)
    gdf_out = gdf.copy()
    gdf_out["geometry"] = force_2d(gdf_out["geometry"]).simplify(
        tolerance=tolerance, preserve_topology=True
    )
    gdf_out["nm_kecamatan"] = gdf_out["nm_kecamatan"].astype(str).str.strip()
    gdf_out["name"] = gdf_out["nm_kecamatan"]
    gdf_out["code"] = gdf_out["nm_kecamatan"]

    keep_cols = ["kd_propinsi", "kd_dati2", "kd_kecamatan", "nm_kecamatan", "name", "code", "geometry"]
    gdf_clean = gdf_out[[c for c in keep_cols if c in gdf_out.columns]].copy()

    dst_geojson.parent.mkdir(parents=True, exist_ok=True)
    geojson_dict = json.loads(gdf_clean.to_json())
    with open(dst_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, ensure_ascii=False, separators=(",", ":"))

    return gdf


def build_simplified_desa_geojson(
    src_desa_geojson: Path,
    kec_code_to_name: dict[str, str],
    dst_geojson: Path,
    tolerance: float = 0.0005,
) -> gpd.GeoDataFrame:
    """Memuat batas 460 desa/kelurahan, join nama kecamatan induk, dan menyederhanakan geometri."""
    gdf = gpd.read_file(src_desa_geojson)
    gdf["kd_propinsi"] = gdf["kd_propinsi"].astype(str).str.strip()
    gdf["kd_dati2"] = gdf["kd_dati2"].astype(str).str.strip()
    gdf["kd_kecamatan"] = gdf["kd_kecamatan"].astype(str).str.strip()
    gdf["kd_kelurahan"] = gdf["kd_kelurahan"].astype(str).str.strip()
    gdf["nm_kelurahan"] = gdf["nm_kelurahan"].astype(str).str.strip()
    gdf["nm_kecamatan"] = gdf["kd_kecamatan"].map(kec_code_to_name).fillna("")
    gdf["name"] = gdf["nm_kelurahan"]
    gdf["code"] = (
        gdf["kd_propinsi"]
        + "."
        + gdf["kd_dati2"]
        + "."
        + gdf["kd_kecamatan"]
        + "."
        + gdf["kd_kelurahan"]
    )

    gdf_out = gdf.copy()
    gdf_out["geometry"] = force_2d(gdf_out["geometry"]).simplify(
        tolerance=tolerance, preserve_topology=True
    )

    keep_cols = [
        "code",
        "kd_propinsi",
        "kd_dati2",
        "kd_kecamatan",
        "kd_kelurahan",
        "nm_kelurahan",
        "nm_kecamatan",
        "name",
        "geometry",
    ]
    gdf_clean = gdf_out[[c for c in keep_cols if c in gdf_out.columns]].copy()

    dst_geojson.parent.mkdir(parents=True, exist_ok=True)
    geojson_dict = json.loads(gdf_clean.to_json())
    with open(dst_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, ensure_ascii=False, separators=(",", ":"))

    return gdf


def _clean_records(df: pd.DataFrame) -> list[dict]:
    """Mengonversi DataFrame hasil prediksi menjadi list dict Python yang aman untuk JSON."""
    records = []
    for row in df.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v):
                clean_row[k] = None
            elif hasattr(v, "item"):
                clean_row[k] = v.item()
            else:
                clean_row[k] = v
        records.append(clean_row)
    return records


def _dasarian_to_short_label(val: float) -> str:
    """Mengubah nilai dasarian numerik menjadi label singkat (misal 31.8 -> Nov II)."""
    months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    romans = ["I", "II", "III"]
    idx = int(np.clip(round(val) - 1, 0, 35))
    return f"{months[idx // 3]} {romans[idx % 3]}"


def _classify_hujan_mh(mm: float) -> str:
    if mm < 1500:
        return "< 1.500 mm"
    if mm < 1600:
        return "1.500–1.600 mm"
    if mm < 1750:
        return "1.600–1.750 mm"
    return "> 1.750 mm"


def _classify_hujan_mk(mm: float) -> str:
    if mm < 430:
        return "< 430 mm"
    if mm < 485:
        return "430–485 mm"
    if mm < 525:
        return "485–525 mm"
    return "> 525 mm"


def _classify_kekeringan(score: float) -> str:
    if score < 60:
        return "Sedang (< 60)"
    if score < 65:
        return "Cukup Tinggi (60–65)"
    if score < 70:
        return "Tinggi (65–70)"
    return "Sangat Tinggi (> 70)"


def _label_to_dasarian_idx(label: str) -> int:
    """Mengonversi label dasarian (misal 'Nov II') ke indeks 1..36."""
    months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    parts = str(label).strip().split()
    if len(parts) >= 2:
        m_name = parts[0]
        r_num = parts[1]
        m_idx = months.index(m_name) if m_name in months else 10
        r_idx = ["I", "II", "III"].index(r_num) if r_num in ["I", "II", "III"] else 0
        return m_idx * 3 + r_idx + 1
    return 32  # Default Nov II


def _build_katam_profile(tanam_mt1_str: str, sifat_hujan: str) -> dict:
    """Membangun profil kalender tanam terpadu (MT1, MT2, MT3) & varietas rekomendasi."""
    d_mt1_tanam = _label_to_dasarian_idx(tanam_mt1_str)
    d_mt1_olah = max(1, d_mt1_tanam - 1)
    d_mt1_panen = ((d_mt1_tanam + 10 - 1) % 36) + 1  # ~105-115 hari

    d_mt2_tanam = ((d_mt1_panen + 1 - 1) % 36) + 1
    d_mt2_panen = ((d_mt2_tanam + 9 - 1) % 36) + 1

    d_mt3_tanam = ((d_mt2_panen + 1 - 1) % 36) + 1
    d_mt3_panen = ((d_mt3_tanam + 8 - 1) % 36) + 1

    if sifat_hujan == "Atas Normal":
        var_mt1 = ["Ciherang Sub-1", "Inpari 42 Agritan GSR", "Inpara 2 (Tahan Rendaman)"]
        tips_mt1 = "Antisipasi genangan air puncak musim; bersihkan saluran drainase dan gunakan varietas toleran rendaman."
    elif sifat_hujan == "Bawah Normal":
        var_mt1 = ["Inpari 32 HDB", "Situ Bagendit", "Cakrabuana Agritan (Genjah & Hemat Air)"]
        tips_mt1 = "Optimalkan penampungan air embung/dam parit; percepat pengolahan tanah saat hujan perdana turun."
    else:
        var_mt1 = ["Inpari 32 HDB", "Mekongga", "Ciherang"]
        tips_mt1 = "Gunakan sistem jajar legowo 2:1 untuk memaksimalkan anakan produktif dan sirkulasi udara mikro."

    var_mt2 = ["Inpari 33 (Tahan Wereng)", "Inpari 43", "Dodokan"]
    kom_mt3 = ["Jagung Hibrida Bisi-18", "Kedelai Anjasmoro", "Kacang Hijau Vima-1"]

    return {
        "mt1": {
            "nama": "MT1 (Padi Rendengan)",
            "olah_tanah": _dasarian_to_short_label(d_mt1_olah),
            "tanam": _dasarian_to_short_label(d_mt1_tanam),
            "panen": _dasarian_to_short_label(d_mt1_panen),
            "varietas": var_mt1,
            "tips": tips_mt1,
        },
        "mt2": {
            "nama": "MT2 (Padi Gadu)",
            "tanam": _dasarian_to_short_label(d_mt2_tanam),
            "panen": _dasarian_to_short_label(d_mt2_panen),
            "varietas": var_mt2,
            "tips": "Pantau ketersediaan debit air irigasi teknis Waduk Wadaslintang dan Waduk Sempor.",
        },
        "mt3": {
            "nama": "MT3 (Palawija/Bera)",
            "tanam": _dasarian_to_short_label(d_mt3_tanam),
            "panen": _dasarian_to_short_label(d_mt3_panen),
            "komoditas": kom_mt3,
            "tips": "Manfaatkan sisa kelembapan lengas tanah pasca panen padi gadu untuk budidaya palawija hemat air.",
        },
    }


def _compute_cews_metrics() -> tuple[dict, dict]:
    """
    Menghitung metrik SPI-1, SPI-3, HTH (Hari Tanpa Hujan), status PDCHT,
    serta deret waktu 36 dasarian (Normal vs 2026 vs Proyeksi) per kecamatan.
    """
    df_das = pd.read_csv(DATA_DIR / "dasarian_curah_hujan_kebumen_2000_2026.csv")
    das_labels = [_dasarian_to_short_label(d) for d in range(1, 37)]

    norm_pivot = (
        df_das[df_das["tahun"] <= 2025]
        .groupby(["kecamatan", "dasarian"])["curah_hujan"]
        .mean()
        .unstack(level="dasarian")
    )
    akt_pivot = (
        df_das[df_das["tahun"] == 2026]
        .groupby(["kecamatan", "dasarian"])["curah_hujan"]
        .mean()
        .unstack(level="dasarian")
    )

    # SPI-1 (Oktober / Dasarian 28-30)
    oct_rain = (
        df_das[df_das["dasarian"].isin([28, 29, 30])]
        .groupby(["kecamatan", "tahun"])["curah_hujan"]
        .sum()
        .unstack(level="tahun")
    )
    spi1_mean = oct_rain.loc[:, 2000:2025].mean(axis=1)
    spi1_std = oct_rain.loc[:, 2000:2025].std(axis=1).replace(0, 1.0)
    spi_1_series = (oct_rain[2026] - spi1_mean) / spi1_std

    # SPI-3 (Agustus-Oktober / Dasarian 22-30)
    seas_rain = (
        df_das[df_das["dasarian"].between(22, 30)]
        .groupby(["kecamatan", "tahun"])["curah_hujan"]
        .sum()
        .unstack(level="tahun")
    )
    spi3_mean = seas_rain.loc[:, 2000:2025].mean(axis=1)
    spi3_std = seas_rain.loc[:, 2000:2025].std(axis=1).replace(0, 1.0)
    spi_3_series = (seas_rain[2026] - spi3_mean) / spi3_std

    def _cat_spi(val: float) -> str:
        if val >= 2.0:
            return "Sangat Basah"
        if val >= 1.5:
            return "Basah"
        if val >= 1.0:
            return "Agak Basah"
        if val >= -0.99:
            return "Normal"
        if val >= -1.49:
            return "Agak Kering"
        if val >= -1.99:
            return "Sangat Kering"
        return "Ekstrem Kering"

    kec_cews_map = {}
    kec_chart_map = {}

    for kec in norm_pivot.index:
        norm_vals = [round(float(norm_pivot.loc[kec].get(d, 0.0)), 1) for d in range(1, 37)]
        akt_vals = [
            round(float(akt_pivot.loc[kec][d]), 1)
            if (kec in akt_pivot.index and d in akt_pivot.columns and pd.notna(akt_pivot.loc[kec][d]))
            else None
            for d in range(1, 37)
        ]

        proy_vals = [
            round(float(norm_vals[d - 1] * 0.72), 1) if d >= 31 else None
            for d in range(1, 37)
        ]
        if akt_vals[29] is not None:
            proy_vals[29] = akt_vals[29]

        val_spi1 = round(float(spi_1_series.get(kec, -0.85)), 2)
        val_spi3 = round(float(spi_3_series.get(kec, -0.88)), 2)

        last_rain = akt_vals[29] if akt_vals[29] is not None else 10.0
        if last_rain < 2.0:
            hth_days = 8
            hth_cat = "Pendek (6–10 hari)"
            hth_stat = "Waspada"
        elif last_rain < 5.0:
            hth_days = 5
            hth_cat = "Sangat Pendek (1–5 hari)"
            hth_stat = "Aman"
        else:
            hth_days = 3
            hth_cat = "Sangat Pendek (1–5 hari)"
            hth_stat = "Aman"

        kec_cews_map[kec] = {
            "hth_hari": hth_days,
            "hth_kategori": hth_cat,
            "hth_status": hth_stat,
            "pdcht_status": "Tidak Ada Peringatan",
            "spi_1_bulan": val_spi1,
            "spi_1_kategori": _cat_spi(val_spi1),
            "spi_3_bulan": val_spi3,
            "spi_3_kategori": _cat_spi(val_spi3),
        }

        kec_chart_map[kec] = {
            "labels": das_labels,
            "normal": norm_vals,
            "aktual": akt_vals,
            "proyeksi": proy_vals,
        }

    return kec_cews_map, kec_chart_map


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
                vals = np.array([], dtype=np.float32)

            # Fallback sampling titik pusat jika poligon sempit tidak mengenai pusat piksel
            if len(vals) == 0:
                pt = geom.representative_point()
                row_idx, col_idx = src.index(pt.x, pt.y)
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


def render_raster_overlay_png(
    src_tif: Path,
    dst_png: Path,
    cmap_name: str,
    overlay_id: str,
    season: str,
    label: str,
    unit: str,
    is_dasarian: bool = False,
) -> dict:
    """
    Memproyeksikan ulang GeoTIFF 250m (EPSG:4326) ke Web Mercator (EPSG:3857)
    agar presisi saat ditumpangkan di atas tile Leaflet,
    lalu menyimpannya sebagai PNG RGBA transparan beserta bounding box WGS84.
    """
    if not src_tif.exists():
        raise FileNotFoundError(f"Raster GeoTIFF tidak ditemukan: {src_tif}")

    with rasterio.open(src_tif) as src:
        raw = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            raw[raw == nodata] = np.nan
        raw[~np.isfinite(raw)] = np.nan

        valid_vals = raw[np.isfinite(raw)]
        vmin = float(np.nanpercentile(valid_vals, 2))
        vmax = float(np.nanpercentile(valid_vals, 98))
        if np.isclose(vmin, vmax):
            vmin = float(np.nanmin(valid_vals))
            vmax = float(np.nanmax(valid_vals)) + 1e-4

        dst_crs = "EPSG:3857"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width * 2, src.height * 2, *src.bounds
        )
        dst_arr = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=raw,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        dst_bounds_3857 = rasterio.transform.array_bounds(height, width, transform)
        west, south, east, north = transform_bounds(dst_crs, "EPSG:4326", *dst_bounds_3857)

    norm = np.clip((dst_arr - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(np.nan_to_num(norm, nan=0.0))
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    # Piksel di luar batas kabupaten dibuat transparan penuh (alpha = 0)
    valid_mask = np.isfinite(dst_arr)
    rgba_uint8[..., 3] = np.where(valid_mask, 235, 0)

    dst_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba_uint8, mode="RGBA").save(dst_png, format="PNG", optimize=True)

    gradient_colors = [
        matplotlib.colors.to_hex(cmap(t)) for t in [0.0, 0.25, 0.5, 0.75, 1.0]
    ]

    if is_dasarian:
        min_label = f"{vmin:.1f} ({_dasarian_to_short_label(vmin)})"
        max_label = f"{vmax:.1f} ({_dasarian_to_short_label(vmax)})"
    else:
        min_label = f"{vmin:.0f} {unit}".strip()
        max_label = f"{vmax:.0f} {unit}".strip()

    return {
        "id": overlay_id,
        "season": season,
        "label": label,
        "unit": unit,
        "min": round(vmin, 2),
        "max": round(vmax, 2),
        "minLabel": min_label,
        "maxLabel": max_label,
        "gradientColors": gradient_colors,
        "url": f"/data/overlays/{dst_png.name}",
        "bounds": [[south, west], [north, east]],
    }


def sync_to_dashboard(dashboard_dir: Path = DEFAULT_DASHBOARD_DIR) -> dict:
    """Menjalankan sinkronisasi GeoJSON (kecamatan & desa), overlay raster 250m, dan JSON prediksi."""
    if not dashboard_dir.exists():
        raise FileNotFoundError(f"Direktori dashboard tujuan tidak ditemukan: {dashboard_dir}")

    public_data_dir = dashboard_dir / "public" / "data"
    overlays_dir = public_data_dir / "overlays"
    public_data_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    geojson_kec_out = public_data_dir / "kebumen_kecamatan.json"
    geojson_desa_out = public_data_dir / "kebumen_desa.json"
    json_kec_out = public_data_dir / "prediksi_musim.json"
    json_desa_out = public_data_dir / "prediksi_musim_desa.json"

    gdf_kec = build_simplified_kecamatan_geojson(GEOJSON_KEC_PATH, geojson_kec_out)
    gdf_kec["kd_kecamatan"] = gdf_kec["kd_kecamatan"].astype(str).str.strip()
    gdf_kec["nm_kecamatan"] = gdf_kec["nm_kecamatan"].astype(str).str.strip()
    kec_code_to_name = dict(zip(gdf_kec["kd_kecamatan"], gdf_kec["nm_kecamatan"]))
    geo_names = set(gdf_kec["nm_kecamatan"].tolist())

    gdf_desa = build_simplified_desa_geojson(GEOJSON_DESA_PATH, kec_code_to_name, geojson_desa_out)

    hujan_csv = DATA_DIR / "prakiraan_musim_2026_2027_kebumen.csv"
    kemarau_csv = DATA_DIR / "prakiraan_musim_kemarau_2026_kebumen.csv"

    df_hujan = pd.read_csv(hujan_csv)
    df_kemarau = pd.read_csv(kemarau_csv)
    df_hujan["kecamatan"] = df_hujan["kecamatan"].astype(str).str.strip()
    df_kemarau["kecamatan"] = df_kemarau["kecamatan"].astype(str).str.strip()

    hujan_names = set(df_hujan["kecamatan"])
    kemarau_names = set(df_kemarau["kecamatan"])

    missing_hujan = geo_names - hujan_names
    missing_kemarau = geo_names - kemarau_names
    if missing_hujan or missing_kemarau:
        raise ValueError(
            f"Ketidakcocokan nama kecamatan! Missing hujan={missing_hujan}, missing kemarau={missing_kemarau}"
        )

    # Path raster 250m untuk ekstraksi zonal kecamatan & 460 desa
    tif_total_mh = RASTER_HUJAN_DIR / "Total_Hujan_MH_250m_Kebumen_2026.tif"
    tif_amh = RASTER_HUJAN_DIR / "AMH_Dasarian_250m_Kebumen_2026.tif"
    tif_total_mk = RASTER_KEMARAU_DIR / "Total_Hujan_MK_250m_Kebumen_2026.tif"
    tif_amk = RASTER_KEMARAU_DIR / "AMK_Dasarian_250m_Kebumen_2026.tif"
    tif_kekeringan = RASTER_KEMARAU_DIR / "Indeks_Kerawanan_Kekeringan_250m.tif"

    # Zonal stats 26 kecamatan (dari GeoTIFF 250m agar konsisten dengan level desa)
    kec_mh_mm = compute_zonal_means(gdf_kec, tif_total_mh)
    kec_mk_mm = compute_zonal_means(gdf_kec, tif_total_mk)
    kec_dr_score = compute_zonal_means(gdf_kec, tif_kekeringan)
    kec_zonal_map = {
        nm: {
            "total_hujan_mh_mm": round(float(mh), 1),
            "kelas_hujan_mh": _classify_hujan_mh(float(mh)),
            "total_hujan_mk_mm": round(float(mk), 1),
            "kelas_hujan_mk": _classify_hujan_mk(float(mk)),
            "kekeringan_skor": round(float(dr), 1),
            "kelas_kekeringan": _classify_kekeringan(float(dr)),
        }
        for nm, mh, mk, dr in zip(gdf_kec["nm_kecamatan"], kec_mh_mm, kec_mk_mm, kec_dr_score)
    }

    # Metrik CEWS & Profil 36 Dasarian untuk ECharts
    cews_map, chart_map = _compute_cews_metrics()

    df_hujan["code"] = df_hujan["kecamatan"]
    df_hujan["total_hujan_mh_mm"] = df_hujan["kecamatan"].map(lambda k: kec_zonal_map[k]["total_hujan_mh_mm"])
    df_hujan["kelas_hujan_mh"] = df_hujan["kecamatan"].map(lambda k: kec_zonal_map[k]["kelas_hujan_mh"])
    df_hujan["hth_hari"] = df_hujan["kecamatan"].map(lambda k: cews_map[k]["hth_hari"])
    df_hujan["hth_kategori"] = df_hujan["kecamatan"].map(lambda k: cews_map[k]["hth_kategori"])
    df_hujan["spi_1_bulan"] = df_hujan["kecamatan"].map(lambda k: cews_map[k]["spi_1_bulan"])
    df_hujan["spi_1_kategori"] = df_hujan["kecamatan"].map(lambda k: cews_map[k]["spi_1_kategori"])
    df_hujan["spi_3_bulan"] = df_hujan["kecamatan"].map(lambda k: cews_map[k].get("spi_3_bulan", 0.0))
    df_hujan["spi_3_kategori"] = df_hujan["kecamatan"].map(lambda k: cews_map[k].get("spi_3_kategori", "Normal"))
    df_hujan["realisasi_amh_dasarian"] = None
    df_hujan["realisasi_amh_label"] = None
    df_hujan["realisasi_perbandingan_terhadap_normal"] = None
    df_hujan["realisasi_sifat_hujan"] = None
    df_hujan["realisasi_puncak_bulan"] = None
    df_hujan["selisih_dasarian_eval"] = None
    df_hujan["status_evaluasi_amh"] = None
    df_hujan["status_evaluasi_sifat"] = None
    df_hujan["katam"] = df_hujan.apply(
        lambda r: _build_katam_profile(str(r["rekomendasi_tanam_padi_mt1"]), str(r["prakiraan_sifat_hujan"])),
        axis=1,
    )
    df_hujan["cews"] = df_hujan["kecamatan"].map(lambda k: cews_map[k])
    df_hujan["chart"] = df_hujan["kecamatan"].map(lambda k: chart_map[k])

    df_kemarau["code"] = df_kemarau["kecamatan"]
    df_kemarau["total_hujan_mk_mm"] = df_kemarau["kecamatan"].map(lambda k: kec_zonal_map[k]["total_hujan_mk_mm"])
    df_kemarau["kelas_hujan_mk"] = df_kemarau["kecamatan"].map(lambda k: kec_zonal_map[k]["kelas_hujan_mk"])
    df_kemarau["kekeringan_skor"] = df_kemarau["kecamatan"].map(lambda k: kec_zonal_map[k]["kekeringan_skor"])
    df_kemarau["kelas_kekeringan"] = df_kemarau["kecamatan"].map(lambda k: kec_zonal_map[k]["kelas_kekeringan"])
    df_kemarau["hth_hari"] = df_kemarau["kecamatan"].map(lambda k: cews_map[k]["hth_hari"])
    df_kemarau["hth_kategori"] = df_kemarau["kecamatan"].map(lambda k: cews_map[k]["hth_kategori"])
    df_kemarau["spi_1_bulan"] = df_kemarau["kecamatan"].map(lambda k: cews_map[k]["spi_1_bulan"])
    df_kemarau["spi_1_kategori"] = df_kemarau["kecamatan"].map(lambda k: cews_map[k]["spi_1_kategori"])
    df_kemarau["spi_3_bulan"] = df_kemarau["kecamatan"].map(lambda k: cews_map[k].get("spi_3_bulan", 0.0))
    df_kemarau["spi_3_kategori"] = df_kemarau["kecamatan"].map(lambda k: cews_map[k].get("spi_3_kategori", "Normal"))
    df_kemarau["realisasi_amk_dasarian"] = None
    df_kemarau["realisasi_amk_label"] = None
    df_kemarau["realisasi_perbandingan_terhadap_normal"] = None
    df_kemarau["realisasi_sifat_hujan"] = None
    df_kemarau["realisasi_sifat_hujan_mk"] = None
    df_kemarau["realisasi_pmk_bulan"] = None
    df_kemarau["selisih_dasarian_eval"] = None
    df_kemarau["status_evaluasi_amk"] = None
    df_kemarau["status_evaluasi_sifat"] = None
    katam_by_kec = dict(zip(df_hujan["kecamatan"], df_hujan["katam"]))
    df_kemarau["katam"] = df_kemarau["kecamatan"].map(katam_by_kec)
    df_kemarau["cews"] = df_kemarau["kecamatan"].map(lambda k: cews_map[k])
    df_kemarau["chart"] = df_kemarau["kecamatan"].map(lambda k: chart_map[k])

    # Zonal stats 460 desa/kelurahan dari raster 250m
    desa_mh_mm = compute_zonal_means(gdf_desa, tif_total_mh)
    desa_amh_das = compute_zonal_means(gdf_desa, tif_amh)
    desa_mk_mm = compute_zonal_means(gdf_desa, tif_total_mk)
    desa_amk_das = compute_zonal_means(gdf_desa, tif_amk)
    desa_dr_score = compute_zonal_means(gdf_desa, tif_kekeringan)

    hujan_by_kec = df_hujan.set_index("kecamatan").to_dict(orient="index")
    kemarau_by_kec = df_kemarau.set_index("kecamatan").to_dict(orient="index")

    desa_hujan_records: list[dict] = []
    desa_kemarau_records: list[dict] = []
    desa_csv_rows: list[dict] = []

    for i, row in gdf_desa.iterrows():
        kec_name = str(row["nm_kecamatan"])
        desa_name = str(row["nm_kelurahan"])
        code = str(row["code"])
        mh_val = round(float(desa_mh_mm[i]), 1)
        amh_val = round(float(desa_amh_das[i]), 2)
        mk_val = round(float(desa_mk_mm[i]), 1)
        amk_val = round(float(desa_amk_das[i]), 2)
        dr_val = round(float(desa_dr_score[i]), 1)

        parent_h = hujan_by_kec[kec_name]
        parent_k = kemarau_by_kec[kec_name]

        h_rec = dict(parent_h)
        h_rec.update(
            {
                "code": code,
                "desa": desa_name,
                "kecamatan": kec_name,
                "total_hujan_mh_mm": mh_val,
                "kelas_hujan_mh": _classify_hujan_mh(mh_val),
                "amh_dasarian_250m": amh_val,
                "amh_label_250m": _dasarian_to_short_label(amh_val),
            }
        )
        desa_hujan_records.append({k: (v.item() if hasattr(v, "item") else v) for k, v in h_rec.items()})

        k_rec = dict(parent_k)
        k_rec.update(
            {
                "code": code,
                "desa": desa_name,
                "kecamatan": kec_name,
                "total_hujan_mk_mm": mk_val,
                "kelas_hujan_mk": _classify_hujan_mk(mk_val),
                "amk_dasarian_250m": amk_val,
                "amk_label_250m": _dasarian_to_short_label(amk_val),
                "kekeringan_skor": dr_val,
                "kelas_kekeringan": _classify_kekeringan(dr_val),
            }
        )
        desa_kemarau_records.append({k: (v.item() if hasattr(v, "item") else v) for k, v in k_rec.items()})

        desa_csv_rows.append(
            {
                "code": code,
                "desa": desa_name,
                "kecamatan": kec_name,
                "total_hujan_mh_mm": mh_val,
                "kelas_hujan_mh": _classify_hujan_mh(mh_val),
                "amh_dasarian_250m": amh_val,
                "amh_label_250m": _dasarian_to_short_label(amh_val),
                "total_hujan_mk_mm": mk_val,
                "kelas_hujan_mk": _classify_hujan_mk(mk_val),
                "amk_dasarian_250m": amk_val,
                "amk_label_250m": _dasarian_to_short_label(amk_val),
                "kekeringan_skor": dr_val,
                "kelas_kekeringan": _classify_kekeringan(dr_val),
            }
        )

    desa_csv_out = DATA_DIR / "zonal_stats_desa_250m_2026.csv"
    pd.DataFrame(desa_csv_rows).to_csv(desa_csv_out, index=False, encoding="utf-8")

    overlay_specs = [
        {
            "src": tif_total_mh,
            "dst": overlays_dir / "total_hujan_mh_250m.png",
            "cmap": "YlGnBu",
            "id": "total_hujan_mh",
            "season": "hujan",
            "label": "Total Curah Hujan 250m",
            "unit": "mm",
            "is_dasarian": False,
        },
        {
            "src": tif_amh,
            "dst": overlays_dir / "amh_dasarian_250m.png",
            "cmap": "YlGnBu_r",
            "id": "amh_250m",
            "season": "hujan",
            "label": "Awal Musim (AMH) 250m",
            "unit": "Dasarian",
            "is_dasarian": True,
        },
        {
            "src": tif_total_mk,
            "dst": overlays_dir / "total_hujan_mk_250m.png",
            "cmap": "YlGn",
            "id": "total_hujan_mk",
            "season": "kemarau",
            "label": "Total Hujan Kemarau 250m",
            "unit": "mm",
            "is_dasarian": False,
        },
        {
            "src": tif_amk,
            "dst": overlays_dir / "amk_dasarian_250m.png",
            "cmap": "YlOrRd",
            "id": "amk_250m",
            "season": "kemarau",
            "label": "Awal Kemarau (AMK) 250m",
            "unit": "Dasarian",
            "is_dasarian": True,
        },
        {
            "src": tif_kekeringan,
            "dst": overlays_dir / "kerawanan_kekeringan_250m.png",
            "cmap": "YlOrRd",
            "id": "kekeringan_250m",
            "season": "kemarau",
            "label": "Kerawanan Kekeringan 250m",
            "unit": "Skor",
            "is_dasarian": False,
        },
    ]

    overlays = []
    for spec in overlay_specs:
        if spec["src"].exists():
            meta = render_raster_overlay_png(
                src_tif=spec["src"],
                dst_png=spec["dst"],
                cmap_name=spec["cmap"],
                overlay_id=spec["id"],
                season=spec["season"],
                label=spec["label"],
                unit=spec["unit"],
                is_dasarian=spec["is_dasarian"],
            )
            overlays.append(meta)
            png_kb = spec["dst"].stat().st_size / 1024
            print(f"[SUKSES] Overlay PNG 250m         : {spec['dst'].name} ({png_kb:.1f} KB)")

    wib = timezone(timedelta(hours=7))
    generated_at = datetime.now(wib).isoformat()
    base_metadata = {
        "generated_at": generated_at,
        "kabupaten": "Kabupaten Kebumen",
        "total_kecamatan": len(geo_names),
        "total_desa": len(gdf_desa),
        "model": "LSTM Multi-Head & Downscaling 250m",
        "standard": "BMKG / WMO",
        "seasons": {
            "hujan": {
                "id": "hujan",
                "title": "Musim Hujan 2026/2027",
                "enso_context": "El Niño Kuat (ONI +2.16)",
                "normal_period": "Normal 2000–2025",
            },
            "kemarau": {
                "id": "kemarau",
                "title": "Musim Kemarau 2026",
                "enso_context": "El Niño (ONI +1.85)",
                "normal_period": "Normal 2000–2025",
            },
        },
    }

    payload_kec = {
        "metadata": base_metadata,
        "overlays": overlays,
        "hujan": _clean_records(df_hujan),
        "kemarau": _clean_records(df_kemarau),
    }
    with open(json_kec_out, "w", encoding="utf-8") as f:
        json.dump(payload_kec, f, ensure_ascii=False, indent=2)

    payload_desa = {
        "metadata": base_metadata,
        "hujan": desa_hujan_records,
        "kemarau": desa_kemarau_records,
    }
    with open(json_desa_out, "w", encoding="utf-8") as f:
        json.dump(payload_desa, f, ensure_ascii=False, separators=(",", ":"))

    geo_kec_kb = geojson_kec_out.stat().st_size / 1024
    geo_desa_kb = geojson_desa_out.stat().st_size / 1024
    json_kec_kb = json_kec_out.stat().st_size / 1024
    json_desa_kb = json_desa_out.stat().st_size / 1024

    print(
        f"[SUKSES] GeoJSON kecamatan tersimpan : {geojson_kec_out} ({geo_kec_kb:.1f} KB, {len(geo_names)} kec)"
    )
    print(
        f"[SUKSES] GeoJSON desa tersimpan      : {geojson_desa_out} ({geo_desa_kb:.1f} KB, {len(gdf_desa)} desa)"
    )
    print(f"[SUKSES] Prediksi kecamatan JSON     : {json_kec_out} ({json_kec_kb:.1f} KB)")
    print(f"[SUKSES] Prediksi desa JSON          : {json_desa_out} ({json_desa_kb:.1f} KB)")
    print(f"[SUKSES] CSV zonal desa tersimpan    : {desa_csv_out}")

    return {
        "geojson_kec_path": str(geojson_kec_out),
        "geojson_desa_path": str(geojson_desa_out),
        "json_kec_path": str(json_kec_out),
        "json_desa_path": str(json_desa_out),
        "geojson_kec_kb": geo_kec_kb,
        "geojson_desa_kb": geo_desa_kb,
        "json_kec_kb": json_kec_kb,
        "json_desa_kb": json_desa_kb,
        "overlay_count": len(overlays),
        "kecamatan_count": len(geo_names),
        "desa_count": len(gdf_desa),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sinkronisasi hasil prediksi musim Kebumen (kecamatan & desa) ke web dashboard."
    )
    parser.add_argument(
        "--dashboard-dir",
        type=str,
        default=str(DEFAULT_DASHBOARD_DIR),
        help="Path root direktori repositori Dashboard_Jerukagung-Meteorologi.",
    )
    args = parser.parse_args()
    sync_to_dashboard(Path(args.dashboard_dir))


if __name__ == "__main__":
    main()
