---
name: geospatial-raster-analysis
description: Reusable workflows for multi-band GeoTIFF spatial time-series visualization, GEE BBox extraction, and zonal statistics in Python.
---

# Geospatial Raster Analysis & Visualization Workflow

## Environment & Execution
Always execute code using the `tensorflow` environment:
- Executable: `D:\conda_env\tensorflow\python.exe`
- Includes: `tensorflow`, `rasterio`, `geopandas`, `shapely`, `xarray`, `xgboost`, `scipy`, `sklearn`, `matplotlib`, `pandas`, `numpy`.

This skill outlines the standard pipeline for processing multi-band climate/satellite rasters (such as NDVI, CHIRPS, GSMaP, ERA5) with administrative vector boundaries.

## 1. Google Earth Engine Bounding Box Extraction
When exporting or filtering GEE image collections for an Area of Interest (AOI):
```python
import json

def get_gee_bbox(geojson_path, buffer_deg=0.01):
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    coords = []
    def extract(geom):
        t, c = geom['type'], geom['coordinates']
        if t == 'Polygon':
            for r in c: coords.extend(r)
        elif t == 'MultiPolygon':
            for poly in c:
                for r in poly: coords.extend(r)
    for feat in data['features']: extract(feat['geometry'])
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    return {
        'exact': f"ee.Geometry.BBox({min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f})",
        'buffered': f"ee.Geometry.BBox({min_lon-buffer_deg:.2f}, {min_lat-buffer_deg:.2f}, {max_lon+buffer_deg:.2f}, {max_lat+buffer_deg:.2f})"
    }
```

## 2. Multi-Band Monthly Grid Plot (3x4 Layout)
- **Masking:** Use `rasterio.mask.mask` with `unary_union(gdf.geometry)` and `shapely.force_2d(gdf.geometry)`.
- **Normalization:** Standardize `Normalize(vmin=..., vmax=...)` across all years.
- **Climatology:** Calculate multi-year monthly mean with `np.nanmean(month_stack, axis=0)`.
- **Zonal Stats:** Rasterize geometries with `rasterio.features.rasterize` to compute fast zonal aggregations.

## 3. GEE Multi-Dataset Hierarchical Pipeline
Structure data downloads systematically by product, year, month, and day/hour:
```
data/<jenis_data>/<tahun>/<tahun_bulan>/<nama_file_harian>.nc
```
Example: `data/chirps/2000/2000_01/chirps_2000_01_01.nc`

Use GEE monthly multi-band stacking (`ee.ImageCollection.toBands()`) to optimize throughput (30x faster), then export or slice into nested monthly/daily files.

## 4. Modular Notebook Isolation Policy
- Keep downloader notebooks dataset-specific (`GEE_CHIRPS.ipynb` for CHIRPS variants, `GEE_GSMAP.ipynb` for GSMaP, `GEE_ERA5_Land.ipynb` for ERA5-Land).
- Create a new dedicated notebook whenever integrating multiple datasets for downscaling or joint modeling.

## 5. GEE 1024-Band Export Limit Avoidance
- Never export single images exceeding 1024 bands via `geemap.ee_export_image` or `getPixels`.
- For multi-variable hourly data (e.g. ERA5-Land), perform per-variable monthly exports ($744 \le 1024$), then combine into NetCDF datasets with `xarray`.

## 6. Multi-Variable Atmospheric & Precipitation Analysis
When integrating reanalysis (e.g. ERA5-Land) with satellite precipitation (e.g. CHIRPS):
1. **Derive Atmospheric Variables:**
   - Relative Humidity ($RH$): $e_s = 6.112 \exp\left(\frac{17.625 \cdot T}{243.04 + T}\right)$, $e = 6.112 \exp\left(\frac{17.625 \cdot T_{dew}}{243.04 + T_{dew}}\right)$, $RH = \text{clip}(100 \times e/e_s, 0, 100)$.
   - Wind Speed ($WS$): $\sqrt{u_{10}^2 + v_{10}^2}$.
2. **Meteorological Visualization Suite:**
   - **3-Panel Hyetograph:** Daily rainfall bar chart + Cumulative curve + Temperature range envelope ($T_{max}, T_{avg}, T_{min}$).
   - **Diurnal Boxplots:** Hourly boxplot per day to inspect intraday temperature variability.
   - **Climate Anomaly Heatmap:** DOY daily anomaly matrix ($31 \times 12$) per year.
   - **Spatial Multi-Variable Map:** 4-panel spatial map overlaid with vector boundary GeoJSON.
3. **Structured Subfolder Storage:**
   - Always save figures to subfolders per year at 300 DPI: `plots_hyetograph_bulanan/<tahun>/`, `plots_boxplot_suhu/<tahun>/`, `plots_spasial_cuaca/<tahun>/`.
