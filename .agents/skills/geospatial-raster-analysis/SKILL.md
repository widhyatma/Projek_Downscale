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
