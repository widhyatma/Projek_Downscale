# Project Rules: Downscaling & Geospatial Analysis

## 0. Python Environment: `tensorflow`
- **Wajib menggunakan conda environment `tensorflow`** untuk menjalankan semua script, pipeline, dan notebook di repositori ini:
  - Python Executable: `D:\conda_env\tensorflow\python.exe`
  - Perintah eksekusi: `D:\conda_env\tensorflow\python.exe <script.py>` atau `conda run -n tensorflow python <script.py>`
  - Environment ini sudah dilengkapi dengan `tensorflow`, `rasterio`, `geopandas`, `shapely`, `xarray`, `xgboost`, `scipy`, `sklearn`, `matplotlib`, `pandas`, dan `numpy`.

## 1. Bounding Box & Coordinate Extraction for Google Earth Engine (GEE)
- When defining `ee.Geometry.BBox(west, south, east, north)` from vector GeoJSON boundaries:
  - Always extract exact minimum/maximum longitude and latitude across all polygon vertices.
  - Remember negative latitude orientation in the Southern Hemisphere: northernmost latitude has a higher (less negative) value than southernmost latitude.
  - Apply a safety buffer ($\pm 0.01^\circ$ to $0.02^\circ$) around boundaries so that edge raster pixels are not cropped during resampling or downscaling.

## 2. Multi-Band Monthly Spatial Raster Analysis
- When processing multi-year monthly rasters (e.g. 12 bands/year):
  - Standardize visualization on a 3x4 grid per year (12 months).
  - Mask rasters using administrative boundaries with 2D conversion (`shapely.force_2d`).
  - Use unified colorbar normalization (`vmin`/`vmax`) across all subplots and years.
  - Overlay vector boundaries and include subplot summary statistics (mean, min, max).
  - Always compute multi-year climatological monthly averages as an overarching reference plot.
  - Calculate zonal statistics (mean, median, min, max, std) per administrative sub-district for downstream statistical downscaling.
