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

## 3. Multi-Dataset Hierarchical Data Architecture
- All GEE and spatial satellite datasets must follow a standardized 4-tier nested folder structure:
  - `data/<jenis_data>/<tahun>/<tahun_bulan>/<file_harian_atau_per_jam>.nc` (e.g. `data/chirps/2000/2000_01/chirps_2000_01_01.nc`).
  - Supports modular scalability across multiple GEE products: `chirps`, `gsmap`, `era5_land`, `modis_ndvi`, etc.
  - Use GEE monthly multi-band stacking (`toBands()`) during retrieval to optimize API throughput 30x faster.

## 4. Modular Notebook Isolation
- Keep dataset downloader notebooks strictly dataset-specific and modular:
  - `GEE_CHIRPS.ipynb`: Exclusively dedicated to CHIRPS product variants (v2.0, v3.0, daily, monthly) for inter-product CHIRPS comparison.
  - `GEE_GSMAP.ipynb`: Exclusively dedicated to GSMaP products.
  - `GEE_ERA5_Land.ipynb`: Exclusively dedicated to ERA5-Land reanalysis.
- Always create a **new dedicated notebook** when combining multiple datasets for joint downscaling or comparative machine learning pipelines.

## 5. GEE Band Export Limits & Multi-Variable Stacking
- Google Earth Engine limits single image export (`getPixels` / `geemap.ee_export_image`) to a maximum of **1024 bands**.
- When retrieving multi-variable hourly datasets (like ERA5-Land with 744 hours x 6 variables = 4464 bands), perform **per-variable monthly stacking** (744 bands/variable <= 1024), then merge variables into a single NetCDF dataset (`xarray.Dataset`).

## 6. Multi-Variable Atmospheric & Precipitation Joint Analysis
- When combining reanalysis atmospheric datasets (e.g. `ERA5-Land`) and satellite precipitation (e.g. `CHIRPS` / `GSMaP`):
  - Standardize atmospheric parameter derivation:
    - **Relative Humidity (RH %):** August-Roche-Magnus equation ($T_{2m}$ and $T_{dew}$).
    - **Wind Speed (m/s):** Resultant vector magnitude ($WS = \sqrt{u_{10}^2 + v_{10}^2}$).
  - Implement standard meteorological visualization suites:
    - **3-Panel Monthly Hyetograph:** Daily rainfall bars, cumulative rainfall line, and Max/Avg/Min temperature envelope.
    - **Hourly Temperature Boxplots:** 24-hour diurnal distribution per day.
    - **Daily Temperature Anomaly Heatmaps:** 31 days x 12 months matrix against DOY climatological baseline.
    - **Spatial Multi-Variable Maps:** 4-panel grid with vector administrative boundaries (`33.05_kecamatan.geojson`).
  - Automatically organize output figures into hierarchical subfolders per year (e.g. `plots_hyetograph_bulanan/<tahun>/`) with 300 DPI print quality.

## 7. Memory-Safe Multi-Year Batch Processing Architecture (Anti-OOM)
- When processing multi-year NetCDF collections (hundreds of monthly files):
  - Do NOT open all multi-decade files simultaneously with un-chunked `open_mfdataset`.
  - Process data year-by-year (12 files/year), extract areal series, resample daily/monthly, and close datasets immediately (`ds.close()`).
  - Always call `plt.close('all')` and `gc.collect()` at the end of each batch loop iteration to eliminate memory leaks and avoid Kaggle `DeadKernelError`.
  - For spatial maps, load individual monthly NetCDF files directly on-the-fly (RAM < 10 MB per plot).

## 8. Social Media & Publication Aspect Ratio Standard (16:9 Standard)
- All meteorological and spatial visualization figures must be formatted with **16:9 Aspect Ratio** (`figsize=(16, 9)`, 300 DPI):
  - Ensures 100% upload compliance on Instagram (within 0.8 to 1.91 ratio limit), YouTube, Twitter/X, and Web presentations.
  - Apply `tight_layout()` and safe padding so titles, rotated annotations, and multi-axis legends never overlap.

## 9. Standardized Meteorological Plotting Suites (Aligned with Open Meteo Analytic)
- **Hyetograph Bulanan (2-Panel Twinx):**
  - Top: Daily rainfall bars (`dodgerblue` / `navy`), BMKG thresholds (20mm gold, 50mm orange, 100mm red), Cumulative line (`darkgreen` ●) with monthly total badge.
  - Bottom: 3-line temperature profile (Max `darkorange` ▲, Avg `limegreen` ■, Min `royalblue` ▼) with exact annotations and shaded temperature envelope.
- **Periode Tertentu (1-Panel Twinx):**
  - Single-panel dual-axis correlation: Left Y Suhu (`darkorange` ●) vs Right Y Curah Hujan (`dodgerblue` bars).
  - Saved to `periode_tertentu_plots/<tahun>/Periode_<target_waktu>.png`.

