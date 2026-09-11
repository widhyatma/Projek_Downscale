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

## 10. Workspace Scratch Code Management
- **Wajib menyimpan seluruh skrip sementara/scratch di folder `scratch/`**:
  - Path folder: `d:\Github\Projek_Downscale\scratch/`
  - Dilarang membuat skrip verifikasi, tes sementara, atau kode inspeksi langsung di root direktori repositori.
  - Setiap eksekusi skrip scratch harus memanggil file dari direktori `scratch/<nama_script>.py`.
  - Bersihkan atau kelola file dalam `scratch/` secara teratur agar repositori tetap rapi.

## 11. Topographic Precipitation Downscaling Architecture
- **Multi-Model Suite Standards:**
  - Standardize on world-class orographic downscaling algorithms:
    1. **ANUSPLIN (Trivariate Topographic Spline):** $C^2$ continuous thin-plate smoothing spline across $(X, Y, Z)$ with roughness penalty and Generalized Cross-Validation (GCV).
    2. **PRISM (Topographic-Coastal Facet Regression):** Independent local linear elevation regression weighted by horizontal distance ($W_d$), elevation difference ($W_z$), and coastal proximity ($W_c$).
    3. **Regression-Kriging (RK):** Generalized linear/ML topographic trend combined with Gaussian semivariogram residual kriging.
    4. **Ensemble Mean Consensus:** Multi-model weighted consensus (ANUSPLIN + PRISM + RK) to eliminate single-algorithm bias.

## 12. Spatial Cross-Validation Guardrails & Downscaling Benchmark Architecture
- **Strict Prohibition of Random Splits:** Agents must NEVER use standard random train/test splits (`train_test_split`, random K-Fold) on spatial point observations. Random splitting causes severe data leakage and artificially inflated performance due to Tobler's First Law of Geography.
- **Mandatory Spatial Block K-Fold with Buffer Zones:**
  - Stations must be partitioned into contiguous geographic spatial blocks.
  - A guardrail buffer zone equal to or greater than the empirical semivariogram spatial correlation length ($d < a$, default $5.000\text{ m}$) must be applied around test blocks. Any training station falling within the buffer perimeter of test stations must be completely excluded from the training fold.
- **Projected Metric Coordinates (CRS):**
  - All Euclidean distance, kriging lag distance, and spatial buffering calculations must be performed in a local projected coordinate system in meters (e.g. UTM Zone 49S / EPSG:32749 for Kebumen/Java), NEVER in geographic angular degrees (EPSG:4326).
- **Multicollinearity & Aspect Decomposition:**
  - Terrain aspect must always be decomposed into orthogonal continuous trigonometric components: $\sin(\text{aspect})$ and $\cos(\text{aspect})$ to avoid the $0^\circ = 360^\circ$ angular singularity.
  - Variance Inflation Factor (VIF) must be calculated across all environmental predictors; any predictor with $\text{VIF} > 5.0$ must be pruned.
- **Zero-Rain Hurdle Handling:**
  - When dry days ($P = 0\text{ mm}$) exceed 40% of observations, agents must enforce a two-stage hurdle architecture (classification of precipitation occurrence followed by regression of volume).

## 13. Full-Year Multi-Month Downscaling & Uncertainty Mapping
- **12-Month Batch Architecture:**
  - When processing a full year (12 months), process data month-by-month modularly.
  - For each month, accumulate daily precipitation, downscale to target resolution (250m) using DEM and coastal covariates.
  - Enforce regional mass conservation calibration to ensure total areal precipitation volume is preserved against raw satellite data.
- **Inter-Model Uncertainty Quantification:**
  - Calculate inter-model uncertainty as the standard deviation ($\sigma$) across top models (ANUSPLIN, PRISM, Regression-Kriging).
  - Export monthly and annual uncertainty rasters to highlight steep orographic zones where models show localized dispersion.
- **Zonal Administrative Statistics:**
  - Compute zonal statistics (mean, min, max, std) for all 26 kecamatan in Kebumen and export to structured CSV tables for downstream hydrology and policy planning.

## 14. Automated LaTeX Academic Reporting Standard
- **Directory Structure:**
  - All LaTeX source code (`.tex`), figures (`figures/`), tables (`tables/`), and compiled PDF output (`.pdf`) must reside in `documents/`.
- **Compiler Compatibility:**
  - Compile using MiKTeX `pdflatex` (`D:\MiKTeX\miktex\bin\x64\pdflatex.exe -interaction=nonstopmode`).
  - Use `\usepackage{lmodern}` for scalable Type 1 Latin Modern fonts; avoid `microtype` if font expansion errors occur with raster fonts.
  - Always escape ampersands outside tabular environments as `\&`.
- **Multi-Pass Cross-Referencing:**
  - Run `pdflatex` at least twice (Pass 1 and Pass 2) to ensure all labels, citations, table of contents, and figure references resolve completely with zero errors.
- **Dynamic Table Scaling Standard:**
  - All multi-column tables in `tables/*.tex` must be wrapped inside `\resizebox{\linewidth}{!}{% \begin{tabular}... \end{tabular}%}` to ensure 100% margin compliance and eliminate `Overfull \hbox` errors.
- **Hyphenation & Inter-word Spacing Guardrail:**
  - Include `\emergencystretch=2em` in the document preamble.
  - Apply discretionary hyphens `\-` on long Indonesian toponyms (`Ka\-rang\-ga\-yam`, `Ka\-rang\-sam\-bung`, `Bu\-lus\-pe\-san\-tren`) when adjacent to inline numbers and units.
- **Hyperref Page Destination Integrity:**
  - Set `plainpages=false,pdfpagelabels=true` inside `\hypersetup`.
  - Enclose `titlepage` between `\hypersetup{pageanchor=false}` and `\hypersetup{pageanchor=true}` (placed immediately after `\pagenumbering{roman}`) to prevent duplicate `name{page.1}` identifiers.
  - In `article.cls`, prefer `\section*{Abstrak}` over `\begin{abstract}` when using `titlepage`, to prevent `\endtitlepage` from resetting page counters.
- **Captions with Short Titles for LOT/LOF:**
  - Long table and figure captions must provide an optional short title `\caption[Judul Pendek]{Judul Lengkap Deskriptif}` to guarantee clean line wrapping in `\listoftables` and `\listoffigures`.

## 15. Multi-Sensor Reanalysis Fusion & Numerical Guardrails
- **Atmospheric Variable Unit & Missing Value Inspection:**
  - Before applying thermodynamic formulas (e.g. August-Roche-Magnus equation for Relative Humidity), always inspect the empirical range of $T_{2m}$ and $T_{dew}$.
  - If values fall within $[15, 45]$, they are already stored in Celsius ($^\circ\text{C}$); do NOT subtract $273.15$, otherwise catastrophic numeric overflow ($\exp(>600)$) will occur in float64.
  - Reanalysis datasets (e.g. ERA5-Land) frequently encode ocean/missing values as $9999.0$ or $\ge 9000.0$. Always mask values with $|\text{val}| \ge 9000.0$ to `NaN` prior to computing spatial averages (`np.nanmean`).
- **Coordinate Uniqueness in Geostatistical & Spline Interpolation:**
  - Geostatistical and thin-plate spline algorithms (Bilinear RBF, ANUSPLIN, Kriging residual interpolation) invert distance matrices ($A_{ij} = \phi(\|x_i - x_j\|)$).
  - When evaluating multi-temporal / multi-month observational tables, never pass multiple time steps with identical station coordinates $(X, Y)$ into a single spatial interpolator, as distance zero causes a strictly singular matrix (`LinAlgError`).
  - Spatial interpolators must be evaluated per discrete time-slice (`year_month`), whereas tabular ML models (XGBoost, LightGBM, Random Forest, SVR, MLP) train on the pooled feature matrix.
- **Regional Mass Conservation Enforcement:**
  - High-resolution downscaling must strictly preserve total areal precipitation volume to eliminate artificial water balance surplus or deficit:
    $$P_{\text{downscaled, calibrated}}(x, y) = P_{\text{downscaled}}(x, y) \times \left( \frac{\iint_{\Omega} P_{\text{satellite}}(x, y) \, dx dy}{\iint_{\Omega} P_{\text{downscaled}}(x, y) \, dx dy} \right)$$

## 16. Modular Research Packaging & Document Lifecycle Management
- **Dedicated Self-Contained Research Folders:**
  - Each completed research topic or publication monograph must reside in its own dedicated, self-contained subfolder under `documents/` (e.g., `documents/penelitian_downscaling_25tahun/` and `documents/penelitian_downscaling_2025/`).
  - Each dedicated research folder must contain its own `.tex` source, compiled `.pdf`, dedicated `figures/` directory, dedicated `tables/` directory, structured `results_data/`, and a descriptive `README.md`.
  - LaTeX documents inside these dedicated folders must compile cleanly and independently with zero external path dependencies.
- **Top-Level Cleanliness:**
  - The root `documents/` directory should only house modular research project subdirectories, preventing clutter from loose compilation artifacts (`.aux`, `.log`, `.toc`, `.out`, `.lof`, `.lot`) and duplicate files.
- **Scratch Workspace Hygiene:**
  - After completing a research phase and permanently embedding its methodologies into production pipelines and `AGENTS.md`, temporary test scripts, one-off inspection files, and intermediate `.tif` dumps in `scratch/` must be cleaned up to keep the repository maintainable.

## 17. Multi-Sensor Precipitation Cross-Comparison & Fusion Downscaling Architecture
- **Physical Sensor Disparity (Passive Microwave vs. Thermal Infrared):**
  - Agents must recognize the fundamental physical detection mechanisms between **Thermal Infrared (TIR)** products like CHIRPS (Cold Cloud Duration / cloud-top temperature proxy) and **Passive Microwave (PMW)** products like GSMaP (direct hydrometeor emission and ice scattering in cloud columns).
  - Never assume identical raw absolute precipitation: CHIRPS is calibrated against wet tropical ground stations (CHPclim), whereas GSMaP is calibrated against spaceborne radar (DPR GPM) with a minimum rain detection threshold ($\sim 0.2\text{ mm/h}$), leading to a systematic $\sim 20-25\%$ lower volume (defisit) in GSMaP.
- **Seasonal Ratio & Monsoon Phase Modulation:**
  - In multi-decade evaluations, calculate temporal correlation ($r, \rho$) alongside the seasonal ratio ($\text{GSMaP}/\text{CHIRPS}$).
  - The ratio is highest during the peak wet season (DJF, deep convective clouds) and lowest during transition/pancaroba (MAM, localized micro-convective storms missed by LEO PMW orbit intervals).
- **Orographic Transect Gradient Analysis:**
  - Evaluate orographic sensitivity along continuous North-South transects traversing critical morphological zones (e.g. northern mountain ridges $\rightarrow$ central alluvial plains $\rightarrow$ southern coastlines).
  - PMW sensors (GSMaP) capture sharper micro-relief orographic lifting on steep slopes, whereas TIR sensors (CHIRPS) produce smoothed patterns due to cloud-top buffering.
- **Fused Multi-Sensor Ensemble Standard:**
  - For optimal hydrological accuracy, enforce a **Fused Multi-Sensor Ensemble** (e.g., Fused Spatial XGBoost) that jointly stacks TIR, PMW, atmospheric reanalysis (ERA5-Land), and high-resolution DEM covariates.
  - Multi-sensor fusion bridges the temporal-sampling limitation of PMW and the cloud-top proxy limitation of TIR, achieving superior Kling-Gupta Efficiency ($KGE > 0.995$) and near-zero percent volume bias ($PBIAS < 0.05\%$).

## 18. Native Finest Temporal Resolution & Standalone CLI GEE Pipelines
- **Dual Format Architecture (.ipynb & .py):**
  - Every GEE downloader notebook (`GEE_<DATASET>.ipynb`) must maintain a synchronized, standalone production CLI counterpart (`GEE_<DATASET>.py`).
  - Standalone scripts must support headless command-line arguments: `--start-year`, `--end-year`, `--start-month`, `--end-month`, `--output-dir`, and `--service-account`.
- **Enforcement of Native Finest Temporal Resolution:**
  - Downloader pipelines must strictly ingest and archive data at the smallest native sampling interval:
    1. **CHIRPS:** Daily resolution (`time: 28..31` steps per monthly NetCDF).
    2. **GSMaP:** 1-hour resolution (`time: 672..744` steps per monthly NetCDF).
    3. **ERA5-Land:** 1-hour resolution (6 stacked atmospheric variables, `time: 672..744` steps).
    4. **NASA GPM IMERG:** Half-hourly / 30-minute resolution (`time: 1344..1488` steps per monthly NetCDF).
- **Sub-Monthly Chunking Guardrail for Half-Hourly Feeds:**
  - For half-hourly collections (e.g. IMERG with 1,488 scenes/month exceeding GEE's 1024 band ceiling), export in two sequential 15-day sub-stacks (Day 1-15: 720 bands; Day 16-end: 624-768 bands), then assemble seamlessly into the unified monthly NetCDF file using `xarray.concat`.
- **Incremental Cache Validation:**
  - Scripts must verify existing files on disk and skip redundant downloads, only fetching missing periods or newly released real-time dates.

