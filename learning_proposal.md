# Learning Proposal: Standarisasi Script CLI GEE Standalone & Penegakan Resolusi Temporal Terkecil Asli

## 1. Identifikasi Masalah & Rationale
Pengunduhan data satelit dan reanalisis cuaca sering kali dilakukan di lingkungan Jupyter Notebook (`.ipynb`). Meskipun interaktif, penggunaan notebook memiliki limitasi untuk eksekusi terjadwal (*cron jobs*), otomatisasi *background pipeline*, dan pembaruan data bertahap (*incremental update*). Selain itu:
1. **Pemisahan Script Standalone CLI (.py):** Setiap notebook downloader (`GEE_CHIRPS`, `GEE_GSMAP`, `GEE_ERA5_Land`, `GEE_IMERG`) wajib memiliki padanan script Python murni (`.py`) yang mendukung argumen baris perintah (`argparse` untuk tahun, bulan, path output, dan kredensial).
2. **Penegakan Resolusi Waktu Terkecil Asli (*Native Finest Temporal Resolution*):**
   - **CHIRPS:** Resolusi waktu asli terkecil adalah **Harian (Daily)** (`UCSB-CHC/CHIRPS/V3/DAILY_SAT` & `DAILY_RNL`).
   - **GSMaP:** Resolusi waktu asli terkecil adalah **Per Jam (Hourly)** (`JAXA/GPM_L3/GSMaP/v8/operational`, band `hourlyPrecipRateGC`, 672--744 jam/bulan).
   - **ERA5-Land:** Resolusi waktu asli terkecil adalah **Per Jam (Hourly)** (`ECMWF/ERA5_LAND/HOURLY`, 6 variabel atmosferik, 672--744 jam/bulan).
   - **NASA GPM IMERG:** Resolusi waktu asli terkecil adalah **Setengah Jam (Half-Hourly / 30-menit)** (`NASA/GPM_L3/IMERG_V07`, 1.344--1.488 time-steps/bulan).
3. **Guardrail Band Export GEE untuk Data Setengah Jam (Rule 5):**
   - Karena citra 30-menit dalam 1 bulan menghasilkan 1.488 scene (>1024 band limit GEE), proses ekspor wajib dibagi menjadi dua interval 15-harian ($\le 768$ band), lalu digabungkan kembali (*concatenate*) secara lokal ke dalam satu file NetCDF bulanan utuh.

---

## 2. Klasifikasi Pembelajaran
- **Tipe:** Rule Update & New Rule Addition
- **Target:** Menambahkan `Rule 18. Native Finest Temporal Resolution & Standalone CLI GEE Pipelines` pada [AGENTS.md](file:///d:/Github/Projek_Downscale/.agents/AGENTS.md).

---

## 3. Rincian Usulan Penambahan Rule (Proposed Addition)

```markdown
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
```

---

## 4. Konfirmasi Pengguna
Apakah Anda menyetujui penambahan Rule 18 pada [AGENTS.md](file:///d:/Github/Projek_Downscale/.agents/AGENTS.md) ini?
