# Learning Proposal: Integrasi Analisis Cuaca & Meteorologi Spasial (ERA5-Land + CHIRPS)

## 1. Konteks & Analisis Kebutuhan
Proyek ini mengintegrasikan dua dataset utama dari Google Earth Engine (GEE) yang telah diunduh ke dalam folder `data/`:
1. **ERA5-Land Reanalysis (Hourly):** 6 parameter atmosfer (`precipitation`, `temperature_2m`, `dewpoint_temperature_2m`, `u_wind_10m`, `v_wind_10m`, `surface_pressure`).
2. **CHIRPS Satellite (Daily/Monthly):** Observasi presipitasi curah hujan beresolusi tinggi (`precipitation`).

Pola analisis mengadopsi standar laporan meteorologi operasional BMKG/WMO (seperti pada `Open Meteo Analytic.ipynb`) dengan peningkatan kemampuan analisis spasial resolusi tinggi untuk Kabupaten Kebumen.

---

## 2. Parameter Fisik & Formula Derivasi Atmosfer
- **Kelembapan Relatif (Relative Humidity - RH %):**
  Dihitung secara presisi dari Suhu Udara ($T$) dan Suhu Titik Embun ($T_{dew}$) menggunakan persamaan August-Roche-Magnus:
  $$e_s(T) = 6.112 \times \exp\left(\frac{17.625 \cdot T}{243.04 + T}\right)$$
  $$e(T_{dew}) = 6.112 \times \exp\left(\frac{17.625 \cdot T_{dew}}{243.04 + T_{dew}}\right)$$
  $$RH = \text{clip}\left(\frac{e(T_{dew})}{e_s(T)} \times 100\%, 0, 100\right)$$

- **Kecepatan Angin ($WS$ m/s):**
  $$WS = \sqrt{u_{10m}^2 + v_{10m}^2}$$

---

## 3. Standar Visualisasi & Laporan Meteorologi
1. **Laporan Bulanan Hyetograph HD 3-Panel:**
   - **Panel 1:** Curah hujan harian (CHIRPS) dengan diagram batang berlabel nilai.
   - **Panel 2:** Akumulasi curah hujan bulanan kumulatif (*fill area*).
   - **Panel 3:** Profil suhu udara harian (Maksimum, Rata-rata, Minimum) dengan area arsir (*shaded area*).
2. **Boxplot Variabilitas Suhu Jam-jaman:**
   - Menampilkan sebaran statistik 24 jam untuk setiap tanggal dalam sebulan.
3. **Heatmap Matriks Anomali Suhu Harian (Hari 1–31 $\times$ Bulan 1–12):**
   - Dihitung terhadap baseline rata-rata harian (DOY climatology) dengan diverging colormap (`RdBu_r`).
4. **Peta Spasial 4-Panel Multi-Variabel:**
   - Menampilkan sebaran spasial Curah Hujan, Suhu Rata-rata, Kelembapan Relatif (RH), dan Kecepatan Angin dengan batas administratif kecamatan Kebumen.
5. **Hierarki Penyimpanan Plot Otomatis:**
   - Seluruh grafik otomatis disimpan ke subfolder per tahun dalam resolusi 300 DPI:
     - `analisis_cuaca_spasial/plots_hyetograph_bulanan/<tahun>/`
     - `analisis_cuaca_spasial/plots_boxplot_suhu/<tahun>/`
     - `analisis_cuaca_spasial/plots_heatmap_anomali_suhu/`
     - `analisis_cuaca_spasial/plots_spasial_cuaca/<tahun>/`

---

## 4. Rencana Pembaruan Rule / Skill
Setelah konfirmasi, pola ini akan ditambahkan ke:
- **`.agents/rules/` atau `.agents/AGENTS.md`**: Menambahkan bagian standardisasi analisis gabungan cuaca & presipitasi.
- **`.agents/skills/geospatial-raster-analysis/SKILL.md`**: Menambahkan alur kerja derivasi atmosfer (RH, Wind Speed) dan template grafik Hyetograph 3-Panel.
