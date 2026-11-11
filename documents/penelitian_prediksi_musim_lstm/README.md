# Monograf Riset: Prediksi Musim Beresolusi Tinggi Berbasis Sequential Deep Learning (LSTM & GRU Multi-Head) dan Spatial Downscaling Topografi 250m di Kabupaten Kebumen

## Ringkasan Eksekutif Penelitian
Penelitian ini bertujuan memecahkan batasan utama prakiraan musim konvensional operasional BMKG berbasis **Zona Musim (ZOM)** makro-regional yang kerap menggeneralisasi Kabupaten Kebumen menjadi satu kesatuan temporal tunggal. Dengan memanfaatkan heterogenitas morfologi Kebumen (0 hingga >1.000 mdpl), penelitian ini membangun sistem peramalan iklim beresolusi ultra-tinggi yang mengintegrasikan:
1. **Evaluasi Fusi Baseline Normal Klimatologi Multi-Sensor (2000–2025):** Memadukan CHIRPS v3.0, JAXA GSMaP v8, ECMWF ERA5-Land, dan NASA GPM IMERG v07.
2. **Sequential Deep Learning (GRU & LSTM Multi-Head, Rule 19):** Memodelkan dinamika temporal perambatan anomali iklim global (ENSO ONI & IOD DMI) berbasis konstruksi tensor sekuensial 3D dasarian `(676 sampel, 27 dasarian, 7 fitur)`.
3. **Validasi Silang Tanpa Bocor (Leave-One-Year-Out / LOYO, Rule 12):** Melakukan evaluasi out-of-fold murni sepanjang 26 tahun historis (2000–2025).
4. **Multi-Model Consensus Spatial Downscaling 250m (Rule 11 & 20):** Memadukan Regression-Kriging (RFK), Trivariate Thin-Plate Spline (ANUSPLIN), dan PRISM Facet Model pada 21.594 grid piksel daratan Kebumen.
5. **Penegakan Konservasi Massa Presipitasi (Rule 15):** Menjamin rasio kalibrasi massa air regional tepat 1,0000 dengan **Volume Bias = 0,00%**.

---

## Struktur Folder Mandiri (Rule 16)
```
documents/penelitian_prediksi_musim_lstm/
├── laporan_penelitian_prediksi_musim_lstm.tex  # Naskah akademik monograf LaTeX
├── README.md                                   # Ringkasan eksekutif & panduan riset
├── figures/                                    # 13 Gambar Publikasi Standar 16:9 (300 DPI)
│   ├── fig1_normal_klimatologi_amh.png         # Peta 1: Baseline Normal AMH 26 Kecamatan
│   ├── fig2_prakiraan_amh_2026_lstm.png        # Peta 2: Prakiraan AMH 2026/2027 (Nov I & Nov II)
│   ├── fig3_perbandingan_normal_mundur.png     # Peta 3: Status Terhadap Normal (Mundur 100%)
│   ├── fig4_prakiraan_sifat_hujan.png          # Peta 4: Prakiraan Sifat Hujan (Bawah Normal)
│   ├── fig5_prakiraan_puncak_musim.png         # Peta 5: Prakiraan Puncak Musim Hujan
│   ├── fig6_zonasi_kalender_tanam.png          # Peta 6: Kalender Tanam Padi MT1 Administrasi
│   ├── fig7_spasial_kontinyu_amh_250m.png      # Peta 7: Downscaling Spasial Kontinu AMH 250m
│   ├── fig8_spasial_kontinyu_total_hujan_250m.png # Peta 8: Akumulasi Hujan 250m Mass-Preserved
│   ├── fig9_zonasi_mikro_kalender_tanam_250m.png  # Peta 9: Zonasi Mikro Kalender Tanam 250m
│   ├── fig10_komparasi_multi_sensor_amh.png    # Peta 10: Komparasi 4 Sensor Penginderaan Jauh
│   ├── fig11_prakiraan_kemarau_amk_2026.png    # Peta AMK 2026: Prakiraan Musim Kemarau
│   ├── fig12_spasial_kontinyu_amk_250m.png     # Peta Downscaling Spasial Kontinu AMK 250m
│   └── fig13_zonasi_mikro_kekeringan_250m.png  # Peta Zonasi Mikro Kerawanan Kekeringan 250m
├── tables/                                     # 5 Tabel LaTeX Mandiri (\resizebox wrapped)
│   ├── tab1_benchmark_algoritma_musim.tex      # Benchmark out-of-fold LOYO Musim Hujan
│   ├── tab2_benchmark_algoritma_kemarau.tex    # Benchmark out-of-fold LOYO Musim Kemarau
│   ├── tab3_komparasi_multi_sensor_normal.tex  # Komparasi Normal AMH 4 Sensor Penginderaan Jauh
│   ├── tab4_prakiraan_musim_hujan_26kecamatan.tex # Rincian Prakiraan AMH 2026/2027 26 Kecamatan
│   └── tab5_prakiraan_musim_kemarau_26kecamatan.tex # Rincian Prakiraan AMK 2026 26 Kecamatan
└── results_data/                               # Dataset Analitik Terstruktur CSV
    ├── prakiraan_musim_2026_2027_kebumen.csv   # Hasil prediksi musim hujan 2026/2027
    ├── prakiraan_musim_kemarau_2026_kebumen.csv# Hasil prediksi musim kemarau 2026
    ├── perbandingan_algoritma_musim.csv        # Metrik evaluasi model musim hujan
    ├── perbandingan_algoritma_kemarau.csv      # Metrik evaluasi model musim kemarau
    ├── normal_klimatologi_komparasi_multi_sensor.csv # Komparasi multi-sensor
    └── hasil_evaluasi_loyo.csv                 # 676 observasi out-of-fold LOYO
```

---

## Ringkasan Kinerja Benchmark LOYO (2000–2025)

| Arsitektur Model | Akurasi Kategori AMH (%) | Macro F1 | MAE Dasarian | MAE Hari | Akurasi Sifat Hujan (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline RF + XGBoost (Tabular)** | 58,28% | 0,520 | 1,67 | 16,7 hari | 62,43% |
| **LightGBM Regressor / Classifier** | 66,42% | 0,540 | 1,26 | 12,6 hari | 56,21% |
| **Bi-LSTM Multi-Head (Bidirectional)** | 69,08% | 0,634 | 1,53 | 15,3 hari | 66,12% |
| **LSTM Multi-Head (Sequential Deep Learning)** | 72,78% | 0,684 | 1,52 | 15,2 hari | 71,60% |
| **GRU Multi-Head (Gated Recurrent Unit)** | **77,51%** | **0,723** | **1,24** | **12,4 hari** | **70,71%** |

---

## Temuan Ilmiah Utama Prakiraan 2026/2027
1. **Diferensiasi Spasial di Bawah Super El Niño:**
   Di bawah anomali iklim global El Niño kuat (NOAA ONI ASO = +2,46), seluruh 26 kecamatan di Kebumen terkonfirmasi mengalami keterlambatan (*Mundur 100%*). Namun, waktu tiba aktual terbelah secara spasial:
   - **21 Kecamatan** (wilayah lereng utara Sadang/Karanggayam, sentra barat, dan dataran tengah) tiba pada **November Dasarian I (1–10 November)**.
   - **5 Kecamatan** (wilayah pesisir tenggara dan timur: Ambal, Mirit, Prembun, Gombong, Bonorowo) mundur hingga **November Dasarian II (11–20 November)**.
2. **Gradien Spasial Kontinu 250m:**
   Model downscaling ensemble memetakan bahwa lereng tertinggi di utara memulai musim hujan paling dini pada **Dasarian 31,4 (~14 November)** dengan total curah hujan musim mencapai **$2.150 - 2.220\text{ mm}$**, sedangkan bibir pantai selatan Urutsewu tiba paling akhir pada **Dasarian 32,8 (~28 November)** dengan akumulasi hujan **$1.250 - 1.400\text{ mm}$**.
