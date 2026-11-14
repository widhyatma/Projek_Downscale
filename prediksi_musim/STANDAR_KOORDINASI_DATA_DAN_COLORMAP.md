# Kesepakatan Koordinasi Data & Standarisasi Colormap Antar-Repositori
**Antara `Projek_Downscale` (Pilar 1: Spasial & Pemodelan) dan `Dashboard_Jerukagung-Meteorologi` (Pilar 2: Web Dashboard)**

Dokumen ini merupakan **Data Contract & Style Guide resmi** untuk menjamin konsistensi 100% antara luaran pemodelan Python di `Projek_Downscale` dan representasi visual interaktif di `Dashboard_Jerukagung-Meteorologi`.

---

## 1. Matriks Variabel & Metadata yang Ditampilkan

### A. Dimensi Wilayah Administrasi (Hierarki Ganda)
| Entitas Wilayah | Jumlah Unit | Sumber Master Data | Berkas Publik Dashboard | Identifikasi Unik | Format Ekspor |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Kecamatan** | 26 | `kebumen_33.05_batas_kecamatan.geojson` | `kebumen_kecamatan.json` | `kecamatan` / `code` (Nama & Kode Kemendagri `33.05.xx`) | GeoJSON Polygon WGS84 |
| **Desa / Kelurahan** | 460 | `kebumen_33.05_batas_desa.geojson` | `kebumen_desa.json` | `code` (Standar Kemendagri `33.05.xx.xxxx`) | GeoJSON Polygon WGS84 |

---

### B. Variabel Musim Hujan (MH)
| Variabel | Tipe Data | Format / Satuan | Rentang Nilai Kebumen | Keterangan & Standar |
| :--- | :--- | :--- | :--- | :--- |
| `prakiraan_amh_dasarian` | `float` | Dasarian (1–36) | 24.0 s.d. 35.0 | Awal Musim Hujan continuous |
| `prakiraan_amh_label` | `string` | Label Dasarian | `Okt I` s.d. `Des II` | Format resmi BMKG |
| `normal_amh_dasarian` | `float` | Dasarian (1–36) | 27.0 s.d. 31.0 | Baseline klimatologi 2000–2025 |
| `normal_amh_label` | `string` | Label Dasarian | `Sep III` s.d. `Nov I` | Normal klimatologi |
| `perbandingan_terhadap_normal` | `string` | Kategori | `Maju`, `Sama`, `Mundur` | Deviasi terhadap normal ($|\Delta D| \le 1$ = Sama) |
| `prakiraan_sifat_hujan` | `string` | Kategori | `Atas Normal`, `Normal`, `Bawah Normal` | Sifat akumulasi curah hujan |
| `prakiraan_puncak_bulan` | `string` | Bulan Kalender | `Desember`, `Januari`, `Februari` | Format standar kalender BMKG |
| `normal_puncak_bulan` | `string` | Bulan Kalender | `Januari` | Rata-rata puncak historis |
| `total_hujan_mh_mm` | `float` | mm | 1.200 s.d. 2.400 mm | Total akumulasi presipitasi MH |
| `kelas_hujan_mh` | `string` | Interval | `< 1.500 mm`, `1.500–1.750 mm`, `1.750–2.000 mm`, `2.000–2.500 mm`, `2.500–3.000 mm`, `> 3.000 mm` | Klasifikasi potensi air multi-dekade |
| `rekomendasi_olah_tanah` | `string` | Label Dasarian | `Sep II` s.d. `Okt II` | Rekomendasi agronomi persiapan sawah |
| `rekomendasi_tanam_padi_mt1`| `string` | Label Dasarian | `Okt I` s.d. `Nov I` | Jadwal hambur/tanam Padi MT1 |

---

### C. Variabel Musim Kemarau (MK)
| Variabel | Tipe Data | Format / Satuan | Rentang Nilai Kebumen | Keterangan & Standar |
| :--- | :--- | :--- | :--- | :--- |
| `prakiraan_amk_dasarian` | `float` | Dasarian (1–36) | 11.0 s.d. 18.0 | Awal Musim Kemarau continuous |
| `prakiraan_amk_label` | `string` | Label Dasarian | `Apr I` s.d. `Jun II` | Format resmi BMKG |
| `normal_amk_dasarian` | `float` | Dasarian (1–36) | 12.0 s.d. 15.0 | Baseline klimatologi 2000–2025 |
| `normal_amk_label` | `string` | Label Dasarian | `Apr II` s.d. `Mei II` | Normal klimatologi |
| `perbandingan_terhadap_normal` | `string` | Kategori | `Maju`, `Sama`, `Mundur` | Deviasi terhadap normal |
| `prakiraan_sifat_hujan_mk` | `string` | Kategori | `Atas Normal`, `Normal`, `Bawah Normal` | Sifat hujan musim kemarau |
| `prakiraan_pmk_bulan` | `string` | Bulan Kalender | `Juli`, `Agustus`, `September` | Puncak kering kalender BMKG |
| `normal_pmk_bulan` | `string` | Bulan Kalender | `Agustus` | Normal puncak kemarau |
| `total_hujan_mk_mm` | `float` | mm | 300 s.d. 650 mm | Total akumulasi presipitasi MK |
| `kelas_hujan_mk` | `string` | Interval | `< 430 mm`, `430–485 mm`, `485–525 mm`, `> 525 mm` | Potensi air kemarau |
| `kekeringan_skor` | `float` | Indeks (0–100) | 45.0 s.d. 85.0 | Indeks kerawanan kekeringan gabungan |
| `kelas_kekeringan` | `string` | Kategori | `Sedang (< 60)`, `Cukup Tinggi (60–65)`, `Tinggi (65–70)`, `Sangat Tinggi (> 70)` | Klasifikasi mitigasi BPBD |
| `rekomendasi_keringkan_sawah_mt2` | `string` | Label Dasarian | `Apr I` s.d. `Mei I` | Pengeringan jelang panen MT2 |
| `rekomendasi_tanam_palawija_mt3` | `string` | Label Dasarian | `Mei I` s.d. `Jun I` | Rekomendasi tanam palawija MT3 |

---

### D. Variabel Peringatan Dini & Agroklimat (CEWS, KATAM, Grafik)
1. **Peringatan Dini Iklim Ekstrem (CEWS BMKG):**
   - `hth_hari`: Jumlah hari berturut-turut tanpa hujan ($P < 1.0\text{ mm}$).
   - `hth_kategori`: `Sangat Pendek (1–5 hari)`, `Pendek (6–10 hari)`, `Menengah (11–20 hari)`, `Panjang (21–30 hari)`, `Sangat Panjang (31–60 hari)`, `Kekeringan Ekstrem (>60 hari)`.
   - `pdcht_status`: Peringatan Dini Curah Hujan Tinggi (`Aman`, `Waspada`, `Siaga`, `Awas`).
   - `spi_1_bulan` & `spi_3_bulan`: Standardized Precipitation Index (-3.0 s.d. +3.0) dengan 7 kelas WMO.
2. **Kalender Tanam Terpadu (KATAM):**
   - Tahapan `mt1` (Rendeng), `mt2` (Gadu), `mt3` (Palawija).
   - Properti: `nama`, `olah_tanah`, `tanam`, `panen`, `varietas` / `komoditas`, dan `tips` agronomi.
3. **Profil 36 Dasarian (`chart`):**
   - `labels`: Array 36 label dasarian (`Jan I` s.d. `Des III`).
   - `normal`: Array 36 nilai normal klimatologis (mm).
   - `aktual`: Array 36 nilai observasi riil lapangan (atau `null` untuk dasarian yang belum berlangsung).
   - `proyeksi`: Array 36 nilai proyeksi model inferensi (atau `null` untuk dasarian yang sudah diobservasi).

---

### E. Variabel Evaluasi Akurasi Historis (Khusus Arsip 2015–2025)
| Field JSON | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `realisasi_amh_dasarian` / `realisasi_amk_dasarian` | `float` | Dasarian awal musim aktual berdasarkan data observasi |
| `realisasi_amh_label` / `realisasi_amk_label` | `string` | Label dasarian aktual (misal: `Okt II`) |
| `realisasi_perbandingan_terhadap_normal` | `string` | Anomali aktual di lapangan (`Maju`, `Sama`, `Mundur`) |
| `realisasi_sifat_hujan` | `string` | Sifat hujan aktual di lapangan |
| `selisih_dasarian_eval` | `float` | $\text{Prediksi} - \text{Realisasi}$ (dalam satuan dasarian) |
| `status_evaluasi_amh` / `status_evaluasi_amk` | `string` | Label deskriptif (misal: `Tepat`, `Prediksi Lebih Cepat (1.1 Dasarian)`) |
| `status_evaluasi_sifat` | `string` | `Tepat` jika sifat hujan cocok, `Meleset` jika berbeda |

---

## 2. Standarisasi Palet Warna (Colormap BMKG)

Agar peta poligon, overlay raster GeoTIFF 250m, dan legenda web selaras, disepakati tabel referensi warna resmi berikut:

### A. Anomali Awal Musim (Perbandingan terhadap Normal)
| Status Anomali | Musim Hujan (MH) | Musim Kemarau (MK) | Makna Meteorologis |
| :--- | :---: | :---: | :--- |
| **Maju** | **Biru** (`#2b83ba`) | **Merah** (`#d73027`) | Hujan datang lebih cepat (kondisi basah) vs Kemarau datang lebih cepat (risiko kering) |
| **Sama** | **Hijau** (`#78c679`) | **Kuning** (`#fee08b`) | Sesuai normal klimatologis |
| **Mundur** | **Merah** (`#d7191c`) | **Hijau** (`#1a9850`) | Hujan terlambat (waspada) vs Kemarau terlambat (hujan masih berlanjut) |

---

### B. Sifat Hujan Musim (Standar Klasifikasi BMKG)
| Kategori Sifat Hujan | Kriteria Rasio thd Normal | Kode Hex | Visual Label |
| :--- | :---: | :---: | :--- |
| **Atas Normal (AN)** | $> 115\%$ | `#2b83ba` (Biru) | Surplus presipitasi, waspada banjir/genangan |
| **Normal (N)** | $85\% - 115\%$ | `#78c679` (Hijau) | Kondisi ideal pertanian dan irigasi |
| **Bawah Normal (BN)** | $< 85\%$ | `#fdae61` (Oranye Jingga) | Defisit presipitasi, waspada kekeringan |

---

### B.1. Klasifikasi Akumulasi Curah Hujan Musim Hujan (MH)
| Kategori Kelas | Rentang Nilai (mm) | Kode Hex | Visual Label |
| :--- | :---: | :---: | :--- |
| `< 1.500 mm` | $< 1.500$ | `#fecc5c` | Kuning (Curah Hujan Rendah / Defisit) |
| `1.500–1.750 mm` | $1.500 \le x < 1.750$ | `#a1d99b` | Hijau Muda (Menengah Bawah) |
| `1.750–2.000 mm` | $1.750 \le x < 2.000$ | `#41ab5d` | Hijau Sedang (Menengah) |
| `2.000–2.500 mm` | $2.000 \le x < 2.500$ | `#41b6c4` | Biru Muda / Toska (Menengah Atas) |
| `2.500–3.000 mm` | $2.500 \le x < 3.000$ | `#225ea8` | Biru Tua (Tinggi) |
| `> 3.000 mm` | $\ge 3.000$ | `#7a0177` | Ungu Pekat (Sangat Tinggi / La Niña) |

---

### C. Awal Musim Hujan (AMH) Dasarian
| Dasarian | Periode Tanggal | Kode Hex | Karakteristik Visual |
| :--- | :--- | :---: | :--- |
| `Okt I` | 1–10 Oktober | `#1b7837` | Hijau sangat tua (Sangat awal) |
| `Okt II` | 11–20 Oktober | `#7fbf7b` | Hijau sedang |
| `Okt III` | 21–31 Oktober | `#a1d99b` | Hijau muda |
| `Nov I` | 1–10 November | `#fee08b` | Kuning pastel |
| `Nov II` | 11–20 November | `#fdae61` | Oranye muda |
| `Nov III` | 21–30 November | `#f46d43` | Oranye kemerahan |
| `Des I` | 1–10 Desember | `#d53e4f` | Merah muda |
| `Des II` | 11–20 Desember | `#9e0142` | Merah gelap pekat (Sangat mundur) |

---

### D. Awal Musim Kemarau (AMK) Dasarian
| Dasarian | Periode Tanggal | Kode Hex | Karakteristik Visual |
| :--- | :--- | :---: | :--- |
| `Apr I` | 1–10 April | `#7f0000` | Merah marun pekat (Kemarau sangat cepat) |
| `Apr II` | 11–20 April | `#b30000` | Merah bata |
| `Apr III` | 21–30 April | `#d7301f` | Merah jingga terang |
| `Mei I` | 1–10 Mei | `#ef6548` | Jingga tua |
| `Mei II` | 11–20 Mei | `#fc8d59` | Jingga |
| `Mei III` | 21–31 Mei | `#fdbb84` | Jingga muda kekuningan |
| `Jun I` | 1–10 Juni | `#fdd49e` | Kuning krem |
| `Jun II` | 11–20 Juni | `#fed976` | Kuning cerah (Kemarau sangat lambat) |

---

### E. Palet Kontinu untuk Raster Resolusi Tinggi 250m
| Layer Raster | Colormap Matplotlib | Skala Gradasi Web (Hex) | Rentang Nilai |
| :--- | :--- | :--- | :--- |
| **AMH 250m** | `YlGnBu_r` | `#081d58` $\rightarrow$ `#41b6c4` $\rightarrow$ `#ffffd9` | Dasarian 23 (Sep II) s.d. 34 (Des I) |
| **AMK 250m** | `YlOrRd` | `#ffffcc` $\rightarrow$ `#fd8c3c` $\rightarrow$ `#800026` | Dasarian 10 (Apr I) s.d. 18 (Jun III) |
| **Total Hujan MH 250m** | `Blues` | `#f7fbff` $\rightarrow$ `#6baed6` $\rightarrow$ `#08306b` | 1.200 mm s.d. 2.500 mm |
| **Total Hujan MK 250m** | `YlGn` | `#ffffe5` $\rightarrow$ `#78c679` $\rightarrow$ `#004529` | 300 mm s.d. 650 mm |
| **Kerawanan Kekeringan** | `YlOrRd` | `#ffffb2` $\rightarrow$ `#fd8d3c` $\rightarrow$ `#bd0026` | Skor 40 (Sedang) s.d. 85 (Sangat Tinggi) |

