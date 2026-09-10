# Learning Proposal: Standarisasi Typesetting, Tabel Dinamis, dan Guardrail LaTeX

## 1. Identifikasi Masalah & Rationale
Selama penulisan dan kompilasi monograf penelitian akademik di folder `documents/` (`penelitian_downscaling_2025`, `penelitian_downscaling_25tahun`, dan `penelitian_downscaling_gsmap_vs_chirps`), ditemukan beberapa masalah typesetting berulang:
1. **Tabel Melebihi Margin (*Overfull \hbox*):** Tabel dengan banyak kolom atau teks nama kecamatan panjang sering kali tumpah ke luar margin kanan kertas A4 jika menggunakan lebar kolom statis.
2. **Spasi dan Teks Spasial Bahasa Indonesia:** Nama geografis majemuk (*Karanggayam*, *Karangsambung*, *Buluspesantren*) yang berdampingan dengan nilai numerik dan satuan presipitasi ($3.030,9$~mm) menyebabkan LaTeX gagal memotong baris secara alami.
3. **Peringatan Duplikasi Halaman Hyperref (*Duplicate Destination Identifier*):** Penomoran halaman romawi pada bagian awal (*front matter*) dan arab pada batang tubuh sering memicu peringatan duplikat `name{page.1}` dan `name{page.i}` jika `titlepage` dan `\begin{abstract}` tidak diatur secara tepat dalam kelas `article`.
4. **Daftar Tabel & Gambar (*LOT/LOF*) yang Melebihi Margin:** Judul caption yang terlalu panjang tumpah di halaman Daftar Tabel/Gambar bila tidak menyertakan judul pendek opsional `\caption[Short]{Long}`.

---

## 2. Klasifikasi Pembelajaran
- **Tipe:** Rule Update
- **Target:** `Rule 14. Automated LaTeX Academic Reporting Standard` pada file [AGENTS.md](file:///d:/Github/Projek_Downscale/.agents/AGENTS.md).

---

## 3. Rincian Usulan Perubahan Rule (Proposed Addition)

```markdown
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
```

---

## 4. Konfirmasi Pengguna
Apakah Anda menyetujui pembaruan Rule 14 di [AGENTS.md](file:///d:/Github/Projek_Downscale/.agents/AGENTS.md) sesuai proposal di atas?
