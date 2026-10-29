# Learning Proposal: Penegasan Manajemen Kode Scratch & Workspace Hygiene (Rule 10)

## 1. Identifikasi Masalah & Rationale
Pengguna memberikan instruksi eksplisit:
> `kalau mau buat kode scratch masuk ke d:\Github\Projek_Downscale\scratch`

Sebelumnya, Rule 10 sudah tercantum pada [`.agents/AGENTS.md`](file:///d:/Github/Projek_Downscale/.agents/AGENTS.md), namun perlu dipertegas sebagai **Zero-Tolerance Behavioral Guardrail** agar agen tidak pernah membuat skrip uji coba, inspeksi data, atau skrip ad-hoc di direktori root atau direktori dokumen proyek.

---

## 2. Klasifikasi Pembelajaran
- **Tipe:** Update Existing Rule
- **Target File:** [`.agents/AGENTS.md`](file:///d:/Github/Projek_Downscale/.agents/AGENTS.md)
- **Target Bagian:** `## 10. Workspace Scratch Code Management`

---

## 3. Rincian Usulan Perubahan (Diff / Proposed Update)

```markdown
## 10. Workspace Scratch Code Management
- **Wajib menyimpan seluruh skrip sementara/scratch di folder `scratch/` (Zero-Tolerance Guardrail)**:
  - Path folder absolut: `d:\Github\Projek_Downscale\scratch/`
  - Dilarang keras membuat skrip verifikasi, tes sementara, inspeksi data (`check_*.py`, `test_*.py`, `inspect_*.py`), atau file intermediate langsung di root direktori repositori maupun di dalam `documents/`.
  - Setiap eksekusi skrip scratch harus memanggil file dari direktori `scratch/<nama_script>.py`.
  - **Siklus Hidup Scratch (Lifecycle Hygiene):** Setelah eksekusi skrip scratch selesai dan hasilnya diperoleh, agen wajib membersihkan file tersebut dari `scratch/` agar workspace tetap bersih dan terpelihara.
```

---

## 4. Konfirmasi Pengguna
Apakah Anda menyetujui penguatan **Rule 10** pada [`.agents/AGENTS.md`](file:///d:/Github/Projek_Downscale/.agents/AGENTS.md) ini?
