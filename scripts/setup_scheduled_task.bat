@echo off
REM ==============================================================================
REM BAT SCRIPT: DAFTARKAN TUGAS TERJADWAL OTOMATIS (WINDOWS TASK SCHEDULER)
REM Menjalankan sync_kaggle_data.py setiap hari jam 06:00 pagi
REM ==============================================================================

echo Mendaftarkan tugas terjadwal 'Downscale_KaggleDataSync'...
schtasks /create /tn "Downscale_KaggleDataSync" /tr "\"d:\Github\Projek_Downscale\scripts\run_sync.bat\"" /sc daily /st 06:00 /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================================
    echo [SUKSES] Tugas terjadwal berhasil didaftarkan!
    echo Skrip akan otomatis memeriksa dan menyinkronkan data setiap jam 06:00 pagi.
    echo Log pembaruan dapat dilihat di: d:\Github\Projek_Downscale\logs\sync_kaggle.log
    echo ========================================================================
) else (
    echo.
    echo [PERINGATAN] Gagal mendaftarkan task. Pastikan dijalankan sebagai Administrator jika diminta.
)
pause

