@echo off
REM ==============================================================================
REM BAT SCRIPT: HAPUS TUGAS TERJADWAL OTOMATIS (WINDOWS TASK SCHEDULER)
REM ==============================================================================

echo Menghapus tugas terjadwal 'Downscale_KaggleDataSync'...
schtasks /delete /tn "Downscale_KaggleDataSync" /f

if %ERRORLEVEL% EQU 0 (
    echo [SUKSES] Tugas terjadwal berhasil dihapus.
) else (
    echo [INFO] Tugas terjadwal tidak ditemukan atau sudah dihapus sebelumnya.
)
pause

