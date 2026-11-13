@echo off
REM ==============================================================================
REM BAT SCRIPT: SETUP WINDOWS TASK SCHEDULER UNTUK RETRAINING & DASHBOARD SYNC
REM Menjalankan otomatis setiap hari Senin pukul 03:00 WIB
REM ==============================================================================

set TASK_NAME=JerisMeteo_AutoRetrain_Sync
set SCRIPT_PATH=D:\Github\Projek_Downscale\scripts\run_retrain_and_sync.bat

echo Mendaftarkan Task Scheduler: %TASK_NAME%
schtasks /create /tn "%TASK_NAME%" /tr "%SCRIPT_PATH%" /sc WEEKLY /d MON /st 03:00 /f /rl HIGHEST

if %ERRORLEVEL% EQU 0 (
    echo [SUKSES] Task Scheduler %TASK_NAME% berhasil didaftarkan!
) else (
    echo [ERROR] Gagal mendaftarkan task. Pastikan dijalankan sebagai Administrator jika diperlukan.
)
pause

