@echo off
REM ==============================================================================
REM BAT SCRIPT: AUTOMATED RETRAINING, 250M DOWNSCALING & DASHBOARD SYNC RUNNER
REM Projek Downscale Kebumen - BMKG / JerisMeteo
REM ==============================================================================

cd /d "d:\Github\Projek_Downscale"
if not exist "logs" mkdir "logs"

echo [%date% %time%] Memulai Master Pipeline Retraining & Web Sync... >> "logs\retrain_runner.log"

"D:\conda_env\tensorflow\python.exe" scripts\auto_retrain_and_sync.py --force --log-file "logs\auto_retrain_and_sync.log" >> "logs\retrain_runner.log" 2>&1

echo [%date% %time%] Selesai Master Pipeline Retraining & Web Sync. >> "logs\retrain_runner.log"
REM ==============================================================================

