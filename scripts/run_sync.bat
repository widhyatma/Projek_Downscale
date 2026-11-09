@echo off
REM ==============================================================================
REM BAT SCRIPT: AUTOMATED KAGGLE DATASET SYNC RUNNER
REM Projek Downscale Kebumen - BMKG / JerisMeteo
REM ==============================================================================

cd /d "d:\Github\Projek_Downscale"
if not exist "logs" mkdir "logs"

echo [%date% %time%] Memulai Sinkronisasi Kaggle Dataset... >> "logs\sync_runner.log"

"D:\conda_env\tensorflow\python.exe" sync_kaggle_data.py --dataset all --log-file "logs\sync_kaggle.log" >> "logs\sync_runner.log" 2>&1

echo [%date% %time%] Selesai Sinkronisasi Kaggle Dataset. >> "logs\sync_runner.log"
REM ==============================================================================

