@echo off
REM Run batch training with validation
REM Usage: scripts\train.bat [epochs] [batch_size]

cd /d "%~dp0\.."
set EPOCHS=%1
if "%EPOCHS%"=="" set EPOCHS=50
set BATCH=%2
if "%BATCH%"=="" set BATCH=4

python -m iternet.scripts.train_batch --data_dir data/processed --epochs %EPOCHS% --batch_size %BATCH% --device cuda
