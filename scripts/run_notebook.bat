@echo off
REM Open Jupyter notebook for pipeline
cd /d "%~dp0\.."
jupyter notebook notebooks/maga_pipe.ipynb
