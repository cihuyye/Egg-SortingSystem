@echo off
title FLASK API - EGG SORTING SYSTEM
color 0B
echo ============================================================
echo   MENJALANKAN FLASK API (YOLOv8-seg + OpenCV + Firebase)
echo ============================================================
echo.
cd /d "%~dp0"
".venv\Scripts\python.exe" app_flask.py
pause
