@echo off
chcp 65001 >nul
cd /d "%~dp0"
title EveAC AI farm
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
".venv\Scripts\python.exe" -X utf8 -u -m eveac_ai.match20
echo.
pause
