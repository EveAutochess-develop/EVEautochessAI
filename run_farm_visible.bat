chcp 65001 >nul
cd /d H:\game_dev\EVEautochessAI-main
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title EVE Autochess AI Farm
echo.
echo === Farm resume (session next_batch) ===
echo Ctrl+C saves after current gen
echo.
".\.venv\Scripts\python.exe" -X utf8 -u -m eveac_ai.match20
echo.
echo Farm exited. Window stays open.
pause
