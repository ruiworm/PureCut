@echo off
cd /d "%~dp0"
title PureCut Studio Pro (Electron Desktop)

echo =======================================================
echo   PureCut Studio Pro - Starting desktop...
echo =======================================================

:: main.js spawns the Python engine (dynamic port) and enables dev hot-reload:
::   - edit scripts/studio_template.html  -> auto refresh page
::   - edit scripts/*.py                  -> auto restart engine
start "" "node_modules\electron\dist\electron.exe" .
exit
