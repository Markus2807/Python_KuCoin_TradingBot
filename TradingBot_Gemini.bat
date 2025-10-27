@echo off
title KuCoin Trading Bot
echo Starting KuCoin Trading Bot...
cd /d "%~dp0"
python Gemini_Optimized.py
echo.
echo KuCoin Trading Bot has stopped.
pause