@echo off
title KuCoin Trading Bot
echo Starting KuCoin Trading Bot...
cd /d "%~dp0"
python complete_trading_bot.py
echo.
echo KuCoin Trading Bot has stopped.
pause