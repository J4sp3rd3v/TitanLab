@echo off
title TITAN COMMAND CENTER
color 0b

echo ========================================================
echo 👾 TITAN VIDEO ENGINE - AUTOMATED LAUNCHER
echo ========================================================
echo.
echo [FASE 1] Apertura Google Colab (Il Motore)...
echo    -> Devi solo premere "Runtime" -> "Run all" (o Play)
echo.
start https://colab.research.google.com/github/J4sp3rd3v/TitanLab/blob/main/TITAN_VIRAL_LAB.ipynb
echo.
echo [FASE 2] Avvio Interfaccia Locale (Il Controllo)...
echo.
python titan_remote.py
pause