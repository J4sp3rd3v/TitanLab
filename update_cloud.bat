@echo off
echo 👾 TITAN CLOUD SYNC PROTOCOL
echo ==========================================
echo Questo script invia i tuoi aggiornamenti locali a GitHub.
echo Google Colab scarichera' automaticamente l'ultima versione al prossimo avvio.
echo.

:: Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo <!> GIT NON INSTALLATO!
    echo Scarica e installa Git da: https://git-scm.com/downloads
    pause
    exit /b
)

:: Check if repo is initialized
if not exist .git (
    echo ⚠️ Repository non inizializzato.
    echo Inserisci l'URL del tuo nuovo repository GitHub (es. https://github.com/tuonome/TITAN.git):
    set /p REPO_URL=URL: 
    
    git init
    git branch -M main
    git remote add origin %REPO_URL%
    echo ✅ Repository collegato.
)

echo.
echo 📦 Preparazione pacchetto aggiornamento...
git add .
set /p COMMIT_MSG="Descrivi l'aggiornamento (es. 'Fix bug audio'): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG="Auto-update from Titan Console"

git commit -m "%COMMIT_MSG%"

echo.
echo 🚀 Invio al Cloud (GitHub)...
git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo <!> ERRORE DURANTE L'INVIO.
    echo Assicurati di avere i permessi o di aver fatto il login su GitHub.
    echo Se e' la prima volta, potresti dover usare: git push --force origin main
) else (
    echo.
    echo ✅ CLOUD AGGIORNATO CON SUCCESSO!
    echo Riavvia la cella su Google Colab per applicare le modifiche.
)

pause
