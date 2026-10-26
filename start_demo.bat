@echo off
rem Starts the ConvoX demo and opens it in your browser. Double-click this file.
rem The program runs inside Ubuntu (WSL); this window must stay open while you demo.
cd /d "%~dp0"
set WSLDIR=
for /f "usebackq delims=" %%i in (`wsl -d Ubuntu wslpath -u "%CD%"`) do set WSLDIR=%%i
if "%WSLDIR%"=="" (
  echo Could not find Ubuntu under WSL. Install or start it first.
  pause
  exit /b 1
)
echo Starting ConvoX demo...
start "" /b cmd /c "timeout /t 5 /nobreak >nul && start http://localhost:8000"
wsl -d Ubuntu bash -lc "cd '%WSLDIR%' && bash scripts/start_ui.sh"
echo.
echo The demo has stopped.
pause
