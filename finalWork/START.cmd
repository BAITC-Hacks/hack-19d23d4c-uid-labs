@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing.
  echo Use Python 3.10-3.13:
  echo python -m venv .venv
  echo .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8520 after the server starts.
echo Press Ctrl+C to stop.
".venv\Scripts\python.exe" -X utf8 -m moneygraph serve --data data --out results/real --port 8520
pause
