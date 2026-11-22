@echo off
rem FormBack launcher for Windows. Needs Python 3.10, 3.11 or 3.12.
cd /d "%~dp0"

if not exist .venv (
  py -3.11 -m venv .venv || py -3 -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

echo FormBack is running at http://localhost:8000  (Ctrl+C to stop)
uvicorn api.main:app --host 127.0.0.1 --port 8000
