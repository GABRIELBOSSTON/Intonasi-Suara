@echo off
rem Klik dua kali untuk membuka halaman tes intonasi di browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtualenv .venv belum ada. Lihat README bagian instalasi.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\demo_server.py %*
pause
