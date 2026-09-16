@echo off
echo Starting labscript suite...
start "runmanager" call "%~dp0launch_runmanager.bat"
timeout /t 3 /nobreak > nul
start "blacs"      call "%~dp0launch_blacs.bat"
timeout /t 3 /nobreak > nul
start "lyse"       call "%~dp0launch_lyse.bat"
echo All three launched. Check each window for errors.
