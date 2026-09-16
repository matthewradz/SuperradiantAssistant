@echo off
set ENV=%USERPROFILE%\anaconda3\envs\python38
set PATH=%ENV%;%ENV%\Library\bin;%ENV%\Library\mingw-w64\bin;%ENV%\Scripts;%PATH%
set QT_PLUGIN_PATH=%ENV%\Library\plugins

"%ENV%\python.exe" -m blacs