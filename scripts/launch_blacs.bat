@echo off
set ENV=C:\Users\radzi\AppData\Local\Anaconda3\envs\ybclock_3_11_24
set PATH=%ENV%;%ENV%\Library\bin;%ENV%\Library\mingw-w64\bin;%ENV%\Scripts;%PATH%
set QT_PLUGIN_PATH=%ENV%\Library\plugins
"%ENV%\python.exe" "%ENV%\Lib\site-packages\blacs\__main__.py"
