@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title Lottery Tool - Check
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
if not exist "%~dp0runtime\python.exe" goto missing
"%~dp0runtime\python.exe" -I -X utf8 -B "%~dp0launcher.py" --check
set "LOTTERY_EXIT=%ERRORLEVEL%"
if not "%LOTTERY_EXIT%"=="0" (
  echo.
  echo 自检未通过。请保留报错和 logs 文件夹，不要安装来历不明的软件。
  echo 缺少 MSVCP140 / VCRUNTIME 时，可运行 installers\VC_redist.x64.exe 后再自检。
  pause
  exit /b %LOTTERY_EXIT%
)
pause
exit /b 0
:missing
echo 未找到内置 Python。请完整解压 ZIP，再运行本文件。
pause
exit /b 1
