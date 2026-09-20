@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title Lottery Tool
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "ALL_PROXY="
set "NO_PROXY=localhost,127.0.0.1,::1"
if not exist "%~dp0runtime\python.exe" goto missing
"%~dp0runtime\python.exe" -I -X utf8 -B "%~dp0launcher.py" %*
set "LOTTERY_EXIT=%ERRORLEVEL%"
if not "%LOTTERY_EXIT%"=="0" (
  echo.
  echo 启动失败：请双击 Check.bat，并查看 logs 文件夹。
  echo 如果提示缺少 MSVCP140 或 VCRUNTIME，请运行 installers 内的微软安装包。
  pause
  exit /b %LOTTERY_EXIT%
)
exit /b 0
:missing
echo 未找到内置 Python。请对 ZIP 右键选择“全部提取”，不要在压缩包内运行。
pause
exit /b 1
