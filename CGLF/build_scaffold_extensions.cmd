@echo off
setlocal

set REPO_ROOT=%~dp0
if "%REPO_ROOT:~-1%"=="\" set REPO_ROOT=%REPO_ROOT:~0,-1%

if "%1"=="" (
  set ENV_NAME=scaffold_gs
) else (
  set ENV_NAME=%1
)

if "%CONDA_EXE%"=="" (
  for /f "delims=" %%i in ('where conda 2^>nul') do (
    set CONDA_EXE=%%i
    goto :found_conda
  )
)
:found_conda

if "%CONDA_EXE%"=="" (
  echo Could not find conda.exe in PATH. Set CONDA_EXE manually or activate the environment before running this script.
  exit /b 1
)

for %%i in ("%CONDA_EXE%") do set CONDA_SCRIPTS=%%~dpi
if not exist "%CONDA_SCRIPTS%activate.bat" (
  echo Could not locate activate.bat next to %CONDA_EXE%.
  exit /b 1
)

if "%CUDA_HOME%"=="" (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6" (
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6
  )
)

if not "%CUDA_HOME%"=="" (
  set CUDA_PATH=%CUDA_HOME%
  set PATH=%CUDA_HOME%\bin;%CUDA_HOME%\libnvvp;%PATH%
)

if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

set DISTUTILS_USE_SDK=1
set MSSdk=1
set CUDA_MODULE_LOADING=LAZY

call "%CONDA_SCRIPTS%activate.bat" %ENV_NAME%
if errorlevel 1 exit /b 1

cd /d "%REPO_ROOT%\submodules\diff-gaussian-rasterization"
python -m pip install . --force-reinstall --no-build-isolation
if errorlevel 1 exit /b 1

cd /d "%REPO_ROOT%\submodules\simple-knn"
python -m pip install . --force-reinstall --no-build-isolation
if errorlevel 1 exit /b 1

echo Finished rebuilding CUDA extensions in environment %ENV_NAME%.
endlocal
