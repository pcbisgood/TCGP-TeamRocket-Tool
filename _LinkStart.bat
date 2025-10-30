@echo off
REM ============================================================================
REM TCGP Team Rocket Tool - Setup & Launch Script
REM ============================================================================
REM Author: pcb.is.good
REM Description: Install Python, create venv, install dependencies and launch app
REM ============================================================================

setlocal enabledelayedexpansion

REM Configuration
set PYTHON_URL=https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe
set PYTHON_INSTALLER=python-installer.exe
set VENV_DIR=venv
set REQUIREMENTS_FILE=requirements.txt
set PYTHON_SCRIPT=discord_bot.py

REM Colors (using findstr for compatibility)
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

cls

echo.
echo ============================================================================
echo  ^?^  TCGP Team Rocket Tool - Setup Launcher
echo ============================================================================
echo.

REM ============================================================================
REM 1. CHECK IF PYTHON IS INSTALLED
REM ============================================================================
echo [INFO] Checking Python installation...

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python NOT found on system!
    echo.
    echo Downloading and installing Python 3.11.7 automatically...
    echo.
    
    REM Download Python
    call :downloadPython
    
    REM Verify download
    if not exist %PYTHON_INSTALLER% (
        echo [ERROR] Download failed!
        echo Please download manually from: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    
    REM Install Python
    echo [INFO] Installing Python...
    %PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1
    
    if errorlevel 1 (
        echo [ERROR] Python installation failed!
        pause
        exit /b 1
    )
    
    echo [SUCCESS] Python installed successfully!
    
    REM Clean up installer
    del /f /q %PYTHON_INSTALLER% >nul 2>&1
    
    REM Reload environment variables
    call refreshenv.bat 2>nul || (
        echo [INFO] Please restart the terminal to complete installation
        pause
        exit /b 0
    )
) else (
    for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo [SUCCESS] !PYTHON_VERSION! found
)

echo.

REM ============================================================================
REM 2. CHECK IF VENV EXISTS
REM ============================================================================
echo [INFO] Checking Virtual Environment...

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [SUCCESS] Venv found
) else (
    echo [INFO] Creating Virtual Environment...
    
    python -m venv %VENV_DIR%
    
    if errorlevel 1 (
        echo [ERROR] Venv creation failed!
        pause
        exit /b 1
    )
    
    echo [SUCCESS] Venv created successfully
)

echo.

REM ============================================================================
REM 3. ACTIVATE VENV
REM ============================================================================
echo [INFO] Activating Virtual Environment...

call "%VENV_DIR%\Scripts\activate.bat"

if errorlevel 1 (
    echo [ERROR] Venv activation failed!
    pause
    exit /b 1
)

echo [SUCCESS] Venv activated

echo.

REM ============================================================================
REM 4. INSTALL/UPDATE DEPENDENCIES
REM ============================================================================
if not exist "%REQUIREMENTS_FILE%" (
    echo [ERROR] requirements.txt file not found!
    echo Please create the file with necessary dependencies
    pause
    exit /b 1
)

echo [INFO] Verifying dependencies...

REM Upgrade pip
python -m pip install --upgrade pip >nul 2>&1

REM Install requirements
python -m pip install -r "%REQUIREMENTS_FILE%" >nul 2>&1

if errorlevel 1 (
    echo [ERROR] Dependency installation failed!
    echo Retrying...
    
    python -m pip install -r "%REQUIREMENTS_FILE%"
    
    if errorlevel 1 (
        echo [ERROR] Unable to install dependencies
        echo Please verify requirements.txt
        pause
        exit /b 1
    )
)

echo [SUCCESS] Dependencies installed successfully

echo.

REM ============================================================================
REM 5. VERIFY CRITICAL LIBRARIES
REM ============================================================================
echo [INFO] Verifying critical libraries...

setlocal enabledelayedexpansion
set LIBS=PyQt5 discord requests pillow beautifulsoup4

for %%L in (!LIBS!) do (
    python -c "import %%L" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Library %%L not found, reinstalling...
        python -m pip install %%L >nul 2>&1
    )
)

echo [SUCCESS] All libraries present

echo.

REM ============================================================================
REM 6. LAUNCH APPLICATION
REM ============================================================================
echo [INFO] Launching TCGP Team Rocket Tool...
echo.

if not exist "%PYTHON_SCRIPT%" (
    echo [ERROR] File %PYTHON_SCRIPT% not found!
    pause
    exit /b 1
)

:begin
python "%PYTHON_SCRIPT%"

REM If app closes, ask if restart
echo.
echo [INFO] Application closed
echo.
set /p RESTART="Do you want to restart? (Y/N): "
if /i "!RESTART!"=="Y" goto begin
if /i "!RESTART!"=="S" goto begin

endlocal
exit /b 0

REM ============================================================================
REM FUNCTIONS
REM ============================================================================

:downloadPython
echo [INFO] Downloading Python from %PYTHON_URL%...

REM Use PowerShell to download
powershell -Command "(New-Object Net.WebClient).DownloadFile('%PYTHON_URL%', '%PYTHON_INSTALLER%')" >nul 2>&1

if errorlevel 1 (
    echo [ERROR] PowerShell download failed
    echo Trying with bitsadmin...
    
    bitsadmin /transfer PyDownload /download /priority foreground "%PYTHON_URL%" "!CD!\%PYTHON_INSTALLER%"
    
    if errorlevel 1 (
        echo [ERROR] Download failed with both methods
        exit /b 1
    )
)

exit /b 0

:refreshenv
REM Reload environment variables
for /f "tokens=2*" %%A in ('reg query HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment /v Path') do (
    set PATH=%%B
)
exit /b 0

endlocal
