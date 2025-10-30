@echo off
REM ============================================================================
REM TCGP Team Rocket Tool - Enhanced Setup & Launch Script
REM ============================================================================
REM Author: pcb.is.good (Enhanced Edition)
REM Description: Fast, robust Python environment setup and application launcher
REM ============================================================================

setlocal enabledelayedexpansion

REM ═══════════════════════════════════════════════════════════════════════════
REM ENABLE ANSI COLORS
REM ═══════════════════════════════════════════════════════════════════════════
for /f "tokens=3" %%a in ('reg query "HKCU\Console" /v VirtualTerminalLevel 2^>nul') do set VT_ENABLED=%%a
if not "!VT_ENABLED!"=="0x1" (
    reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul 2>&1
)

REM ═══════════════════════════════════════════════════════════════════════════
REM CONFIGURATION
REM ═══════════════════════════════════════════════════════════════════════════
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
set "PYTHON_INSTALLER=python-installer.exe"
set "VENV_DIR=venv"
set "REQUIREMENTS_FILE=requirements.txt"
set "PYTHON_SCRIPT=discord_bot.py"

REM ANSI Colors (ESC[XXm format)
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

set "C_SUCCESS=%ESC%[92m"
set "C_ERROR=%ESC%[91m"
set "C_WARNING=%ESC%[93m"
set "C_INFO=%ESC%[96m"
set "C_HEADER=%ESC%[95m"
set "C_BOLD=%ESC%[1m"
set "C_RESET=%ESC%[0m"

REM Symbols
set "SYM_CHECK=OK"
set "SYM_CROSS=X"
set "SYM_ARROW=->"
set "SYM_STAR=*"

REM ═══════════════════════════════════════════════════════════════════════════
REM HEADER
REM ═══════════════════════════════════════════════════════════════════════════
cls
chcp 65001 >nul 2>&1

echo.
echo %C_HEADER%%C_BOLD%=========================================================================%C_RESET%
echo %C_HEADER%%C_BOLD%                                                                         %C_RESET%
echo %C_HEADER%%C_BOLD%         %SYM_STAR% TCGP TEAM ROCKET TOOL %SYM_STAR% Enhanced Launcher             %C_RESET%
echo %C_HEADER%%C_BOLD%                                                                         %C_RESET%
echo %C_HEADER%%C_BOLD%=========================================================================%C_RESET%
echo.

REM ═══════════════════════════════════════════════════════════════════════════
REM STEP 1: PYTHON DETECTION & INSTALLATION
REM ═══════════════════════════════════════════════════════════════════════════
call :printStep "1/5" "Detecting Python Installation"

where python >nul 2>&1
if errorlevel 1 (
    call :printError "Python not found in PATH"
    call :installPython
) else (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "DETECTED_VERSION=%%v"
    call :printSuccess "Python !DETECTED_VERSION! detected"
    
    REM Check if Python version is too new (3.13+)
    for /f "tokens=1 delims=." %%M in ("!DETECTED_VERSION!") do set "MAJOR=%%M"
    for /f "tokens=2 delims=." %%m in ("!DETECTED_VERSION!") do set "MINOR=%%m"
    
    if !MAJOR! GEQ 3 if !MINOR! GEQ 13 (
        echo.
        call :printWarning "Python 3.13+ detected - discord.py requires Python 3.8-3.12"
        call :printWarning "audioop module was removed in Python 3.13"
        echo.
        call :printInfo "Recommended: Install Python 3.11 or 3.12"
        echo.
        choice /c YN /n /m "%C_WARNING%Continue anyway? [Y/N]: %C_RESET%"
        if errorlevel 2 (
            call :printInfo "Installation cancelled"
            pause
            exit /b 0
        )
    )
)

REM ═══════════════════════════════════════════════════════════════════════════
REM STEP 2: VIRTUAL ENVIRONMENT
REM ═══════════════════════════════════════════════════════════════════════════
call :printStep "2/5" "Virtual Environment Setup"

if exist "%VENV_DIR%\Scripts\python.exe" (
    call :printSuccess "Virtual environment exists"
) else (
    call :printInfo "Creating new virtual environment..."
    python -m venv "%VENV_DIR%" --clear 2>nul
    if errorlevel 1 (
        call :printError "Failed to create virtual environment"
        pause & exit /b 1
    )
    call :printSuccess "Virtual environment created"
)

REM ═══════════════════════════════════════════════════════════════════════════
REM STEP 3: ACTIVATE VIRTUAL ENVIRONMENT
REM ═══════════════════════════════════════════════════════════════════════════
call :printStep "3/5" "Activating Virtual Environment"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    call :printError "Activation script not found"
    pause & exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"

REM Verify activation by checking Python path
"%VENV_DIR%\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 (
    call :printError "Virtual environment activation failed"
    pause & exit /b 1
)

call :printSuccess "Environment activated"

REM ═══════════════════════════════════════════════════════════════════════════
REM STEP 4: DEPENDENCIES MANAGEMENT
REM ═══════════════════════════════════════════════════════════════════════════
call :printStep "4/5" "Managing Dependencies"

if not exist "%REQUIREMENTS_FILE%" (
    call :printError "requirements.txt not found"
    pause & exit /b 1
)

REM Use venv Python explicitly
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"

REM Upgrade pip first (silent)
call :printInfo "Upgrading pip..."
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel -q --disable-pip-version-check 2>nul

REM Get installed packages from venv
call :printInfo "Checking installed packages in virtual environment..."
"%PIP_EXE%" list --format=freeze >"%TEMP%\venv_packages.txt" 2>nul

REM Check if we need to install anything
set "NEEDS_INSTALL=0"
for /f "usebackq delims=" %%p in ("%REQUIREMENTS_FILE%") do (
    set "LINE=%%p"
    REM Skip comments and empty lines
    if not "!LINE:~0,1!"=="#" if not "!LINE!"=="" (
        REM Extract package name (before ==, >=, <=, etc.)
        for /f "tokens=1 delims==><!~" %%n in ("!LINE!") do set "PKG_NAME=%%n"
        REM Trim spaces
        set "PKG_NAME=!PKG_NAME: =!"
        
        REM Check if package exists (case insensitive)
        findstr /i /b "!PKG_NAME!" "%TEMP%\venv_packages.txt" >nul 2>&1
        if errorlevel 1 (
            set "NEEDS_INSTALL=1"
        )
    )
)

del "%TEMP%\venv_packages.txt" 2>nul

if !NEEDS_INSTALL!==1 (
    call :printInfo "Installing missing dependencies..."
    echo.
    "%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%" --disable-pip-version-check
    if errorlevel 1 (
        echo.
        call :printError "Dependency installation failed"
        pause & exit /b 1
    )
    echo.
    call :printSuccess "All dependencies installed"
) else (
    call :printSuccess "All dependencies already installed"
)

REM Verify critical modules can be imported
call :printInfo "Verifying critical modules..."
set "ALL_OK=1"

REM First check for audioop (required by discord.py)
"%PYTHON_EXE%" -c "import audioop" 2>nul
if errorlevel 1 (
    call :printWarning "audioop module not found (removed in Python 3.13+)"
    call :printInfo "Installing audioop-lts compatibility package..."
    "%PYTHON_EXE%" -m pip install audioop-lts --disable-pip-version-check 2>nul
    if errorlevel 1 (
        call :printError "Failed to install audioop-lts"
        call :printWarning "Discord voice features may not work"
        set "ALL_OK=0"
    ) else (
        call :printSuccess "audioop-lts installed"
    )
)

for %%M in (discord dotenv PyQt5 requests PIL bs4) do (
    "%PYTHON_EXE%" -c "import %%M" 2>nul
    if errorlevel 1 (
        call :printWarning "Module '%%M' import failed"
        set "ALL_OK=0"
        
        REM Try to install the package
        if "%%M"=="dotenv" (
            "%PYTHON_EXE%" -m pip install python-dotenv --disable-pip-version-check 2>nul
        ) else if "%%M"=="PIL" (
            "%PYTHON_EXE%" -m pip install pillow --disable-pip-version-check 2>nul
        ) else if "%%M"=="bs4" (
            "%PYTHON_EXE%" -m pip install beautifulsoup4 --disable-pip-version-check 2>nul
        ) else (
            "%PYTHON_EXE%" -m pip install %%M --disable-pip-version-check 2>nul
        )
        
        REM Check again
        "%PYTHON_EXE%" -c "import %%M" 2>nul
        if errorlevel 1 (
            call :printError "Failed to fix module '%%M'"
        ) else (
            call :printSuccess "Module '%%M' fixed"
        )
    )
)

if !ALL_OK!==1 (
    call :printSuccess "All critical modules verified"
) else (
    echo.
    call :printWarning "Some modules had issues but may have been fixed"
    echo.
)

REM ═══════════════════════════════════════════════════════════════════════════
REM STEP 5: LAUNCH APPLICATION
REM ═══════════════════════════════════════════════════════════════════════════
call :printStep "5/5" "Launching Application"

if not exist "%PYTHON_SCRIPT%" (
    call :printError "%PYTHON_SCRIPT% not found"
    pause & exit /b 1
)

echo.
echo %C_HEADER%=========================================================================%C_RESET%
echo.

:launch_app
"%PYTHON_EXE%" "%PYTHON_SCRIPT%"
set "EXIT_CODE=!errorlevel!"

echo.
echo %C_HEADER%=========================================================================%C_RESET%
echo.

if !EXIT_CODE! NEQ 0 (
    call :printWarning "Application exited with code: !EXIT_CODE!"
    echo.
)

choice /c YN /n /m "Restart application? [Y/N]: "
if errorlevel 2 goto :end
if errorlevel 1 (
    cls
    echo.
    call :printInfo "Restarting application..."
    echo.
    echo %C_HEADER%=========================================================================%C_RESET%
    echo.
    goto :launch_app
)

:end
call :printInfo "Shutting down..."
timeout /t 2 /nobreak >nul
endlocal
exit /b 0

REM ═══════════════════════════════════════════════════════════════════════════
REM HELPER FUNCTIONS
REM ═══════════════════════════════════════════════════════════════════════════

:printStep
echo %C_HEADER%%C_BOLD%[STEP %~1]%C_RESET% %C_BOLD%%~2%C_RESET%
echo.
exit /b 0

:printSuccess
echo %C_SUCCESS%%SYM_CHECK% %~1%C_RESET%
echo.
exit /b 0

:printError
echo %C_ERROR%%SYM_CROSS% ERROR: %~1%C_RESET%
echo.
exit /b 0

:printWarning
echo %C_WARNING%! WARNING: %~1%C_RESET%
echo.
exit /b 0

:printInfo
echo %C_INFO%%SYM_ARROW% %~1%C_RESET%
exit /b 0

:installPython
call :printInfo "Preparing Python installation..."

REM Download using PowerShell (faster and more reliable)
call :printInfo "Downloading Python 3.11.7..."
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ProgressPreference = 'SilentlyContinue'; ^
    try { ^
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; ^
        Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%' -UseBasicParsing; ^
        exit 0 ^
    } catch { ^
        Write-Host $_.Exception.Message; ^
        exit 1 ^
    }"

if errorlevel 1 (
    call :printError "Download failed"
    call :printInfo "Please download Python manually from: https://www.python.org/downloads/"
    pause & exit /b 1
)

if not exist "%PYTHON_INSTALLER%" (
    call :printError "Installer file not found after download"
    pause & exit /b 1
)

call :printInfo "Installing Python (this may take a few minutes)..."
start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

if errorlevel 1 (
    call :printError "Python installation failed"
    pause & exit /b 1
)

REM Cleanup
del /f /q "%PYTHON_INSTALLER%" >nul 2>&1

REM Refresh PATH
call :refreshPath

call :printSuccess "Python installed successfully"
call :printWarning "Please restart this script to complete setup"
pause
exit /b 0

:refreshPath
REM Refresh environment variables for current session
for /f "skip=2 tokens=3*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%a %%b"
for /f "skip=2 tokens=3*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USR_PATH=%%a %%b"
set "PATH=%SYS_PATH%;%USR_PATH%"
exit /b 0