@echo off
REM ============================================================================
REM TCGP TEAM ROCKET TOOL - Enhanced Setup & Launch Script
REM ============================================================================
REM Author: pcb.is.good (Black & Yellow Enhanced Edition)
REM Description: Fast, robust Python environment setup and application launcher
REM Theme: Black & Yellow Terminal UI
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
set "PYTHON_SCRIPT=main.py"
set "LOG_FILE=launcher.log"

REM ANSI Colors - Black & Yellow Theme
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

set "C_PRIMARY=%ESC%[93m"
set "C_SECONDARY=%ESC%[33m"
set "C_SUCCESS=%ESC%[92m"
set "C_ERROR=%ESC%[91m"
set "C_WARNING=%ESC%[93m"
set "C_INFO=%ESC%[96m"
set "C_DIM=%ESC%[90m"
set "C_BOLD=%ESC%[1m"
set "C_RESET=%ESC%[0m"
set "C_BG_BLACK=%ESC%[40m"
set "C_BG_YELLOW=%ESC%[43m%ESC%[30m"

REM Box Drawing Characters (UTF-8)
set "BOX_TL=╔"
set "BOX_TR=╗"
set "BOX_BL=╚"
set "BOX_BR=╝"
set "BOX_H=═"
set "BOX_V=║"
set "BOX_VR=╠"
set "BOX_VL=╣"
set "BOX_HU=╩"
set "BOX_HD=╦"

REM Symbols
set "SYM_CHECK=✓"
set "SYM_CROSS=✗"
set "SYM_ARROW=►"
set "SYM_STAR=★"
set "SYM_ROCKET=🚀"
set "SYM_GEAR=⚙"
set "SYM_PACKAGE=📦"
set "SYM_PYTHON=🐍"
set "SYM_PLAY=▶"

REM ═══════════════════════════════════════════════════════════════════════════
REM INITIALIZE LOG
REM ═══════════════════════════════════════════════════════════════════════════
echo [%date% %time%] Launcher started > "%LOG_FILE%"

REM ═══════════════════════════════════════════════════════════════════════════
REM HEADER
REM ═══════════════════════════════════════════════════════════════════════════
cls
chcp 65001 >nul 2>&1

call :drawBox "TCGP TEAM ROCKET TOOL" "Enhanced Launcher v2.0"

REM ═══════════════════════════════════════════════════════════════════════════
REM STEP 1: PYTHON DETECTION & INSTALLATION
REM ═══════════════════════════════════════════════════════════════════════════
call :printStepHeader "1" "5" "PYTHON DETECTION"

where python >nul 2>&1
if errorlevel 1 (
    call :printError "Python not found in PATH" >> "%LOG_FILE%" 2>&1
    call :printStatus "ERROR" "Python not found in system PATH"
    echo.
    call :installPython
) else (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "DETECTED_VERSION=%%v"
    call :printStatus "SUCCESS" "Python !DETECTED_VERSION! detected"
    echo [%time%] Python !DETECTED_VERSION! found >> "%LOG_FILE%"
    
    REM Check if Python version is too new (3.13+)
    for /f "tokens=1 delims=." %%M in ("!DETECTED_VERSION!") do set "MAJOR=%%M"
    for /f "tokens=2 delims=." %%m in ("!DETECTED_VERSION!") do set "MINOR=%%m"
    
    if !MAJOR! GEQ 3 if !MINOR! GEQ 13 (
        echo.
        call :printStatus "WARNING" "Python 3.13+ detected"
        call :printWarning "  discord.py requires Python 3.8-3.12"
        call :printWarning "  audioop module was removed in Python 3.13"
        echo.
        call :printInfo "  %SYM_ARROW% Recommended: Install Python 3.11 or 3.12"
        echo.
        choice /c YN /n /m "  %C_WARNING%Continue anyway? [Y/N]: %C_RESET%"
        if errorlevel 2 (
            call :printInfo "  Installation cancelled by user"
            pause
            exit /b 0
        )
    )
)
echo.

REM ═══════════════════════════════════════════════════════════════════════════
REM STEP 2: VIRTUAL ENVIRONMENT
REM ═══════════════════════════════════════════════════════════════════════════
call :printStepHeader "2" "5" "VIRTUAL ENVIRONMENT"

if exist "%VENV_DIR%\Scripts\python.exe" (
    call :printStatus "SUCCESS" "Virtual environment exists"
    echo [%time%] Venv found >> "%LOG_FILE%"
) else (
    call :printStatus "INFO" "Creating virtual environment..."
    python -m venv "%VENV_DIR%" --clear 2>nul
    if errorlevel 1 (
        call :printStatus "ERROR" "Failed to create virtual environment"
        echo [%time%] ERROR: Venv creation failed >> "%LOG_FILE%"
        pause & exit /b 1
    )
    call :printStatus "SUCCESS" "Virtual environment created"
    echo [%time%] Venv created successfully >> "%LOG_FILE%"
)
echo.

REM ═══════════════════════════════════════════════════════════════════════════
REM STEP 3: ACTIVATE VIRTUAL ENVIRONMENT
REM ═══════════════════════════════════════════════════════════════════════════
call :printStepHeader "3" "5" "ENVIRONMENT ACTIVATION"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    call :printStatus "ERROR" "Activation script not found"
    echo [%time%] ERROR: activate.bat missing >> "%LOG_FILE%"
    pause & exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat" >nul 2>&1

REM Verify activation
"%VENV_DIR%\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 (
    call :printStatus "ERROR" "Environment activation failed"
    echo [%time%] ERROR: Venv activation failed >> "%LOG_FILE%"
    pause & exit /b 1
)

call :printStatus "SUCCESS" "Environment activated"
echo [%time%] Venv activated >> "%LOG_FILE%"
echo.

REM ═══════════════════════════════════════════════════════════════════════════
REM STEP 4: DEPENDENCIES MANAGEMENT
REM ═══════════════════════════════════════════════════════════════════════════
call :printStepHeader "4" "5" "DEPENDENCY MANAGEMENT"

if not exist "%REQUIREMENTS_FILE%" (
    call :printStatus "ERROR" "requirements.txt not found"
    echo [%time%] ERROR: requirements.txt missing >> "%LOG_FILE%"
    pause & exit /b 1
)

set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"

REM Upgrade pip
call :printStatus "INFO" "Upgrading pip, setuptools, wheel..."
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel -q --disable-pip-version-check >nul 2>&1
if errorlevel 1 (
    call :printWarning "  Pip upgrade had issues (continuing anyway)"
) else (
    call :printSuccess "  Pip upgraded successfully"
)

REM Check installed packages
call :printStatus "INFO" "Analyzing installed packages..."
"%PIP_EXE%" list --format=freeze >"%TEMP%\venv_packages.txt" 2>nul

set "NEEDS_INSTALL=0"
set "MISSING_COUNT=0"
for /f "usebackq delims=" %%p in ("%REQUIREMENTS_FILE%") do (
    set "LINE=%%p"
    if not "!LINE:~0,1!"=="#" if not "!LINE!"=="" (
        for /f "tokens=1 delims==><!~" %%n in ("!LINE!") do set "PKG_NAME=%%n"
        set "PKG_NAME=!PKG_NAME: =!"
        
        findstr /i /b "!PKG_NAME!" "%TEMP%\venv_packages.txt" >nul 2>&1
        if errorlevel 1 (
            set "NEEDS_INSTALL=1"
            set /a MISSING_COUNT+=1
        )
    )
)

del "%TEMP%\venv_packages.txt" 2>nul

if !NEEDS_INSTALL!==1 (
    echo.
    call :printStatus "INFO" "Installing !MISSING_COUNT! missing dependencies..."
    echo.
    call :printDivider
    echo.
    "%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS_FILE%" --disable-pip-version-check
    set "PIP_EXIT=!errorlevel!"
    echo.
    call :printDivider
    echo.
    if !PIP_EXIT! NEQ 0 (
        call :printStatus "ERROR" "Dependency installation failed"
        echo [%time%] ERROR: Pip install failed >> "%LOG_FILE%"
        pause & exit /b 1
    )
    call :printStatus "SUCCESS" "All dependencies installed"
    echo [%time%] Dependencies installed >> "%LOG_FILE%"
) else (
    call :printStatus "SUCCESS" "All dependencies already satisfied"
    echo [%time%] Dependencies check passed >> "%LOG_FILE%"
)

echo.
call :printStatus "INFO" "Verifying critical modules..."

REM Check audioop for Python 3.13+
"%PYTHON_EXE%" -c "import audioop" 2>nul
if errorlevel 1 (
    call :printWarning "  audioop module missing (Python 3.13+)"
    call :printInfo "  %SYM_ARROW% Installing audioop-lts compatibility..."
    "%PYTHON_EXE%" -m pip install audioop-lts --disable-pip-version-check -q 2>nul
    if errorlevel 1 (
        call :printError "  Failed to install audioop-lts"
        call :printWarning "  Discord voice features may not work"
    ) else (
        call :printSuccess "  audioop-lts installed"
    )
)

set "FAILED_MODULES="
for %%M in (discord dotenv PyQt5 requests PIL bs4) do (
    "%PYTHON_EXE%" -c "import %%M" 2>nul
    if errorlevel 1 (
        call :printWarning "  Module '%%M' import failed - attempting fix..."
        
        if "%%M"=="dotenv" (
            "%PYTHON_EXE%" -m pip install python-dotenv --disable-pip-version-check -q 2>nul
        ) else if "%%M"=="PIL" (
            "%PYTHON_EXE%" -m pip install pillow --disable-pip-version-check -q 2>nul
        ) else if "%%M"=="bs4" (
            "%PYTHON_EXE%" -m pip install beautifulsoup4 --disable-pip-version-check -q 2>nul
        ) else (
            "%PYTHON_EXE%" -m pip install %%M --disable-pip-version-check -q 2>nul
        )
        
        "%PYTHON_EXE%" -c "import %%M" 2>nul
        if errorlevel 1 (
            call :printError "  Failed to fix module '%%M'"
            set "FAILED_MODULES=!FAILED_MODULES! %%M"
        ) else (
            call :printSuccess "  Module '%%M' fixed"
        )
    )
)

if "!FAILED_MODULES!"=="" (
    call :printStatus "SUCCESS" "All critical modules verified"
    echo [%time%] All modules verified >> "%LOG_FILE%"
) else (
    echo.
    call :printStatus "WARNING" "Some modules failed:!FAILED_MODULES!"
    echo [%time%] WARNING: Failed modules:!FAILED_MODULES! >> "%LOG_FILE%"
)
echo.

REM ═══════════════════════════════════════════════════════════════════════════
REM STEP 5: LAUNCH APPLICATION
REM ═══════════════════════════════════════════════════════════════════════════
call :printStepHeader "5" "5" "APPLICATION LAUNCH"

if not exist "%PYTHON_SCRIPT%" (
    call :printStatus "ERROR" "%PYTHON_SCRIPT% not found"
    echo [%time%] ERROR: main.py missing >> "%LOG_FILE%"
    pause & exit /b 1
)

call :printStatus "SUCCESS" "Ready to launch"
echo.
timeout /t 2 /nobreak >nul

:launch_app
cls
call :printLaunchHeader

"%PYTHON_EXE%" "%PYTHON_SCRIPT%"
set "EXIT_CODE=!errorlevel!"

echo.
call :printDivider
echo.

if !EXIT_CODE! NEQ 0 (
    call :printStatus "WARNING" "Application exited with code: !EXIT_CODE!"
    echo [%time%] App exit code: !EXIT_CODE! >> "%LOG_FILE%"
) else (
    call :printStatus "INFO" "Application closed normally"
    echo [%time%] App closed normally >> "%LOG_FILE%"
)

echo.
choice /c YN /n /m "  %C_PRIMARY%%C_BOLD%Restart application? [Y/N]: %C_RESET%"
if errorlevel 2 goto :end
if errorlevel 1 (
    echo [%time%] Restarting app >> "%LOG_FILE%"
    goto :launch_app
)

:end
echo.
call :printStatus "INFO" "Shutting down..."
echo [%time%] Launcher closed >> "%LOG_FILE%"
timeout /t 2 /nobreak >nul
endlocal
exit /b 0

REM ═══════════════════════════════════════════════════════════════════════════
REM VISUAL HELPER FUNCTIONS
REM ═══════════════════════════════════════════════════════════════════════════

:drawBox
echo.
echo  %C_PRIMARY%%BOX_TL%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_TR%%C_RESET%
echo  %C_PRIMARY%%BOX_V%                                                                    %BOX_V%%C_RESET%
echo  %C_PRIMARY%%BOX_V%  %C_BOLD%%C_BG_YELLOW% %SYM_STAR% %~1 %SYM_STAR% %C_RESET%  %C_PRIMARY%                                          %BOX_V%%C_RESET%
echo  %C_PRIMARY%%BOX_V%                                                                    %BOX_V%%C_RESET%
echo  %C_PRIMARY%%BOX_V%  %C_SECONDARY%%~2%C_RESET%                                                    %C_PRIMARY%%BOX_V%%C_RESET%
echo  %C_PRIMARY%%BOX_V%                                                                    %BOX_V%%C_RESET%
echo  %C_PRIMARY%%BOX_BL%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_BR%%C_RESET%
echo.
exit /b 0

:printStepHeader
echo  %C_PRIMARY%%C_BOLD%%BOX_VR%%BOX_H%%BOX_H%%BOX_H%%BOX_H% [STEP %~1/%~2] %~3 %C_RESET%
echo.
exit /b 0

:printLaunchHeader
echo.
echo  %C_PRIMARY%%BOX_TL%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_TR%%C_RESET%
echo  %C_PRIMARY%%BOX_V%  %C_BOLD%%C_BG_YELLOW% %SYM_PLAY% APPLICATION RUNNING %C_RESET%                                              %C_PRIMARY%%BOX_V%%C_RESET%
echo  %C_PRIMARY%%BOX_BL%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_BR%%C_RESET%
echo.
exit /b 0

:printDivider
echo  %C_DIM%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%BOX_H%%C_RESET%
exit /b 0

:printStatus
if /i "%~1"=="SUCCESS" (
    echo  %C_SUCCESS%%C_BOLD%%SYM_CHECK%%C_RESET% %C_SUCCESS%%~2%C_RESET%
) else if /i "%~1"=="ERROR" (
    echo  %C_ERROR%%C_BOLD%%SYM_CROSS%%C_RESET% %C_ERROR%%~2%C_RESET%
) else if /i "%~1"=="WARNING" (
    echo  %C_WARNING%%C_BOLD%!%C_RESET% %C_WARNING%%~2%C_RESET%
) else if /i "%~1"=="INFO" (
    echo  %C_INFO%%SYM_ARROW%%C_RESET% %C_INFO%%~2%C_RESET%
)
exit /b 0

:printSuccess
echo  %C_SUCCESS%%~1%C_RESET%
exit /b 0

:printError
echo  %C_ERROR%%~1%C_RESET%
exit /b 0

:printWarning
echo  %C_WARNING%%~1%C_RESET%
exit /b 0

:printInfo
echo  %C_INFO%%~1%C_RESET%
exit /b 0

REM ═══════════════════════════════════════════════════════════════════════════
REM INSTALLATION FUNCTIONS
REM ═══════════════════════════════════════════════════════════════════════════

:installPython
call :printInfo "  %SYM_PYTHON% Preparing Python installation..."
echo.

call :printInfo "  %SYM_ARROW% Downloading Python 3.11.7..."
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
    call :printStatus "ERROR" "Download failed"
    echo.
    call :printInfo "  %SYM_ARROW% Please download Python manually from:"
    call :printInfo "     https://www.python.org/downloads/"
    echo.
    pause & exit /b 1
)

if not exist "%PYTHON_INSTALLER%" (
    call :printStatus "ERROR" "Installer not found after download"
    pause & exit /b 1
)

echo.
call :printInfo "  %SYM_GEAR% Installing Python (this may take a few minutes)..."
echo.
start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

if errorlevel 1 (
    call :printStatus "ERROR" "Python installation failed"
    pause & exit /b 1
)

del /f /q "%PYTHON_INSTALLER%" >nul 2>&1

call :refreshPath

echo.
call :printStatus "SUCCESS" "Python installed successfully"
echo.
call :printWarning "  Please restart this script to complete setup"
echo.
pause
exit /b 0

:refreshPath
for /f "skip=2 tokens=3*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%a %%b"
for /f "skip=2 tokens=3*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USR_PATH=%%a %%b"
set "PATH=%SYS_PATH%;%USR_PATH%"
exit /b 0