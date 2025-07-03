@echo off
echo Setting up ASL Recognition Docker environment...
echo Detected Windows
echo.

REM Check if Python 3.11 virtual environment exists, create if not
if not exist "..\.venv311" (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv ..\.venv311
    echo Virtual environment created successfully!
) else (
    echo Python 3.11 virtual environment already exists.
)

REM Activate the virtual environment
call ..\.venv311\Scripts\activate.bat

echo.
echo Please make sure you have an X server running for GUI support:
echo.
echo 1. Download VcXsrv from: https://sourceforge.net/projects/vcxsrv/
echo 2. Install VcXsrv
echo 3. Run XLaunch with these settings:
echo    - Display settings: "Multiple windows"
echo    - Client startup: "Start no client"
echo    - Extra settings: Check "Disable access control"
echo.

pause

echo Starting with Windows configuration...
echo Using Python 3.11 for Docker build...

docker-compose -f ./../docker-compose-dev.yml up --build

echo.
echo If you see camera errors, make sure no other applications are using your webcam.
pause