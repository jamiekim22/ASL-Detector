@echo off
echo Setting up ASL Recognition Docker environment...
echo Detected Windows
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
docker-compose -f docker-compose-windows.yml up --build

echo.
echo If you see camera errors, make sure no other applications are using your webcam.
pause