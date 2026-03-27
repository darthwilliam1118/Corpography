@echo off
echo Building Corpography...
pyinstaller --onefile --windowed --name Corpography --paths src src\main.py
if %ERRORLEVEL% == 0 (
    echo.
    echo Build succeeded: dist\Corpography.exe
) else (
    echo.
    echo Build FAILED.
    exit /b 1
)
