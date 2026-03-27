@echo off
echo Building Corpography...
pyinstaller --onefile --windowed --name Corpography --paths src src\main.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Build FAILED (Corpography).
    exit /b 1
)
echo.
echo Build succeeded: dist\Corpography.exe

echo.
echo Building CorpographyEditor...
pyinstaller --onefile --windowed --name CorpographyEditor --paths src src\editor.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Build FAILED (CorpographyEditor).
    exit /b 1
)
echo.
echo Build succeeded: dist\CorpographyEditor.exe
