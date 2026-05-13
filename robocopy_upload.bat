@echo off
setlocal

rem Copy new and updated files from one folder to another.
rem Change these two paths before running.
set "SOURCE=C:\Users\Aria\Desktop\ML_V2"
set "DESTINATION=C:\Users\Aria\OneDrive - The University of Western Ontario\MSc\Thesis\Data\ML_V2"

rem /E       Copy subfolders, including empty ones.
rem /XO      Exclude older source files; keeps newer destination files.
rem /R:3     Retry failed copies 3 times.
rem /W:5     Wait 5 seconds between retries.
rem /FFT     Use tolerant file timestamp comparison for mixed filesystems.
rem /LOG     Write a copy log next to this script.
robocopy "%SOURCE%" "%DESTINATION%" /E /XO /R:3 /W:5 /FFT /LOG:"%~dp0robocopy-copy.log"

rem Robocopy exit codes 0-7 are usually success or non-fatal copy differences.
if %ERRORLEVEL% LEQ 7 (
    echo Copy completed. See robocopy-copy.log for details.
    exit /b 0
) else (
    echo Copy failed with robocopy exit code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)
