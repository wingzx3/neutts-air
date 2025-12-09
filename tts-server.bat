@echo off
setlocal enabledelayedexpansion

echo ========================================
echo TTS Server - Voice Selection
echo ========================================
echo.
echo Available voices:
echo.

set index=1
set voice_list=

for %%f in ("%~dp0samples\*.wav") do (
    if exist "%~dp0samples\%%~nf.txt" (
        echo [!index!] %%~nf
        set "voice[!index!]=%%~nf"
        set /a index+=1
    )
)

echo.
set /p choice="Select voice (1-%index:~-1%): "

if not defined voice[%choice%] (
    echo Invalid choice. Defaulting to scarlett.
    set selected_voice=scarlett
) else (
    set selected_voice=!voice[%choice%]!
)

echo.
echo Starting TTS Server with voice: !selected_voice!
echo This will keep the model loaded in memory for fast responses.
echo Press Ctrl+C to stop the server.
echo.

"%~dp0venv\Scripts\python.exe" "%~dp0tts_server.py" --voice !selected_voice!
