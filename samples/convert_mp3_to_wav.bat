@echo off
setlocal enabledelayedexpansion

echo Converting MP3 files to WAV format (44.1kHz stereo PCM)...
echo.

for %%f in (*.mp3) do (
    echo Converting %%f...
    ffmpeg -i "%%f" -ar 44100 -ac 2 -c:a pcm_s16le "%%~nf.wav" -y -loglevel error
    if !errorlevel! equ 0 (
        echo   ✓ Created %%~nf.wav
    ) else (
        echo   ✗ Failed to convert %%f
    )
)

echo.
echo Done!
pause
