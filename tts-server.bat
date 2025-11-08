@echo off
echo Starting TTS Server...
echo This will keep the model loaded in memory for fast responses.
echo Press Ctrl+C to stop the server.
echo.
"%~dp0venv\Scripts\python.exe" "%~dp0tts_server.py"
