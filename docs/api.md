# TTS Server API Documentation

## Overview

The TTS Server is an HTTP API that provides text-to-speech synthesis using NeuTTS Air with voice cloning. The server keeps the model loaded in memory for fast, repeated requests.

## Server Details

- **Base URL**: `http://127.0.0.1:5555`
- **Port**: 5555
- **Protocol**: HTTP

## Endpoints

### POST /

Synthesize text to speech and play to the default audio device.

#### Request

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "text": "string (required) - Text to synthesize",
  "voice": "string (optional) - Voice name. Default: 'jo'",
  "output": "string (optional) - Output file path. Default: 'tts_output.wav'"
}
```

#### Response

**Status Code**: 200 (success) | 400 (missing text field)

**Body:**
```json
{
  "status": "ok",
  "output": "path/to/output.wav"
}
```

## Voice Options

Available voices are determined by the sample files in the `samples/` directory.

### Pre-loaded Voices

- **dave** - Male voice
- **jo** - Female voice (default)

### Adding Custom Voices

To add a new voice:

1. Create a reference audio file: `samples/[voice_name].wav`
2. Create a text file: `samples/[voice_name].txt` with the exact text spoken in the audio
3. Restart the server

The server will automatically load and cache the new voice on startup.

**Requirements:**
- Audio format: WAV
- Quality: Clearer audio produces better voice cloning results
- Sample rate: 24kHz (automatically handled by the server)

## Examples

### Basic Usage (curl)

```bash
curl -X POST http://127.0.0.1:5555 \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

Uses the default voice ("jo") and saves to `tts_output.wav`.

### With Custom Voice

```bash
curl -X POST http://127.0.0.1:5555 \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "dave"}'
```

### With Custom Output Path

```bash
curl -X POST http://127.0.0.1:5555 \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "output": "my_audio.wav"}'
```

### Using Python Client

```bash
python tts_client.py "Hello world" --voice dave --save output.wav
python tts_client.py "Hello world" --no-play  # Don't play audio
```

## Behavior

1. **Audio Synthesis**: Text is synthesized using the NeuTTS Air model with the specified voice
2. **Audio Playback**: Generated audio is automatically played to the default audio device with 1 second of silence appended (prevents last syllable cutoff)
3. **File Output**: Audio is also saved to the specified output file at 24kHz sample rate

## Notes

- The server runs single-threaded; requests are processed sequentially
- Audio playback blocks until the audio finishes playing
- The 1-second silence padding is included in both playback and saved files
- Model loading is done once at startup for fast inference
