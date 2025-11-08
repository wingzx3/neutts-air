#!/usr/bin/env python3
"""TTS Server - keeps model loaded in memory for fast responses."""
import os
import sys
import socketserver
import json
import numpy as np
import soundfile as sf
import sounddevice as sd
from http.server import BaseHTTPRequestHandler

# Configure espeak for Windows
if os.name == 'nt':
    import platform
    if platform.machine().endswith('64'):
        espeak_lib = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'
    else:
        espeak_lib = r'C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll'
    if os.path.exists(espeak_lib):
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeak_lib)

from neuttsair.neutts import NeuTTSAir
import re

def split_text_into_chunks(text, max_tokens=1500):
    """Split text into chunks that fit within token limit.

    Uses sentence boundaries to maintain naturalness.
    Conservative max_tokens (1500) leaves room for reference text and safety margin.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Rough estimate: ~1 token per word, add 10% margin
        sentence_tokens = len(sentence.split()) * 1.1

        if current_length + sentence_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Load model once at startup
print("Starting TTS server...")
print("Loading model (this takes a moment)...")
TTS = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air-q4-gguf",
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec",
    codec_device="cpu"
)

# Pre-load reference voices
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES = {}
for voice in ["javis"]:
    ref_audio = os.path.join(SCRIPT_DIR, "samples", f"{voice}.wav")
    ref_text_path = os.path.join(SCRIPT_DIR, "samples", f"{voice}.txt")
    with open(ref_text_path, "r") as f:
        ref_text = f.read().strip()
    VOICES[voice] = {
        "codes": TTS.encode_reference(ref_audio),
        "text": ref_text
    }

print(f"✓ Server ready on port 5555")
print(f"✓ Loaded voices: {', '.join(VOICES.keys())}")


class TTSHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        text = data.get("text", "")
        voice = data.get("voice", "javis")
        output = data.get("output", "tts_output.wav")

        if not text:
            self.send_response(400)
            self.end_headers()
            return

        # Generate audio with chunking for long texts
        voice_data = VOICES.get(voice, VOICES["javis"])
        sample_rate = 24000

        # Split text into chunks if needed
        chunks = split_text_into_chunks(text)
        print(f"Processing {len(chunks)} chunk(s)...")

        wav_parts = []
        for i, chunk in enumerate(chunks, 1):
            print(f"  Generating chunk {i}/{len(chunks)}: {chunk[:50]}...")
            chunk_wav = TTS.infer(chunk, voice_data["codes"], voice_data["text"])
            wav_parts.append(chunk_wav)

        # Concatenate all audio chunks
        wav = np.concatenate(wav_parts) if wav_parts else np.array([])

        # Add silence at the end to prevent cutoff
        audio_duration = len(wav) / sample_rate
        silence_duration = max(2.0, audio_duration * 0.1)  # At least 2 seconds or 10% of audio length
        silence = np.zeros(int(silence_duration * sample_rate), dtype=wav.dtype)
        wav_with_silence = np.concatenate([wav, silence])

        # Play to default audio device
        sd.play(wav_with_silence, sample_rate, blocking=True)

        # Also save to file if requested
        sf.write(output, wav_with_silence, sample_rate)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "output": output}).encode())

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


if __name__ == "__main__":
    PORT = 5555
    with socketserver.TCPServer(("127.0.0.1", PORT), TTSHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
