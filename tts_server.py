#!/usr/bin/env python3
"""TTS Server - keeps model loaded in memory for fast responses."""
import os
import sys
import socketserver
import json
import argparse
import numpy as np
import soundfile as sf
import sounddevice as sd
from http.server import BaseHTTPRequestHandler

# Setup CUDA paths BEFORE importing GPU-dependent libraries (like llama-cpp)
def setup_cuda_paths():
    """Add PyTorch CUDA DLLs to PATH for llama-cpp-python to use."""
    paths_to_add = []

    # Add PyTorch lib directory (contains cublas, cudart, etc.)
    try:
        import torch
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib_path):
            paths_to_add.append(torch_lib_path)
            print(f"Found PyTorch CUDA libraries at: {torch_lib_path}", flush=True)
    except ImportError:
        pass

    # Update PATH
    if paths_to_add:
        current_path = os.environ.get('PATH', '')
        os.environ['PATH'] = os.pathsep.join(paths_to_add) + os.pathsep + current_path
        print(f"Added CUDA paths to environment", flush=True)

# Setup CUDA paths first
setup_cuda_paths()

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
import warnings

# Suppress phonemizer and torch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='phonemizer')
warnings.filterwarnings('ignore', category=UserWarning, module='neucodec')
warnings.filterwarnings('ignore', category=FutureWarning)

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

# Parse command line arguments
parser = argparse.ArgumentParser(description='TTS Server')
parser.add_argument('--voice', type=str, default='scarlett', help='Default voice to use')
args = parser.parse_args()

DEFAULT_VOICE = args.voice

# Load model once at startup
print("Starting TTS server...", flush=True)
print("Loading model (this takes a moment)...", flush=True)

# Check GPU availability
import torch
gpu_available = torch.cuda.is_available()
if gpu_available:
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"✓ CUDA version: {torch.version.cuda}", flush=True)
    # CPU backbone optimal: 1.6s vs GPU 6.0s (even after warmup)
    # GPU warmup improves 7.3% (7.6s→6.0s) but still 3.7x slower than CPU
    backbone_device = "cpu"   # 1.6s latency (optimal)
    codec_device = "cuda"      # GPU for audio decode
else:
    print("⚠ No GPU detected, falling back to CPU", flush=True)
    backbone_device = "cpu"
    codec_device = "cpu"

print(f"Loading NeuTTSAir with backbone_device={backbone_device}, codec_device={codec_device}...", flush=True)
TTS = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air-q4-gguf",  # Quantized GGUF for best CPU performance
    backbone_device=backbone_device,
    codec_repo="neuphonic/neucodec",
    codec_device=codec_device
)
print("✓ Model loaded successfully!", flush=True)

# Pre-load reference voices
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES = {}

# Auto-discover all voices in samples folder
voice_files = [f[:-4] for f in os.listdir(os.path.join(SCRIPT_DIR, "samples"))
               if f.endswith('.wav') and os.path.exists(os.path.join(SCRIPT_DIR, "samples", f[:-4] + '.txt'))]

for voice in voice_files:
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
print(f"✓ Default voice: {DEFAULT_VOICE}")


class TTSHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        text = data.get("text", "")
        voice = data.get("voice", DEFAULT_VOICE)
        output = data.get("output", "tts_output.wav")

        if not text:
            self.send_response(400)
            self.end_headers()
            return

        # Generate audio with streaming for better responsiveness
        voice_data = VOICES.get(voice, VOICES.get(DEFAULT_VOICE, VOICES[list(VOICES.keys())[0]]))
        sample_rate = 24000

        # Use streaming inference for better perceived latency
        print(f"Generating (streaming): {text[:100]}...")

        wav_parts = []
        chunk_count = 0
        first_chunk_time = None

        # Stream audio chunks as they're generated
        import time
        start_time = time.time()

        for audio_chunk in TTS.infer_stream(text, voice_data["codes"], voice_data["text"]):
            chunk_count += 1
            wav_parts.append(audio_chunk)

            if chunk_count == 1:
                first_chunk_time = (time.time() - start_time) * 1000
                print(f"  First chunk ready in {first_chunk_time:.0f}ms, starting playback...")

        print(f"  Received {chunk_count} chunks total in {(time.time() - start_time) * 1000:.0f}ms")

        # Concatenate all audio chunks
        wav = np.concatenate(wav_parts) if wav_parts else np.array([])

        # Add silence at the end to prevent cutoff
        audio_duration = len(wav) / sample_rate
        silence_duration = max(2.0, audio_duration * 0.1)  # At least 2 seconds or 10% of audio length
        silence = np.zeros(int(silence_duration * sample_rate), dtype=wav.dtype)
        wav_with_silence = np.concatenate([wav, silence])

        # Play complete audio to default device
        sd.play(wav_with_silence, sample_rate, blocking=True)

        # Save complete audio to file
        sf.write(output, wav_with_silence, sample_rate)

        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "output": output}).encode())
        except (ConnectionAbortedError, BrokenPipeError):
            # Client disconnected after audio played - this is normal
            pass

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


if __name__ == "__main__":
    PORT = 5555
    with socketserver.TCPServer(("127.0.0.1", PORT), TTSHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            # Explicitly close TTS model to avoid llama-cpp cleanup errors
            if hasattr(TTS, 'backbone') and hasattr(TTS.backbone, 'close'):
                try:
                    TTS.backbone.close()
                except:
                    pass
            sys.exit(0)
