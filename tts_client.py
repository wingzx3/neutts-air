#!/usr/bin/env python3
"""Fast TTS client - connects to tts_server.py"""
import argparse
import json
import sys
import urllib.request
import urllib.error

try:
    import sounddevice as sd
    import soundfile as sf
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


def main():
    parser = argparse.ArgumentParser(description="Fast TTS client")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("--voice", default="javis", choices=["javis"])
    parser.add_argument("--save", help="Output file")
    parser.add_argument("--no-play", action="store_true")
    args = parser.parse_args()

    output_file = args.save or "tts_temp.wav"

    data = json.dumps({
        "text": args.text,
        "voice": args.voice,
        "output": output_file
    }).encode()

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:5555",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        urllib.request.urlopen(req, timeout=30)

        if not args.no_play and HAS_SOUNDDEVICE:
            wav, sr = sf.read(output_file)
            sd.play(wav, sr)
            sd.wait()

    except urllib.error.URLError:
        print("Error: TTS server not running. Start it with: python tts_server.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
