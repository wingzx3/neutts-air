#!/usr/bin/env python3
"""Simple TTS CLI - Generate and play text-to-speech audio."""
import argparse
import sys
import os
import soundfile as sf

# Configure espeak for Windows
if os.name == 'nt':  # Windows
    import platform
    if platform.machine().endswith('64'):
        espeak_lib = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'
    else:
        espeak_lib = r'C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll'

    if os.path.exists(espeak_lib):
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(espeak_lib)

from neuttsair.neutts import NeuTTSAir

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


def play_audio(wav, sample_rate=24000):
    """Play audio using sounddevice."""
    if not HAS_SOUNDDEVICE:
        raise RuntimeError("sounddevice not installed. Install with: pip install sounddevice")

    sd.play(wav, sample_rate)
    sd.wait()


def main():
    parser = argparse.ArgumentParser(
        description="Simple TTS CLI - Generate and play text-to-speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tts "Hello, this is the computer talking"
  tts "Hello world" --voice jo
  tts "Hello world" --save output.wav --no-play
        """
    )
    parser.add_argument("text", nargs="?", help="Text to convert to speech")
    parser.add_argument(
        "--voice",
        type=str,
        default="javis",
        choices=["javis"],
        help="Voice to use (default: javis)"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save audio to file (e.g., output.wav)"
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Don't play audio (only save)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="neuphonic/neutts-air-q4-gguf",
        help="Model to use (default: q4-gguf)"
    )

    args = parser.parse_args()

    # Read from stdin if no text provided
    if not args.text:
        if sys.stdin.isatty():
            parser.print_help()
            sys.exit(1)
        args.text = sys.stdin.read().strip()

    if not args.text:
        print("Error: No text provided")
        sys.exit(1)

    # Initialize TTS
    print("Loading TTS model...")
    tts = NeuTTSAir(
        backbone_repo=args.model,
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu"
    )

    # Load reference voice
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_audio_path = os.path.join(script_dir, "samples", f"{args.voice}.wav")
    ref_text_path = os.path.join(script_dir, "samples", f"{args.voice}.txt")

    with open(ref_text_path, "r") as f:
        ref_text = f.read().strip()

    print(f"Encoding reference ({args.voice})...")
    ref_codes = tts.encode_reference(ref_audio_path)

    print(f"Generating: '{args.text}'")
    wav = tts.infer(args.text, ref_codes, ref_text)

    # Save if requested
    if args.save:
        print(f"Saving to {args.save}")
        sf.write(args.save, wav, 24000)

    # Play unless disabled
    if not args.no_play:
        print("Playing...")
        play_audio(wav, sample_rate=24000)

    print("Done!")


if __name__ == "__main__":
    main()
