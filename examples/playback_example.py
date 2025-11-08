import os
import time
import soundfile as sf
from neuttsair.neutts import NeuTTSAir

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("sounddevice not installed. Install with: pip install sounddevice")


def play_audio(wav, sample_rate=24000):
    """Play audio using sounddevice."""
    if not HAS_SOUNDDEVICE:
        print("Cannot play audio - sounddevice not installed")
        return

    print(f"\n▶ Playing audio ({len(wav)/sample_rate:.2f} seconds)...")
    sd.play(wav, sample_rate)
    sd.wait()  # Wait until playback is finished
    print("✓ Playback complete")


def main(input_text, ref_audio_path, ref_text, backbone, output_path="output.wav", play=False, save=True):
    if not ref_audio_path or not ref_text:
        print("No reference audio or text provided.")
        return None

    # Initialize NeuTTSAir with the desired model and codec
    print("Loading TTS model...")
    tts = NeuTTSAir(
        backbone_repo=backbone,
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu"
    )

    # Check if ref_text is a path if it is read it if not just return string
    if ref_text and os.path.exists(ref_text):
        with open(ref_text, "r") as f:
            ref_text = f.read().strip()

    print("\nEncoding reference audio...")
    ref_codes = tts.encode_reference(ref_audio_path)

    print(f"\nGenerating audio for: '{input_text}'")
    start_time = time.time()

    wav = tts.infer(input_text, ref_codes, ref_text)

    generation_time = time.time() - start_time
    audio_duration = len(wav) / 24000  # Sample rate is 24kHz
    rtf = generation_time / audio_duration if audio_duration > 0 else 0

    print(f"\n📊 Performance Metrics:")
    print(f"   Generation time: {generation_time:.2f}s")
    print(f"   Audio duration:  {audio_duration:.2f}s")
    print(f"   Real-Time Factor (RTF): {rtf:.2f}x")

    if rtf < 1.0:
        print(f"   ✓ Faster than real-time! ({1/rtf:.2f}x speed)")
    else:
        print(f"   ⚠ Slower than real-time ({rtf:.2f}x slower)")

    if save:
        print(f"\n💾 Saving output to {output_path}")
        sf.write(output_path, wav, 24000)

    if play:
        play_audio(wav, sample_rate=24000)

    return wav


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuTTSAir Example with Playback")
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help="Input text to be converted to speech"
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        default="./samples/dave.wav",
        help="Path to reference audio file"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default="./samples/dave.txt",
        help="Reference text corresponding to the reference audio",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.wav",
        help="Path to save the output audio"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="neuphonic/neutts-air-q4-gguf",
        help="Huggingface repo containing the backbone checkpoint"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play audio after generation"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save audio to file (useful with --play)"
    )

    args = parser.parse_args()

    main(
        input_text=args.input_text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        backbone=args.backbone,
        output_path=args.output_path,
        play=args.play,
        save=not args.no_save,
    )
