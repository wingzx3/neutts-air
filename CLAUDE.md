# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuTTS Air is an on-device TTS (Text-to-Speech) model with instant voice cloning. Built on Qwen 0.5B LLM backbone + NeuCodec audio codec. Supports multiple formats: PyTorch, GGUF (quantized), ONNX decoder.

## Core Architecture

### Three-Component System

1. **Backbone**: LLM that generates speech tokens from phonemized text
   - Default: `neuphonic/neutts-air` (PyTorch)
   - Quantized: `neuphonic/neutts-air-q4-gguf` or `neuphonic/neutts-air-q8-gguf` (llama-cpp)
   - Loaded via `_load_backbone()` in `neuttsair/neutts.py`

2. **Codec**: Encodes/decodes audio ↔ speech tokens
   - Default: `neuphonic/neucodec` (PyTorch)
   - Alternative: `neuphonic/distill-neucodec` (PyTorch)
   - ONNX: `neuphonic/neucodec-onnx-decoder` (onnxruntime)
   - Loaded via `_load_codec()` in `neuttsair/neutts.py`

3. **Phonemizer**: Converts text → phonemes using espeak backend
   - Always required (system dependency)
   - espeak library path must be configured on Windows (see tts_server.py:13-21)

### Inference Flow

```
Text → Phonemize → Chat Template → Backbone (LLM) → Speech Tokens → Codec Decode → Audio → Watermark
```

- `infer()`: Standard inference (neutts.py:383)
- `infer_stream()`: Streaming inference using GGUF backbone only (neutts.py:408)
- `encode_reference()`: Encodes reference audio to codes for voice cloning (neutts.py:426)

### Device Selection Logic

Device resolution is complex due to multiple backends (PyTorch, GGUF, ONNX):
- `_select_torch_device()` (neutts.py:91): Auto-selects cuda→mps→cpu for PyTorch
- `_select_backbone_device()` (neutts.py:129): GGUF uses "gpu"/"mps" (maps to llama-cpp n_gpu_layers)
- `_configure_onnx_codec_session()` (neutts.py:235): Maps devices to ONNX execution providers

**IMPORTANT**: Don't assume device strings are uniform across backends. GGUF uses "gpu", PyTorch uses "cuda", ONNX uses provider names.

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: GGUF support
pip install llama-cpp-python

# Optional: ONNX decoder support
pip install onnxruntime
```

### Running Examples
```bash
# Basic PyTorch example
python -m examples.basic_example \
  --input_text "My name is Dave" \
  --ref_audio samples/dave.wav \
  --ref_text samples/dave.txt

# GGUF quantized model
python -m examples.basic_example \
  --input_text "My name is Dave" \
  --ref_audio samples/dave.wav \
  --ref_text samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf

# ONNX decoder (requires pre-encoded reference)
python -m examples.encode_reference \
  --ref_audio samples/dave.wav \
  --output_path encoded_reference.pt

python -m examples.onnx_example \
  --input_text "My name is Dave" \
  --ref_codes encoded_reference.pt \
  --ref_text samples/dave.txt
```

### Server Mode
```bash
# Start persistent server (keeps model loaded)
python tts_server.py

# Client usage (from another terminal)
python tts_client.py
```

Server at tts_server.py:
- Loads model once at startup (fast subsequent requests)
- Pre-loads reference voices from samples/
- Handles long text via chunking (split_text_into_chunks:26)
- Auto-plays audio + saves to file
- HTTP endpoint on localhost:5555

## Key Implementation Details

### Chat Template System
- Uses special tokens: `<|SPEECH_REPLACE|>`, `<|TEXT_REPLACE|>`, `<|SPEECH_GENERATION_START|>`, etc.
- Template in `_apply_chat_template()` (neutts.py:463) for PyTorch
- GGUF uses string-based prompt in `_infer_ggml()` (neutts.py:514)

### Streaming Implementation
- Only works with GGUF backbone (`_infer_stream_ggml()` at neutts.py:533)
- Uses `_linear_overlap_add()` to blend chunks smoothly (neutts.py:15)
- Configurable via: `streaming_frames_per_chunk`, `streaming_lookback`, `streaming_lookforward` (neutts.py:57-59)

### Reference Audio Requirements
Reference samples should be:
- Mono, 16-44kHz sample rate
- 3-15 seconds duration
- `.wav` format
- Clean audio (minimal background noise)
- Natural continuous speech

### Watermarking
All outputs are watermarked via Perth (Perceptual Threshold Watermarker):
- Applied in `infer()` (neutts.py:404) and streaming (neutts.py:583, 615)
- Uses `perth.PerthImplicitWatermarker()` (neutts.py:84)

## Common Gotchas

1. **espeak path on Windows**: Must manually set library path (see tts_server.py or README)
2. **GGUF device strings**: Use "gpu" not "cuda" for GGUF backbones
3. **Streaming requires GGUF**: PyTorch backbone doesn't support streaming
4. **ONNX requires pre-encoded references**: Can't use raw audio with ONNX decoder
5. **Token limits**: Max context is 32768 tokens (neutts.py:54) - use chunking for long text

## Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```
