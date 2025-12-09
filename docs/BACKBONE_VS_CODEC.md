# Backbone vs Codec - TTS Architecture Explained

## Overview

NeuTTS Air uses a **two-stage pipeline** for text-to-speech:

```
Text → Backbone → Speech Tokens → Codec → Audio
```

## The Two Components

### 1. Backbone (LLM - Language Model)

**What it does**: Converts text into "speech tokens" (abstract audio representation)

**Input**: Phonemized text (e.g., "hɛˈloʊ wɜrld")
**Output**: Sequence of integer tokens (e.g., [42, 17, 89, 23, ...])

**How it works**:
- Takes text and reference voice codes
- Uses a **Qwen 0.5B LLM** (748M parameters)
- Generates speech tokens autoregressively (token-by-token)
- Similar to ChatGPT generating text, but generates "audio tokens" instead

**Models available**:
- `neuphonic/neutts-air` - Full precision PyTorch (1.5GB)
- `neuphonic/neutts-air-q4-gguf` - Quantized GGUF (495MB) ✅ **We use this**
- `neuphonic/neutts-air-q8-gguf` - Higher quality quantized (larger)

**Performance characteristics**:
- **CPU-bound** for quantized models
- Token-by-token generation (sequential, not parallel)
- Accounts for ~70% of total inference time

---

### 2. Codec (Neural Audio Codec)

**What it does**: Converts speech tokens into actual audio waveform

**Input**: Speech tokens from backbone (e.g., [42, 17, 89, ...])
**Output**: Raw audio waveform (24kHz .wav file)

**How it works**:
- Uses **NeuCodec** neural network
- Decodes token sequences → audio samples
- Similar to how MP3/JPEG decompress data, but learned via neural network
- Also used in reverse to encode reference audio for voice cloning

**Models available**:
- `neuphonic/neucodec` - Standard codec ✅ **We use this**
- `neuphonic/distill-neucodec` - Distilled version (smaller/faster)
- `neuphonic/neucodec-onnx-decoder` - ONNX optimized

**Performance characteristics**:
- **GPU-accelerated** (benefits from parallelism)
- Processes audio frames in parallel
- Accounts for ~30% of total inference time

---

## Full Pipeline Visualization

```
1. INPUT TEXT
   "Hello world"
   ↓
2. PHONEMIZER (espeak)
   "hɛˈloʊ wɜrld"
   ↓
3. BACKBONE (Qwen LLM 0.5B)
   [CPU - GGUF Q4 - 495MB]
   ↓
   Speech Tokens: [42, 17, 89, 23, 156, 78, ...]
   ↓
4. CODEC (NeuCodec)
   [GPU - PyTorch - Neural Network]
   ↓
   Audio Waveform: [0.02, -0.15, 0.34, ...]
   ↓
5. WATERMARK (Perth)
   Inaudible watermark added
   ↓
6. OUTPUT AUDIO
   24kHz .wav file
```

---

## Why Separate Backbone & Codec?

### Flexibility
- Swap backbone: Use full precision or quantized
- Swap codec: Use standard, distilled, or ONNX

### Optimization
- **Backbone**: Optimize for CPU (quantization, GGUF format)
- **Codec**: Optimize for GPU (PyTorch CUDA)

### Resource Allocation
- Backbone needs **memory** (large model)
- Codec needs **compute** (parallel processing)

---

## Current Configuration

```python
backbone_repo = "neuphonic/neutts-air-q4-gguf"  # 495MB quantized
backbone_device = "cpu"                          # Optimal for sequential generation

codec_repo = "neuphonic/neucodec"               # Neural codec
codec_device = "cuda"                            # GPU acceleration
```

### Why This Is Optimal

| Component | Device | Reason |
|-----------|--------|--------|
| **Backbone** | CPU | Token-by-token generation doesn't benefit from GPU parallelism. CPU has lower latency and better cache locality. |
| **Codec** | GPU | Audio decoding is inherently parallel. GPU excels at matrix operations needed for neural audio synthesis. |

---

## Performance Breakdown

For a typical 10-word sentence:

### Backbone (CPU GGUF Q4)
- **Time**: ~1.1 seconds
- **Work**: Generate ~100-200 speech tokens
- **Process**: Sequential (token-by-token)
- **Bottleneck**: Model size + memory bandwidth

### Codec (GPU PyTorch)
- **Time**: ~0.5 seconds
- **Work**: Decode tokens → 2-3 seconds of 24kHz audio
- **Process**: Parallel (frames processed simultaneously)
- **Bottleneck**: GPU compute

### Total Latency
- **Combined**: ~1.6 seconds ✅

---

## Analogy

Think of it like video encoding:

**Backbone = Script Writer**
- Writes a "script" (speech tokens) describing what audio to make
- Slow, thoughtful, sequential process
- Like a human writing: one word at a time

**Codec = Video Renderer**
- Takes the script and renders actual pixels/audio
- Fast, parallel, GPU-optimized
- Like GPU rendering a video: processes many frames at once

---

## Voice Cloning

The codec also works in **reverse** for voice cloning:

```
Reference Audio → Codec Encode → Speech Codes → Stored for inference

Later:
New Text + Stored Codes → Backbone → New Speech Tokens → Codec Decode → Audio (in reference voice)
```

The backbone learns to mimic the "style" represented by the reference codes.

---

## Key Takeaways

1. **Backbone** = LLM that generates "what to say" as tokens
2. **Codec** = Neural network that renders tokens into audio
3. **Optimal split**: CPU backbone + GPU codec
4. **Why separate**: Different compute characteristics (sequential vs parallel)
5. **Quantization helps backbone**, GPU helps codec

This architecture enables **fast, on-device TTS with voice cloning** - the backbone being quantized (495MB) makes it small enough to run on CPU, while codec on GPU keeps audio quality high.
