# TTS Latency Analysis

## Current Performance (RTX 4060 Laptop GPU)

### Measured Latency Breakdown

**Server-side (via HTTP)**: 7-10 seconds
- Actual inference: ~1.6s
- Audio playback (blocking): ~5-8s
- Network/overhead: minimal

**Direct inference**: ~1.6 seconds
- "Hello world" (1.1s audio): 1502-1683ms
- Real-time factor: 1.37x (slower than real-time)

### What's Using GPU

✓ **Codec (PyTorch)**: Running on CUDA
- Audio encoding/decoding on GPU
- ~2.2s for reference encoding

✗ **Backbone (GGUF)**: Running on CPU only
- llama-cpp-python built without CUDA support
- Using AVX512 CPU instructions instead
- `n_gpu_layers=-1` has no effect

## Why llama-cpp-python Isn't Using GPU

The pip install with CMAKE_ARGS doesn't work reliably on Windows. The compilation only built CPU backend with AVX512.

### Options to Fix

#### Option 1: Use PyTorch Backbone (Recommended for GPU)
Switch from GGUF to full PyTorch model for true GPU acceleration:

```python
TTS = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air",  # Not -q4-gguf
    backbone_device="cuda",
    codec_repo="neuphonic/neucodec",
    codec_device="cuda"
)
```

**Pros**:
- Full GPU acceleration for backbone
- Likely 2-5x faster inference
- No compilation issues

**Cons**:
- Larger model size (~2GB vs ~600MB)
- Higher VRAM usage

#### Option 2: Build llama-cpp-python with CUDA (Complex)
Manually compile llama-cpp-python with CUDA support on Windows:

1. Install CMake and Visual Studio Build Tools
2. Install CUDA Toolkit 12.1
3. Clone llama-cpp-python repo
4. Build with: `cmake -DLLAMA_CUDA=ON ..`
5. Install the wheel

**Pros**:
- Keeps quantized model (lower VRAM)
- GPU acceleration for GGUF

**Cons**:
- Complex Windows build process
- Requires CUDA toolkit install
- May have compatibility issues

#### Option 3: Accept Current Performance
1.6s latency for CPU GGUF is reasonable:

**Pros**:
- Works out of the box
- Low VRAM usage
- Stable

**Cons**:
- Slower than GPU potential
- Not utilizing full RTX 4060 capability

### Server Latency Fix

The 7-10s server latency is mostly audio playback. The server uses `blocking=True`:

```python
# tts_server.py:120
sd.play(wav_with_silence, sample_rate, blocking=True)
```

**Fix**: Make non-blocking:
```python
sd.play(wav_with_silence, sample_rate, blocking=False)
```

This would drop server response to ~1.6s (just inference time).

## Recommendation

**For lowest latency with your RTX 4060**:

1. **Switch to PyTorch backbone** for full GPU acceleration
2. **Make audio playback non-blocking** in server
3. Expected result: **~500-800ms inference** (2-3x faster)

**If VRAM is limited**:

Keep current GGUF setup but fix server blocking. Accept 1.6s latency or attempt manual CUDA compilation.

## Test Results Summary

| Configuration | Latency | Real-time Factor |
|--------------|---------|------------------|
| Current (GGUF CPU + PyTorch codec GPU) | 1.6s | 1.37x |
| PyTorch backbone + codec (both GPU) | ~0.5-0.8s* | 0.4-0.6x* |
| Server with blocking playback | 7-10s | N/A |

*Estimated based on typical GPU speedup

## Files

- `profile_latency.py` - Benchmark inference latency
- `benchmark_latency.py` - Test server HTTP latency
- `tts_server.py:120` - Blocking audio playback
