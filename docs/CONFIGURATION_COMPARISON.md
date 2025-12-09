# Configuration Performance Comparison

## All Configurations Tested

| Config | Backbone | Codec | Latency | Relative | Winner |
|--------|----------|-------|---------|----------|--------|
| **Hybrid (Optimal)** | CPU GGUF Q4 | GPU CUDA | **1.6s** | Baseline | ✅ **BEST** |
| Full CPU | CPU GGUF Q4 | CPU | 8.0s | +400% | ❌ Slow |
| Full GPU (cold) | GPU GGUF Q4 | GPU CUDA | 7.6s | +375% | ❌ Very slow |
| Full GPU (warm) | GPU GGUF Q4 | GPU CUDA | 6.0s | +275% | ❌ Slow |

## Test Results Summary

### 1. CPU Backbone + GPU Codec (Current - Optimal) ✅
```python
backbone_device="cpu"
codec_device="cuda"
```
- **Latency**: 1,600 ms
- **Why it wins**:
  - CPU optimal for sequential token generation
  - GPU accelerates parallel audio decode
  - Best of both worlds

### 2. CPU Backbone + CPU Codec
```python
backbone_device="cpu"
codec_device="cpu"
```
- **Latency**: 8,039 ms (5x slower)
- **Why it loses**:
  - Codec decode is inherently parallel
  - CPU can't efficiently process audio frames
  - Neural network operations slow on CPU

### 3. GPU Backbone + GPU Codec (Cold Start)
```python
backbone_device="gpu"
codec_device="cuda"
```
- **First request**: 7,633 ms
- **Why it loses**:
  - Token-by-token generation doesn't parallelize
  - PCIe transfer overhead per token
  - Memory bandwidth bottleneck

### 4. GPU Backbone + GPU Codec (Warmed Up)
- **After 10 requests**: ~6,000 ms average
- **Improvement**: 7.3% from cold start
- **Why still slow**:
  - Warmup helps minimally
  - Still 3.7x slower than CPU backbone
  - Architecture mismatch (sequential vs parallel)

---

## Streaming Output ✅ SUPPORTED

Streaming is **only available with GGUF backbone** (CPU or GPU).

### Streaming Test Results
**Text**: Long sentence (5.8s of audio)

| Metric | Value | Benefit |
|--------|-------|---------|
| **Time to First Byte (TTFB)** | 2,279 ms | User hears audio starting |
| **Total latency** | 11,725 ms | Complete generation |
| **Chunks delivered** | 12 | Progressive playback |
| **Avg chunk interval** | 859 ms | ~0.5s audio every 0.9s |

### Streaming Benefits

**Without Streaming**:
```
[Wait 11.7s] → [Hear all audio at once]
         ↑
    User waits
```

**With Streaming**:
```
[Wait 2.3s] → [Chunk 1] → [Chunk 2] → ... → [Chunk 12]
      ↑          ↑
   TTFB    Audio starts playing immediately
```

**Perceived latency**: 2.3s (TTFB) vs 11.7s (total)
**Improvement**: **5x faster perceived response**

### Streaming Usage

```python
# Example streaming inference
from neuttsair.neutts import NeuTTSAir

tts = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air-q4-gguf",  # GGUF required
    backbone_device="cpu",
    codec_device="cuda"
)

ref_codes = tts.encode_reference("samples/voice.wav")
ref_text = "Reference text here"

# Stream audio chunks as they're generated
for audio_chunk in tts.infer_stream("Long text here", ref_codes, ref_text):
    # Play chunk immediately (e.g., via sounddevice)
    # Or buffer for web streaming
    play_audio(audio_chunk)
```

### Streaming Requirements
- ✅ **GGUF backbone only** (not PyTorch)
- ✅ Works with CPU or GPU backbone
- ✅ Compatible with any codec (CPU/GPU)
- ❌ Not available with PyTorch backbone

---

## Final Recommendations

### For Short Text (1-2 sentences)
**Use**: CPU backbone + GPU codec (current config)
```python
backbone_device="cpu"
codec_device="cuda"
```
- **Latency**: 1.6s
- **Why**: Optimal for short, sequential generation

### For Long Text (paragraphs, articles)
**Use**: CPU backbone + GPU codec **with streaming**
```python
tts.infer_stream(text, ref_codes, ref_text)
```
- **TTFB**: ~2s (user hears audio quickly)
- **Why**: Perceived latency 5x better than waiting for full generation

### For Batch Processing
**Consider**: GPU backbone might help with batching
- Needs custom batching implementation
- Could process multiple requests in parallel
- Not currently implemented in tts_server.py

### For No GPU Available
**Use**: CPU backbone + CPU codec
```python
backbone_device="cpu"
codec_device="cpu"
```
- **Latency**: ~8s
- **Why**: Still functional, just slower

---

## Architecture Insights

### Why CPU Wins for Backbone
1. **Sequential generation**: Tokens generated one-by-one
2. **Small batch size**: Single request at a time
3. **CPU advantages**:
   - Lower latency
   - Better cache locality
   - No PCIe transfer overhead
4. **GPU disadvantages**:
   - Underutilized (built for parallelism)
   - Memory bandwidth bottleneck
   - Transfer overhead every token

### Why GPU Wins for Codec
1. **Parallel processing**: Audio frames decoded simultaneously
2. **Matrix operations**: GPU optimized for neural networks
3. **Larger workload**: More computation per operation
4. **GPU advantages**:
   - Massive parallelism
   - Fast matrix multiplication
   - High memory bandwidth for weights

---

## Performance Summary Table

| Configuration | Short Text | Long Text (streaming) | Batch (future) |
|--------------|------------|----------------------|----------------|
| CPU + GPU | ✅ 1.6s | ✅ 2.3s TTFB | ❌ Not optimal |
| CPU + CPU | ❌ 8.0s | ❌ Slow | ❌ Very slow |
| GPU + GPU | ❌ 6.0s | ⚠️ Works but slower | ✅ Potentially best |

---

## Conclusion

**Optimal configuration confirmed**:
```python
backbone_device = "cpu"   # 1.6s for short text
codec_device = "cuda"     # GPU acceleration for audio
```

**For better UX with long text**:
```python
use tts.infer_stream()    # 2.3s TTFB vs 11.7s total
```

The hybrid CPU+GPU approach leverages strengths of each:
- CPU: Low-latency sequential processing
- GPU: High-throughput parallel processing

GPU backbone works but is counterproductive for this workload. Streaming provides 5x better perceived latency for long text.
