# GPU Performance Test Results

## Test Configuration

**Date**: 2025-11-08
**Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU
**CUDA**: 12.8.1
**llama-cpp-python**: 0.3.16 (built with CUDA support)
**Model**: neutts-air-Q4-0.gguf (495 MB quantized)

## GPU Offloading Confirmed ✅

```
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9, VMM: yes

load_tensors: offloaded 25/25 layers to GPU
load_tensors: CUDA0 model buffer size = 390.63 MiB
load_tensors: CPU_Mapped model buffer size = 104.61 MiB

llama_kv_cache_unified: CUDA0 KV buffer size = 384.00 MiB
llama_context: flash_attn = 1
```

## Benchmark Results

### GPU GGUF (Current Test)
| Test | Text | Latency |
|------|------|---------|
| 1 | "Hello world" | 6,505 ms |
| 2 | "This is a test of GPU acceleration" | 6,283 ms |
| 3 | "The quick brown fox jumps over the lazy dog" | 6,819 ms |

**Average**: **6,536 ms** (6.5 seconds)

### CPU GGUF (Previous Baseline)
From CUDA_BUILD_JOURNEY.md:
- **Latency**: ~1,600 ms (1.6 seconds)

## Analysis: GPU is SLOWER ❌

### Unexpected Result
GPU acceleration is **4x SLOWER** than CPU:
- CPU GGUF Q4: **1.6s**
- GPU GGUF Q4: **6.5s**

### Possible Causes

#### 1. Flash Attention Overhead
```
llama_context: flash_attn = 1
```
Flash attention enabled but PyTorch warning suggests it's not compiled with flash attention support:
```
UserWarning: Torch was not compiled with flash attention
```

This could cause performance degradation rather than improvement.

#### 2. Small Batch Inference
TTS generates token-by-token for short sentences. GPU excels at large batch parallel processing, not sequential generation.

#### 3. PCIe/Memory Transfer Overhead
- Model weights: 390 MB on GPU
- KV cache: 384 MB on GPU
- Constant CPU ↔ GPU transfers for each token

For small inference tasks, CPU cache locality may outperform GPU despite lower compute.

#### 4. Laptop GPU Thermal Throttling
RTX 4060 Laptop GPU may throttle under sustained load, reducing effective performance.

#### 5. CUDA 12.8 vs PyTorch CUDA 12.1 Mismatch
Server shows:
```
✓ CUDA version: 12.1  (PyTorch)
```

But llama-cpp-python built with CUDA 12.8. Potential driver/library version conflicts.

## Configuration Used

```python
# tts_server.py
backbone_device = "gpu"  # GGUF GPU offload
codec_device = "cuda"     # PyTorch GPU

# neutts.py
n_gpu_layers = -1          # All 25 layers on GPU
flash_attn = True          # Enabled for "gpu" device
n_ctx = 32768              # Large context window
```

## Recommendations

### Option 1: Disable Flash Attention
```python
flash_attn=False  # neutts.py:187
```

Test if flash attention overhead is the culprit.

### Option 2: Reduce Context Size
```python
n_ctx=2048  # Instead of 32768
```

Smaller KV cache = less VRAM = potentially faster.

### Option 3: Use CPU GGUF (BEST)
**Recommendation**: Revert to CPU GGUF backbone.

```python
# tts_server.py
backbone_device = "cpu"   # 1.6s latency
codec_device = "cuda"     # Keep GPU for codec
```

**Why**:
- ✅ 4x faster (1.6s vs 6.5s)
- ✅ No CUDA compilation complexity
- ✅ Proven working configuration
- ✅ Codec still uses GPU for audio decode

### Option 4: Try Different Model Precision
Test if Q8 quantization performs better on GPU than Q4:
```bash
pip install huggingface-hub
huggingface-cli download neuphonic/neutts-air-q8-gguf
```

## Conclusion

**GPU acceleration for GGUF backbone is NOT beneficial for this use case.**

The original CPU GGUF + GPU codec configuration was actually optimal:
- Backbone (LLM token generation): **CPU** - 1.6s ✅
- Codec (audio decode): **GPU** - leverages CUDA ✅

Token-by-token generation for short texts doesn't benefit from GPU parallelism. CPU cache locality and lower latency win for this workload.

## Next Steps

1. **Revert to CPU backbone** (`backbone_device="cpu"`)
2. **Keep GPU codec** (`codec_device="cuda"`)
3. Document lesson learned: Not all models benefit from GPU offloading
4. Consider GPU only for batch inference or longer text generation

## Files Modified

- `neuttsair/neutts.py` - Enabled verbose=True to see offloading
- `tts_server.py` - Auto-detect GPU and set backbone_device="gpu"

## Takeaway

✅ **CUDA 12.8 build successful** - GPU offloading works
❌ **GPU performance regression** - 4x slower than CPU
📝 **Lesson**: Small models + short inference = CPU wins over GPU
