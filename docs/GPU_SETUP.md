# GPU Setup for NeuTTS Air

## Status: ✓ GPU Enabled

Your RTX 4060 Laptop GPU is now configured and working with NeuTTS Air.

## Configuration Summary

### Hardware Detected
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **CUDA Version**: 12.1
- **Status**: CUDA available and working

### Software Configuration
The TTS server (`tts_server.py`) is now configured to automatically use GPU when available:

- **Backbone Device**: `"gpu"` (GGUF with llama-cpp GPU layer offloading)
- **Codec Device**: `"cuda"` (PyTorch GPU acceleration)

If no GPU is detected, it automatically falls back to CPU.

## Testing

### Quick GPU Test
Run the included test script to verify GPU is working:

```bash
python -u test_gpu.py
```

Expected output:
```
GPU Test for NeuTTS Air
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
✓ Model loaded successfully!
✓ Generated audio with 52320 samples
GPU test completed successfully!
```

### Running the Server
Start the server with GPU support:

```bash
python -u tts_server.py
```

Or in a new window:
```bash
start cmd /k "python -u tts_server.py"
```

The server will display:
```
✓ GPU detected: NVIDIA GeForce RTX 4060 Laptop GPU
✓ CUDA version: 12.1
Loading NeuTTSAir with backbone_device=gpu, codec_device=cuda...
```

### Test the Server
Once loaded (takes ~1-2 minutes first time), test with:

```bash
curl -X POST http://127.0.0.1:5555 -H "Content-Type: application/json" -d "{\"text\": \"GPU acceleration is working\", \"voice\": \"javis\", \"output\": \"test.wav\"}"
```

Or use the client:
```bash
python tts_client.py
```

## Performance Expectations

With your RTX 4060 Laptop GPU, you should see:
- **2-5x speedup** over CPU-only inference
- **Sub-100ms latency** for voice cloning
- **Real-time or faster** generation (audio duration ≤ generation time)

## How It Works

### Device Mapping
1. **GGUF Backbone** (neuphonic/neutts-air-q4-gguf)
   - Uses `device="gpu"` (not "cuda"!)
   - Offloads layers to GPU via llama-cpp-python
   - Quantized (Q4) for efficient memory usage

2. **PyTorch Codec** (neuphonic/neucodec)
   - Uses `device="cuda"`
   - Standard PyTorch GPU acceleration
   - Handles audio encoding/decoding on GPU

3. **Phonemizer**
   - Runs on CPU (no GPU support)
   - Pre-configured for Windows espeak

### Code Changes
The server now includes:
```python
import torch
gpu_available = torch.cuda.is_available()
if gpu_available:
    backbone_device = "gpu"   # For GGUF
    codec_device = "cuda"      # For PyTorch
else:
    backbone_device = "cpu"
    codec_device = "cpu"

TTS = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air-q4-gguf",
    backbone_device=backbone_device,
    codec_repo="neuphonic/neucodec",
    codec_device=codec_device
)
```

## Troubleshooting

### If GPU not detected:
1. Verify CUDA with: `python -c "import torch; print(torch.cuda.is_available())"`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### If llama-cpp-python not using GPU:
Reinstall with CUDA support:
```bash
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Common Issues
- **"AssertionError: Torch not compiled with CUDA"**: Reinstall PyTorch with CUDA (see above)
- **Slow performance despite GPU**: Check Task Manager > Performance > GPU to verify usage
- **Out of memory errors**: Reduce batch size or use Q4 quantized model (already default)

## Files Modified
- `tts_server.py`: Added GPU auto-detection and device configuration
- `test_gpu.py`: New test script for verifying GPU setup

## Notes
- First model load takes 1-2 minutes (downloads models to cache)
- Subsequent loads are faster (~30-60 seconds)
- The warnings about `torch.load` and `weight_norm` are harmless
- The llama cleanup exception at exit is a known llama-cpp-python issue (harmless)
