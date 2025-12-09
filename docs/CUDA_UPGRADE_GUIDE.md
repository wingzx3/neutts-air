# CUDA 12.8 Upgrade Guide

## Why Upgrade from 12.1 to 12.8?

CUDA 12.1 missing critical C++ headers:
- `cub/cub.cuh` (CUB library)
- `thrust/` (Thrust library)
- Incomplete libcudacxx headers

CUDA 12.8 includes CCCL (CUDA Core Compute Libraries) with all headers bundled.

## Step-by-Step Upgrade Process

### 1. Download CUDA Toolkit 12.8.1

**Download Link**: https://developer.nvidia.com/cuda-12-8-1-download-archive

**Select**:
- Operating System: Windows
- Architecture: x86_64
- Version: 11 (or your Windows version)
- Installer Type: **exe (local)** - recommended (3.9GB) for offline install

**Alternative**: Network installer (~30MB, downloads components during install)

### 2. Uninstall CUDA 12.1

**Option A - Windows Settings** (Recommended):
1. Open Settings → Apps → Installed apps
2. Search for "CUDA"
3. Uninstall all CUDA 12.1 components:
   - NVIDIA CUDA Runtime 12.1
   - NVIDIA CUDA Development 12.1
   - NVIDIA CUDA Documentation 12.1
   - NVIDIA CUDA Visual Studio Integration 12.1
   - Any other CUDA 12.1 entries

**Option B - Control Panel**:
1. Control Panel → Programs → Programs and Features
2. Uninstall all NVIDIA CUDA 12.1 components

**Important**: Do NOT uninstall:
- NVIDIA Graphics Driver
- NVIDIA GeForce Experience
- Other NVIDIA display components

### 3. Install CUDA Toolkit 12.8.1

1. Run the downloaded installer (as Administrator if prompted)
2. **Installation Type**: Choose **Custom (Advanced)**
3. **Components to Install** (CRITICAL):
   - ✅ CUDA → Development → Compiler → nvcc
   - ✅ CUDA → Development → Libraries → CUDA Runtime
   - ✅ CUDA → Development → Libraries → cuBLAS
   - ✅ CUDA → Development → Libraries → CCCL (libcudacxx, CUB, Thrust)
   - ✅ CUDA → Integration → Visual Studio Integration
   - ✅ CUDA → Documentation (optional)
   - ❌ Driver components (skip if you have recent driver)

4. **Installation Path**: Use default `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`

5. Click **Next** → **Install**

6. Wait for installation (10-20 minutes)

### 4. Verify Installation

Open new Command Prompt (restart terminal to refresh PATH):

```cmd
nvcc --version
```

**Expected output**:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on <date>
Cuda compilation tools, release 12.8, V12.8.x
```

### 5. Verify CCCL Headers

Check for CUB headers:
```cmd
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include\cub"
```

Check for Thrust headers:
```cmd
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include\thrust"
```

Check for libcudacxx headers:
```cmd
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include\cuda\std"
```

All three should exist with multiple header files.

### 6. Copy Visual Studio Integration Files

**Purpose**: Allow CMake to detect CUDA Toolkit in Visual Studio projects.

**Source**:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\visual_studio_integration\MSBuildExtensions\
```

**Destinations** (copy to ALL that exist):

For VS 2017 Community:
```
C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\Microsoft\VC\v141\BuildCustomizations\
```

For VS 2017 BuildTools:
```
C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\MSBuild\Microsoft\VC\v141\BuildCustomizations\
```

For VS 2019 Community:
```
C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations\
```

For VS 2022 Community:
```
C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\
```

**Files to copy** (4 files):
- `CUDA 12.8.props`
- `CUDA 12.8.targets`
- `CUDA 12.8.xml`
- `Nvda.Build.CudaTasks.v12.8.dll`

**Command** (run as Administrator / with gsudo):
```cmd
gsudo powershell -Command "Copy-Item 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\visual_studio_integration\MSBuildExtensions\*' -Destination 'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\Microsoft\VC\v141\BuildCustomizations\' -Force"
```

Repeat for all VS installations you have.

### 7. Build llama-cpp-python with CUDA

Open **NEW** Command Prompt (to get updated CUDA paths):

```cmd
cd C:\apps\neutts-air

REM Set environment variables
set FORCE_CMAKE=1
set CMAKE_ARGS=-DGGML_CUDA=ON

REM Optional: Parallel build
set CMAKE_BUILD_PARALLEL_LEVEL=%NUMBER_OF_PROCESSORS%

REM Build and install
venv\Scripts\pip.exe install llama-cpp-python --no-cache-dir --force-reinstall --upgrade --verbose
```

Watch for:
- ✅ "-- Found CUDAToolkit"
- ✅ "-- CUDA found"
- ✅ "-- GGML CUDA: ON"
- ✅ Compiling .cu files (mean.cu, ssm-scan.cu, sum.cu should work now)

### 8. Verify GPU Support

Test if llama-cpp-python was built with CUDA:

```cmd
venv\Scripts\python.exe -c "from llama_cpp import Llama; print('CUDA build successful!')"
```

Test loading model on GPU:
```python
from llama_cpp import Llama

model = Llama(
    model_path="C:/Users/wingz/.cache/huggingface/hub/models--neuphonic--neutts-air-q4-gguf/snapshots/*/neutts-air-Q4_K_M.gguf",
    n_gpu_layers=-1,  # -1 = all layers on GPU
    verbose=True
)

# Should show: "llama_init_from_gpt_params: using CUDA for GPU acceleration"
# Should show: "llama_init_from_gpt_params: offloading XX layers to GPU"
```

### 9. Update tts_server.py

Change GGUF backend device to use GPU:
```python
# In tts_server.py, update backbone_device:
backbone_device="gpu",  # Changed from None to "gpu"
```

### 10. Test Performance

Run benchmark:
```cmd
python benchmark_latency.py
```

**Expected improvement**:
- Before (CPU GGUF): ~1.6s
- After (GPU GGUF): ~500-800ms (2-3x faster)

## Troubleshooting

### "No CUDA toolset found" during build
- Verify Visual Studio Integration files were copied
- Restart terminal/IDE after CUDA install
- Check CMAKE_ARGS is set correctly

### "cub/cub.cuh not found"
- Verify CUDA 12.8 installed (not 12.1)
- Check `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include\cub\` exists

### Build fails with CUDA errors
- Ensure you have C++ compiler (`cl.exe` from Visual Studio)
- Try disabling CUDA curl: `set CMAKE_ARGS=-DGGML_CUDA=ON -DLLAMA_CURL=OFF`

### Model doesn't use GPU
- Check n_gpu_layers parameter (-1 for all layers)
- Verify verbose output shows "using CUDA for GPU acceleration"
- Ensure RTX 4060 is visible: `nvidia-smi`

## Rollback Plan

If CUDA 12.8 causes issues:
1. Uninstall CUDA 12.8 (same process as uninstalling 12.1)
2. Reinstall CUDA 12.1
3. Use CPU GGUF (current working config at 1.6s latency)

## References

- CUDA 12.8.1 Download: https://developer.nvidia.com/cuda-12-8-1-download-archive
- llama-cpp-python Build Guide: https://github.com/abetlen/llama-cpp-python/issues/1963
- CCCL Documentation: https://github.com/NVIDIA/cccl
