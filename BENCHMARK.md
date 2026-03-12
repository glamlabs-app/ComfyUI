# Z-Image Diffusion Model Benchmark

`benchmark_z_image_fp8.py` is a standalone benchmarking script for the Z-Image Turbo diffusion model (Lumina2-based architecture). It measures latency, throughput, and VRAM usage without requiring a text encoder or VAE — dummy conditioning tensors are synthesized so the benchmark isolates diffusion-model compute only.

## Quick Start

```bash
# Basic benchmark (FP16 weights, SageAttention, torch.compile)
python benchmark_z_image_fp8.py \
  --unet_path Z-Image-Turbo-fp16.safetensors \
  --resolutions 2048x1024 \
  --attention sage \
  --raw --compile

# A/B comparison: PyTorch SDPA vs SageAttention
python benchmark_z_image_fp8.py \
  --unet_path Z-Image-Turbo-fp16.safetensors \
  --attention both \
  --repeats 5

# Full optimization stack with compile cache
python benchmark_z_image_fp8.py \
  --unet_path Z-Image-Turbo-fp16.safetensors \
  --resolutions 2048x1024 \
  --attention sage \
  --raw --compile --cuda-graph \
  --cache-file compile_cache.bin \
  --warmup 2 --repeats 5
```

## Features

### Execution Modes

| Flag | Mode | Description |
|------|------|-------------|
| *(none)* | Normal | Runs through the full `comfy.sample.sample` pipeline including CFGGuider, model patching/unpatching per step, etc. Measures end-to-end ComfyUI performance. |
| `--raw` | Raw | Minimal Euler denoising loop calling `apply_model` directly. The model is loaded to GPU and patched once, eliminating ComfyUI's per-step overhead. Isolates pure model compute. |
| `--profile` | Profile | Runs `torch.profiler` and prints a CUDA kernel time breakdown categorized into Attention, Linear/FFN, Norms, and Other. |

### Attention Backends

| `--attention` | Backend | Notes |
|---------------|---------|-------|
| `default` | Whatever ComfyUI selected at startup | Usually PyTorch SDPA |
| `pytorch` | PyTorch's `scaled_dot_product_attention` (SDPA) | Baseline |
| `sage` | [SageAttention](https://github.com/thu-ml/SageAttention) | Optimized attention kernels; requires `pip install sageattention --no-build-isolation` |
| `both` | Runs pytorch then sage sequentially | Prints speedup comparison at the end |

The script patches attention functions globally across all loaded modules (not just `comfy.ldm.modules.attention`) to ensure the selected backend is actually used everywhere, including modules that did `from ... import optimized_attention`.

### torch.compile

`--compile [MODE]` compiles all 34 `JointTransformerBlock` modules in the Lumina2 architecture using `torch.compile`. This fuses elementwise operations (norms, activations, scaling) that would otherwise launch many small CUDA kernels.

Available modes: `default` (recommended), `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`.

The `default` mode provides the best balance of compile time and runtime performance, achieving ~15% speedup over uncompiled execution by fusing elementwise kernels.

**ComfyUI modifications required for torch.compile compatibility:**

- `comfy/ldm/modules/attention.py`: The `wrap_attn` decorator includes an early return when `torch.compiler.is_compiling()` to prevent Dynamo from tracing dynamic `transformer_options` dict lookups, which cause recompilation.
- `comfy/ldm/lumina/model.py`: `torch.compiler.cudagraph_mark_step_begin()` calls are inserted before each `JointTransformerBlock` invocation in `NextDiT._forward` (for `context_refiner`, `noise_refiner`, and the main layer loop) to resolve CUDA graph tree conflicts in `reduce-overhead` mode.

### CUDA Graphs

`--cuda-graph` wraps the diffusion model's forward pass in a manually captured CUDA graph via the `CUDAGraphRunner` class. After a warmup + capture phase, subsequent calls replay the graph with zero CPU kernel-launch overhead. Requires fixed resolution (use a single `--resolutions` value).

### Compile Cache

Compilation of 34 transformer blocks takes significant time on the first run. Two caching strategies are available to speed up subsequent runs:

**Approach 1: Environment variables (zero code changes)**

```bash
TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
TORCHINDUCTOR_CACHE_DIR=.compile_cache \
python benchmark_z_image_fp8.py ... --compile
```

PyTorch automatically caches FX graphs, AOTAutograd results, Triton cubins, and autotuning results to disk. Second run uses the cache transparently.

**Approach 2: Portable mega-cache file (`--cache-file`)**

```bash
# First run: compiles and saves cache
python benchmark_z_image_fp8.py ... --compile --cache-file compile_cache.bin

# Second run: loads cache, ~2x faster startup
python benchmark_z_image_fp8.py ... --compile --cache-file compile_cache.bin
```

Uses `torch.compiler.save_cache_artifacts()` / `load_cache_artifacts()` to serialize all compilation artifacts into a single portable binary (~10 MB). Can be copied between machines with the same GPU, PyTorch version, and Triton version.

Debug cache hits/misses with `TORCH_LOGS=+torch._inductor.codecache`.

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--unet_path` | *(required)* | Path to the diffusion model `.safetensors` file |
| `--weight_dtype` | `default` | Weight dtype: `default`, `fp8_e4m3fn`, `fp8_e4m3fn_fast`, `fp8_e5m2` |
| `--resolutions` | `1024x1024` | Space-separated resolutions, e.g. `512x512 1024x1024 2048x1024` |
| `--steps` | `4` | Denoising step counts (turbo models typically use 1-8) |
| `--batch_sizes` | `1` | Batch sizes to benchmark |
| `--sampler` | `euler` | Sampler name (`euler`, `dpmpp_2m`, etc.) |
| `--scheduler` | `simple` | Scheduler name (`simple`, `karras`, `normal`, etc.) |
| `--cfg` | `1.0` | CFG scale (turbo models often use 1.0) |
| `--warmup` | `1` | Number of warmup iterations (not timed) |
| `--repeats` | `3` | Number of timed iterations per config |
| `--attention` | `default` | Attention backend: `default`, `pytorch`, `sage`, `both` |
| `--compile` | *(off)* | torch.compile mode: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs` |
| `--cuda-graph` | `false` | Capture forward pass in a CUDA graph |
| `--raw` | `false` | Raw denoising loop bypassing ComfyUI overhead |
| `--profile` | `false` | Run torch.profiler for kernel-level breakdown |
| `--cache-file` | *(none)* | Path to save/load torch.compile mega-cache binary |
| `--output_json` | *(none)* | Path to write results as JSON |

## Benchmark Results (RTX 3090, 2048x1024, 4 steps)

Representative results on an RTX 3090 with Z-Image-Turbo-fp16:

| Configuration | ms/step | Speedup | Notes |
|--------------|---------|---------|-------|
| Normal mode, PyTorch SDPA | ~2200 | baseline | Full ComfyUI pipeline overhead |
| Raw mode, PyTorch SDPA | ~2040 | 1.08x | Eliminates ComfyUI per-step overhead |
| Raw mode, SageAttention | ~2040 | 1.08x | Attention is not the bottleneck at this resolution |
| Raw + SageAttention + torch.compile | ~1718 | 1.28x | Fuses elementwise ops across transformer blocks |
| Raw + SageAttention + compile + CUDA graph | ~1718 | 1.28x | CPU dispatch is minimal; CUDA graph adds little here |

### Profiler Breakdown (compiled, raw mode)

| Category | Time | Share |
|----------|------|-------|
| Linear/FFN (FP16 matmul) | 5.23s | 76% |
| Attention (SageAttention) | 0.98s | 14% |
| Elementwise (fused by compile) | 0.65s | 9% |
| Other | ~0.06s | 1% |

The model is compute-bound on FP16 matmuls, running near the RTX 3090's 35.6 TFLOPS FP16 limit. Further speedups would require weight quantization (INT8 at 142 TOPS) or hardware upgrade.

## Architecture Notes

The Z-Image Turbo model uses a Lumina2 (`NextDiT`) architecture with:

- 34 `JointTransformerBlock` layers (main denoising path)
- Context refiner and noise refiner sub-networks
- FP16 weights with CUTLASS `tensorop_f16_s16816` kernels for matmuls

The benchmark synthesizes dummy conditioning (random tensors matching the expected shapes) so no text encoder is needed. This means the output images are meaningless — only timing and memory measurements are valid.
