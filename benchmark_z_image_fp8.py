#!/usr/bin/env python3
"""
Standalone benchmark for z_image (turbo) diffusion model with FP8 scaled e4m3fn weights.

Measures: latency per step, end-to-end generation time, peak VRAM, throughput (images/sec).
Supports A/B comparison between attention backends (pytorch SDPA vs SageAttention).

Usage:
    python benchmark_z_image_fp8.py --unet_path <path_to_safetensors>
    python benchmark_z_image_fp8.py --unet_path models/diffusion_models/z_image_turbo_fp8.safetensors
    python benchmark_z_image_fp8.py --unet_path /path.safetensors --attention sage --resolutions 1024x1024 --steps 4
    python benchmark_z_image_fp8.py --unet_path /path.safetensors --attention both --repeats 5

The script does NOT require a text encoder or VAE — it synthesises dummy conditioning
tensors so the benchmark isolates diffusion-model throughput only.
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.ldm.modules.attention as attn_module
import comfy.ldm.lumina.model as lumina_module


# ---------------------------------------------------------------------------
# Attention backend switching
# ---------------------------------------------------------------------------

_ORIGINAL_ATTN = attn_module.optimized_attention
_ORIGINAL_ATTN_MASKED = attn_module.optimized_attention_masked

_SAGE_AVAILABLE = attn_module.SAGE_ATTENTION_IS_AVAILABLE

# Modules that do `from ... import optimized_attention[_masked]` — these hold
# stale local references that won't update when we reassign the module-level
# variable.  We must patch them explicitly.
_MODULES_USING_ATTN = []
_MODULES_USING_ATTN_MASKED = []

def _discover_patch_targets():
    """Find all loaded modules that imported optimized_attention directly."""
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is attn_module:
            continue
        if hasattr(mod, "optimized_attention_masked") and not hasattr(mod, "optimized_attention"):
            if callable(getattr(mod, "optimized_attention_masked", None)):
                _MODULES_USING_ATTN_MASKED.append(mod)
        if hasattr(mod, "optimized_attention") and callable(getattr(mod, "optimized_attention", None)):
            _MODULES_USING_ATTN.append(mod)
        if hasattr(mod, "optimized_attention_masked") and callable(getattr(mod, "optimized_attention_masked", None)):
            if mod not in _MODULES_USING_ATTN_MASKED:
                _MODULES_USING_ATTN_MASKED.append(mod)

_discover_patch_targets()


def set_attention_backend(name: str):
    """Swap the attention function everywhere — module globals AND direct imports."""
    if name == "sage":
        if not _SAGE_AVAILABLE:
            raise RuntimeError(
                "SageAttention requested but sageattention package is not installed.\n"
                "Install with: pip install sageattention --no-build-isolation"
            )
        fn = attn_module.attention_sage
        fn_masked = attn_module.attention_sage
        label = "SageAttention"
    elif name == "pytorch":
        fn = attn_module.attention_pytorch
        fn_masked = attn_module.attention_pytorch
        label = "PyTorch SDPA"
    elif name == "default":
        fn = _ORIGINAL_ATTN
        fn_masked = _ORIGINAL_ATTN_MASKED
        label = f"default ({_ORIGINAL_ATTN.__name__})"
    else:
        raise ValueError(f"Unknown attention backend: {name}")

    attn_module.optimized_attention = fn
    attn_module.optimized_attention_masked = fn_masked

    for mod in _MODULES_USING_ATTN:
        mod.optimized_attention = fn
    for mod in _MODULES_USING_ATTN_MASKED:
        mod.optimized_attention_masked = fn_masked

    print(f"  Attention backend: {label} (patched {len(_MODULES_USING_ATTN)}+{len(_MODULES_USING_ATTN_MASKED)} modules)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    resolution: str
    steps: int
    batch_size: int
    sampler: str
    scheduler: str
    weight_dtype: str
    attention_backend: str
    warmup_runs: int
    total_time_s: float = 0.0
    median_time_s: float = 0.0
    min_time_s: float = 0.0
    max_time_s: float = 0.0
    median_step_time_ms: float = 0.0
    throughput_img_per_s: float = 0.0
    peak_vram_gb: float = 0.0
    device_name: str = ""
    run_times: list = field(default_factory=list)


def parse_resolution(res_str: str) -> tuple[int, int]:
    parts = res_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Resolution must be WxH, got: {res_str}")
    return int(parts[0]), int(parts[1])


@contextmanager
def cuda_memory_tracker():
    """Track peak GPU memory during a block."""
    if not torch.cuda.is_available():
        yield {"peak_gb": 0.0}
        return
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    info = {}
    try:
        yield info
    finally:
        torch.cuda.synchronize()
        info["peak_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)


def make_dummy_conditioning(device, dtype, batch_size=1, seq_len=64, dim=2560):
    """
    Build minimal conditioning tensors that satisfy the Lumina2 / ZImage model's
    extra_conds expectations (cross_attn + num_tokens).
    """
    cond_tensor = torch.randn(batch_size, seq_len, dim, device="cpu", dtype=dtype)
    positive = [[cond_tensor, {"pooled_output": None}]]
    negative = [[torch.zeros_like(cond_tensor), {"pooled_output": None}]]
    return positive, negative


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def _dequant_fp8_to_bf16(model_patcher):
    """Convert FP8 quantized weights to bf16 in-place so torch.compile works.

    FP8 checkpoints use custom cast ops (comfy_kitchen / fp8_linear) that break
    torch.compile's graph tracing.  Dequantizing trades memory for compile
    compatibility — weights go from ~0.5x to 1x size but all linear ops become
    plain bf16 matmuls that the compiler can fuse and optimize.
    """
    diffusion_model = model_patcher.model.diffusion_model
    converted = 0
    for name, module in diffusion_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        w = module.weight
        if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            scale = getattr(module, "scale_weight", None)
            new_w = w.to(torch.bfloat16)
            if scale is not None and scale is not None:
                new_w = new_w * scale.to(torch.bfloat16)
            module.weight = torch.nn.Parameter(new_w, requires_grad=False)
            module.scale_weight = None
            module.scale_input = None
            if hasattr(module, "comfy_cast_weights"):
                module.comfy_cast_weights = False
            if hasattr(module, "weight_function"):
                module.weight_function = []
            converted += 1
    if converted:
        print(f"  Dequantized {converted} FP8 layers to bf16 for torch.compile")
    return converted


class CUDAGraphRunner:
    """Captures a CUDA graph of the diffusion model's forward pass.

    First call: warmup + graph capture (records all CUDA ops).
    Subsequent calls: copy inputs into static buffers, replay graph.
    Eliminates all CPU kernel-launch overhead on replay.
    """

    def __init__(self, original_forward):
        self.original_forward = original_forward
        self.graph = None
        self.static_args = None
        self.static_kwargs = None
        self.static_output = None
        self._tensor_arg_idx = []
        self._tensor_kwarg_keys = []

    def __call__(self, *args, **kwargs):
        if self.graph is None:
            return self._warmup_and_capture(*args, **kwargs)
        return self._replay(*args, **kwargs)

    def _warmup_and_capture(self, *args, **kwargs):
        print("  [CUDAGraph] warmup ...", end="", flush=True)
        torch.cuda.synchronize()
        self.original_forward(*args, **kwargs)
        torch.cuda.synchronize()
        print(" done.  capturing ...", end="", flush=True)

        static_args = list(args)
        for i, a in enumerate(static_args):
            if isinstance(a, torch.Tensor):
                static_args[i] = a.clone()
                self._tensor_arg_idx.append(i)
        self.static_args = static_args

        static_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                static_kwargs[k] = v.clone()
                self._tensor_kwarg_keys.append(k)
            else:
                static_kwargs[k] = v
        self.static_kwargs = static_kwargs

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.original_forward(
                *self.static_args, **self.static_kwargs
            )
        torch.cuda.synchronize()
        print(" done")
        return self.static_output

    def _replay(self, *args, **kwargs):
        for i in self._tensor_arg_idx:
            self.static_args[i].copy_(args[i])
        for k in self._tensor_kwarg_keys:
            if k in kwargs:
                self.static_kwargs[k].copy_(kwargs[k])
        self.graph.replay()
        return self.static_output


def _load_compile_cache(cache_path: str) -> bool:
    """Load torch.compile mega-cache from a binary file. Returns True on success."""
    p = Path(cache_path)
    if not p.exists():
        return False
    t0 = time.perf_counter()
    artifact_bytes = p.read_bytes()
    torch.compiler.load_cache_artifacts(artifact_bytes)
    elapsed = time.perf_counter() - t0
    print(f"  Loaded compile cache from {p} ({len(artifact_bytes)/1024/1024:.1f} MB) in {elapsed:.1f}s")
    return True


def _save_compile_cache(cache_path: str):
    """Save torch.compile mega-cache to a binary file."""
    p = Path(cache_path)
    if p.exists():
        return
    t0 = time.perf_counter()
    result = torch.compiler.save_cache_artifacts()
    if result is None:
        print("  Warning: save_cache_artifacts() returned None — no cache to save")
        return
    artifact_bytes, cache_info = result
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(artifact_bytes)
    elapsed = time.perf_counter() - t0
    print(f"  Saved compile cache to {p} ({len(artifact_bytes)/1024/1024:.1f} MB) in {elapsed:.1f}s")


def load_model(unet_path: str, weight_dtype_name: str, compile_mode: str = None,
               cuda_graph: bool = False, cache_file: str = None):
    model_options = {}
    if weight_dtype_name == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype_name == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype_name == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    print(f"Loading model from: {unet_path}")
    print(f"  weight_dtype option: {weight_dtype_name}")
    t0 = time.perf_counter()
    model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.2f}s")

    if compile_mode:
        if cache_file:
            _load_compile_cache(cache_file)

        from comfy.ldm.lumina.model import JointTransformerBlock
        import torch._dynamo.config
        diffusion_model = model.model.diffusion_model
        n_blocks = sum(1 for _, m in diffusion_model.named_modules()
                       if isinstance(m, JointTransformerBlock))
        torch._dynamo.config.cache_size_limit = max(n_blocks + 8,
                                                     torch._dynamo.config.cache_size_limit)
        print(f"  torch.compile mode: {compile_mode} (full JointTransformerBlock)")
        print(f"  dynamo cache_size_limit raised to {torch._dynamo.config.cache_size_limit}")
        compiled_count = 0
        for name, module in diffusion_model.named_modules():
            if isinstance(module, JointTransformerBlock):
                module.forward = torch.compile(module.forward, mode=compile_mode, dynamic=False)
                compiled_count += 1
        print(f"  Compiled {compiled_count} JointTransformerBlock modules")

    if cuda_graph:
        print("  CUDA Graph: will capture on first forward pass")
        dm = model.model.diffusion_model
        runner = CUDAGraphRunner(dm.forward)
        dm.forward = runner

    return model


def run_single_sample(
    model,
    width: int,
    height: int,
    steps: int,
    batch_size: int,
    sampler_name: str,
    scheduler: str,
    seed: int,
    cfg: float,
):
    """Run one sampling pass and return elapsed seconds."""
    device = model.load_device
    latent_format = model.get_model_object("latent_format")
    latent_channels = latent_format.latent_channels
    downscale = latent_format.spacial_downscale_ratio

    latent_h = height // downscale
    latent_w = width // downscale

    latent_image = torch.zeros(
        [batch_size, latent_channels, latent_h, latent_w],
        device=comfy.model_management.intermediate_device(),
    )
    noise = comfy.sample.prepare_noise(latent_image, seed)

    model_dtype = model.model_dtype()
    cond_dim = 2560
    positive, negative = make_dummy_conditioning(
        device, model_dtype, batch_size=batch_size, seq_len=64, dim=cond_dim
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
        disable_noise=False,
        force_full_denoise=True,
        seed=seed,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed


# ---------------------------------------------------------------------------
# Raw denoising loop — bypasses comfy.sample.sample entirely
# ---------------------------------------------------------------------------

def _ensure_model_on_gpu(model_patcher):
    """Load model to GPU and patch it once. Returns the inner BaseModel."""
    comfy.model_management.load_models_gpu([model_patcher])
    model_patcher.patch_model(load_weights=False)
    return model_patcher.model


def run_raw_denoising(
    model_patcher,
    real_model,
    width: int,
    height: int,
    steps: int,
    batch_size: int,
    scheduler: str,
    seed: int,
):
    """Euler denoising loop calling apply_model directly. Returns elapsed seconds."""
    device = model_patcher.load_device
    dtype = model_patcher.model_dtype()
    latent_format = real_model.latent_format
    latent_channels = latent_format.latent_channels
    downscale = latent_format.spacial_downscale_ratio
    model_sampling = real_model.model_sampling

    latent_h = height // downscale
    latent_w = width // downscale

    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(batch_size, latent_channels, latent_h, latent_w,
                        generator=g, device="cpu", dtype=torch.float32).to(device)
    latent_image = torch.zeros_like(noise)

    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).to(device)
    x = model_sampling.noise_scaling(sigmas[0], noise, latent_image, max_denoise=True)

    seq_len = 64
    cond_dim = 2560
    context = torch.randn(batch_size, seq_len, cond_dim, device=device, dtype=dtype)
    transformer_options = {"cond_or_uncond": [0]}

    extra_conds = {"num_tokens": seq_len}

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    s_in = x.new_ones([x.shape[0]])
    with torch.no_grad():
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            timestep = sigma * s_in
            denoised = real_model.apply_model(
                x, timestep,
                c_crossattn=context,
                transformer_options=transformer_options,
                **extra_conds,
            )
            d = (x - denoised) / sigma
            dt = sigmas[i + 1] - sigma
            x = x + d * dt

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def benchmark_config(
    model,
    width: int,
    height: int,
    steps: int,
    batch_size: int,
    sampler_name: str,
    scheduler: str,
    cfg: float,
    warmup: int,
    repeats: int,
    weight_dtype_name: str,
    attention_backend: str = "default",
    is_compiled: bool = False,
):
    res_tag = f"{width}x{height}"
    print(f"\n{'='*60}")
    print(f"  Resolution: {res_tag}  |  Steps: {steps}  |  Batch: {batch_size}")
    print(f"  Sampler: {sampler_name}  |  Scheduler: {scheduler}  |  CFG: {cfg}")
    set_attention_backend(attention_backend)
    print(f"{'='*60}")

    # Warmup
    for i in range(warmup):
        print(f"  Warmup {i+1}/{warmup} ...", end="", flush=True)
        run_single_sample(model, width, height, steps, batch_size, sampler_name, scheduler, seed=42+i, cfg=cfg)
        print(" done")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Timed runs
    times = []
    with cuda_memory_tracker() as mem:
        for i in range(repeats):
            print(f"  Run {i+1}/{repeats} ...", end="", flush=True)
            t = run_single_sample(model, width, height, steps, batch_size, sampler_name, scheduler, seed=100+i, cfg=cfg)
            times.append(t)
            print(f" {t:.3f}s")

    # When compiled, drop the first timed run — it may still include
    # residual compilation / CUDA graph capture overhead.
    if is_compiled and len(times) > 2:
        dropped = times[0]
        times = times[1:]
        print(f"  (dropped first run {dropped:.3f}s — compile overhead)")

    result = BenchResult(
        resolution=res_tag,
        steps=steps,
        batch_size=batch_size,
        sampler=sampler_name,
        scheduler=scheduler,
        weight_dtype=weight_dtype_name,
        attention_backend=attention_backend,
        warmup_runs=warmup,
        total_time_s=sum(times),
        median_time_s=statistics.median(times),
        min_time_s=min(times),
        max_time_s=max(times),
        median_step_time_ms=(statistics.median(times) / steps) * 1000,
        throughput_img_per_s=batch_size / statistics.median(times),
        peak_vram_gb=mem.get("peak_gb", 0.0),
        device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        run_times=times,
    )
    return result


def print_summary(results: list[BenchResult]):
    print(f"\n{'='*92}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*92}")

    header = (
        f"{'Attention':<10} {'Resolution':<12} {'Steps':>5} {'Batch':>5} "
        f"{'Med(s)':>8} {'Min(s)':>8} {'Max(s)':>8} {'ms/step':>8} {'img/s':>7} {'VRAM(GB)':>9}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.attention_backend:<10} {r.resolution:<12} {r.steps:>5} {r.batch_size:>5} "
            f"{r.median_time_s:>8.3f} {r.min_time_s:>8.3f} {r.max_time_s:>8.3f} "
            f"{r.median_step_time_ms:>8.1f} {r.throughput_img_per_s:>7.2f} "
            f"{r.peak_vram_gb:>9.2f}"
        )

    if results:
        print(f"\nDevice: {results[0].device_name}")
        print(f"Weight dtype: {results[0].weight_dtype}")

    # Print speedup comparison when both backends are present
    backends = {r.attention_backend for r in results}
    if "pytorch" in backends and "sage" in backends:
        print(f"\n  --- Speedup (SageAttention vs PyTorch SDPA) ---")
        pytorch_map = {(r.resolution, r.steps, r.batch_size): r for r in results if r.attention_backend == "pytorch"}
        for r in results:
            if r.attention_backend == "sage":
                key = (r.resolution, r.steps, r.batch_size)
                baseline = pytorch_map.get(key)
                if baseline:
                    speedup = baseline.median_time_s / r.median_time_s
                    saved_ms = (baseline.median_time_s - r.median_time_s) * 1000
                    print(f"  {r.resolution} steps={r.steps} batch={r.batch_size}: "
                          f"{speedup:.2f}x faster ({saved_ms:+.0f}ms)")


# ---------------------------------------------------------------------------
# Profiling — shows where time actually goes (attention vs FFN vs other)
# ---------------------------------------------------------------------------

def profile_one_run(model, width, height, steps, batch_size, sampler_name, scheduler, cfg, attention_backend, is_compiled=False):
    """Run one sampling pass under torch.profiler and print a kernel-level breakdown."""
    set_attention_backend(attention_backend)

    warmup_count = 4 if is_compiled else 2
    for i in range(warmup_count):
        print(f"  Profiler warmup {i+1}/{warmup_count} ...", end="", flush=True)
        run_single_sample(model, width, height, steps, batch_size, sampler_name, scheduler, seed=i, cfg=cfg)
        print(" done")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("  Profiling (steady-state) ...", end="", flush=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        run_single_sample(model, width, height, steps, batch_size, sampler_name, scheduler, seed=1, cfg=cfg)
    print(" done\n")

    # Build a breakdown by kernel category
    events = prof.key_averages()

    attn_keywords = ["attention", "sageattn", "sdpa", "flash", "scaled_dot", "bmm", "_attn", "fmha"]
    norm_keywords = ["layer_norm", "layernorm", "rms_norm", "rmsnorm", "group_norm"]
    ffn_keywords = ["linear", "addmm", "gemm", "matmul", "cublas", "mm_", "fp8"]

    total_cuda_us = 0
    attn_us = 0
    ffn_us = 0
    norm_us = 0
    other_us = 0

    for evt in events:
        cuda_time = evt.device_time_total
        total_cuda_us += cuda_time
        key_lower = evt.key.lower()

        if any(k in key_lower for k in attn_keywords):
            attn_us += cuda_time
        elif any(k in key_lower for k in ffn_keywords):
            ffn_us += cuda_time
        elif any(k in key_lower for k in norm_keywords):
            norm_us += cuda_time
        else:
            other_us += cuda_time

    total_s = total_cuda_us / 1e6
    print(f"  CUDA Time Breakdown ({attention_backend}, {width}x{height}, {steps} steps):")
    print(f"  {'Total:':<16} {total_s:>8.3f}s")
    if total_cuda_us > 0:
        for label, us in [("Attention", attn_us), ("Linear/FFN", ffn_us), ("Norms", norm_us), ("Other", other_us)]:
            pct = us / total_cuda_us * 100
            print(f"  {label + ':':<16} {us/1e6:>8.3f}s  ({pct:>5.1f}%)")

    print(f"\n  Top 15 CUDA kernels:")
    print(events.table(sort_by="self_device_time_total", row_limit=15))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark z_image (turbo) FP8 scaled e4m3fn diffusion model"
    )
    parser.add_argument(
        "--unet_path", type=str, required=True,
        help="Path to the diffusion model safetensors file"
    )
    parser.add_argument(
        "--weight_dtype", type=str, default="default",
        choices=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
        help="Weight dtype to load the model with (default: auto-detect from checkpoint)"
    )
    parser.add_argument(
        "--resolutions", nargs="+", default=["1024x1024"],
        help="Resolutions to benchmark, e.g. 512x512 1024x1024 1536x1536"
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[4],
        help="Step counts to benchmark (turbo models typically use 1-8)"
    )
    parser.add_argument(
        "--batch_sizes", nargs="+", type=int, default=[1],
        help="Batch sizes to benchmark"
    )
    parser.add_argument(
        "--sampler", type=str, default="euler",
        help="Sampler name (euler, dpmpp_2m, etc.)"
    )
    parser.add_argument(
        "--scheduler", type=str, default="simple",
        help="Scheduler name (simple, karras, normal, etc.)"
    )
    parser.add_argument(
        "--cfg", type=float, default=1.0,
        help="CFG scale (turbo models often use 1.0)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup iterations (not timed)"
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Number of timed iterations per config"
    )
    parser.add_argument(
        "--attention", type=str, default="default",
        choices=["default", "pytorch", "sage", "both"],
        help="Attention backend: pytorch (SDPA), sage (SageAttention), "
             "both (runs pytorch then sage for A/B comparison), or default (whatever ComfyUI picked)"
    )
    parser.add_argument(
        "--compile", type=str, default=None, nargs="?", const="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile the transformer blocks (default mode: 'default' — fuses elementwise ops)"
    )
    parser.add_argument(
        "--cuda-graph", action="store_true", default=False,
        help="Capture the diffusion model forward in a CUDA graph for zero CPU-launch overhead. "
             "Requires fixed resolution (one --resolutions value recommended)."
    )
    parser.add_argument(
        "--raw", action="store_true", default=False,
        help="Raw denoising loop: call apply_model directly with a minimal euler sampler, "
             "bypassing comfy.sample.sample overhead (patch/unpatch, CFGGuider, etc.)"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Run torch.profiler to show CUDA kernel time breakdown (attention vs FFN vs other)"
    )
    parser.add_argument(
        "--cache-file", type=str, default=None,
        help="Path to a torch.compile mega-cache file (.bin). "
             "If the file exists, compiled artifacts are loaded from it before compilation "
             "(speeding up warm start). After the first run, artifacts are saved to this file "
             "if it doesn't already exist. Requires --compile."
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Optional path to write results as JSON"
    )

    args = parser.parse_args()

    if args.attention in ("sage", "both") and not _SAGE_AVAILABLE:
        print("ERROR: SageAttention requested but sageattention is not installed.", file=sys.stderr)
        print(f"  Install: {sys.executable} -m pip install sageattention --no-build-isolation", file=sys.stderr)
        sys.exit(1)

    cache_file = getattr(args, "cache_file", None)
    if cache_file and args.compile is None:
        print("ERROR: --cache-file requires --compile", file=sys.stderr)
        sys.exit(1)

    unet_path = args.unet_path
    if not os.path.isabs(unet_path):
        unet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), unet_path)

    if not os.path.isfile(unet_path):
        print(f"ERROR: Model file not found: {unet_path}", file=sys.stderr)
        sys.exit(1)

    model = load_model(unet_path, args.weight_dtype, compile_mode=args.compile,
                       cuda_graph=args.cuda_graph, cache_file=cache_file)

    if args.profile:
        w, h = parse_resolution(args.resolutions[0])
        backends = ["pytorch", "sage"] if args.attention == "both" else [args.attention]
        for backend in backends:
            has_capture_overhead = (args.compile is not None) or args.cuda_graph
            profile_one_run(model, w, h, args.steps[0], args.batch_sizes[0],
                            args.sampler, args.scheduler, args.cfg, backend,
                            is_compiled=has_capture_overhead)
        if cache_file:
            _save_compile_cache(cache_file)
        return

    # ---- Raw mode: direct apply_model loop, no comfy.sample overhead ----
    if args.raw:
        set_attention_backend(args.attention if args.attention != "both" else "sage")
        print("\n[RAW MODE] Loading model to GPU and patching once ...")
        real_model = _ensure_model_on_gpu(model)

        for res_str in args.resolutions:
            w, h = parse_resolution(res_str)
            for steps in args.steps:
                for bs in args.batch_sizes:
                    tag = f"{w}x{h} steps={steps} batch={bs}"
                    print(f"\n{'='*60}")
                    print(f"  {tag}  |  Scheduler: {args.scheduler}")
                    print(f"{'='*60}")

                    for i in range(args.warmup):
                        print(f"  Warmup {i+1}/{args.warmup} ...", end="", flush=True)
                        run_raw_denoising(model, real_model, w, h, steps, bs,
                                          args.scheduler, seed=42+i)
                        print(" done")

                    gc.collect()
                    torch.cuda.empty_cache()

                    times = []
                    with cuda_memory_tracker() as mem:
                        for i in range(args.repeats):
                            t = run_raw_denoising(model, real_model, w, h, steps, bs,
                                                  args.scheduler, seed=100+i)
                            times.append(t)
                            print(f"  Run {i+1}/{args.repeats}: {t:.3f}s  "
                                  f"({t/steps*1000:.0f} ms/step)")

                    has_overhead = (args.compile is not None) or args.cuda_graph
                    if has_overhead and len(times) > 2:
                        dropped = times[0]
                        times = times[1:]
                        print(f"  (dropped first run {dropped:.3f}s)")

                    med = statistics.median(times)
                    print(f"\n  Median: {med:.3f}s  ({med/steps*1000:.0f} ms/step)  "
                          f"Min: {min(times):.3f}s  Max: {max(times):.3f}s  "
                          f"VRAM: {mem.get('peak_gb', 0):.2f} GB")

        if cache_file:
            _save_compile_cache(cache_file)
        return

    # ---- Normal mode: through comfy.sample.sample ----
    has_capture_overhead = (args.compile is not None) or args.cuda_graph

    if args.attention == "both":
        attention_backends = ["pytorch", "sage"]
    else:
        attention_backends = [args.attention]

    results = []
    for backend in attention_backends:
        for res_str in args.resolutions:
            w, h = parse_resolution(res_str)
            for steps in args.steps:
                for bs in args.batch_sizes:
                    r = benchmark_config(
                        model,
                        width=w,
                        height=h,
                        steps=steps,
                        batch_size=bs,
                        sampler_name=args.sampler,
                        scheduler=args.scheduler,
                        cfg=args.cfg,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        weight_dtype_name=args.weight_dtype,
                        attention_backend=backend,
                        is_compiled=has_capture_overhead,
                    )
                    results.append(r)

    if cache_file:
        _save_compile_cache(cache_file)

    print_summary(results)

    if args.output_json:
        out_data = [asdict(r) for r in results]
        with open(args.output_json, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nResults written to: {args.output_json}")


if __name__ == "__main__":
    main()
