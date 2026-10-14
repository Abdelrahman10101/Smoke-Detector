"""Benchmark latency and throughput (FPS) across GPU and CPU.

Measures:
- GPU: BS=1 FP16 Latency, BS=1 FP32 Latency, BS=8 Throughput
- CPU: BS=1 FP32 Latency, BS=1 Throughput (FPS)
- Model size (MB) and Parameter count (M)

Usage:
    python benchmark_latency.py [ckpt1 ckpt2 ...]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from model import load_checkpoint_model


def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def benchmark_device(model: torch.nn.Module, device: torch.device, img_size: int = 224,
                     batch_size: int = 1, num_warmup: int = 20, num_runs: int = 100,
                     use_fp16: bool = False):
    model.eval()
    model.to(device)

    dtype = torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=dtype)

    if use_fp16 and device.type == "cuda":
        model = model.half()
    else:
        model = model.float()

    # 1. Warm-up runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # 2. Timed runs
    latencies = []
    with torch.no_grad():
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(num_runs):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))  # milliseconds
        else:
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # milliseconds

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    fps = float(batch_size / (mean_lat / 1000.0))

    return {"mean_ms": mean_lat, "fps": fps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", default=[
        "/content/outputs/checkpoints/efficientnet_b0_best.pt",
        "/content/outputs/checkpoints/efficientnet_v2_s_best.pt",
        "/content/outputs/checkpoints/mobilenet_v3_large_best.pt",
    ])
    ap.add_argument("--gpu-runs", type=int, default=150, help="Iterations on GPU")
    ap.add_argument("--cpu-runs", type=int, default=50, help="Iterations on CPU")
    args = ap.parse_args()

    has_cuda = torch.cuda.is_available()
    cuda_device = torch.device("cuda") if has_cuda else None
    cpu_device = torch.device("cpu")

    print("\n" + "=" * 96)
    print(" Complete Model Benchmark: GPU vs CPU")
    if has_cuda:
        print(f" GPU: {torch.cuda.get_device_name(0)} ({args.gpu_runs} runs)")
    else:
        print(" GPU: Not available")
    print(f" CPU: Host Processor ({args.cpu_runs} runs)")
    print("=" * 96 + "\n")

    results = []

    for ckpt_str in args.checkpoints:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.exists():
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue

        file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        model, cfg, _ = load_checkpoint_model(str(ckpt_path), device="cpu")
        params_m = count_parameters(model)
        arch = cfg.arch

        print(f"-> Testing {arch} ({ckpt_path.name}) ...")

        # --- GPU BENCHMARKS ---
        gpu_fp16 = None
        gpu_fp32 = None
        gpu_bs8 = None
        if has_cuda:
            # GPU BS=1 FP16
            model, cfg, _ = load_checkpoint_model(str(ckpt_path), device="cpu")
            gpu_fp16 = benchmark_device(model, cuda_device, cfg.img_size, batch_size=1,
                                        num_warmup=25, num_runs=args.gpu_runs, use_fp16=True)
            # GPU BS=1 FP32
            model, cfg, _ = load_checkpoint_model(str(ckpt_path), device="cpu")
            gpu_fp32 = benchmark_device(model, cuda_device, cfg.img_size, batch_size=1,
                                        num_warmup=25, num_runs=args.gpu_runs, use_fp16=False)
            # GPU BS=8 (Throughput)
            model, cfg, _ = load_checkpoint_model(str(ckpt_path), device="cpu")
            gpu_bs8 = benchmark_device(model, cuda_device, cfg.img_size, batch_size=8,
                                       num_warmup=20, num_runs=args.gpu_runs, use_fp16=True)

        # --- CPU BENCHMARKS ---
        model, cfg, _ = load_checkpoint_model(str(ckpt_path), device="cpu")
        cpu_res = benchmark_device(model, cpu_device, cfg.img_size, batch_size=1,
                                   num_warmup=10, num_runs=args.cpu_runs, use_fp16=False)

        results.append({
            "arch": arch,
            "params_m": params_m,
            "size_mb": file_size_mb,
            "gpu_fp16_ms": gpu_fp16["mean_ms"] if gpu_fp16 else None,
            "gpu_fp32_ms": gpu_fp32["mean_ms"] if gpu_fp32 else None,
            "gpu_bs8_fps": gpu_bs8["fps"] if gpu_bs8 else None,
            "cpu_ms": cpu_res["mean_ms"],
            "cpu_fps": cpu_res["fps"],
        })

    if not results:
        print("No checkpoints were tested.")
        return

    # Print Formatted Comparison Table
    print("\n" + "=" * 105)
    print(f"{'Model Architecture':<22} | {'Params':<7} | {'Size':<6} | {'GPU FP16':<10} | {'GPU FP32':<10} | {'GPU BS=8':<9} | {'CPU (BS=1)':<11} | {'CPU FPS':<8}")
    print(f"{'':<22} | {'(M)':<7} | {'(MB)':<6} | {'Latency':<10} | {'Latency':<10} | {'FPS':<9} | {'Latency':<11} | {'':<8}")
    print("-" * 105)
    for r in results:
        g_fp16 = f"{r['gpu_fp16_ms']:.2f} ms" if r['gpu_fp16_ms'] is not None else "N/A"
        g_fp32 = f"{r['gpu_fp32_ms']:.2f} ms" if r['gpu_fp32_ms'] is not None else "N/A"
        g_bs8  = f"{r['gpu_bs8_fps']:.1f}" if r['gpu_bs8_fps'] is not None else "N/A"
        c_lat  = f"{r['cpu_ms']:.2f} ms"
        c_fps  = f"{r['cpu_fps']:.1f}"
        print(f"{r['arch']:<22} | {r['params_m']:<5.2f}M | {r['size_mb']:<5.1f}M | {g_fp16:<10} | {g_fp32:<10} | {g_bs8:<9} | {c_lat:<11} | {c_fps:<8}")
    print("=" * 105)
    print("Note: Latency is in milliseconds per frame (lower is better). FPS is frames per second (higher is better).\n")


if __name__ == "__main__":
    main()
