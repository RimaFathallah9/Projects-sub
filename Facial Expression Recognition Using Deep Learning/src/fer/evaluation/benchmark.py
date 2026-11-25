"""Computational-efficiency measurements: params, MACs, model size, latency, throughput."""
from __future__ import annotations

import copy
import io
import time

import numpy as np
import torch

from fer.models.factory import count_parameters


def model_size_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def count_macs(model: torch.nn.Module, input_shape: tuple[int, int, int]) -> float | None:
    """Multiply-accumulate count via fvcore (note: fvcore reports MACs, often loosely called FLOPs)."""
    try:
        from fvcore.nn import FlopCountAnalysis
        m = copy.deepcopy(model).cpu().eval()
        fca = FlopCountAnalysis(m, torch.randn(1, *input_shape))
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return float(fca.total())
    except Exception as e:  # fvcore missing or tracing failure
        print(f"[benchmark] MAC counting skipped: {e}")
        return None


@torch.no_grad()
def measure_latency(model, input_shape, device: torch.device, batch_size: int = 1,
                    n_warmup: int = 20, n_iter: int = 100) -> dict:
    model = copy.deepcopy(model).to(device).eval()
    x = torch.randn(batch_size, *input_shape, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(n_warmup):
        model(x)
    sync()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        model(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
            "throughput_img_per_s": float(batch_size / (np.median(t) / 1000))}


def benchmark_model(model, input_shape, gpu_batch: int = 256) -> dict:
    res = {"params": count_parameters(model), "model_size_mb": model_size_mb(model),
           "macs": count_macs(model, input_shape), "input_shape": list(input_shape)}
    res["latency_cpu_bs1"] = measure_latency(model, input_shape, torch.device("cpu"), 1, 10, 50)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        res["latency_gpu_bs1"] = measure_latency(model, input_shape, dev, 1)
        res[f"throughput_gpu_bs{gpu_batch}"] = measure_latency(model, input_shape, dev, gpu_batch, 10, 30)
        res["gpu"] = torch.cuda.get_device_name(0)
    res["torch_threads_cpu"] = torch.get_num_threads()
    return res
