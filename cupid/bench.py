# cupid/bench.py
from __future__ import annotations
import torch
import triton.testing as tt  # for F2 only :contentReference[oaicite:3]{index=3}


def bench_ms_cuda_event(fn, warmup_iters=5, iters=10) -> float:
    """
    Very cheap timing using CUDA events.
    Returns milliseconds.
    """
    for _ in range(warmup_iters):
        fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    # elapsed_time is in milliseconds :contentReference[oaicite:4]{index=4}
    return start.elapsed_time(end) / iters


def bench_ms_do_bench(fn) -> float:
    """
    Stable timing (but slower). Returns milliseconds.
    do_bench warmup/rep are in ms. :contentReference[oaicite:5]{index=5}
    """
    res = tt.do_bench(fn, warmup=25, rep=100, quantiles=(0.5,))
    return float(res[0] if isinstance(res, (tuple, list)) else res)


def bench_us(fn, fidelity: str) -> float:
    """
    Return microseconds (us) for convenience in tuner code.
    """
    if fidelity == "F1":
        return bench_ms_cuda_event(fn, warmup_iters=3, iters=10) * 1000.0
    elif fidelity == "F2":
        return bench_ms_do_bench(fn) * 1000.0
    else:
        raise ValueError(f"Unknown fidelity: {fidelity}")
