# cupid/bench.py
from __future__ import annotations
import triton.testing as tt  # official Triton helper :contentReference[oaicite:3]{index=3}


def bench_us(fn, fidelity: str) -> float:
    """
    Returns microseconds for a callable `fn`.
    Fidelity:
      F1 = cheap/noisy
      F2 = expensive/stable
    """
    if fidelity == "F1":
        # quick and noisy: small warmup/rep
        us = tt.do_bench(fn, warmup=5, rep=20, quantiles=(0.5,))  # median only :contentReference[oaicite:4]{index=4}
        if isinstance(us, (tuple, list)):
            us = us[0]
        return float(us)
    elif fidelity == "F2":
        # stable: more warmup/rep, with quantiles
        us = tt.do_bench(fn, warmup=25, rep=100, quantiles=(0.5, 0.2, 0.8))  # :contentReference[oaicite:5]{index=5}
        # If quantiles is set, Triton returns tuple-like
        if isinstance(us, (tuple, list)):
            med = float(us[0])
            return med
        return float(us)
    else:
        raise ValueError(f"Unknown fidelity: {fidelity}")
