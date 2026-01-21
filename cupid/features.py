# cupid/features.py
from __future__ import annotations
import math
import torch
from typing import Dict, Any, Tuple


def bucket_pow2(x: int) -> int:
    """Power-of-two bucketing for stable caching/transfer keys."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def shape_bucket_2d(m: int, n: int) -> Tuple[int, int]:
    return bucket_pow2(m), bucket_pow2(n)


def device_fingerprint() -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "cc": f"{props.major}.{props.minor}",
        "sm": props.multi_processor_count,
        "mem_gb": round(props.total_memory / (1024**3), 1),
    }


def featurize(
    m: int,
    n: int,
    dtype: torch.dtype,
    config: Dict[str, Any],
) -> torch.Tensor:
    """
    Light-weight numeric features for a Bayesian linear model.
    You can expand this later (strides, layout, more meta-params).
    """
    # config keys:
    #   meta: dict -> e.g. {"BLOCK": 1024}
    #   num_warps, num_stages
    meta = config.get("meta", {})
    num_warps = float(config.get("num_warps", 4))
    num_stages = float(config.get("num_stages", 3))

    # Pull a few common params if present
    block = float(meta.get("BLOCK", 0))
    block_m = float(meta.get("BLOCK_M", 0))
    block_n = float(meta.get("BLOCK_N", 0))

    dtype_code = {
        torch.float16: 0.0,
        torch.bfloat16: 1.0,
        torch.float32: 2.0,
    }.get(dtype, 3.0)

    # Use log scale for shapes
    fm = math.log2(max(1, m))
    fn = math.log2(max(1, n))

    feat = torch.tensor(
        [fm, fn, dtype_code, num_warps, num_stages, block, block_m, block_n],
        dtype=torch.float32,
    )
    return feat
