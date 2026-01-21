# kernels/softmax_row_autotune.py
import torch
import triton
import triton.language as tl


# Candidate configurations to benchmark.
# Triton will try them when key=['n_cols'] changes. :contentReference[oaicite:1]{index=1}
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 512},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=4,  num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK": 2048}, num_warps=4,  num_stages=3),
    triton.Config({"BLOCK": 2048}, num_warps=8,  num_stages=4),
    triton.Config({"BLOCK": 4096}, num_warps=8,  num_stages=4),
    triton.Config({"BLOCK": 4096}, num_warps=16, num_stages=5),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["n_cols"],  # retune when N changes :contentReference[oaicite:2]{index=2}
)
@triton.jit
def softmax_row_kernel_autotuned(
    X_ptr, Y_ptr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    n_cols,                     # runtime key
    BLOCK: tl.constexpr,        # chosen by autotune config
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x_ptrs = X_ptr + row * stride_xm + offs * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    y_ptrs = Y_ptr + row * stride_ym + offs * stride_yn
    tl.store(y_ptrs, y.to(tl.float16), mask=mask)


def softmax_triton_autotune(x: torch.Tensor) -> torch.Tensor:
    """
    Triton built-in autotune baseline.
    First call triggers tuning for a given n_cols. :contentReference[oaicite:3]{index=3}
    """
    assert x.is_cuda and x.ndim == 2
    M, N = x.shape
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)

    grid = (M,)
    softmax_row_kernel_autotuned[grid](
        x, y,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        N,  # n_cols key
    )
    return y
