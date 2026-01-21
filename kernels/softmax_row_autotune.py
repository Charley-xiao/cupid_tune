# kernels/softmax_row_autotune.py
import torch
import triton
import triton.language as tl

_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK": 512},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=4,  num_stages=2),
    triton.Config({"BLOCK": 1024}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK": 2048}, num_warps=4,  num_stages=3),
    triton.Config({"BLOCK": 2048}, num_warps=8,  num_stages=4),
    triton.Config({"BLOCK": 4096}, num_warps=8,  num_stages=4),
    triton.Config({"BLOCK": 4096}, num_warps=16, num_stages=5),
]

def _early_prune(configs, named_args, **kwargs):
    # named_args contains runtime args like n_cols
    n_cols = int(named_args["n_cols"])
    pruned = [c for c in configs if int(c.kwargs["BLOCK"]) >= n_cols]
    # Triton requires at least one config returned :contentReference[oaicite:1]{index=1}
    return pruned if len(pruned) > 0 else [max(configs, key=lambda c: int(c.kwargs["BLOCK"]))]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["n_cols"],
    prune_configs_by={"early_config_prune": _early_prune},  # ✅ key fix :contentReference[oaicite:2]{index=2}
)
@triton.jit
def softmax_row_kernel_autotuned(
    X_ptr, Y_ptr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    n_cols,                # runtime
    BLOCK: tl.constexpr,   # autotuned
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
    assert x.is_cuda and x.ndim == 2
    M, N = x.shape
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    softmax_row_kernel_autotuned[(M,)](
        x, y,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        N,   # n_cols key
    )
    return y
