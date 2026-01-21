# kernels/softmax_row.py
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_row_kernel(X_ptr, Y_ptr, stride_xm, stride_xn, stride_ym, stride_yn,
                       n_cols: tl.constexpr, BLOCK: tl.constexpr):
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


def make_runner(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda
    M, N = x.shape

    def run_with_config(cfg: dict):
        meta = cfg.get("meta", {})
        # For this kernel, BLOCK must be >= N; tune over powers of two
        BLOCK = int(meta.get("BLOCK", 1024))
        if BLOCK < N:
            # force compile fail early => filtered by feasibility stage
            raise ValueError("BLOCK must be >= n_cols for this kernel.")
        num_warps = int(cfg.get("num_warps", 4))
        num_stages = int(cfg.get("num_stages", 3))
        grid = (M,)

        def _call():
            softmax_row_kernel[grid](
                x, y,
                x.stride(0), x.stride(1),
                y.stride(0), y.stride(1),
                n_cols=N,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        return _call

    return run_with_config
