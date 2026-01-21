# kernels/vadd.py
import torch
import triton
import triton.language as tl


@triton.jit
def vadd_kernel(X_ptr, Y_ptr, Z_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offs, mask=mask, other=0.0)
    tl.store(Z_ptr + offs, x + y, mask=mask)


def make_runner(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    assert x.is_cuda and y.is_cuda and z.is_cuda
    N = x.numel()

    def run_with_config(cfg: dict):
        meta = cfg.get("meta", {})
        BLOCK = int(meta.get("BLOCK", 1024))
        num_warps = int(cfg.get("num_warps", 4))
        num_stages = int(cfg.get("num_stages", 3))
        grid = (triton.cdiv(N, BLOCK),)

        def _call():
            vadd_kernel[grid](
                x, y, z,
                N=N,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        return _call

    return run_with_config
