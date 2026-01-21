import os
import time
import torch
import triton.testing as tt

from cupid.tuner import CUPIDTuner
from kernels.softmax_row import make_runner as make_runner_cupid
from kernels.softmax_row_autotune import softmax_triton_autotune

def bench_median_ms(fn):
    res = tt.do_bench(fn, warmup=25, rep=100, quantiles=(0.5,))
    return float(res[0] if isinstance(res, (tuple, list)) else res)

def main():
    torch.manual_seed(0)
    M, N = 8192, 2048
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)

    # -------------------------
    # CUPID: COLD START
    # -------------------------
    y_cupid = torch.empty((M, N), device="cuda", dtype=torch.float16)

    # Use a fresh DB file so it's guaranteed cold-start
    tuner = CUPIDTuner(kernel_name="softmax_row", db_path="cupid_tmp.sqlite")
    run_with_config = make_runner_cupid(x, y_cupid)

    t0 = time.perf_counter()
    cupid_res = tuner.tune(
        x=x,
        run_with_config=run_with_config,
        kernel_family="softmax",
        budget_f1=60,
        budget_f2=12,
        extra_key=f"M={M},N={N}",
        use_cache=False,  # ✅ force tuning
    )
    torch.cuda.synchronize()
    cupid_tune_ms = (time.perf_counter() - t0) * 1e3

    fn_best = run_with_config(cupid_res.best_config)
    cupid_ms = bench_median_ms(fn_best)

    # -------------------------
    # Triton autotune baseline
    # -------------------------
    t0 = time.perf_counter()
    y_auto = softmax_triton_autotune(x)
    torch.cuda.synchronize()
    auto_first_ms = (time.perf_counter() - t0) * 1e3

    auto_ms = bench_median_ms(lambda: softmax_triton_autotune(x))

    # correctness
    y_ref = torch.softmax(x.float(), dim=1).half()
    torch.testing.assert_close(y_cupid, y_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(y_auto, y_ref, rtol=1e-2, atol=1e-2)

    # report (correct units)
    print(f"\n=== Softmax Row (M={M}, N={N}) ===")
    print(f"[CUPID] tune_wall: {cupid_tune_ms:.1f} ms (F1={cupid_res.tried_f1}, F2={cupid_res.tried_f2})")
    print(f"[CUPID] steady:    {cupid_ms:.4f} ms ({cupid_ms*1000:.1f} us)")
    print(f"[AUTO]  first:     {auto_first_ms:.1f} ms (includes tuning)")
    print(f"[AUTO]  steady:    {auto_ms:.4f} ms ({auto_ms*1000:.1f} us)")
    print(f"[CUPID] best_cfg:  {cupid_res.best_config}")

    tuner.close()

if __name__ == "__main__":
    main()
