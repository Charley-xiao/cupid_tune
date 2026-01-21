# experiments/run_softmax.py
import time
import torch
import triton.testing as tt

from cupid.tuner import CUPIDTuner
from kernels.softmax_row import make_runner as make_runner_cupid
from kernels.softmax_row_autotune import softmax_triton_autotune


def main():
    torch.manual_seed(0)
    M, N = 8192, 2048
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)

    # -------------------------
    # CUPID-Tune
    # -------------------------
    y_cupid = torch.empty((M, N), device="cuda", dtype=torch.float16)
    tuner = CUPIDTuner(kernel_name="softmax_row")

    run_with_config = make_runner_cupid(x, y_cupid)

    cupid_res = tuner.tune(
        x=x,
        run_with_config=run_with_config,
        kernel_family="softmax",
        budget_f1=60,
        budget_f2=12,
        extra_key=f"M={M},N={N}",
    )

    # Measure CUPID steady-state with best config (F2-style benchmark)
    best_cfg = cupid_res.best_config
    fn_best = run_with_config(best_cfg)
    cupid_us = tt.do_bench(fn_best, warmup=25, rep=100, quantiles=(0.5,))[0]

    # -------------------------
    # Triton autotune baseline
    # -------------------------
    # 1) First call wall time (includes tuning overhead)
    t0 = time.perf_counter()
    y_auto = softmax_triton_autotune(x)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t0) * 1e3

    # 2) Steady-state latency AFTER tuning is done
    def run_auto():
        softmax_triton_autotune(x)

    auto_us = tt.do_bench(run_auto, warmup=25, rep=100, quantiles=(0.5,))[0]

    # -------------------------
    # Correctness checks
    # -------------------------
    y_ref = torch.softmax(x.float(), dim=1).half()
    torch.testing.assert_close(y_cupid, y_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(y_auto, y_ref, rtol=1e-2, atol=1e-2)

    # -------------------------
    # Report
    # -------------------------
    print("\n=== Results: Softmax Row (M={}, N={}) ===".format(M, N))
    print("[CUPID]  best_us(F2 median): {:.3f} us".format(float(cupid_res.best_us)))
    print("[CUPID]  steady_us:          {:.3f} us".format(float(cupid_us)))
    print("[AUTO]   first_call_total:   {:.3f} ms (includes tuning)".format(float(first_call_ms)))
    print("[AUTO]   steady_us:          {:.3f} us".format(float(auto_us)))
    print("[CUPID]  from_cache:         {}".format(cupid_res.from_cache))
    print("[CUPID]  tried_f1 / tried_f2: {} / {}".format(cupid_res.tried_f1, cupid_res.tried_f2))
    print("[CUPID]  best_config:        {}".format(cupid_res.best_config))

    tuner.close()


if __name__ == "__main__":
    main()
