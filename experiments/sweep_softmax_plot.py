# experiments/sweep_softmax_plot.py
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import triton.testing as tt

from cupid.tuner import CUPIDTuner
from kernels.softmax_row import make_runner as make_runner_cupid
from kernels.softmax_row_autotune import softmax_triton_autotune

def bench_median_ms(fn):
    res = tt.do_bench(fn, warmup=25, rep=100, quantiles=(0.5,))
    return float(res[0] if isinstance(res, (tuple, list)) else res)

def run_one(M, N):
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)

    # CUPID cold-start (force no cache)
    y_cupid = torch.empty((M, N), device="cuda", dtype=torch.float16)
    tuner = CUPIDTuner(kernel_name="softmax_row", db_path="cupid_tmp.sqlite")
    run_with_config = make_runner_cupid(x, y_cupid)

    t0 = time.perf_counter()
    res = tuner.tune(
        x=x,
        run_with_config=run_with_config,
        kernel_family="softmax",
        budget_f1=30,
        budget_f2=8,
        use_cache=False,
        extra_key=f"M={M},N={N}",
    )
    torch.cuda.synchronize()
    cupid_tune_ms = (time.perf_counter() - t0) * 1e3
    cupid_steady_ms = bench_median_ms(run_with_config(res.best_config))
    tuner.close()

    # Triton autotune baseline
    t0 = time.perf_counter()
    softmax_triton_autotune(x)
    torch.cuda.synchronize()
    auto_first_ms = (time.perf_counter() - t0) * 1e3
    auto_steady_ms = bench_median_ms(lambda: softmax_triton_autotune(x))

    return cupid_tune_ms, cupid_steady_ms, auto_first_ms, auto_steady_ms

def main():
    M = 8192
    Ns = [256, 512, 1024, 2048, 4096, 8192]

    cupid_tune = []
    auto_tune = []
    cupid_steady = []
    auto_steady = []

    for N in Ns:
        ct, cs, at, a_s = run_one(M, N)
        cupid_tune.append(ct)
        auto_tune.append(at)
        cupid_steady.append(cs)
        auto_steady.append(a_s)
        print(f"N={N:5d} | CUPID tune {ct:7.1f} ms, steady {cs*1000:6.1f} us | AUTO first {at:7.1f} ms, steady {a_s*1000:6.1f} us")

    # Plot tuning overhead
    plt.figure()
    plt.plot(Ns, cupid_tune, marker="o", label="CUPID tune wall (ms)")
    plt.plot(Ns, auto_tune, marker="o", label="Triton autotune first-call (ms)")
    plt.xscale("log", base=2)
    plt.xlabel("N (columns)")
    plt.ylabel("Tuning overhead (ms)")
    plt.legend()
    plt.title("Cold-start tuning overhead vs shape")
    plt.savefig("sweep_softmax_tuning_overhead.png")
    # plt.show()
    plt.close()

    # Plot steady-state
    plt.figure()
    plt.plot(Ns, np.array(cupid_steady)*1000, marker="o", label="CUPID steady (us)")
    plt.plot(Ns, np.array(auto_steady)*1000, marker="o", label="Triton steady (us)")
    plt.xscale("log", base=2)
    plt.xlabel("N (columns)")
    plt.ylabel("Kernel time (us)")
    plt.legend()
    plt.title("Steady-state kernel latency")
    plt.savefig("sweep_softmax_steady_state.png")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    main()
