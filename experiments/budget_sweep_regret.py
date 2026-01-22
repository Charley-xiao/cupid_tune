import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import triton.testing as tt

from cupid.tuner import CUPIDTuner
from kernels.softmax_row import make_runner as make_runner_cupid
from kernels.softmax_row_autotune import softmax_triton_autotune


def bench_median_ms(fn):
    """Robust across Triton versions (float or tuple)."""
    res = tt.do_bench(fn, warmup=25, rep=100, quantiles=(0.5,))
    return float(res[0] if isinstance(res, (tuple, list)) else res)


def get_oracle_ms(x):
    """
    Oracle runtime = Triton autotune steady-state median (ms).
    First call triggers tuning for this shape.
    """
    # autotune cold-start
    y = softmax_triton_autotune(x)
    torch.cuda.synchronize()

    # correctness sanity check (optional but recommended once)
    y_ref = torch.softmax(x.float(), dim=1).half()
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    # steady-state timing
    return bench_median_ms(lambda: softmax_triton_autotune(x))


def run_cupid_once(x, budget_f2, budget_f1=30, seed=0):
    """
    One cold-start CUPID run (no cache).
    Returns: (tune_wall_ms, steady_ms, best_config)
    """
    M, N = x.shape
    y = torch.empty((M, N), device="cuda", dtype=torch.float16)

    # Make tuning "cold start" by using a unique db each run
    db_path = f"cupid_tmp_b{budget_f2}_seed{seed}.sqlite"
    tuner = CUPIDTuner(kernel_name="softmax_row", db_path=db_path, seed=seed)

    run_with_config = make_runner_cupid(x, y)

    t0 = time.perf_counter()
    res = tuner.tune(
        x=x,
        run_with_config=run_with_config,
        kernel_family="softmax",
        budget_f1=budget_f1,
        budget_f2=budget_f2,
        use_cache=False,  # IMPORTANT: cold-start
        extra_key=f"M={M},N={N}",
    )
    torch.cuda.synchronize()
    tune_wall_ms = (time.perf_counter() - t0) * 1e3

    # steady-state of best config CUPID found
    fn_best = run_with_config(res.best_config)
    steady_ms = bench_median_ms(fn_best)

    # correctness check (CUPID output buffer y must match ref)
    y_ref = torch.softmax(x.float(), dim=1).half()
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    tuner.close()
    return tune_wall_ms, steady_ms, res.best_config


def main():
    torch.manual_seed(0)

    # Choose one shape first. Later you can wrap this in an outer loop over N.
    M, N = 8192, 2048
    x = torch.randn((M, N), device="cuda", dtype=torch.float16)

    budgets = [2, 4, 6, 8, 12]
    repeats = 3  # average over randomness/noise

    # Oracle from Triton autotune (best among its configs)
    oracle_ms = get_oracle_ms(x)
    print(f"[ORACLE] Triton autotune steady: {oracle_ms:.4f} ms ({oracle_ms*1000:.1f} us)")

    avg_regrets = []
    avg_tune_wall = []
    avg_steady = []

    for B in budgets:
        regs = []
        tune_walls = []
        steadies = []

        for r in range(repeats):
            tune_wall_ms, steady_ms, cfg = run_cupid_once(
                x, budget_f2=B, budget_f1=30, seed=r
            )
            regret = (steady_ms - oracle_ms) / oracle_ms
            regs.append(regret)
            tune_walls.append(tune_wall_ms)
            steadies.append(steady_ms)

            print(
                f"B={B:2d} rep={r} | tune={tune_wall_ms:7.1f} ms | "
                f"steady={steady_ms*1000:6.1f} us | regret={regret*100:6.2f}% | cfg={cfg}"
            )

        avg_regrets.append(float(np.mean(regs)))
        avg_tune_wall.append(float(np.mean(tune_walls)))
        avg_steady.append(float(np.mean(steadies)))

    # ---- Plot: regret vs budget ----
    plt.figure()
    plt.plot(budgets, np.array(avg_regrets) * 100.0, marker="o")
    plt.xlabel("CUPID budget_f2 (number of expensive evaluations)")
    plt.ylabel("Regret (%) vs Triton autotune oracle")
    plt.title(f"Regret vs budget (M={M}, N={N})")
    plt.grid(True)
    plt.show()

    # ---- Plot: tuning overhead vs budget ----
    plt.figure()
    plt.plot(budgets, avg_tune_wall, marker="o")
    plt.xlabel("CUPID budget_f2")
    plt.ylabel("CUPID tuning wall time (ms)")
    plt.title(f"Tuning overhead vs budget (M={M}, N={N})")
    plt.grid(True)
    plt.show()

    # ---- Plot: steady-state time vs budget (optional) ----
    plt.figure()
    plt.plot(budgets, np.array(avg_steady) * 1000.0, marker="o", label="CUPID steady (us)")
    plt.axhline(oracle_ms * 1000.0, linestyle="--", label="Triton oracle (us)")
    plt.xlabel("CUPID budget_f2")
    plt.ylabel("Kernel time (us)")
    plt.title(f"Steady-state latency vs budget (M={M}, N={N})")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
