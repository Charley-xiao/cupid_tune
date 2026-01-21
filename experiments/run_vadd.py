# experiments/run_vadd.py
import torch
from cupid.tuner import CUPIDTuner
from kernels.vadd import make_runner


def main():
    torch.manual_seed(0)
    x = torch.randn(1_000_000, device="cuda", dtype=torch.float16)
    y = torch.randn(1_000_000, device="cuda", dtype=torch.float16)
    z = torch.empty_like(x)

    tuner = CUPIDTuner(kernel_name="vadd")
    run_with_config = make_runner(x, y, z)

    res = tuner.tune(
        x=x.view(-1, 1),  # fake 2D for bucketing convenience
        run_with_config=run_with_config,
        kernel_family="vadd",
        budget_f1=50,
        budget_f2=10,
        extra_key="N=1e6",
    )

    print("=== CUPID-Tune vadd ===")
    print("from_cache:", res.from_cache)
    print("best_us:", res.best_us)
    print("best_config:", res.best_config)
    print("tried_f1:", res.tried_f1, "tried_f2:", res.tried_f2)
    tuner.close()


if __name__ == "__main__":
    main()
