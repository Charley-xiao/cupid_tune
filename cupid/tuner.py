# cupid/tuner.py
from __future__ import annotations
import json
import random
import torch
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List, Tuple

from .db import TuningDB
from .features import shape_bucket_2d, device_fingerprint, featurize
from .bayeslin import BayesLinReg
from .bench import bench_us
from .candidate_gen import CandidateGenerator


@dataclass
class TuneResult:
    best_config: Dict[str, Any]
    best_us: float
    from_cache: bool
    tried_f2: int
    tried_f1: int


class CUPIDTuner:
    """
    CUPID-Tune:
      - DejaVu-style on-disk cache
      - Multi-fidelity evaluation
      - Uncertainty-aware model (Bayesian linear regression)
      - Adaptive candidate space rewrite (light-weight bottleneck heuristic)
    """

    def __init__(
        self,
        kernel_name: str,
        db_path: str = "cupid_tuning.sqlite",
        seed: int = 0,
    ):
        self.kernel_name = kernel_name
        self.db = TuningDB(db_path)
        self.rng = random.Random(seed)
        self.dev = device_fingerprint()

    def _make_cache_key(
        self,
        m: int,
        n: int,
        dtype: torch.dtype,
        extra_key: Optional[str] = None,
    ) -> str:
        bm, bn = shape_bucket_2d(m, n)
        dtype_str = str(dtype).replace("torch.", "")
        extra = extra_key or ""
        # include device fingerprint for correctness
        return json.dumps(
            {
                "kernel": self.kernel_name,
                "device": self.dev,
                "bucket": [bm, bn],
                "dtype": dtype_str,
                "extra": extra,
            },
            sort_keys=True,
        )

    def tune(
        self,
        x: torch.Tensor,
        run_with_config: Callable[[Dict[str, Any]], Callable[[], None]],
        kernel_family: str,
        budget_f2: int = 12,
        budget_f1: int = 60,
        extra_key: Optional[str] = None,
        use_cache: bool = True,
    ) -> TuneResult:
        """
        x: input tensor (we infer shape from it)
        run_with_config(cfg) -> callable that runs kernel under cfg (no args)

        budget_f1: how many cheap trials
        budget_f2: how many expensive trials
        """
        assert x.is_cuda
        m = int(x.shape[0])
        n = int(x.shape[1]) if x.ndim == 2 else int(x.numel())
        dtype = x.dtype

        cache_key = self._make_cache_key(m, n, dtype, extra_key)
        if use_cache:
            rec = self.db.get(cache_key)
            if rec is not None:
                return TuneResult(
                    best_config=json.loads(rec.best_config_json),
                    best_us=float(rec.best_us),
                    from_cache=True,
                    tried_f2=0,
                    tried_f1=0,
                )

        gen = CandidateGenerator()
        candidates = gen.generate(kernel_family, n_cols=n)
        self.rng.shuffle(candidates)

        # Model for uncertainty-aware selection
        model = BayesLinReg(dim=8, prior_var=50.0, noise_var=0.5)

        # Track best
        best_cfg = None
        best_us = float("inf")

        tried_f1 = 0
        tried_f2 = 0

        # --- Stage 0: F0 filter by compile feasibility
        # feasible: List[Dict[str, Any]] = []
        # for cfg in candidates:
        #     try:
        #         # A cheap feasibility test: attempt to build callable, no timing yet
        #         fn = run_with_config(cfg)
        #         # run 1 warm execution to trigger compile
        #         fn()
        #         feasible.append(cfg)
        #     except Exception:
        #         continue

        # if len(feasible) == 0:
        #     raise RuntimeError("No feasible configs compiled successfully.")

        feasible = candidates  # skip feasibility for now

        # --- Stage 1: F1 cheap timing
        scored_f1 = []
        for cfg in feasible:
            if tried_f1 >= budget_f1:
                break
            try:
                fn = run_with_config(cfg)
                us = bench_us(fn, fidelity="F1")  # will compile if needed
                tried_f1 += 1
                scored_f1.append((us, cfg))

                feat = featurize(m, n, dtype, cfg)
                model.add(feat, us)
            except Exception:
                continue

        if len(scored_f1) == 0:
            raise RuntimeError("No configs succeeded in F1 benchmarking.")


        scored_f1.sort(key=lambda t: t[0])

        # --- Stage 2: Pick F2 candidates via uncertainty-aware acquisition
        # We take a mix: top cheap ones + UCB exploration
        top_pool = [cfg for _, cfg in scored_f1[: min(10, len(scored_f1))]]

        # build extra candidates using UCB
        ucb_pool = []
        for _, cfg in scored_f1:
            feat = featurize(m, n, dtype, cfg)
            mean, std = model.predict(feat)
            # lower latency is better => acquisition = mean - beta*std
            beta = 1.5
            score = mean - beta * std
            ucb_pool.append((score, cfg))
        ucb_pool.sort(key=lambda t: t[0])
        ucb_pick = [cfg for _, cfg in ucb_pool[: min(10, len(ucb_pool))]]

        # merge unique while preserving order
        f2_list = []
        seen = set()
        for cfg in top_pool + ucb_pick:
            k = json.dumps(cfg, sort_keys=True)
            if k not in seen:
                seen.add(k)
                f2_list.append(cfg)

        # --- Adaptive rewrite checkpoint (lightweight heuristic)
        # If F1 results are very close, we suspect memory-bound; widen blocks.
        if len(scored_f1) >= 5:
            ratio = scored_f1[0][0] / max(1e-6, scored_f1[4][0])
            if ratio > 0.92:
                gen.rewrite_space("memory_bound")
            elif ratio < 0.70:
                gen.rewrite_space("reg_bound")

        # Optionally expand a bit after rewrite
        if tried_f2 < budget_f2:
            extra = gen.generate(kernel_family)
            self.rng.shuffle(extra)
            for cfg in extra[:20]:
                k = json.dumps(cfg, sort_keys=True)
                if k not in seen:
                    seen.add(k)
                    f2_list.append(cfg)

        # --- Stage 3: F2 stable timing (few)
        for cfg in f2_list[:budget_f2]:
            fn = run_with_config(cfg)
            us = bench_us(fn, fidelity="F2")
            tried_f2 += 1
            if us < best_us:
                best_us = us
                best_cfg = cfg

        assert best_cfg is not None
        self.db.put(cache_key, best_cfg, best_us)

        return TuneResult(
            best_config=best_cfg,
            best_us=best_us,
            from_cache=False,
            tried_f2=tried_f2,
            tried_f1=tried_f1,
        )

    def close(self):
        self.db.close()
