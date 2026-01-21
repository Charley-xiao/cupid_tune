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
    CUPID-Tune (v2):
      - Lazy compilation: no Stage-0 "compile everything"
      - Multi-fidelity:
          F1 = cheap CUDA-event timing (should be fast)
          F2 = stable timing via triton.testing.do_bench
      - Uncertainty-aware selection via Bayesian linear regression
      - SQLite caching (DejaVu-style behavior)
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
        x: input tensor (CUDA)
        run_with_config(cfg) -> fn() that launches the kernel with cfg
        kernel_family: e.g. "softmax", "vadd", etc.

        budget_f1: number of *successful* cheap measurements
        budget_f2: number of *successful* stable measurements
        """

        assert x.is_cuda, "x must be CUDA tensor"
        assert x.ndim in (1, 2), "x must be 1D or 2D"

        if x.ndim == 2:
            m = int(x.shape[0])
            n = int(x.shape[1])
        else:
            m = int(x.numel())
            n = 1

        dtype = x.dtype

        # -------------------------
        # Cache check
        # -------------------------
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

        # -------------------------
        # Candidate generation (shape-aware)
        # -------------------------
        gen = CandidateGenerator()
        # IMPORTANT: pass n_cols so softmax can create BLOCK >= n
        candidates = gen.generate(kernel_family, n_cols=n)  # <--- needs updated candidate_gen.py
        self.rng.shuffle(candidates)

        # -------------------------
        # Surrogate model for UCB
        # -------------------------
        model = BayesLinReg(dim=8, prior_var=50.0, noise_var=0.5)

        best_cfg: Optional[Dict[str, Any]] = None
        best_us = float("inf")

        tried_f1 = 0
        tried_f2 = 0

        # -------------------------
        # Stage 1: F1 cheap timing (lazy compile)
        # -------------------------
        scored_f1: List[Tuple[float, Dict[str, Any]]] = []

        for cfg in candidates:
            if tried_f1 >= budget_f1:
                break
            try:
                fn = run_with_config(cfg)
                us = bench_us(fn, fidelity="F1")  # should be quick
                tried_f1 += 1

                scored_f1.append((us, cfg))
                feat = featurize(m, n, dtype, cfg)
                model.add(feat, us)

            except Exception:
                # Compilation failure or runtime error => skip
                continue

        if len(scored_f1) == 0:
            raise RuntimeError("No configs succeeded during F1. Check candidate space / kernel constraints.")

        model.fit()
        scored_f1.sort(key=lambda t: t[0])

        # -------------------------
        # Stage 2: Select configs for F2 using:
        #   - exploitation: top-k from F1
        #   - exploration: UCB (mean - beta*std)
        # -------------------------
        top_k = min(10, len(scored_f1))
        exploit_pool = [cfg for _, cfg in scored_f1[:top_k]]

        ucb_pool: List[Tuple[float, Dict[str, Any]]] = []
        beta = 1.5  # exploration weight
        for _, cfg in scored_f1:
            feat = featurize(m, n, dtype, cfg)
            mean, std = model.predict(feat)
            # lower runtime is better
            score = mean - beta * std
            ucb_pool.append((score, cfg))
        ucb_pool.sort(key=lambda t: t[0])
        explore_pool = [cfg for _, cfg in ucb_pool[:top_k]]

        # Merge unique (keep order)
        f2_list: List[Dict[str, Any]] = []
        seen = set()
        for cfg in exploit_pool + explore_pool:
            k = json.dumps(cfg, sort_keys=True)
            if k not in seen:
                seen.add(k)
                f2_list.append(cfg)

        # If still too few, fill with more from F1 ranking
        for _, cfg in scored_f1:
            if len(f2_list) >= max(budget_f2 * 2, 20):
                break
            k = json.dumps(cfg, sort_keys=True)
            if k not in seen:
                seen.add(k)
                f2_list.append(cfg)

        # -------------------------
        # Stage 3: F2 stable timing (THIS is the "similarly for F2" part)
        # Wrap in try/except just like F1 so failures don't crash tuning.
        # -------------------------
        for cfg in f2_list:
            if tried_f2 >= budget_f2:
                break
            try:
                fn = run_with_config(cfg)
                us = bench_us(fn, fidelity="F2")  # stable (do_bench-based)

                tried_f2 += 1
                if us < best_us:
                    best_us = us
                    best_cfg = cfg

            except Exception:
                # Compilation failure or runtime error => skip
                continue

        if best_cfg is None:
            raise RuntimeError("No configs succeeded during F2. Consider reducing constraints or increasing F1 pool.")

        # Save to DB for warm-start
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
