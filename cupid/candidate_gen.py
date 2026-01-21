# cupid/candidate_gen.py
from __future__ import annotations
from typing import Dict, Any, List
import itertools
import math 


def next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


class CandidateGenerator:
    """
    Generates Triton config dictionaries.
    The "rewrite" mechanism updates the candidate space based on observed bottlenecks.
    """

    def __init__(self):
        # Initial search space (reasonable defaults)
        self.num_warps_list = [1, 2, 4, 8]
        self.num_stages_list = [2, 3, 4, 5]
        self.block_list = [128, 256, 512, 1024, 2048]

    def generate(self, kernel_family: str, n_cols: int | None = None):
        # ✅ make softmax search space valid for all N
        if kernel_family == "softmax" and n_cols is not None:
            b = next_pow2(n_cols)
            self.block_list = sorted(set([b, max(128, b // 2), min(8192, b * 2)]))
            # keep it small like Triton autotune config list
            self.num_warps_list = [2, 4, 8]
            self.num_stages_list = [2, 3, 4]

        configs = []
        for nw in self.num_warps_list:
            for ns in self.num_stages_list:
                for blk in self.block_list:
                    configs.append({
                        "num_warps": int(nw),
                        "num_stages": int(ns),
                        "meta": {"BLOCK": int(blk)},
                    })
        return configs

    def rewrite_space(self, bottleneck: str):
        """
        Simple adaptive rewrite:
          - memory_bound: try bigger blocks + more warps
          - reg_bound: reduce stages + reduce blocks
          - occupancy_bound: reduce warps/stages
        """
        if bottleneck == "memory_bound":
            self.block_list = sorted(set(self.block_list + [4096, 8192]))
            self.num_warps_list = sorted(set(self.num_warps_list + [8]))
        elif bottleneck == "reg_bound":
            self.num_stages_list = [2, 3]
            self.block_list = [128, 256, 512]
        elif bottleneck == "occupancy_bound":
            self.num_warps_list = [1, 2, 4]
            self.num_stages_list = [2, 3]
        # else: no change
