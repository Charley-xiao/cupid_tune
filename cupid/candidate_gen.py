# cupid/candidate_gen.py
from __future__ import annotations
from typing import Dict, Any, List
import itertools


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

    def generate(self, kernel_family: str) -> List[Dict[str, Any]]:
        """
        kernel_family lets you tailor meta parameters
        e.g., "vadd" uses BLOCK; softmax might use BLOCK_N etc.
        """
        configs = []
        for nw, ns, blk in itertools.product(self.num_warps_list, self.num_stages_list, self.block_list):
            cfg = {
                "num_warps": int(nw),
                "num_stages": int(ns),
                "meta": {"BLOCK": int(blk)},
            }
            configs.append(cfg)
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
