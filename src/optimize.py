"""
Influence-maximization seed selection (greedy with CELF speedup).

Given the diffusion model, find the set of k seed nodes that maximizes
expected total reach. This is the classic Kempe-Kleinberg-Tardos (2003)
influence maximization problem — NP-hard in general, but greedy with a
(1 - 1/e) ≈ 63% approximation guarantee thanks to submodularity.

CELF (Cost-Effective Lazy Forward, Leskovec 2007) avoids re-evaluating
every candidate at every step. It exploits the fact that marginal gains
are monotonically non-increasing across iterations: if a candidate's gain
was already evaluated at iteration i and was lower than another candidate's
*next* evaluation, you can prune without recomputing.
"""
from __future__ import annotations

import heapq
import random
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

from .diffusion import simulate_sir_run


def estimate_reach(G: nx.Graph,
                   seeds: Sequence[str],
                   beta: float,
                   gamma: float,
                   alpha: float,
                   steps: int,
                   runs: int,
                   rng: random.Random) -> float:
    """Monte Carlo estimate of expected total reach (I + R at end)."""
    if not seeds:
        return 0.0
    total = 0.0
    for _ in range(runs):
        run_rng = random.Random(rng.random())
        curve, _, _ = simulate_sir_run(G, seeds, beta, gamma, alpha, steps, run_rng)
        total += curve[-1]["I"] + curve[-1]["R"]
    return total / runs


def find_optimal_seeds(G: nx.Graph,
                        k: int = 5,
                        beta: float = 0.15,
                        gamma: float = 0.05,
                        alpha: float = 1.0,
                        steps: int = 30,
                        runs_per_eval: int = 60,
                        seed: int = 1729) -> Dict:
    """Greedy + CELF influence maximization.

    Returns:
        seeds:           list of k node IDs in the order they were chosen
        marginal_gains:  the marginal gain of each seed when it was added
        expected_reach:  reach of the final seed set as a fraction of |V|
        all_first_gains: every node's initial single-node reach (for inspection)
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    n = len(nodes)

    # First pass: evaluate every node as a single seed
    first_gains: Dict[str, float] = {}
    for v in nodes:
        first_gains[v] = estimate_reach(G, [v], beta, gamma, alpha, steps, runs_per_eval, rng)

    # CELF priority queue: (-gain, node, last_eval_iter)
    heap: List[Tuple[float, str, int]] = [(-g, v, 0) for v, g in first_gains.items()]
    heapq.heapify(heap)

    seeds: List[str] = []
    marginal_gains: List[float] = []
    current_reach = 0.0

    while len(seeds) < k and heap:
        neg_gain, candidate, last_iter = heapq.heappop(heap)
        if last_iter == len(seeds):
            # Gain is up-to-date — accept this candidate
            seeds.append(candidate)
            marginal_gains.append(-neg_gain)
            current_reach += -neg_gain
        else:
            # Recompute marginal gain with current seed set
            new_reach = estimate_reach(G, seeds + [candidate], beta, gamma, alpha,
                                       steps, runs_per_eval, rng)
            new_gain = new_reach - current_reach
            heapq.heappush(heap, (-new_gain, candidate, len(seeds)))

    return {
        "seeds": seeds,
        "marginal_gains": [round(g, 3) for g in marginal_gains],
        "expected_reach": round(current_reach / max(n, 1), 4),
        "all_first_gains": {v: round(g, 3) for v, g in first_gains.items()},
    }
