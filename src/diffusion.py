"""
SIR information-diffusion simulation on the person-person co-affiliation graph.

We model information spread as a contagion. Each node is in one of three states:

    S (Susceptible):  has not heard
    I (Informed):     knows the information AND is actively sharing it
    R (Retained):     knows the information but has stopped sharing it
                      (yesterday's news / boredom / has shared with everyone they cared to)

Transmission across an edge (u, v) where u ∈ I, v ∈ S happens with
probability p = 1 - (1 - β)^(α · w(u,v)). The exponent shape means
heavier edges yield disproportionately higher transmission probability —
strong ties are not just slightly better than weak ones, they are
qualitatively different information channels.

Each timestep, an Informed node also has probability γ of moving to R
(losing interest / stopping outreach).

Results are aggregated over many Monte Carlo runs because outcomes are
stochastic and a single run is not representative.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np


def simulate_sir_run(G: nx.Graph,
                     seeds: Sequence[str],
                     beta: float,
                     gamma: float,
                     alpha: float,
                     steps: int,
                     rng: random.Random) -> Tuple[List[Dict[str, int]], List[Dict[str, str]], List[int]]:
    """Run a single stochastic SIR simulation.

    Returns:
        sir_curve:    per-timestep counts {S, I, R}
        node_states:  per-timestep dict node_id -> state in {"S","I","R"}
        infection_times: per-node first timestep at which each node
                         transitioned from S to I. None if never infected.
                         Returned as a list aligned to G.nodes() order.
    """
    nodes = list(G.nodes())
    state = {n: "S" for n in nodes}
    for s in seeds:
        if s in state:
            state[s] = "I"

    inf_time: Dict[str, int] = {}
    for s in seeds:
        if s in state:
            inf_time[s] = 0

    sir_curve: List[Dict[str, int]] = []
    node_states: List[Dict[str, str]] = []

    def record(t: int) -> None:
        sir_curve.append({
            "t": t,
            "S": sum(1 for v in state.values() if v == "S"),
            "I": sum(1 for v in state.values() if v == "I"),
            "R": sum(1 for v in state.values() if v == "R"),
        })
        node_states.append(dict(state))

    record(0)

    for t in range(1, steps + 1):
        new_state = dict(state)
        # Transmission: each informed node tries to infect its susceptible neighbors
        for u in nodes:
            if state[u] != "I":
                continue
            for v in G.neighbors(u):
                if state[v] == "S" and new_state[v] == "S":
                    w = G[u][v].get("weight", 1.0)
                    p = 1.0 - (1.0 - beta) ** (alpha * w)
                    if rng.random() < p:
                        new_state[v] = "I"
                        inf_time.setdefault(v, t)
        # Recovery: informed -> retained
        for u in nodes:
            if state[u] == "I":
                if rng.random() < gamma:
                    new_state[u] = "R"
        state = new_state
        record(t)

    aligned_inf_times = [inf_time.get(n) for n in nodes]
    return sir_curve, node_states, aligned_inf_times


def simulate_sir(G: nx.Graph,
                 seeds: Sequence[str],
                 beta: float,
                 gamma: float,
                 alpha: float = 1.0,
                 steps: int = 40,
                 runs: int = 200,
                 seed: int = 42) -> Dict:
    """Monte Carlo SIR simulation aggregated over many runs.

    Returns:
        params:           the parameters used
        sir_curve:        list of dicts per timestep with mean/std of S, I, R
        node_stats:       per-node {mean_inf_time, std_inf_time, prob_infected}
        representative_run: single run nearest the median total reach
                           (for animation in the dashboard)
        expected_reach:   mean fraction of the network that learned the info
    """
    nodes = list(G.nodes())
    n = len(nodes)
    rng_master = random.Random(seed)

    all_curves: List[List[Dict[str, int]]] = []
    all_inf_times: List[List] = []  # per run, list aligned to nodes
    representative: Tuple[List[Dict[str, int]], List[Dict[str, str]]] | None = None
    final_reach: List[int] = []
    cached_runs: List[Tuple[List[Dict[str, int]], List[Dict[str, str]]]] = []

    for r in range(runs):
        run_rng = random.Random(rng_master.random())
        curve, state_history, inf_times = simulate_sir_run(
            G, seeds, beta, gamma, alpha, steps, run_rng)
        all_curves.append(curve)
        all_inf_times.append(inf_times)
        # Total reach = informed + retained at the final step
        reach = curve[-1]["I"] + curve[-1]["R"]
        final_reach.append(reach)
        cached_runs.append((curve, state_history))

    # Pick the run whose final reach is closest to the median
    median_reach = float(np.median(final_reach))
    rep_idx = int(np.argmin([abs(r - median_reach) for r in final_reach]))
    rep_curve, rep_state_history = cached_runs[rep_idx]

    # Aggregate SIR curve: mean and std per timestep
    aggregated: List[Dict] = []
    for t in range(steps + 1):
        S = [c[t]["S"] for c in all_curves]
        I = [c[t]["I"] for c in all_curves]
        R = [c[t]["R"] for c in all_curves]
        aggregated.append({
            "t": t,
            "S_mean": float(np.mean(S)), "S_std": float(np.std(S)),
            "I_mean": float(np.mean(I)), "I_std": float(np.std(I)),
            "R_mean": float(np.mean(R)), "R_std": float(np.std(R)),
        })

    # Per-node stats: probability of infection and mean infection time
    inf_array = np.array([[t if t is not None else -1 for t in row]
                          for row in all_inf_times])
    node_stats = {}
    for i, node_id in enumerate(nodes):
        col = inf_array[:, i]
        infected_runs = col[col >= 0]
        prob = len(infected_runs) / runs if runs else 0.0
        if len(infected_runs):
            mean_t = float(np.mean(infected_runs))
            std_t = float(np.std(infected_runs))
        else:
            mean_t, std_t = None, None
        node_stats[node_id] = {
            "prob_infected": round(prob, 4),
            "mean_inf_time": round(mean_t, 2) if mean_t is not None else None,
            "std_inf_time": round(std_t, 2) if std_t is not None else None,
        }

    return {
        "params": {"beta": beta, "gamma": gamma, "alpha": alpha,
                    "steps": steps, "runs": runs, "seeds": list(seeds)},
        "sir_curve": aggregated,
        "node_stats": node_stats,
        "representative_run": {
            "sir_curve": rep_curve,
            "node_states": rep_state_history,
        },
        "expected_reach": float(np.mean(final_reach)) / max(n, 1),
    }
