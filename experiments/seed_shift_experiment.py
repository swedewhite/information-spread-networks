"""
Seed-Shift Experiment
=====================
Tests empirically whether the OPTIMAL SEED SET shifts when the baseline
simple-contagion SIR model (M0) is extended with:
  (a) source-credibility weighting (M1)
  (b) count-threshold complex contagion (M2)

Usage:
    python3 experiments/seed_shift_experiment.py

All randomness is seeded from np.random.seed(42) for reproducibility.
No existing src/, data/, or docs/ files are modified.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow imports from repo root without installing as a package
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.graph import load_data, build_person_graph  # noqa: E402

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
np.random.seed(42)

DATA_DIR = REPO_ROOT / "data"
BETA = 0.035
GAMMA = 0.08
ALPHA = 3.0
STEPS = 40
MC_RUNS = 300          # per evaluation during optimization
SIM_RUNS = 300         # for final reach evaluation
SEED_K = 5
GLOBAL_SEED = 42


# ===========================================================================
# STEP 1 — SOURCE CREDIBILITY
# ===========================================================================

def compute_credibility(
    people: Dict,
    entities: Dict,
    memberships: List[Dict],
) -> Tuple[Dict[str, float], Dict[str, float], str]:
    """Compute cred_global and cred_domain for every person.

    cred_global(p)         = sum of p's membership weights across all entities,
                             normalized to [0,1] over people.
    cred_domain(p, topic)  = max membership weight of p on entities whose
                             category == topic, normalized to [0,1].

    Dominant topic = entity category with the most distinct associated people.

    Returns:
        cred_global_norm:  pid -> [0,1]
        cred_domain_norm:  pid -> [0,1]  (using dominant topic)
        dominant_topic:    the winning category string
    """
    # ── cred_global: sum of membership weights per person ──────────────────
    raw_global: Dict[str, float] = {pid: 0.0 for pid in people}
    for m in memberships:
        raw_global[m["person"]] = raw_global.get(m["person"], 0.0) + float(m.get("weight", 1))

    max_g = max(raw_global.values()) if raw_global else 1.0
    cred_global_norm = {pid: v / max_g for pid, v in raw_global.items()}

    # ── Identify dominant topic ─────────────────────────────────────────────
    # For each category, count distinct people who have ANY membership in
    # an entity of that category.
    category_people: Dict[str, Set[str]] = {}
    for m in memberships:
        cat = entities[m["entity"]].get("category", "unknown")
        category_people.setdefault(cat, set()).add(m["person"])

    dominant_topic = max(category_people, key=lambda c: len(category_people[c]))

    # ── cred_domain: max weight on dominant-topic entities ──────────────────
    domain_entities = {
        eid for eid, e in entities.items()
        if e.get("category") == dominant_topic
    }

    raw_domain: Dict[str, float] = {pid: 0.0 for pid in people}
    for m in memberships:
        if m["entity"] in domain_entities:
            pid = m["person"]
            raw_domain[pid] = max(raw_domain.get(pid, 0.0), float(m.get("weight", 1)))

    max_d = max(raw_domain.values()) if any(v > 0 for v in raw_domain.values()) else 1.0
    cred_domain_norm = {pid: v / max_d for pid, v in raw_domain.items()}

    return cred_global_norm, cred_domain_norm, dominant_topic


# ===========================================================================
# STEP 2 — DIFFUSION MODELS
# ===========================================================================

def _rng_from_master(master_rng: random.Random) -> random.Random:
    return random.Random(master_rng.random())


# ── M0: Baseline SIR (replicates src/diffusion.py logic exactly) ───────────

def _run_m0(
    G: nx.Graph,
    seeds: Sequence[str],
    beta: float,
    gamma: float,
    alpha: float,
    steps: int,
    rng: random.Random,
) -> int:
    """Single M0 SIR run. Returns total reach (I+R at end)."""
    nodes = list(G.nodes())
    state = {n: "S" for n in nodes}
    for s in seeds:
        if s in state:
            state[s] = "I"

    for _ in range(steps):
        new_state = dict(state)
        for u in nodes:
            if state[u] != "I":
                continue
            for v in G.neighbors(u):
                if state[v] == "S" and new_state[v] == "S":
                    w = G[u][v].get("weight", 1.0)
                    p = 1.0 - (1.0 - beta) ** (alpha * w)
                    if rng.random() < p:
                        new_state[v] = "I"
        for u in nodes:
            if state[u] == "I" and rng.random() < gamma:
                new_state[u] = "R"
        state = new_state

    return sum(1 for s in state.values() if s in ("I", "R"))


def reach_m0(
    G: nx.Graph,
    seeds: Sequence[str],
    beta: float = BETA,
    gamma: float = GAMMA,
    alpha: float = ALPHA,
    steps: int = STEPS,
    runs: int = MC_RUNS,
    master_rng: Optional[random.Random] = None,
) -> float:
    """MC estimate of M0 reach as absolute node count."""
    if master_rng is None:
        master_rng = random.Random(GLOBAL_SEED)
    total = 0
    for _ in range(runs):
        total += _run_m0(G, seeds, beta, gamma, alpha, steps, _rng_from_master(master_rng))
    return total / runs


# ── M1: Credibility-Weighted SIR ───────────────────────────────────────────
# Transmission from u to v: p = 1 - (1 - beta * cred(u))^(alpha * w_uv)
# This scales the effective base infection probability by the SOURCE's credibility.

def _run_m1(
    G: nx.Graph,
    seeds: Sequence[str],
    cred: Dict[str, float],
    beta: float,
    gamma: float,
    alpha: float,
    steps: int,
    rng: random.Random,
) -> int:
    """Single M1 credibility-weighted SIR run."""
    nodes = list(G.nodes())
    state = {n: "S" for n in nodes}
    for s in seeds:
        if s in state:
            state[s] = "I"

    for _ in range(steps):
        new_state = dict(state)
        for u in nodes:
            if state[u] != "I":
                continue
            beta_u = beta * cred.get(u, 0.0)  # scale by source credibility
            for v in G.neighbors(u):
                if state[v] == "S" and new_state[v] == "S":
                    w = G[u][v].get("weight", 1.0)
                    p = 1.0 - (1.0 - beta_u) ** (alpha * w)
                    if rng.random() < p:
                        new_state[v] = "I"
        for u in nodes:
            if state[u] == "I" and rng.random() < gamma:
                new_state[u] = "R"
        state = new_state

    return sum(1 for s in state.values() if s in ("I", "R"))


def reach_m1(
    G: nx.Graph,
    seeds: Sequence[str],
    cred: Dict[str, float],
    beta: float = BETA,
    gamma: float = GAMMA,
    alpha: float = ALPHA,
    steps: int = STEPS,
    runs: int = MC_RUNS,
    master_rng: Optional[random.Random] = None,
) -> float:
    if master_rng is None:
        master_rng = random.Random(GLOBAL_SEED)
    total = 0
    for _ in range(runs):
        total += _run_m1(G, seeds, cred, beta, gamma, alpha, steps, _rng_from_master(master_rng))
    return total / runs


# ── M2: Count-Threshold Complex Contagion ──────────────────────────────────
# v becomes Informed only after k DISTINCT informed neighbors have each
# successfully exposed it, counted CUMULATIVELY ACROSS STEPS.
#
# Implementation:
#   exposure_sources[v] = set of distinct node IDs that have transmitted to v
#   Each step, for each Informed u neighboring Susceptible v:
#     draw Bernoulli(p = 1-(1-beta)^(alpha*w_uv))
#     if success: exposure_sources[v].add(u)
#   If |exposure_sources[v]| >= k: v transitions S->I
#
# This faithfully requires k distinct sources because:
#   - We track a SET (not a count), so the same u contributing twice
#     only counts once.
#   - v does NOT transition on any single exposure; it waits for the
#     accumulation to reach threshold k.
#   - The set persists across steps (cumulative, not per-step reset).

def _run_m2(
    G: nx.Graph,
    seeds: Sequence[str],
    beta: float,
    gamma: float,
    alpha: float,
    steps: int,
    k: int,
    rng: random.Random,
) -> int:
    """Single M2 count-threshold complex contagion run."""
    nodes = list(G.nodes())
    state = {n: "S" for n in nodes}
    for s in seeds:
        if s in state:
            state[s] = "I"

    # Cumulative set of distinct informed neighbors that have transmitted to each node.
    # Seeds start Informed, so they are not susceptible to further infection.
    exposure_sources: Dict[str, Set[str]] = {n: set() for n in nodes}

    for _ in range(steps):
        new_state = dict(state)
        newly_exposed: Dict[str, Set[str]] = {}  # collect exposure additions this step

        for u in nodes:
            if state[u] != "I":
                continue
            for v in G.neighbors(u):
                if state[v] != "S":
                    continue
                w = G[u][v].get("weight", 1.0)
                p = 1.0 - (1.0 - beta) ** (alpha * w)
                if rng.random() < p:
                    # u successfully transmits to v this step
                    newly_exposed.setdefault(v, set()).add(u)

        # Apply accumulated exposures and check threshold
        for v, new_sources in newly_exposed.items():
            if state[v] != "S":
                continue
            exposure_sources[v].update(new_sources)
            if len(exposure_sources[v]) >= k:
                new_state[v] = "I"

        # Recovery: Informed -> Retained
        for u in nodes:
            if state[u] == "I" and rng.random() < gamma:
                new_state[u] = "R"

        state = new_state

    return sum(1 for s in state.values() if s in ("I", "R"))


def reach_m2(
    G: nx.Graph,
    seeds: Sequence[str],
    k: int = 2,
    beta: float = BETA,
    gamma: float = GAMMA,
    alpha: float = ALPHA,
    steps: int = STEPS,
    runs: int = MC_RUNS,
    master_rng: Optional[random.Random] = None,
) -> float:
    if master_rng is None:
        master_rng = random.Random(GLOBAL_SEED)
    total = 0
    for _ in range(runs):
        total += _run_m2(G, seeds, beta, gamma, alpha, steps, k, _rng_from_master(master_rng))
    return total / runs


# ── M3: Credibility-Weighted Complex Contagion (bonus) ─────────────────────
# Combines M1 and M2: cred-scaled beta AND count threshold k.

def _run_m3(
    G: nx.Graph,
    seeds: Sequence[str],
    cred: Dict[str, float],
    beta: float,
    gamma: float,
    alpha: float,
    steps: int,
    k: int,
    rng: random.Random,
) -> int:
    """Single M3 credibility-weighted complex contagion run."""
    nodes = list(G.nodes())
    state = {n: "S" for n in nodes}
    for s in seeds:
        if s in state:
            state[s] = "I"

    exposure_sources: Dict[str, Set[str]] = {n: set() for n in nodes}

    for _ in range(steps):
        new_state = dict(state)
        newly_exposed: Dict[str, Set[str]] = {}

        for u in nodes:
            if state[u] != "I":
                continue
            beta_u = beta * cred.get(u, 0.0)
            for v in G.neighbors(u):
                if state[v] != "S":
                    continue
                w = G[u][v].get("weight", 1.0)
                p = 1.0 - (1.0 - beta_u) ** (alpha * w)
                if rng.random() < p:
                    newly_exposed.setdefault(v, set()).add(u)

        for v, new_sources in newly_exposed.items():
            if state[v] != "S":
                continue
            exposure_sources[v].update(new_sources)
            if len(exposure_sources[v]) >= k:
                new_state[v] = "I"

        for u in nodes:
            if state[u] == "I" and rng.random() < gamma:
                new_state[u] = "R"

        state = new_state

    return sum(1 for s in state.values() if s in ("I", "R"))


def reach_m3(
    G: nx.Graph,
    seeds: Sequence[str],
    cred: Dict[str, float],
    k: int = 2,
    beta: float = BETA,
    gamma: float = GAMMA,
    alpha: float = ALPHA,
    steps: int = STEPS,
    runs: int = MC_RUNS,
    master_rng: Optional[random.Random] = None,
) -> float:
    if master_rng is None:
        master_rng = random.Random(GLOBAL_SEED)
    total = 0
    for _ in range(runs):
        total += _run_m3(G, seeds, cred, beta, gamma, alpha, steps, k, _rng_from_master(master_rng))
    return total / runs


# ===========================================================================
# STEP 3 — GREEDY SEED OPTIMIZATION
# ===========================================================================

def greedy_seeds(
    G: nx.Graph,
    reach_fn,               # callable(seeds) -> float (absolute node count)
    k: int = 5,
    people: Optional[Dict] = None,
    label: str = "",
) -> Dict:
    """Plain greedy influence maximization (no CELF — each candidate is cheap enough).

    Greedy IM is (1-1/e)-approximate due to submodularity of reach.
    We use plain greedy (not CELF) for simplicity and because n=34 nodes
    makes the full first-pass scan inexpensive.

    Returns:
        seeds:          ordered list of node IDs
        seed_names:     ordered list of human-readable names (if people dict provided)
        marginal_gains: marginal reach added by each seed
        total_reach:    total expected reach (absolute)
    """
    nodes = list(G.nodes())
    selected: List[str] = []
    current_reach = 0.0
    marginal_gains: List[float] = []

    for i in range(k):
        best_node = None
        best_gain = -1.0
        for v in nodes:
            if v in selected:
                continue
            r = reach_fn(selected + [v])
            gain = r - current_reach
            if gain > best_gain:
                best_gain = gain
                best_node = v
        if best_node is None:
            break
        selected.append(best_node)
        current_reach += best_gain
        marginal_gains.append(best_gain)
        name = people[best_node]["name"] if people else best_node
        print(f"      [{label}] seed {i+1}: {name:28s}  marginal gain={best_gain:.2f}")

    return {
        "seeds": selected,
        "seed_names": [people[s]["name"] if people else s for s in selected],
        "marginal_gains": marginal_gains,
        "total_reach": current_reach,
    }


# ===========================================================================
# STEP 4 — SHIFT METRICS
# ===========================================================================

def jaccard(a: Sequence, b: Sequence) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def seed_metrics(
    seeds: List[str],
    G: nx.Graph,
    betweenness: Dict[str, float],
    clustering: Dict[str, float],
    cred_domain: Dict[str, float],
    cred_global: Dict[str, float],
) -> Dict:
    return {
        "mean_betweenness": np.mean([betweenness.get(s, 0.0) for s in seeds]),
        "mean_clustering": np.mean([clustering.get(s, 0.0) for s in seeds]),
        "mean_cred_domain": np.mean([cred_domain.get(s, 0.0) for s in seeds]),
        "mean_cred_global": np.mean([cred_global.get(s, 0.0) for s in seeds]),
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    t_start = time.time()

    print("=" * 70)
    print("  SEED-SHIFT EXPERIMENT")
    print("  Simple contagion vs credibility-weighted vs complex contagion")
    print("=" * 70)
    print(f"\nParameters: β={BETA}, γ={GAMMA}, α={ALPHA}, steps={STEPS}, "
          f"MC runs/eval={MC_RUNS}, seeds k={SEED_K}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[0] Loading data and building graph...")
    people, entities, memberships = load_data(DATA_DIR)
    G, W = build_person_graph(people, entities, memberships)
    n = G.number_of_nodes()
    print(f"    {n} people, {G.number_of_edges()} person-person edges")

    # ── STEP 1: Credibility ──────────────────────────────────────────────────
    print("\n[1] Computing source credibility...")
    cred_global_norm, cred_domain_norm, dominant_topic = compute_credibility(
        people, entities, memberships)

    # Report dominant topic and category-people counts
    category_people_count: Dict[str, Set[str]] = {}
    for m in memberships:
        cat = entities[m["entity"]].get("category", "unknown")
        category_people_count.setdefault(cat, set()).add(m["person"])
    cat_counts = sorted(category_people_count.items(), key=lambda x: -len(x[1]))
    print(f"\n    Entity-category people counts (top 8):")
    for cat, pset in cat_counts[:8]:
        marker = " <-- DOMINANT" if cat == dominant_topic else ""
        print(f"      {cat:22s}: {len(pset)} people{marker}")
    print(f"\n    Dominant topic chosen: '{dominant_topic}'")

    # Top-5 by each credibility
    top_cred_domain = sorted(cred_domain_norm.items(), key=lambda x: -x[1])[:5]
    top_cred_global = sorted(cred_global_norm.items(), key=lambda x: -x[1])[:5]
    print(f"\n    Top-5 by cred_domain ({dominant_topic}):")
    for pid, v in top_cred_domain:
        print(f"      {people[pid]['name']:28s}  {v:.4f}")
    print(f"\n    Top-5 by cred_global:")
    for pid, v in top_cred_global:
        print(f"      {people[pid]['name']:28s}  {v:.4f}")

    # ── Graph-level structural metrics ───────────────────────────────────────
    print("\n[2] Computing structural metrics...")
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clustering = nx.clustering(G, weight="weight")

    top5_betweenness = sorted(betweenness.items(), key=lambda x: -x[1])[:5]
    top5_clustering = sorted(clustering.items(), key=lambda x: -x[1])[:5]
    print(f"\n    Top-5 by betweenness centrality (weight-aware):")
    for pid, v in top5_betweenness:
        print(f"      {people[pid]['name']:28s}  {v:.4f}")
    print(f"\n    Top-5 by clustering coefficient:")
    for pid, v in top5_clustering:
        print(f"      {people[pid]['name']:28s}  {v:.4f}")

    # ── STEP 3: Seed optimization — one model at a time ──────────────────────
    # Use a fresh Random seeded deterministically for each model so runs are
    # comparable but independent. The global np.random.seed(42) was set at
    # module load; we derive Python random seeds from numpy's rng.
    rng_seeds = [int(x) for x in np.random.randint(0, 2**31, size=10)]

    def make_rng(idx: int) -> random.Random:
        return random.Random(rng_seeds[idx])

    print("\n" + "=" * 70)
    print("[3] GREEDY SEED OPTIMIZATION — M0 Baseline (simple SIR)")
    print("=" * 70)

    rng_m0 = make_rng(0)

    def rf_m0(seeds):
        return reach_m0(G, seeds, master_rng=random.Random(rng_m0.random()))

    result_m0 = greedy_seeds(G, rf_m0, k=SEED_K, people=people, label="M0")

    print("\n" + "=" * 70)
    print("[3] GREEDY SEED OPTIMIZATION — M1a Credibility-Weighted (cred_domain)")
    print("=" * 70)

    rng_m1a = make_rng(1)

    def rf_m1a(seeds):
        return reach_m1(G, seeds, cred_domain_norm, master_rng=random.Random(rng_m1a.random()))

    result_m1a = greedy_seeds(G, rf_m1a, k=SEED_K, people=people, label="M1-domain")

    print("\n" + "=" * 70)
    print("[3] GREEDY SEED OPTIMIZATION — M1b Credibility-Weighted (cred_global)")
    print("=" * 70)

    rng_m1b = make_rng(2)

    def rf_m1b(seeds):
        return reach_m1(G, seeds, cred_global_norm, master_rng=random.Random(rng_m1b.random()))

    result_m1b = greedy_seeds(G, rf_m1b, k=SEED_K, people=people, label="M1-global")

    print("\n" + "=" * 70)
    print("[3] GREEDY SEED OPTIMIZATION — M2 Complex Contagion (k=2)")
    print("=" * 70)

    rng_m2k2 = make_rng(3)

    def rf_m2k2(seeds):
        return reach_m2(G, seeds, k=2, master_rng=random.Random(rng_m2k2.random()))

    result_m2k2 = greedy_seeds(G, rf_m2k2, k=SEED_K, people=people, label="M2-k2")

    print("\n" + "=" * 70)
    print("[3] GREEDY SEED OPTIMIZATION — M2 Complex Contagion (k=3)")
    print("=" * 70)

    rng_m2k3 = make_rng(4)

    def rf_m2k3(seeds):
        return reach_m2(G, seeds, k=3, master_rng=random.Random(rng_m2k3.random()))

    result_m2k3 = greedy_seeds(G, rf_m2k3, k=SEED_K, people=people, label="M2-k3")

    print("\n" + "=" * 70)
    print("[3] GREEDY SEED OPTIMIZATION — M3 Credibility-Weighted Complex (k=2)")
    print("=" * 70)

    rng_m3 = make_rng(5)

    def rf_m3(seeds):
        return reach_m3(G, seeds, cred_domain_norm, k=2,
                        master_rng=random.Random(rng_m3.random()))

    result_m3 = greedy_seeds(G, rf_m3, k=SEED_K, people=people, label="M3")

    # ── STEP 4: Shift metrics ─────────────────────────────────────────────────
    baseline_seeds = result_m0["seeds"]

    variants = [
        ("M0  baseline (SIR)",                result_m0),
        ("M1a cred_domain (SIR+domain-cred)", result_m1a),
        ("M1b cred_global (SIR+global-cred)", result_m1b),
        ("M2  complex k=2",                   result_m2k2),
        ("M2  complex k=3",                   result_m2k3),
        ("M3  cred+complex k=2",              result_m3),
    ]

    print("\n" + "=" * 70)
    print("[4] SHIFT METRICS vs M0 Baseline")
    print("=" * 70)

    metrics_all = {}
    for label, result in variants:
        m = seed_metrics(
            result["seeds"], G, betweenness, clustering,
            cred_domain_norm, cred_global_norm)
        m["jaccard_vs_m0"] = jaccard(result["seeds"], baseline_seeds)
        m["total_reach_frac"] = result["total_reach"] / n
        m["seeds"] = result["seeds"]
        m["seed_names"] = result["seed_names"]
        m["marginal_gains"] = result["marginal_gains"]
        metrics_all[label] = m

    # Print combined table
    print(f"\n{'Model':<38} {'Reach%':>7} {'Jaccard':>8} {'Betw':>8} {'Clust':>7} {'CredD':>7} {'CredG':>7}")
    print("-" * 90)
    for label, m in metrics_all.items():
        print(f"{label:<38} {m['total_reach_frac']*100:>6.1f}% "
              f"{m['jaccard_vs_m0']:>8.3f} "
              f"{m['mean_betweenness']:>8.4f} "
              f"{m['mean_clustering']:>7.4f} "
              f"{m['mean_cred_domain']:>7.4f} "
              f"{m['mean_cred_global']:>7.4f}")

    # Per-model seed lists
    print("\n── Seed sets (ordered by selection) ──────────────────────────────────")
    for label, result in variants:
        names = result["seed_names"]
        gains = result["marginal_gains"]
        gain_strs = [f"{g:.1f}" for g in gains]
        print(f"\n  {label}")
        for i, (nm, gs) in enumerate(zip(names, gain_strs)):
            print(f"    {i+1}. {nm:28s}  (marginal gain: {gs})")

    # Reference lists
    print("\n── Reference: top-5 by betweenness (weight-aware) ───────────────────")
    for pid, v in top5_betweenness:
        print(f"    {people[pid]['name']:28s}  betweenness={v:.4f}")

    print("\n── Reference: top-5 by clustering coefficient ───────────────────────")
    for pid, v in top5_clustering:
        print(f"    {people[pid]['name']:28s}  clustering={v:.4f}")

    # ── STEP 5: Plain-English summary ────────────────────────────────────────
    elapsed = time.time() - t_start

    # Compute direction statistics for summary
    m0_betw = metrics_all["M0  baseline (SIR)"]["mean_betweenness"]
    m0_clust = metrics_all["M0  baseline (SIR)"]["mean_clustering"]
    m0_cred_d = metrics_all["M0  baseline (SIR)"]["mean_cred_domain"]
    m0_cred_g = metrics_all["M0  baseline (SIR)"]["mean_cred_global"]

    m2k2_betw = metrics_all["M2  complex k=2"]["mean_betweenness"]
    m2k2_clust = metrics_all["M2  complex k=2"]["mean_clustering"]

    m1a_cred_d = metrics_all["M1a cred_domain (SIR+domain-cred)"]["mean_cred_domain"]
    m1b_cred_g = metrics_all["M1b cred_global (SIR+global-cred)"]["mean_cred_global"]

    jac_m1a = metrics_all["M1a cred_domain (SIR+domain-cred)"]["jaccard_vs_m0"]
    jac_m1b = metrics_all["M1b cred_global (SIR+global-cred)"]["jaccard_vs_m0"]
    jac_m2k2 = metrics_all["M2  complex k=2"]["jaccard_vs_m0"]
    jac_m2k3 = metrics_all["M2  complex k=3"]["jaccard_vs_m0"]

    print("\n" + "=" * 70)
    print("  PLAIN-ENGLISH SUMMARY")
    print("=" * 70)

    print(f"""
1. DOMINANT TOPIC: '{dominant_topic}' had the most people across its entities and
   was used for cred_domain scoring.

2. COMPLEX CONTAGION SEED SHIFT (brokers vs cluster-interior):
   M0 baseline seeds had mean betweenness={m0_betw:.4f}, mean clustering={m0_clust:.4f}.
   M2 (k=2)        seeds had mean betweenness={m2k2_betw:.4f}, mean clustering={m2k2_clust:.4f}.
   Jaccard overlap M0 vs M2-k2={jac_m2k2:.3f}, M0 vs M2-k3={jac_m2k3:.3f}.
   {"Complex contagion SHIFTED seeds away from high-betweenness brokers TOWARD" if m2k2_betw < m0_betw else "Complex contagion did NOT reduce betweenness; brokers remained preferable"}
   {"cluster-interior nodes (lower betweenness, higher clustering), confirming the hypothesis." if m2k2_clust > m0_clust else "lower-clustering nodes (hypothesis not clearly supported by this dataset)."}

3. CREDIBILITY WEIGHTING:
   M1-domain seeds had mean cred_domain={m1a_cred_d:.4f} vs M0 baseline {m0_cred_d:.4f}.
   {"Credibility weighting DID shift toward higher-credibility nodes as expected." if m1a_cred_d > m0_cred_d else "Credibility weighting did not strongly shift seed credibility."}
   Jaccard M0 vs M1-domain={jac_m1a:.3f}.

4. COLLINEARITY CHECK (cred_domain vs cred_global):
   Jaccard M1-domain vs M1-global (directly): {jaccard(result_m1a['seeds'], result_m1b['seeds']):.3f}.
   M0 vs M1-global Jaccard={jac_m1b:.3f}.
   {"cred_global and cred_domain produced NEARLY IDENTICAL seed sets (high Jaccard), confirming collinearity." if jaccard(result_m1a['seeds'], result_m1b['seeds']) > 0.6 else "cred_global and cred_domain produced DIFFERENT seed sets; less collinear than expected."}

5. TOTAL REACH COMPARISON:
   M0 baseline: {metrics_all["M0  baseline (SIR)"]["total_reach_frac"]*100:.1f}%
   M1-domain:   {metrics_all["M1a cred_domain (SIR+domain-cred)"]["total_reach_frac"]*100:.1f}%
   M1-global:   {metrics_all["M1b cred_global (SIR+global-cred)"]["total_reach_frac"]*100:.1f}%
   M2-k=2:      {metrics_all["M2  complex k=2"]["total_reach_frac"]*100:.1f}%
   M2-k=3:      {metrics_all["M2  complex k=3"]["total_reach_frac"]*100:.1f}%
   M3 (combo):  {metrics_all["M3  cred+complex k=2"]["total_reach_frac"]*100:.1f}%
   (Reach figures are WITHIN each model's own dynamics, not cross-model comparable.)

Total runtime: {elapsed:.1f}s
""")


if __name__ == "__main__":
    main()
