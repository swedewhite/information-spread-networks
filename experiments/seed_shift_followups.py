"""
Seed-Shift Follow-Up Experiments
================================
Two follow-up experiments that REUSE the audited model functions from
seed_shift_experiment.py UNCHANGED:

  EXP 1 — "Public announcement" diffusion regime (beta=0.07, gamma=0.30,
          alpha=1.0) on the REAL 34-node graph, compared side-by-side with
          the gossip regime (beta=0.035, gamma=0.08, alpha=3.0).

  EXP 2 — Controlled test of the cluster-interior claim on a LARGER, genuinely
          MODULAR synthetic graph (>=6 communities, >=150 nodes), with planted
          interior vs bridge node labels.

No model mechanics are modified. reach_m0 / reach_m1 / reach_m2 / reach_m3 /
greedy_seeds / compute_credibility / seed_metrics / jaccard are imported as-is.

Usage:
    python3 experiments/seed_shift_followups.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import the AUDITED, UNCHANGED model + helper functions. Importing this module
# also executes its module-level np.random.seed(42).
from experiments.seed_shift_experiment import (  # noqa: E402
    reach_m0, reach_m1, reach_m2, reach_m3,
    greedy_seeds, compute_credibility, seed_metrics, jaccard,
    BETA, GAMMA, ALPHA, STEPS, SEED_K,
)
from src.graph import load_data, build_person_graph  # noqa: E402

# Re-assert determinism at the top of THIS script (defensive; the import
# already set it, but we want a clean known state before our own draws).
np.random.seed(42)

DATA_DIR = REPO_ROOT / "data"

# Gossip regime (matches prior run / module defaults)
GOSSIP = dict(beta=BETA, gamma=GAMMA, alpha=ALPHA, steps=STEPS)          # 0.035 / 0.08 / 3.0
# Announcement regime
ANNOUNCE = dict(beta=0.07, gamma=0.30, alpha=1.0, steps=STEPS)           # 0.07 / 0.30 / 1.0


# ===========================================================================
# EXPERIMENT 1 — Announcement regime on the real graph
# ===========================================================================

def run_regime_comparison(
    G: nx.Graph,
    people: Dict,
    cred_domain: Dict[str, float],
    cred_global: Dict[str, float],
    regime: Dict,
    regime_name: str,
    mc_runs: int,
    rng_pool: List[int],
) -> Dict[str, Dict]:
    """Run the full 6-model comparison under one diffusion regime.

    Returns dict: variant_label -> greedy result dict.
    Each model gets its own deterministic Python RNG stream from rng_pool.
    Model functions are called UNCHANGED with explicit regime params + master_rng.
    """
    b, g, a, s = regime["beta"], regime["gamma"], regime["alpha"], regime["steps"]

    def mk(idx):
        return random.Random(rng_pool[idx])

    print(f"\n{'#'*70}")
    print(f"#  REGIME: {regime_name}  (beta={b}, gamma={g}, alpha={a}, steps={s})")
    print(f"{'#'*70}")

    results: Dict[str, Dict] = {}

    # M0
    print(f"\n  -- {regime_name} / M0 baseline (simple SIR) --")
    r0 = mk(0)
    results["M0"] = greedy_seeds(
        G,
        lambda seeds: reach_m0(G, seeds, beta=b, gamma=g, alpha=a, steps=s,
                               runs=mc_runs, master_rng=random.Random(r0.random())),
        k=SEED_K, people=people, label=f"{regime_name}/M0")

    # M1-domain
    print(f"\n  -- {regime_name} / M1-domain (cred_domain) --")
    r1a = mk(1)
    results["M1-domain"] = greedy_seeds(
        G,
        lambda seeds: reach_m1(G, seeds, cred_domain, beta=b, gamma=g, alpha=a, steps=s,
                               runs=mc_runs, master_rng=random.Random(r1a.random())),
        k=SEED_K, people=people, label=f"{regime_name}/M1-domain")

    # M1-global
    print(f"\n  -- {regime_name} / M1-global (cred_global) --")
    r1b = mk(2)
    results["M1-global"] = greedy_seeds(
        G,
        lambda seeds: reach_m1(G, seeds, cred_global, beta=b, gamma=g, alpha=a, steps=s,
                               runs=mc_runs, master_rng=random.Random(r1b.random())),
        k=SEED_K, people=people, label=f"{regime_name}/M1-global")

    # M2 k=2
    print(f"\n  -- {regime_name} / M2 complex k=2 --")
    r2a = mk(3)
    results["M2-k2"] = greedy_seeds(
        G,
        lambda seeds: reach_m2(G, seeds, k=2, beta=b, gamma=g, alpha=a, steps=s,
                               runs=mc_runs, master_rng=random.Random(r2a.random())),
        k=SEED_K, people=people, label=f"{regime_name}/M2-k2")

    # M2 k=3
    print(f"\n  -- {regime_name} / M2 complex k=3 --")
    r2b = mk(4)
    results["M2-k3"] = greedy_seeds(
        G,
        lambda seeds: reach_m2(G, seeds, k=3, beta=b, gamma=g, alpha=a, steps=s,
                               runs=mc_runs, master_rng=random.Random(r2b.random())),
        k=SEED_K, people=people, label=f"{regime_name}/M2-k3")

    # M3 cred+complex k=2
    print(f"\n  -- {regime_name} / M3 cred+complex k=2 --")
    r3 = mk(5)
    results["M3"] = greedy_seeds(
        G,
        lambda seeds: reach_m3(G, seeds, cred_domain, k=2, beta=b, gamma=g, alpha=a, steps=s,
                               runs=mc_runs, master_rng=random.Random(r3.random())),
        k=SEED_K, people=people, label=f"{regime_name}/M3")

    return results


def experiment_1():
    print("=" * 70)
    print("  EXPERIMENT 1 — ANNOUNCEMENT vs GOSSIP on the REAL 34-node graph")
    print("=" * 70)

    people, entities, memberships = load_data(DATA_DIR)
    G, W = build_person_graph(people, entities, memberships)
    n = G.number_of_nodes()
    print(f"\n  Graph: {n} people, {G.number_of_edges()} edges")

    cred_global, cred_domain, dominant_topic = compute_credibility(people, entities, memberships)
    print(f"  Dominant topic (cred_domain): '{dominant_topic}'")

    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clustering = nx.clustering(G, weight="weight")

    MC = 150  # trimmed for tractable runtime; directional comparison is robust to this

    # Independent RNG pools per regime so the two regimes don't share streams.
    pool_gossip = [int(x) for x in np.random.randint(0, 2**31, size=10)]
    pool_announce = [int(x) for x in np.random.randint(0, 2**31, size=10)]

    res_gossip = run_regime_comparison(
        G, people, cred_domain, cred_global, GOSSIP, "GOSSIP", MC, pool_gossip)
    res_announce = run_regime_comparison(
        G, people, cred_domain, cred_global, ANNOUNCE, "ANNOUNCE", MC, pool_announce)

    # ----- Side-by-side M0 and M2 tables -----
    def fmt_seedset(result):
        return [f"{name} ({g:.1f})" for name, g in
                zip(result["seed_names"], result["marginal_gains"])]

    print("\n" + "=" * 70)
    print("  [1a] BASELINE SATURATION — M0 gossip vs announcement")
    print("=" * 70)
    for rname, res in [("GOSSIP", res_gossip), ("ANNOUNCE", res_announce)]:
        m0 = res["M0"]
        total = m0["total_reach"]
        seed1_gain = m0["marginal_gains"][0]
        share = seed1_gain / total if total > 0 else 0.0
        print(f"\n  {rname} M0:")
        print(f"    total reach        = {total/n*100:.1f}% of network ({total:.2f} nodes)")
        print(f"    seed-1 marginal    = {seed1_gain:.2f} nodes")
        print(f"    seed-1 share       = {share*100:.1f}% of total reach")
        print(f"    marginal gains     = {[round(x,2) for x in m0['marginal_gains']]}")
        gains_2to5 = sum(m0["marginal_gains"][1:])
        print(f"    seeds 2-5 combined = {gains_2to5:.2f} nodes "
              f"({gains_2to5/total*100:.1f}% of total)")
        print(f"    ordered seeds      = {m0['seed_names']}")

    print("\n" + "=" * 70)
    print("  [1b] DO BROKERS ENTER M0's SEED SET UNDER ANNOUNCEMENT?")
    print("=" * 70)
    top_betw = sorted(betweenness.items(), key=lambda x: -x[1])[:5]
    print("\n  Top-5 betweenness brokers:")
    for pid, v in top_betw:
        print(f"    {people[pid]['name']:28s}  betweenness={v:.4f}")
    sloane = "sloane-whitaker"
    julian = "julian-park"
    for rname, res in [("GOSSIP", res_gossip), ("ANNOUNCE", res_announce)]:
        seeds = res["M0"]["seeds"]
        print(f"\n  {rname} M0 seeds: {res['M0']['seed_names']}")
        print(f"    Sloane Whitaker (betw 0.342) in seed set? "
              f"{'YES' if sloane in seeds else 'NO'}")
        print(f"    Julian Park (betw 0.174) in seed set?     "
              f"{'YES' if julian in seeds else 'NO'}")

    print("\n" + "=" * 70)
    print("  [1c] DOES COMPLEX CONTAGION STILL SHIFT SEEDS UNDER ANNOUNCEMENT?")
    print("=" * 70)
    ann_m0_seeds = res_announce["M0"]["seeds"]
    for variant in ["M2-k2", "M2-k3", "M1-domain", "M1-global", "M3"]:
        jac = jaccard(res_announce[variant]["seeds"], ann_m0_seeds)
        print(f"    Jaccard(ANNOUNCE {variant:10s} vs ANNOUNCE M0) = {jac:.3f}")

    # ----- Full side-by-side metric tables (gossip vs announce) -----
    def metric_row(label, result, regime_params):
        m = seed_metrics(result["seeds"], G, betweenness, clustering,
                         cred_domain, cred_global)
        m["jaccard_vs_m0"] = jaccard(result["seeds"],
                                     (res_gossip if regime_params == "G" else res_announce)["M0"]["seeds"])
        m["reach_frac"] = result["total_reach"] / n
        return m

    print("\n" + "=" * 70)
    print("  FULL COMPARISON TABLE — GOSSIP regime")
    print("=" * 70)
    print(f"\n{'Model':<14} {'Reach%':>7} {'Jac/M0':>7} {'Betw':>8} {'Clust':>7} {'CredD':>7} {'CredG':>7}")
    print("-" * 64)
    for label in ["M0", "M1-domain", "M1-global", "M2-k2", "M2-k3", "M3"]:
        m = metric_row(label, res_gossip[label], "G")
        print(f"{label:<14} {m['reach_frac']*100:>6.1f}% {m['jaccard_vs_m0']:>7.3f} "
              f"{m['mean_betweenness']:>8.4f} {m['mean_clustering']:>7.4f} "
              f"{m['mean_cred_domain']:>7.4f} {m['mean_cred_global']:>7.4f}")

    print("\n" + "=" * 70)
    print("  FULL COMPARISON TABLE — ANNOUNCEMENT regime")
    print("=" * 70)
    print(f"\n{'Model':<14} {'Reach%':>7} {'Jac/M0':>7} {'Betw':>8} {'Clust':>7} {'CredD':>7} {'CredG':>7}")
    print("-" * 64)
    for label in ["M0", "M1-domain", "M1-global", "M2-k2", "M2-k3", "M3"]:
        m = metric_row(label, res_announce[label], "A")
        print(f"{label:<14} {m['reach_frac']*100:>6.1f}% {m['jaccard_vs_m0']:>7.3f} "
              f"{m['mean_betweenness']:>8.4f} {m['mean_clustering']:>7.4f} "
              f"{m['mean_cred_domain']:>7.4f} {m['mean_cred_global']:>7.4f}")

    # ----- M0 & M2 seed sets side by side -----
    print("\n" + "=" * 70)
    print("  M0 SEED SETS SIDE BY SIDE")
    print("=" * 70)
    print("\n  GOSSIP   M0:", fmt_seedset(res_gossip["M0"]))
    print("  ANNOUNCE M0:", fmt_seedset(res_announce["M0"]))
    print("\n  GOSSIP   M2-k2:", fmt_seedset(res_gossip["M2-k2"]))
    print("  ANNOUNCE M2-k2:", fmt_seedset(res_announce["M2-k2"]))
    print("\n  GOSSIP   M2-k3:", fmt_seedset(res_gossip["M2-k3"]))
    print("  ANNOUNCE M2-k3:", fmt_seedset(res_announce["M2-k3"]))

    return {
        "res_gossip": res_gossip,
        "res_announce": res_announce,
        "betweenness": betweenness,
        "clustering": clustering,
        "people": people,
        "n": n,
    }


# ===========================================================================
# EXPERIMENT 2 — Modular synthetic graph + cluster-interior test
# ===========================================================================

def build_modular_graph(
    n_communities: int = 8,
    interior_per_comm: int = 18,
    n_bridges: int = 12,
    p_within: float = 0.55,
    p_between: float = 0.012,
    seed: int = 42,
):
    """Construct a genuinely modular weighted graph with planted interior/bridge roles.

    Construction (relaxed planted-partition with explicit bridge nodes):
      - n_communities dense communities, each with `interior_per_comm` INTERIOR nodes.
      - INTERIOR-INTERIOR edges within the same community: present w.p. p_within,
        weight ~ Uniform(0.6, 1.0)  (STRONG within-community ties).
      - INTERIOR-INTERIOR edges across communities: present w.p. p_between,
        weight ~ Uniform(0.05, 0.25)  (WEAK between-community ties).
      - n_bridges BRIDGE nodes, each attached to 3 distinct communities. A bridge
        connects to ~40% of the interior nodes in each of its assigned communities,
        with MODERATE weight ~ Uniform(0.15, 0.45). Bridges therefore span clusters
        (=> high betweenness, low clustering) while interior nodes sit inside one
        dense cluster (=> high clustering, low betweenness).

    All randomness is from a local np.random.default_rng(seed) so this graph is
    independent of, and does not perturb, the global np.random stream used by the
    Monte-Carlo simulations.

    Returns:
        G:           weighted networkx graph, node attr 'role' in {interior,bridge},
                     node attr 'community' (int; bridges get community=-1).
        communities: list of sets of interior-node ids (for modularity Q).
        roles:       dict node -> 'interior' | 'bridge'
        comm_of:     dict node -> community index (bridges -> -1)
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    # ---- Interior nodes ----
    comm_members: List[List[str]] = [[] for _ in range(n_communities)]
    comm_of: Dict[str, int] = {}
    roles: Dict[str, str] = {}
    for c in range(n_communities):
        for j in range(interior_per_comm):
            nid = f"c{c}_i{j}"
            G.add_node(nid, role="interior", community=c)
            comm_members[c].append(nid)
            comm_of[nid] = c
            roles[nid] = "interior"

    # ---- Within-community dense edges (strong weights) ----
    for c in range(n_communities):
        members = comm_members[c]
        for ii in range(len(members)):
            for jj in range(ii + 1, len(members)):
                if rng.random() < p_within:
                    w = float(rng.uniform(0.6, 1.0))
                    G.add_edge(members[ii], members[jj], weight=w)

    # ---- Between-community sparse edges (weak weights) ----
    all_interior = [nid for c in range(n_communities) for nid in comm_members[c]]
    for ii in range(len(all_interior)):
        for jj in range(ii + 1, len(all_interior)):
            a, b = all_interior[ii], all_interior[jj]
            if comm_of[a] == comm_of[b]:
                continue
            if rng.random() < p_between:
                w = float(rng.uniform(0.05, 0.25))
                G.add_edge(a, b, weight=w)

    # ---- Bridge nodes spanning 3 communities each ----
    for k in range(n_bridges):
        nid = f"bridge_{k}"
        G.add_node(nid, role="bridge", community=-1)
        roles[nid] = "bridge"
        comm_of[nid] = -1
        # pick 3 distinct communities
        chosen = list(rng.choice(n_communities, size=3, replace=False))
        for c in chosen:
            members = comm_members[c]
            # connect to ~40% of that community's interior nodes
            n_attach = max(2, int(0.40 * len(members)))
            picks = rng.choice(len(members), size=n_attach, replace=False)
            for idx in picks:
                w = float(rng.uniform(0.15, 0.45))  # moderate, weaker than within
                G.add_edge(nid, members[idx], weight=w)

    communities = [set(m) for m in comm_members]
    return G, communities, roles, comm_of


def experiment_2():
    print("\n\n" + "=" * 70)
    print("  EXPERIMENT 2 — CLUSTER-INTERIOR TEST on a MODULAR synthetic graph")
    print("=" * 70)

    # ---- Build graph ----
    G, communities, roles, comm_of = build_modular_graph(
        n_communities=8, interior_per_comm=18, n_bridges=12,
        p_within=0.55, p_between=0.012, seed=42)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    n_interior = sum(1 for r in roles.values() if r == "interior")
    n_bridge = sum(1 for r in roles.values() if r == "bridge")

    # ---- Validity report: modularity, sizes ----
    # Modularity Q using the planted interior communities. Bridges belong to no
    # single community; assign each bridge to the community it has the most edges
    # into, purely for the Q computation (does not affect simulation).
    comm_assign: Dict[str, int] = dict(comm_of)
    for b in [x for x, r in roles.items() if r == "bridge"]:
        counts: Dict[int, int] = {}
        for nb in G.neighbors(b):
            c = comm_of[nb]
            if c >= 0:
                counts[c] = counts.get(c, 0) + 1
        comm_assign[b] = max(counts, key=counts.get) if counts else 0

    # Build community node-sets for modularity (interior + assigned bridges)
    q_communities: Dict[int, set] = {}
    for node, c in comm_assign.items():
        q_communities.setdefault(c, set()).add(node)
    Q = nx.algorithms.community.modularity(
        G, list(q_communities.values()), weight="weight")

    print(f"\n  [VALIDITY REPORT]")
    print(f"    nodes                = {n}  ({n_interior} interior + {n_bridge} bridge)")
    print(f"    edges                = {m}")
    print(f"    communities          = {len(communities)} (interior planted)")
    print(f"    community size       = {len(communities[0])} interior nodes each")
    print(f"    modularity Q (weighted, planted partition) = {Q:.4f}")
    print(f"    avg weighted degree  = {2*sum(d['weight'] for _,_,d in G.edges(data=True))/n:.2f}")
    print(f"    is connected         = {nx.is_connected(G)}")

    # ---- Centrality / clustering distributions, split by role ----
    print("\n  Computing betweenness (weight-aware) and clustering...")
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clustering = nx.clustering(G, weight="weight")

    int_clust = [clustering[x] for x, r in roles.items() if r == "interior"]
    brg_clust = [clustering[x] for x, r in roles.items() if r == "bridge"]
    int_betw = [betweenness[x] for x, r in roles.items() if r == "interior"]
    brg_betw = [betweenness[x] for x, r in roles.items() if r == "bridge"]

    def dist(label, arr):
        arr = np.array(arr)
        print(f"    {label:28s} mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")

    print("\n  [CLUSTERING distribution by role]")
    dist("INTERIOR clustering", int_clust)
    dist("BRIDGE   clustering", brg_clust)
    print("\n  [BETWEENNESS distribution by role]")
    dist("INTERIOR betweenness", int_betw)
    dist("BRIDGE   betweenness", brg_betw)

    interior_ok = np.mean(int_clust) > np.mean(brg_clust) and np.mean(int_betw) < np.mean(brg_betw)
    print(f"\n  Planted-role validity check: "
          f"interior=high-clustering/low-betweenness, bridge=reverse -> "
          f"{'CONFIRMED' if interior_ok else 'FAILED'}")

    # Top-5 references
    top_betw = sorted(betweenness.items(), key=lambda x: -x[1])[:5]
    top_clust = sorted(clustering.items(), key=lambda x: -x[1])[:5]
    print("\n  Top-5 betweenness (expect bridges):")
    for nid, v in top_betw:
        print(f"    {nid:14s} role={roles[nid]:8s} betw={v:.4f} clust={clustering[nid]:.4f}")
    print("\n  Top-5 clustering (expect interior):")
    for nid, v in top_clust:
        print(f"    {nid:14s} role={roles[nid]:8s} clust={v:.4f} betw={betweenness[nid]:.4f}")

    # ---- Seed optimization (CELF lazy greedy for tractability) ----
    # Tractable settings: fewer MC runs and steps; CELF prunes re-evaluation.
    MC = 100
    STEPS_SYN = 20
    B, Gm, A = 0.05, 0.10, 2.0   # diffusion params tuned so spread is partial (not 0/100%)

    print(f"\n  Diffusion params for EXP2: beta={B}, gamma={Gm}, alpha={A}, "
          f"steps={STEPS_SYN}, MC runs={MC} (CELF lazy greedy)")

    rng_pool = [int(x) for x in np.random.randint(0, 2**31, size=10)]

    def celf_seeds(reach_fn, k, label):
        """CELF (lazy-forward) greedy. reach_fn(seeds)->float. Submodular reach
        => marginal gains are non-increasing, so a stale top-of-heap can be
        lazily re-evaluated and re-inserted instead of rescanning all nodes."""
        import heapq
        nodes = list(G.nodes())
        # First pass: single-node reach
        first = {v: reach_fn([v]) for v in nodes}
        heap = [(-gn, v, 0) for v, gn in first.items()]
        heapq.heapify(heap)
        selected: List[str] = []
        gains: List[float] = []
        current = 0.0
        while len(selected) < k and heap:
            neg_g, cand, last_iter = heapq.heappop(heap)
            if last_iter == len(selected):
                selected.append(cand)
                gains.append(-neg_g)
                current += -neg_g
                print(f"      [{label}] seed {len(selected)}: {cand:14s} "
                      f"role={roles[cand]:8s} marginal={-neg_g:.2f}")
            else:
                new_reach = reach_fn(selected + [cand])
                heapq.heappush(heap, (-(new_reach - current), cand, len(selected)))
        return {"seeds": selected, "seed_names": selected,
                "marginal_gains": gains, "total_reach": current}

    print(f"\n  -- EXP2 / M0 baseline (simple SIR) --")
    r0 = random.Random(rng_pool[0])
    res_m0 = celf_seeds(
        lambda seeds: reach_m0(G, seeds, beta=B, gamma=Gm, alpha=A, steps=STEPS_SYN,
                               runs=MC, master_rng=random.Random(r0.random())),
        SEED_K, "M0")

    print(f"\n  -- EXP2 / M2 complex k=2 --")
    r2a = random.Random(rng_pool[1])
    res_m2k2 = celf_seeds(
        lambda seeds: reach_m2(G, seeds, k=2, beta=B, gamma=Gm, alpha=A, steps=STEPS_SYN,
                               runs=MC, master_rng=random.Random(r2a.random())),
        SEED_K, "M2-k2")

    print(f"\n  -- EXP2 / M2 complex k=3 --")
    r2b = random.Random(rng_pool[2])
    res_m2k3 = celf_seeds(
        lambda seeds: reach_m2(G, seeds, k=3, beta=B, gamma=Gm, alpha=A, steps=STEPS_SYN,
                               runs=MC, master_rng=random.Random(r2b.random())),
        SEED_K, "M2-k3")

    # ---- Decisive output ----
    def seed_detail(result, label):
        print(f"\n  {label} seed set:")
        cl, bt = [], []
        n_int, n_brg = 0, 0
        for s in result["seeds"]:
            c = comm_of[s]
            comm_str = f"comm{c}" if c >= 0 else "SPANS"
            cl.append(clustering[s])
            bt.append(betweenness[s])
            if roles[s] == "interior":
                n_int += 1
            else:
                n_brg += 1
            print(f"    {s:14s}  {comm_str:7s}  role={roles[s]:8s}  "
                  f"clust={clustering[s]:.4f}  betw={betweenness[s]:.4f}")
        print(f"    -> mean clustering = {np.mean(cl):.4f}, "
              f"mean betweenness = {np.mean(bt):.4f}")
        print(f"    -> composition: {n_int} interior, {n_brg} bridge")
        return np.mean(cl), np.mean(bt), n_int, n_brg

    print("\n" + "=" * 70)
    print("  [EXP2 DECISIVE OUTPUT] per-model seed sets")
    print("=" * 70)
    m0_cl, m0_bt, m0_int, m0_brg = seed_detail(res_m0, "M0 (simple SIR)")
    m2k2_cl, m2k2_bt, m2k2_int, m2k2_brg = seed_detail(res_m2k2, "M2 (complex k=2)")
    m2k3_cl, m2k3_bt, m2k3_int, m2k3_brg = seed_detail(res_m2k3, "M2 (complex k=3)")

    print("\n" + "=" * 70)
    print("  [EXP2 SUMMARY TABLE]")
    print("=" * 70)
    print(f"\n  {'Model':<18} {'MeanClust':>10} {'MeanBetw':>10} {'#Interior':>10} {'#Bridge':>9} {'Reach%':>8}")
    print("  " + "-" * 67)
    for label, res, cl, bt, ni, nb in [
        ("M0 simple",      res_m0,   m0_cl,   m0_bt,   m0_int,   m0_brg),
        ("M2 complex k=2", res_m2k2, m2k2_cl, m2k2_bt, m2k2_int, m2k2_brg),
        ("M2 complex k=3", res_m2k3, m2k3_cl, m2k3_bt, m2k3_int, m2k3_brg),
    ]:
        print(f"  {label:<18} {cl:>10.4f} {bt:>10.4f} {ni:>10} {nb:>9} "
              f"{res['total_reach']/n*100:>7.1f}%")

    # ---- Verdict ----
    print("\n" + "=" * 70)
    print("  [EXP2 VERDICT] Does complex contagion shift toward interior nodes?")
    print("=" * 70)
    clust_up = m2k2_cl > m0_cl
    betw_down = m2k2_bt < m0_bt
    interior_up = m2k2_int > m0_int
    print(f"""
  M0 simple SIR:   mean clustering={m0_cl:.4f}, mean betweenness={m0_bt:.4f}, {m0_brg} bridges in seeds
  M2 complex k=2:  mean clustering={m2k2_cl:.4f}, mean betweenness={m2k2_bt:.4f}, {m2k2_brg} bridges in seeds
  M2 complex k=3:  mean clustering={m2k3_cl:.4f}, mean betweenness={m2k3_bt:.4f}, {m2k3_brg} bridges in seeds

  Directional cluster-interior prediction:
    clustering INCREASES M0->M2(k2)?  {clust_up}  ({m0_cl:.4f} -> {m2k2_cl:.4f})
    betweenness DECREASES M0->M2(k2)? {betw_down}  ({m0_bt:.4f} -> {m2k2_bt:.4f})
    more INTERIOR nodes in M2 seeds?  {interior_up}  ({m0_int} -> {m2k2_int})

  VERDICT: {'CONFIRMED — complex contagion shifts toward high-clustering interior nodes and away from high-betweenness bridges on a properly modular graph.' if (clust_up and betw_down) else 'NOT CONFIRMED in this configuration — see numbers above.'}
""")

    # ---- GAMMA ROBUSTNESS SWEEP (mechanism check) ----
    # Guards against "tuned-until-confirmed": directly compare M2 (k=2) reach from a
    # CONCENTRATED single-community INTERIOR seed set vs a SPREAD BRIDGE seed set across
    # recovery rates. Expectation from the earlier diagnostic: at gamma=0 bridges can
    # slowly accumulate exposures forever (artifact, ratio ~1); with recovery, only
    # dense/fast local reinforcement (interior) sustains spread (ratio rises).
    print("\n" + "=" * 70)
    print("  [EXP2 ROBUSTNESS] M2 k=2 reach: concentrated INTERIOR vs SPREAD BRIDGE seeds, vs recovery")
    print("=" * 70)
    interior_seeds = list(communities[0])[:SEED_K]
    bridge_seeds = [x for x, r in roles.items() if r == "bridge"][:SEED_K]
    print(f"  interior seeds (all in one community): {interior_seeds}")
    print(f"  bridge   seeds (span communities):     {bridge_seeds}")
    print(f"\n  {'gamma':>7} {'interior reach%':>16} {'bridge reach%':>15} {'ratio int/brg':>15}")
    print("  " + "-" * 56)
    sweep_rng = random.Random(rng_pool[5])
    for g in [0.0, 0.02, 0.05, 0.10, 0.20]:
        ri = reach_m2(G, interior_seeds, k=2, beta=B, gamma=g, alpha=A, steps=STEPS_SYN,
                      runs=MC, master_rng=random.Random(sweep_rng.random())) / n * 100
        rb = reach_m2(G, bridge_seeds, k=2, beta=B, gamma=g, alpha=A, steps=STEPS_SYN,
                      runs=MC, master_rng=random.Random(sweep_rng.random())) / n * 100
        ratio = (ri / rb) if rb > 0 else float("inf")
        print(f"  {g:>7.2f} {ri:>16.1f} {rb:>15.1f} {ratio:>15.2f}")
    print("\n  Reading: ratio ~1 at gamma=0 (bridges accumulate freely = the artifact);")
    print("  ratio climbs with recovery => interior's dense, fast reinforcement is what wins.")

    return {
        "Q": Q, "n": n, "m": m, "n_communities": len(communities),
        "interior_ok": interior_ok,
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    t0 = time.time()
    exp1 = experiment_1()
    exp2 = experiment_2()
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  ALL FOLLOW-UP EXPERIMENTS COMPLETE in {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
