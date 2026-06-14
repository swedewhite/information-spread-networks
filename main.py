"""
Tech Influence Network — main orchestrator.

Loads YAML data, runs the full analytical pipeline (centrality →
structural holes → diffusion → seed optimization), assembles a JSON
payload, and renders the interactive HTML dashboard.

Usage:
    python3 main.py
    python3 main.py --open       # generate and open in browser
    python3 main.py --quick      # use small Monte Carlo run counts (for dev)
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

import yaml

from src import graph, centrality, diffusion, optimize, dashboard


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_PATH = ROOT / "output" / "dashboard.html"


def load_scenario(name: str) -> dict:
    with open(DATA_DIR / "scenarios" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true",
                     help="Open the dashboard in the default browser")
    ap.add_argument("--quick", action="store_true",
                     help="Use small Monte Carlo run counts (faster iteration)")
    args = ap.parse_args()

    print("┌─────────────────────────────────────────────")
    print("│ Tech Influence Network — Pipeline")
    print("└─────────────────────────────────────────────")

    t0 = time.time()
    print("\n[1/6] Loading data...")
    people, entities, memberships = graph.load_data(DATA_DIR)
    print(f"      {len(people)} people, {len(entities)} entities, "
          f"{len(memberships)} memberships")

    print("\n[2/6] Building graphs...")
    B = graph.build_bipartite(people, entities, memberships)
    G, W = graph.build_person_graph(people, entities, memberships)
    M = graph.entity_cooccurrence(W)
    person_entities = graph.person_entity_summary(people, entities, memberships)
    print(f"      Bipartite: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
    print(f"      Person-person: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n[3/6] Computing centralities...")
    cents = centrality.compute_centralities(G)
    composite = centrality.composite_broker_score(cents)
    sh = centrality.structural_holes(G, top_k=5)
    print(f"      Top 3 composite brokers:")
    for name, score in centrality.top_n(composite, 3):
        print(f"        {people[name]['name']:25s} {score:.4f}")
    print(f"      Removing brokers: {[people[n]['name'] for n in sh['removed']]}")
    print(f"        density: {sh['full']['density']} → {sh['reduced']['density']}")
    print(f"        components: {sh['full']['components']} → {sh['reduced']['components']}")

    # Choose seed set for diffusion: top-5 by composite (used in scenario animation)
    composite_top5 = [n for n, _ in centrality.top_n(composite, 5)]

    runs = 50 if args.quick else 200
    opt_runs = 25 if args.quick else 60

    print(f"\n[4/6] Running diffusion simulations ({runs} runs each)...")

    gossip_cfg = load_scenario("insider_gossip")
    announce_cfg = load_scenario("public_announcement")

    print(f"      Gossip scenario: β={gossip_cfg['beta']}, γ={gossip_cfg['gamma']}, α={gossip_cfg['alpha']}")
    diff_gossip = diffusion.simulate_sir(
        G, composite_top5,
        beta=gossip_cfg["beta"], gamma=gossip_cfg["gamma"],
        alpha=gossip_cfg["alpha"], steps=gossip_cfg["steps"], runs=runs)
    print(f"        expected reach: {diff_gossip['expected_reach']*100:.1f}% of network")

    print(f"      Announcement scenario: β={announce_cfg['beta']}, γ={announce_cfg['gamma']}, α={announce_cfg['alpha']}")
    diff_announce = diffusion.simulate_sir(
        G, composite_top5,
        beta=announce_cfg["beta"], gamma=announce_cfg["gamma"],
        alpha=announce_cfg["alpha"], steps=announce_cfg["steps"], runs=runs)
    print(f"        expected reach: {diff_announce['expected_reach']*100:.1f}% of network")

    print(f"\n[5/6] Finding optimal seeds (CELF greedy, {opt_runs} runs/eval)...")
    seeds = optimize.find_optimal_seeds(
        G, k=5,
        beta=gossip_cfg["beta"], gamma=gossip_cfg["gamma"],
        alpha=gossip_cfg["alpha"], steps=gossip_cfg["steps"],
        runs_per_eval=opt_runs)
    print(f"      Optimal seeds: {[people[s]['name'] for s in seeds['seeds']]}")
    print(f"      Marginal gains: {seeds['marginal_gains']}")
    print(f"      Total expected reach: {seeds['expected_reach']*100:.1f}% of network")

    print("\n[6/6] Assembling dashboard payload...")

    # Persons (richly attributed)
    persons_data = []
    for pid, p in people.items():
        persons_data.append({
            "id": pid, "name": p["name"],
            "affiliation": p.get("affiliation", ""),
            "domains": p.get("domains", []),
            "betweenness": round(cents["betweenness"].get(pid, 0), 6),
            "closeness": round(cents["closeness"].get(pid, 0), 6),
            "eigenvector": round(cents["eigenvector"].get(pid, 0), 6),
            "composite": round(composite.get(pid, 0), 6),
            "degree": G.degree(pid) if pid in G else 0,
            "entities": person_entities.get(pid, []),
        })

    entities_data = [
        {"id": eid, "name": e["name"], "type": e["type"],
         "category": e.get("category", "")}
        for eid, e in entities.items()
    ]

    # Bipartite edges
    bipartite_edges = [
        {"source": m["person"], "target": m["entity"],
         "weight": int(m.get("weight", 1)), "role": m.get("role", "")}
        for m in memberships
    ]

    # Person-person edges
    person_edges = [
        {"source": u, "target": v, "weight": d["weight"]}
        for u, v, d in G.edges(data=True)
    ]

    # Reduced graph
    G_reduced = sh["G_reduced"]
    reduced_nodes_data = []
    for pid in G_reduced.nodes():
        reduced_nodes_data.append({
            "id": pid, "name": people[pid]["name"],
            "betweenness": round(cents["betweenness"].get(pid, 0), 6),
            "degree": G_reduced.degree(pid),
        })
    reduced_edges = [
        {"source": u, "target": v, "weight": d["weight"]}
        for u, v, d in G_reduced.edges(data=True)
    ]

    # Top-N tables
    def name_score_list(scores, n=10):
        return [{"name": people[k]["name"], "id": k, "score": float(v)}
                for k, v in centrality.top_n(scores, n)]

    # Heatmap
    entity_ids = list(entities.keys())
    entity_names = [entities[eid]["name"] for eid in entity_ids]
    entity_types = {entities[eid]["name"]: entities[eid]["type"] for eid in entity_ids}
    heatmap = []
    for i, eid_i in enumerate(entity_ids):
        for j, eid_j in enumerate(entity_ids):
            heatmap.append({
                "x": entities[eid_j]["name"],
                "y": entities[eid_i]["name"],
                "value": int(M[i, j]),
            })

    # Diffusion data — convert representative_run.node_states from list of dicts
    # (already dict from simulator) to ensure JSON-friendly
    def package_diffusion(d):
        return {
            "params": d["params"],
            "sir_curve": d["sir_curve"],
            "nodeStats": d["node_stats"],
            "expected_reach": d["expected_reach"],
            "representative_run": {
                "sir_curve": d["representative_run"]["sir_curve"],
                "node_states": d["representative_run"]["node_states"],
            },
        }

    payload = {
        "persons": persons_data,
        "entities": entities_data,
        "bipartiteEdges": bipartite_edges,
        "personEdges": person_edges,
        "reducedNodes": reduced_nodes_data,
        "reducedEdges": reduced_edges,
        "topComposite": name_score_list(composite),
        "topBetweenness": name_score_list(cents["betweenness"]),
        "topCloseness": name_score_list(cents["closeness"]),
        "topEigenvector": name_score_list(cents["eigenvector"]),
        "fullMetrics": sh["full"],
        "reducedMetrics": sh["reduced"],
        "removedBrokers": [people[n]["name"] for n in sh["removed"]],
        "entityNames": entity_names,
        "entityTypes": entity_types,
        "heatmap": heatmap,
        "diffusion": {
            "gossip": package_diffusion(diff_gossip),
            "announcement": package_diffusion(diff_announce),
        },
        "seeds": {
            "nodes": seeds["seeds"],
            "marginalGains": seeds["marginal_gains"],
            "expectedReach": seeds["expected_reach"],
        },
    }

    out = dashboard.render_dashboard(payload, OUTPUT_PATH)
    elapsed = time.time() - t0
    print(f"      Dashboard: {out}")
    print(f"\nDone in {elapsed:.1f}s")

    if args.open:
        webbrowser.open(f"file://{out}")


if __name__ == "__main__":
    main()
