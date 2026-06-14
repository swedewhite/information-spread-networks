"""
Centrality + structural-hole analysis.

Three classical centrality measures plus a composite broker score that
weights betweenness more heavily (since brokerage — the ability to move
information between otherwise-disconnected groups — is exactly what we
care about for information diffusion).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx


def compute_centralities(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    """Return betweenness, closeness, and eigenvector centralities."""
    bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
    cc = nx.closeness_centrality(G, distance=None)  # treat unweighted shortest path
    try:
        ec = nx.eigenvector_centrality(G, max_iter=2000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        ec = nx.eigenvector_centrality_numpy(G, weight="weight")
    return {"betweenness": bc, "closeness": cc, "eigenvector": ec}


def composite_broker_score(centralities: Dict[str, Dict[str, float]],
                            weights: Tuple[float, float, float] = (0.5, 0.25, 0.25)
                            ) -> Dict[str, float]:
    """Weighted composite of normalized centrality measures.

    Default weights (0.5, 0.25, 0.25) emphasize betweenness because that
    measure captures brokerage between communities directly. Closeness
    and eigenvector are rolled in to break ties and reward both
    reachability and prestige.
    """
    bc, cc, ec = centralities["betweenness"], centralities["closeness"], centralities["eigenvector"]

    def normalize(d: Dict[str, float]) -> Dict[str, float]:
        if not d:
            return {}
        m = max(d.values()) or 1.0
        return {k: v / m for k, v in d.items()}

    bcn, ccn, ecn = normalize(bc), normalize(cc), normalize(ec)
    w_bc, w_cc, w_ec = weights
    return {n: w_bc * bcn.get(n, 0) + w_cc * ccn.get(n, 0) + w_ec * ecn.get(n, 0)
            for n in bc.keys()}


def top_n(scores: Dict[str, float], n: int = 10) -> List[Tuple[str, float]]:
    """Return the top-n nodes by score, sorted descending."""
    return sorted(scores.items(), key=lambda kv: -kv[1])[:n]


def structural_holes(G: nx.Graph, top_k: int = 5) -> Dict:
    """Compare the network with and without the top-k betweenness brokers.

    Returns a dict with full and reduced metrics plus the removed node IDs.
    """
    bc = nx.betweenness_centrality(G, weight="weight")
    removed = [n for n, _ in sorted(bc.items(), key=lambda kv: -kv[1])[:top_k]]
    G_red = G.copy()
    G_red.remove_nodes_from(removed)

    def metrics(g: nx.Graph) -> Dict:
        return {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "components": nx.number_connected_components(g),
            "density": round(nx.density(g), 4),
            "clustering": round(nx.average_clustering(g, weight="weight"), 4),
            "largest_component": (
                len(max(nx.connected_components(g), key=len))
                if g.number_of_nodes() else 0
            ),
        }

    return {
        "removed": removed,
        "full": metrics(G),
        "reduced": metrics(G_red),
        "G_reduced": G_red,
    }
