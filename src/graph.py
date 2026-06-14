"""
Graph construction module.

Loads YAML-defined people, entities, and memberships into a weighted
bipartite graph and projects it to a person-person co-affiliation graph.

The projection uses weighted matrix multiplication: edge weights between
two people equal the sum of the products of their membership weights
across all shared entities. So two keynote speakers at the same conference
have a stronger tie than two casual attendees.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
import yaml


def load_data(data_dir: str | Path) -> Tuple[Dict, Dict, List[Dict]]:
    """Load people, entities, and memberships YAML files.

    Returns:
        people:      dict id -> person attrs
        entities:    dict id -> entity attrs
        memberships: list of {person, entity, role, weight}
    """
    data_dir = Path(data_dir)
    with open(data_dir / "people.yaml") as f:
        people = {p["id"]: p for p in yaml.safe_load(f)["people"]}
    with open(data_dir / "entities.yaml") as f:
        entities = {e["id"]: e for e in yaml.safe_load(f)["entities"]}
    with open(data_dir / "memberships.yaml") as f:
        memberships = yaml.safe_load(f)["memberships"]
    return people, entities, memberships


def build_bipartite(people: Dict, entities: Dict, memberships: List[Dict]) -> nx.Graph:
    """Build the bipartite graph of people ↔ entities with edge weights."""
    B = nx.Graph()
    for pid, p in people.items():
        B.add_node(pid, bipartite=0, kind="person", **p)
    for eid, e in entities.items():
        B.add_node(eid, bipartite=1, kind="entity", **e)
    for m in memberships:
        if m["person"] not in people:
            raise KeyError(f"Unknown person in memberships: {m['person']}")
        if m["entity"] not in entities:
            raise KeyError(f"Unknown entity in memberships: {m['entity']}")
        B.add_edge(m["person"], m["entity"],
                   weight=int(m.get("weight", 1)),
                   role=m.get("role", ""))
    return B


def build_person_graph(people: Dict, entities: Dict,
                        memberships: List[Dict]) -> Tuple[nx.Graph, np.ndarray]:
    """Project the bipartite graph to a weighted person-person graph.

    Edge weight between two people p1 and p2 = Σ_e (w(p1,e) * w(p2,e))
    over all entities e they share. Diagonal is zeroed.

    Returns:
        G:   weighted person-person graph
        W:   |people| × |entities| weight matrix (numpy)
    """
    person_ids = list(people.keys())
    entity_ids = list(entities.keys())
    p_idx = {pid: i for i, pid in enumerate(person_ids)}
    e_idx = {eid: j for j, eid in enumerate(entity_ids)}

    W = np.zeros((len(person_ids), len(entity_ids)), dtype=float)
    for m in memberships:
        i = p_idx[m["person"]]
        j = e_idx[m["entity"]]
        W[i, j] = float(m.get("weight", 1))

    A = W @ W.T
    np.fill_diagonal(A, 0)

    # Normalize edge weights to [0, 1] so diffusion parameters are interpretable.
    # Without this, the matrix product can produce weights of 20+ (two keynotes
    # at three shared events), and the SIR model with α=1 saturates instantly.
    max_w = float(A.max()) if A.max() > 0 else 1.0

    G = nx.Graph()
    for pid in person_ids:
        G.add_node(pid, **people[pid])
    n = len(person_ids)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                G.add_edge(person_ids[i], person_ids[j],
                            weight=float(A[i, j] / max_w),
                            raw_weight=float(A[i, j]))
    return G, W


def entity_cooccurrence(W: np.ndarray) -> np.ndarray:
    """Compute the entity × entity co-occurrence matrix from the weight matrix.

    cell (i, j) = number of people who belong to both entity i and entity j
    (counting binary co-membership; weights ignored for clarity).
    """
    binary = (W > 0).astype(int)
    M = binary.T @ binary
    np.fill_diagonal(M, 0)
    return M


def person_entity_summary(people: Dict, entities: Dict,
                          memberships: List[Dict]) -> Dict[str, List[Dict]]:
    """For each person, list their entity affiliations with weights and roles."""
    out: Dict[str, List[Dict]] = {pid: [] for pid in people}
    for m in memberships:
        out[m["person"]].append({
            "entity_id": m["entity"],
            "entity_name": entities[m["entity"]]["name"],
            "entity_type": entities[m["entity"]].get("type", "unknown"),
            "category": entities[m["entity"]].get("category", "unknown"),
            "role": m.get("role", ""),
            "weight": int(m.get("weight", 1)),
        })
    return out
