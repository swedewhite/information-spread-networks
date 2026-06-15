# Diffusion-model experiments: findings

This folder documents an investigation into whether the dashboard's information-diffusion model *should* be extended with **trust, credibility, and platform dynamics**, and whether switching from simple to **complex contagion** changes who you'd optimally seed information with.

The short version: **we tested the extensions and decided not to ship them.** The conceptual decomposition is sound, but it is not identifiable from single-source affiliation data, and the more realistic contagion model breaks the seed optimizer. This note records why—because "we considered making this more sophisticated and deliberately didn't" is itself a result worth keeping.

> The shipped tool (simple SIR + CELF seed optimization in [`src/`](../src/)) is unaffected and correct. Everything here is research that lives *outside* the published model.

## Contents

| File | What it does |
|---|---|
| `seed_shift_experiment.py` | Baseline comparison on the real 34-node graph (gossip regime): simple SIR vs credibility-weighted vs complex contagion. |
| `seed_shift_followups.py` | (1) the announcement regime on the real graph; (2) a controlled cluster-interior test on a planted modular graph + a recovery-rate sweep. |

Run them with the repo's existing deps (`numpy`, `networkx`, `pyyaml`):

```bash
python3 experiments/seed_shift_experiment.py
python3 experiments/seed_shift_followups.py
```

Both set `np.random.seed(42)` and isolate per-model RNG streams, so results are reproducible.

---

## The question

The dashboard footnote concedes the model "does not capture trust, credibility, or platform dynamics." Those are real forces. So: can we add them, and does it matter for seeding?

We first decomposed a single transmission event into factors attached to different objects:

```
P(A→B) = g( base · tieStrength · sourceCredibility · messageCredibility · trust · platform )
         gated by a receiver threshold
```

with trust on the **edge**, source credibility on the **sender node**, message credibility on the **message**, platform on the **channel**, and a skepticism **threshold** on the receiver. Then we tried to operationalize it from the data we actually have.

## The models tested

All share the existing per-edge transmission form `p = 1 − (1 − β)^(α·w)`, where `w` is the normalized co-affiliation edge weight. Only the activation/weighting changes, so comparisons are apples-to-apples.

- **M0 — simple SIR (baseline).** Identical to [`src/diffusion.py`](../src/diffusion.py). An Informed node infects each Susceptible neighbor with probability `p`; Informed→Retained with rate `γ`.
- **M1 — credibility-weighted.** Source credibility scales the base rate: `p = 1 − (1 − β·cred(u))^(α·w)`. Credibility is derived from the role weights already in the data (1 = attendee, 2 = recurring, 3 = keynote/governance/maintainer):
  - `cred_global(p)` = sum of a person's role weights, normalized to [0, 1].
  - `cred_domain(p, topic)` = max role weight on entities whose `category` matches the message topic, normalized. (Dominant topic in the demo data: `software-engineering`.)
- **M2 — complex contagion (count-threshold).** A Susceptible node activates **only after `k` distinct informed neighbors have each transmitted to it, accumulated across time.** Each node carries a persistent set of distinct successful sources; it never resets. A single hub—however strongly connected—**cannot** ignite anyone alone. Tested at `k = 2` and `k = 3`, with the same `γ` recovery.
- **M3 — credibility-weighted complex contagion** (combination).

Seed optimization uses the same protocol across models: greedy / CELF influence maximization, seed-set size N = 5.

## What we found

### 1. Saturation depends entirely on the regime

| Regime (real graph) | M0 reach | seed-1 share of reach | brokers selected? |
|---|---|---|---|
| Gossip (β .035 / γ .08 / α 3.0) | 95.6% | **88%** (one hub dominates) | No |
| Announcement (β .07 / γ .30 / α 1.0) | 39.6% | **33%** (spread across all 5) | No |

Under gossip, a single super-connector (Ada Okafor) reaches ~88% of the network alone, so seeds 2–5 are within Monte-Carlo noise and the "seed set" beyond #1 is barely meaningful. Announcement's faster burnout breaks that degeneracy and makes the full seed set informative.

### 2. Structural brokers are never the optimal seeds — betweenness ≠ reach

The highest-betweenness nodes (Sloane Whitaker 0.34, Julian Park 0.17) are **not selected by any model in any regime.** Influence-maximization rewards high-degree hubs embedded in dense neighborhoods, not the bridges *between* neighborhoods. This corrected a premise we started with: "complex contagion shifts *away from* brokers" was wrong, because brokers were never the target.

### 3. Complex contagion does change the seeds — and its signature confirms the implementation is faithful

Complex contagion produces a different optimal seed set in both regimes (Jaccard 0.11 gossip, 0.25 announcement). The marginal-gain pattern is the tell: under `k = 2`, the **first** seed has marginal gain exactly **1.0** (it reaches only itself—one seed can never give anyone two distinct sources); under `k = 3`, seeds #1 *and* #2 both gain 1.0. That is complex contagion behaving exactly as theory requires.

### 4. The cluster-interior effect is real — but the optimizer can't find it

On a planted **modular** graph (156 nodes, 8 communities, **modularity Q = 0.77**, with interior nodes high-clustering/low-betweenness and bridge nodes the reverse), the greedy optimizer returned a paradox:

| Model | Optimal seeds | mean clustering | mean betweenness | reach |
|---|---|---|---|---|
| M0 simple | 5 interior | 0.302 | 0.007 | 98.3% |
| M2 complex k=2 (greedy) | **5 bridges** | 0.068 | 0.062 | 6.3% |

Greedy says complex contagion prefers *bridges*—the opposite of the hypothesis. **It's an optimizer artifact.** A trivially hand-picked seed set of 5 interior nodes concentrated in **one** community reaches **11.7%** at the same parameters—nearly double the 6.3% the "optimizer" found. A heuristic beating the optimized solution proves greedy did not find the optimum.

A recovery-rate sweep reveals the true mechanism (M2 k=2 reach, concentrated-interior vs spread-bridge seeds):

| γ (recovery) | concentrated interior | spread bridges | ratio |
|---|---|---|---|
| 0.00 | 16.0% | 20.9% | 0.76 |
| 0.02 | 14.9% | 15.4% | 0.96 |
| 0.05 | 13.7% | 11.6% | 1.18 |
| 0.10 | 11.7% | 7.0% | 1.67 |
| 0.20 | 8.0% | 4.1% | 1.95 |

With no recovery, bridges accumulate exposures forever and the distinction vanishes (an artifact). **With recovery, concentrated dense-local seeding wins, and its advantage grows with the recovery rate.** That *is* the cluster-interior mechanism: recovery forces spread to be fast and local, and only a dense neighborhood can deliver `k` corroborating exposures before nodes decay.

**Why greedy fails:** complex contagion (`k ≥ 2`) is **non-submodular**, and degenerate on the first move (every single seed reaches only itself, so the first greedy pick is an arbitrary tie). CELF locks onto whatever it grabs first and extends myopically; it structurally cannot discover the "place ≥2 seeds together to clear the threshold" strategy. This is the empirical confirmation of a known caveat: CELF's (1 − 1/e) guarantee assumes submodularity, which the threshold model violates.

## Scorecard

| Claim | Verdict |
|---|---|
| Complex contagion changes the optimal seeds | ✅ Confirmed (both regimes, both graphs) |
| Baseline favors brokers; complex contagion shifts away | ❌ False premise — brokers are never reach-optimal |
| Complex contagion favors concentrated cluster-interior seeding | ✅ Confirmed as a mechanism; scales with recovery rate |
| The CELF/greedy optimizer can find that optimum | ❌ No — non-submodular + degenerate first move |

## Why these are *not* in the shipped model

This is the point of the note. We are deliberately **not** adding trust/credibility/platform or a complex-contagion mode to the dashboard, because:

1. **Identifiability.** Every candidate factor (tie strength, source credibility, dyadic trust) is derived from the *same* co-affiliation signal, so multiplying them squares one dimension rather than adding new ones. They are collinear by construction in single-source data.
2. **False precision.** Bolting on factors we cannot measure would make the demonstration look more sophisticated while being less honest. The simple model's limitations are visible; a richer model's would be hidden.
3. **Optimizer invalidity.** A complex-contagion mode would silently invalidate the existing seed optimizer (finding 4). It would need a non-submodular-aware optimizer to be correct.
4. **Ethics.** A credibility-weighted seed optimizer is, functionally, a tool for selecting a community's most-trusted voices as delivery vehicles—closer to manufacturing consensus than to understanding structure. That is precisely the misuse [`ETHICS.md`](../ETHICS.md) exists to discourage.

The boundary this maps—what affiliation metadata *can* and *cannot* support—is the useful artifact.

## Caveats

- All data is fictional/synthetic; the modular graph is generated. Diffusion parameters are stylized, not calibrated.
- The graphs are small; Monte-Carlo estimates carry variance (mitigated by fixed seeds and modest run counts).
- "Credibility from role weight" is itself a proxy that reifies institutional prominence; on the demo data the dominant-topic proxy degenerated to a near-binary score, so the credibility/collinearity question is suggestive, not settled.

## Lineage

This work extends the framework described in the project [`README`](../README.md) and [`ETHICS.md`](../ETHICS.md). Method references: Kempe, Kleinberg & Tardos (2003) on influence maximization; Leskovec et al. (2007) on CELF; Centola & Macy (2007) on complex contagion.
