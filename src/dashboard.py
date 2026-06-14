"""
Dashboard HTML generator.

Emits a single self-contained HTML file with embedded D3.js visualizations.
The Python pipeline pre-computes all analytics and bundles them as JSON;
the HTML/JS handles only rendering and interaction (no in-browser numerical
work). Eight sections:

    1. Affiliation Network — bipartite people ↔ entities
    2. Centrality Analysis — temperature-mapped force-directed graph
    3. Centrality Rankings — top-10 tables for each measure + composite
    4. Entity Overlap — heatmap of co-membership
    5. Structural Holes — full vs. broker-removed networks
    6. Diffusion Simulation — animated SIR replay + curve
    7. Seed Optimization — optimal information-seeding visualization
    8. Scenario Comparison — gossip vs. public-announcement curves
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Modeling Information Spread in Networks</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700;800&display=swap');

:root {
  --bg: #0a0e14;
  --bg-card: #111820;
  --bg-hover: #1a2232;
  --border: #1e2a3a;
  --text: #c5cdd8;
  --text-dim: #6b7a8d;
  --text-bright: #eef1f5;
  --accent: #ec6a4d;
  --accent-glow: rgba(236,106,77,0.3);
  --gold: #d4a853;
  --blue: #4a9eed;
  --green: #4ecb8d;
  --purple: #9b7fe6;
  --cyan: #2bbaa0;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.hero {
  text-align: center;
  padding: 5rem 2rem 3rem;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -50%; left: -50%; width: 200%; height: 200%;
  background: radial-gradient(ellipse at 50% 20%, rgba(236,106,77,0.08) 0%, transparent 60%);
  pointer-events: none;
}
.hero h1 {
  font-family: 'Playfair Display', serif;
  font-size: 3.2rem; font-weight: 800;
  color: var(--text-bright);
  letter-spacing: -0.02em;
  margin-bottom: 0.6rem;
}
.hero .subtitle {
  font-size: 1.15rem; color: var(--text-dim);
  max-width: 720px; margin: 0 auto 1rem;
  font-weight: 300;
}
.hero .ethics-banner {
  display: inline-block;
  margin-top: 0.5rem;
  font-size: 0.78rem;
  color: var(--text-dim);
  background: rgba(236,106,77,0.05);
  border: 1px solid rgba(236,106,77,0.2);
  padding: 0.4rem 1rem;
  border-radius: 100px;
}

nav {
  position: sticky; top: 0; z-index: 1000;
  background: rgba(10,14,20,0.92);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  display: flex; flex-wrap: wrap; justify-content: center;
  gap: 0.25rem; padding: 0.8rem 1.5rem;
}
nav a {
  color: var(--text-dim); text-decoration: none;
  padding: 0.4rem 1rem; border-radius: 8px;
  font-size: 0.82rem; font-weight: 500;
  transition: all 0.25s;
}
nav a:hover, nav a.active {
  background: var(--bg-hover); color: var(--text-bright);
}
nav a.active { border-bottom: 2px solid var(--accent); }

.section { max-width: 1300px; margin: 0 auto; padding: 3rem 2rem; }
.section-header { margin-bottom: 2rem; }
.section-header h2 {
  font-family: 'Playfair Display', serif;
  font-size: 1.8rem; font-weight: 700;
  color: var(--text-bright); margin-bottom: 0.3rem;
}
.section-header p {
  color: var(--text-dim); font-size: 0.95rem; max-width: 700px;
}
.section + .section { border-top: 1px solid var(--border); }

.graph-container {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
  position: relative;
}
.graph-container svg { display: block; width: 100%; }

.graph-controls {
  position: absolute; top: 12px; right: 12px;
  display: flex; gap: 6px; z-index: 10;
}
.graph-controls button {
  background: rgba(17,24,32,0.85);
  border: 1px solid var(--border);
  color: var(--text-dim);
  width: 32px; height: 32px; border-radius: 8px;
  cursor: pointer; font-size: 1rem;
  display: flex; align-items: center; justify-content: center;
  transition: all 0.2s; backdrop-filter: blur(8px);
}
.graph-controls button:hover {
  background: var(--bg-hover); color: var(--text-bright);
  border-color: var(--accent);
}

.dual-graph {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}
@media (max-width: 600px) { .dual-graph { grid-template-columns: 1fr; } }
.dual-graph .graph-container { min-height: 550px; }
.graph-label {
  position: absolute; top: 14px; left: 18px;
  font-size: 0.82rem; font-weight: 600;
  color: var(--text-dim); z-index: 5;
  background: rgba(17,24,32,0.7);
  padding: 4px 12px; border-radius: 6px;
  backdrop-filter: blur(8px);
}

.tables-row {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1.5rem; margin-top: 2rem;
}
@media (max-width: 1100px) { .tables-row { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) { .tables-row { grid-template-columns: 1fr; } }

.ranking-table {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}
.ranking-table h3 {
  padding: 1rem 1.2rem 0.6rem;
  font-size: 0.85rem; font-weight: 600;
  color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.06em;
}
.ranking-table h3 .desc {
  display: block;
  font-size: 0.7rem;
  text-transform: none;
  letter-spacing: 0;
  font-weight: 400;
  margin-top: 2px;
  color: var(--text-dim);
  opacity: 0.7;
}
.ranking-table table { width: 100%; border-collapse: collapse; }
.ranking-table th {
  text-align: left;
  padding: 0.5rem 1.2rem;
  font-size: 0.72rem;
  color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.08em;
  border-bottom: 1px solid var(--border);
}
.ranking-table td {
  padding: 0.55rem 1.2rem;
  font-size: 0.85rem;
  border-bottom: 1px solid rgba(30,42,58,0.5);
}
.ranking-table tr:last-child td { border-bottom: none; }
.ranking-table tr:hover { background: var(--bg-hover); }
.ranking-table .rank { color: var(--text-dim); font-weight: 500; width: 24px; }
.ranking-table .name { color: var(--text-bright); font-weight: 500; }
.ranking-table .score { color: var(--text-dim); font-variant-numeric: tabular-nums; text-align: right; }
.ranking-table .bar-cell { width: 35%; }
.ranking-table .bar-bg {
  height: 5px; border-radius: 3px;
  background: rgba(30,42,58,0.6); overflow: hidden;
}
.ranking-table .bar-fill {
  height: 100%; border-radius: 3px;
  transition: width 0.6s ease;
}

.metrics-row {
  display: grid; grid-template-columns: repeat(6, 1fr);
  gap: 1rem; margin-top: 1.5rem;
}
@media (max-width: 1000px) { .metrics-row { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 600px) { .metrics-row { grid-template-columns: repeat(2, 1fr); } }
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.2rem;
  text-align: center;
}
.metric-card .label {
  font-size: 0.7rem; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.06em;
  margin-bottom: 0.3rem;
}
.metric-card .value {
  font-size: 1.4rem; font-weight: 700;
  color: var(--text-bright);
}
.metric-card .delta { font-size: 0.72rem; margin-top: 0.2rem; }
.metric-card .delta.down { color: var(--accent); }
.metric-card .delta.up { color: var(--green); }

.heatmap-container { margin-top: 1rem; }
#overlap { max-width: 1600px; }
#heatmapGraph { width: 100%; overflow-x: auto; overflow-y: hidden; }
#heatmapGraph svg { width: auto; max-width: none; margin: 0 auto; }

.tooltip {
  position: fixed; pointer-events: none;
  background: rgba(17,24,32,0.95);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 14px;
  font-size: 0.82rem; color: var(--text);
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  z-index: 9999;
  max-width: 320px;
  opacity: 0; transition: opacity 0.15s;
}
.tooltip .tt-name {
  font-weight: 600; color: var(--text-bright);
  font-size: 0.92rem; margin-bottom: 4px;
}
.tooltip .tt-affil { color: var(--text-dim); font-size: 0.78rem; margin-bottom: 4px; }
.tooltip .tt-row {
  display: flex; justify-content: space-between; gap: 1.5rem;
}
.tooltip .tt-label { color: var(--text-dim); }
.tooltip .tt-value { font-weight: 500; font-variant-numeric: tabular-nums; }
.tooltip .tt-orgs {
  margin-top: 6px; padding-top: 6px;
  border-top: 1px solid var(--border);
}
.tooltip .org-tag {
  display: inline-block;
  padding: 1px 7px;
  border-radius: 4px;
  font-size: 0.72rem;
  margin: 2px 3px 2px 0;
  font-weight: 500;
}

.color-legend {
  position: absolute;
  bottom: 16px; left: 18px;
  display: flex; align-items: center; gap: 8px;
  z-index: 5;
  background: rgba(17,24,32,0.7);
  padding: 8px 14px; border-radius: 8px;
  backdrop-filter: blur(8px);
}
.color-legend canvas { width: 120px; height: 10px; border-radius: 5px; }
.color-legend span { font-size: 0.72rem; color: var(--text-dim); }

.entity-legend {
  display: flex; flex-wrap: wrap;
  gap: 8px; margin-top: 1rem;
  justify-content: center;
}
.entity-pill {
  display: flex; align-items: center; gap: 6px;
  padding: 5px 14px;
  border-radius: 20px;
  font-size: 0.78rem; font-weight: 500;
  background: var(--bg-card);
  border: 1px solid var(--border);
  cursor: pointer;
  transition: all 0.2s;
}
.entity-pill:hover { border-color: var(--text-dim); }
.entity-pill .dot { width: 9px; height: 9px; border-radius: 50%; }
.entity-pill .shape {
  width: 11px; height: 11px;
  display: inline-block;
}

.centrality-selector {
  position: absolute; top: 14px; left: 18px;
  display: flex; gap: 4px; z-index: 10;
  background: rgba(17,24,32,0.85);
  padding: 4px; border-radius: 10px;
  border: 1px solid var(--border);
  backdrop-filter: blur(8px);
}
.centrality-selector button {
  background: transparent; border: none;
  color: var(--text-dim);
  padding: 6px 14px; border-radius: 7px;
  cursor: pointer; font-size: 0.78rem;
  font-weight: 500; font-family: inherit;
  transition: all 0.2s;
}
.centrality-selector button:hover { color: var(--text-bright); }
.centrality-selector button.active {
  background: var(--accent); color: white;
}

/* ── Diffusion section ── */
.diffusion-grid {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 1.5rem;
  margin-top: 1rem;
}
@media (max-width: 1000px) { .diffusion-grid { grid-template-columns: 1fr; } }
.diffusion-controls {
  display: flex; flex-wrap: wrap; gap: 0.8rem;
  align-items: center;
  padding: 1rem 1.2rem;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  margin-bottom: 1rem;
}
.scenario-toggle {
  display: flex;
  background: var(--bg);
  padding: 4px;
  border-radius: 10px;
  border: 1px solid var(--border);
}
.scenario-toggle button {
  background: transparent; border: none;
  color: var(--text-dim);
  padding: 8px 18px; border-radius: 7px;
  cursor: pointer; font-size: 0.85rem;
  font-weight: 500; font-family: inherit;
  transition: all 0.2s;
}
.scenario-toggle button.active {
  background: var(--accent); color: white;
}
.play-btn {
  background: var(--accent); color: white;
  border: none; padding: 8px 18px;
  border-radius: 8px; cursor: pointer;
  font-weight: 600; font-size: 0.85rem;
  font-family: inherit;
}
.play-btn:hover { filter: brightness(1.1); }
.timeline {
  flex: 1; min-width: 200px;
  display: flex; align-items: center; gap: 10px;
}
.timeline input[type=range] { flex: 1; accent-color: var(--accent); }
.timeline .step-label {
  font-size: 0.82rem;
  color: var(--text-dim);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.sir-stats {
  display: flex; gap: 1rem;
  padding: 0.6rem 1.2rem;
  font-size: 0.85rem;
}
.sir-stats .stat {
  display: flex; align-items: center; gap: 6px;
}
.sir-stats .dot {
  width: 10px; height: 10px; border-radius: 50%;
}
.sir-stats .stat-label { color: var(--text-dim); }
.sir-stats .stat-value {
  color: var(--text-bright); font-weight: 600;
  font-variant-numeric: tabular-nums;
}

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
</head>
<body>

<div class="tooltip" id="tooltip"></div>

<div class="hero">
  <h1>Modeling Information Spread in Networks</h1>
  <p class="subtitle">How information moves through a network of affiliations — who bridges
  communities, who is best positioned to move information, and how it spreads
  through informal channels versus public broadcast.</p>
  <div class="ethics-banner">Fictional demonstration dataset · See ETHICS.md</div>
</div>

<nav id="nav">
  <a href="#affiliation">Affiliation Network</a>
  <a href="#centrality">Centrality</a>
  <a href="#rankings">Rankings</a>
  <a href="#overlap">Entity Overlap</a>
  <a href="#structural">Structural Holes</a>
  <a href="#diffusion">Diffusion</a>
  <a href="#seeds">Seed Optimization</a>
  <a href="#comparison">Scenario Comparison</a>
</nav>

<div class="section" id="affiliation">
  <div class="section-header">
    <h2>Affiliation Network</h2>
    <p>People connected to the entities (conferences, podcasts, governance bodies, projects)
    they participate in. Click an entity to isolate its members.</p>
  </div>
  <div class="entity-legend" id="entityLegend"></div>
  <div class="graph-container" id="bipartiteGraph" style="margin-top:1rem; min-height:720px;">
    <div class="graph-controls">
      <button onclick="resetBipartite()" title="Reset">&#8634;</button>
    </div>
  </div>
</div>

<div class="section" id="centrality">
  <div class="section-header">
    <h2>Centrality Analysis</h2>
    <p>The person-to-person co-affiliation network, colored by centrality.
    Toggle between measures to see who the network's brokers, connectors, and influencers are.</p>
  </div>
  <div class="graph-container" id="centralityGraph" style="min-height:700px;">
    <div class="centrality-selector" id="centralitySelector">
      <button class="active" data-metric="composite">Composite</button>
      <button data-metric="betweenness">Betweenness</button>
      <button data-metric="closeness">Closeness</button>
      <button data-metric="eigenvector">Eigenvector</button>
    </div>
    <div class="color-legend">
      <span>Low</span>
      <canvas id="legendCanvas" width="120" height="10"></canvas>
      <span>High</span>
    </div>
    <div class="graph-controls">
      <button onclick="resetCentrality()" title="Reset">&#8634;</button>
    </div>
  </div>
</div>

<div class="section" id="rankings">
  <div class="section-header">
    <h2>Centrality Rankings</h2>
    <p>The top influencers under each measure. The composite score is a weighted blend
    that emphasizes brokerage — useful for finding who to seed information with.</p>
  </div>
  <div class="tables-row" id="tablesRow"></div>
</div>

<div class="section" id="overlap">
  <div class="section-header">
    <h2>Entity Overlap</h2>
    <p>How many people belong to each pair of entities? Heavy overlap reveals which
    communities are most tightly intertwined.</p>
  </div>
  <div class="heatmap-container">
    <div class="graph-container" id="heatmapGraph"></div>
  </div>
</div>

<div class="section" id="structural">
  <div class="section-header">
    <h2>Structural Holes</h2>
    <p>What happens to the network if you remove the top brokers? The fragmentation
    reveals how dependent the network is on a small number of people for information flow.</p>
  </div>
  <div class="metrics-row" id="metricsRow"></div>
  <div class="dual-graph" style="margin-top: 1.5rem;">
    <div class="graph-container" id="fullGraph" style="min-height:520px;">
      <div class="graph-label">Full Network</div>
    </div>
    <div class="graph-container" id="reducedGraph" style="min-height:520px;">
      <div class="graph-label">Without Top Brokers</div>
    </div>
  </div>
</div>

<div class="section" id="diffusion">
  <div class="section-header">
    <h2>Diffusion Simulation</h2>
    <p>An SIR epidemiological model simulates information spreading through the
    network. Susceptible (gray) &rarr; Informed (red) &rarr; Retained (dim).
    Watch a representative simulation play out, or scrub through it.</p>
  </div>
  <div class="diffusion-controls">
    <div class="scenario-toggle" id="scenarioToggle">
      <button class="active" data-scenario="gossip">Insider Gossip</button>
      <button data-scenario="announcement">Public Announcement</button>
    </div>
    <button class="play-btn" id="playBtn">Play</button>
    <div class="timeline">
      <input type="range" id="stepSlider" min="0" max="30" value="0">
      <span class="step-label" id="stepLabel">step 0</span>
    </div>
  </div>
  <div class="sir-stats">
    <div class="stat"><span class="dot" style="background:#5a6577"></span>
      <span class="stat-label">Susceptible</span><span class="stat-value" id="statS">0</span></div>
    <div class="stat"><span class="dot" style="background:#ec6a4d"></span>
      <span class="stat-label">Informed</span><span class="stat-value" id="statI">0</span></div>
    <div class="stat"><span class="dot" style="background:#9b7fe6"></span>
      <span class="stat-label">Retained</span><span class="stat-value" id="statR">0</span></div>
  </div>
  <div class="diffusion-grid">
    <div class="graph-container" id="diffusionGraph" style="min-height:560px;"></div>
    <div class="graph-container" id="sirCurveChart" style="min-height:560px; padding:1rem;"></div>
  </div>
</div>

<div class="section" id="seeds">
  <div class="section-header">
    <h2>Seed Optimization</h2>
    <p>If you could give exclusive information to N people to maximize spread,
    who would you pick? The greedy algorithm answers this — but the answer depends
    heavily on what kind of information it is.</p>
  </div>
  <div class="dual-graph" style="margin-top: 1rem;">
    <div class="graph-container" id="seedGraph" style="min-height:540px;">
      <div class="graph-label">Optimal Seeds (Insider Gossip)</div>
    </div>
    <div class="graph-container" id="marginalChart" style="min-height:540px; padding:1rem;"></div>
  </div>
</div>

<div class="section" id="comparison">
  <div class="section-header">
    <h2>Scenario Comparison</h2>
    <p>The same seed set produces dramatically different spread patterns when the
    information is framed as private gossip vs. a public announcement. Note how
    gossip reaches further through stronger ties even though it spreads more slowly.</p>
  </div>
  <div class="graph-container" id="comparisonChart" style="min-height:520px; padding:1rem;"></div>
</div>

<div style="text-align:center; padding: 3rem; color: var(--text-dim); font-size: 0.78rem; max-width: 800px; margin: 0 auto;">
  <strong>Demonstration · fictional data.</strong> Every person and entity in this
  dataset is invented to illustrate the analysis; none are real and any resemblance
  is coincidental. The diffusion model is a simplified abstraction — it does not
  capture trust, credibility, or platform dynamics. Results inform intuition,
  not strategy. See ETHICS.md for guardrails.
</div>

<script>
const DATA = __GRAPH_DATA__;

// ── Entity-type styling ──────────────────────────────────────────────────────
const ENTITY_TYPE_COLORS = {
  conference: '#ec6a4d',
  podcast: '#9b7fe6',
  standards_body: '#4ecb8d',
  board: '#2bbaa0',
  oss_project: '#4a9eed',
  vc_firm: '#d4a853',
  publication: '#e88dbb',
};
const ENTITY_TYPE_SHAPES = {
  conference: 'circle',
  podcast: 'hexagon',
  standards_body: 'diamond',
  board: 'square',
  oss_project: 'triangle',
  vc_firm: 'pentagon',
  publication: 'star',
};
const TYPE_LABELS = {
  conference: 'Conference',
  podcast: 'Podcast',
  standards_body: 'Standards Body',
  board: 'Board',
  oss_project: 'OSS Project',
  vc_firm: 'VC Firm',
  publication: 'Publication',
};

// ── Temperature colormap ─────────────────────────────────────────────────────
function tempColor(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [0.0,  [20, 30, 80]],
    [0.15, [30, 80, 160]],
    [0.3,  [40, 160, 180]],
    [0.45, [60, 190, 100]],
    [0.6,  [180, 200, 50]],
    [0.75, [240, 180, 40]],
    [0.88, [230, 100, 40]],
    [1.0,  [180, 30, 30]],
  ];
  let i = 0;
  while (i < stops.length - 1 && stops[i + 1][0] < t) i++;
  if (i >= stops.length - 1) return `rgb(${stops[stops.length-1][1].join(',')})`;
  const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
  const f = (t - t0) / (t1 - t0);
  return `rgb(${Math.round(c0[0] + f*(c1[0]-c0[0]))},${Math.round(c0[1] + f*(c1[1]-c0[1]))},${Math.round(c0[2] + f*(c1[2]-c0[2]))})`;
}

const legendCtx = document.getElementById('legendCanvas').getContext('2d');
for (let x = 0; x < 120; x++) {
  legendCtx.fillStyle = tempColor(x / 119);
  legendCtx.fillRect(x, 0, 1, 10);
}

// ── Polygon shapes for entities ──────────────────────────────────────────────
function symbolPath(shape, size) {
  const r = size;
  switch (shape) {
    case 'square':
      return `M${-r},${-r} L${r},${-r} L${r},${r} L${-r},${r} Z`;
    case 'diamond':
      return `M0,${-r*1.2} L${r*1.2},0 L0,${r*1.2} L${-r*1.2},0 Z`;
    case 'triangle':
      return `M0,${-r*1.2} L${r*1.1},${r*0.85} L${-r*1.1},${r*0.85} Z`;
    case 'hexagon': {
      const a = r;
      return `M${a},0 L${a/2},${a*0.866} L${-a/2},${a*0.866} L${-a},0 L${-a/2},${-a*0.866} L${a/2},${-a*0.866} Z`;
    }
    case 'pentagon': {
      const pts = [];
      for (let i = 0; i < 5; i++) {
        const a = -Math.PI/2 + i * 2*Math.PI/5;
        pts.push([Math.cos(a)*r, Math.sin(a)*r]);
      }
      return 'M' + pts.map(p => p.join(',')).join(' L') + ' Z';
    }
    case 'star': {
      const pts = [];
      for (let i = 0; i < 10; i++) {
        const a = -Math.PI/2 + i * Math.PI/5;
        const rr = i % 2 === 0 ? r : r * 0.45;
        pts.push([Math.cos(a)*rr, Math.sin(a)*rr]);
      }
      return 'M' + pts.map(p => p.join(',')).join(' L') + ' Z';
    }
    default: // circle (handled as <circle>)
      return null;
  }
}

const tooltip = d3.select('#tooltip');
function showTooltip(evt, html) {
  tooltip.html(html).style('opacity', 1)
    .style('left', (evt.clientX + 16) + 'px')
    .style('top', (evt.clientY - 10) + 'px');
}
function hideTooltip() { tooltip.style('opacity', 0); }

// Nav active state
const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('nav a');
new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      navLinks.forEach(a => a.classList.remove('active'));
      const link = document.querySelector(`nav a[href="#${e.target.id}"]`);
      if (link) link.classList.add('active');
    }
  });
}, { threshold: 0.3 }).observe = (function(orig){
  return orig;
})(IntersectionObserver.prototype.observe);
const navObs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      navLinks.forEach(a => a.classList.remove('active'));
      const link = document.querySelector(`nav a[href="#${e.target.id}"]`);
      if (link) link.classList.add('active');
    }
  });
}, { threshold: 0.3 });
sections.forEach(s => navObs.observe(s));

function personTooltipHTML(d) {
  const orgs = (d.entities || []).map(e =>
    `<span class="org-tag" style="background:${ENTITY_TYPE_COLORS[e.entity_type] || '#666'}22;color:${ENTITY_TYPE_COLORS[e.entity_type] || '#aaa'}">${e.entity_name}${e.role ? ' · ' + e.role : ''}</span>`
  ).join('');
  return `<div class="tt-name">${d.name || d.id}</div>
    ${d.affiliation ? `<div class="tt-affil">${d.affiliation}</div>` : ''}
    <div class="tt-row"><span class="tt-label">Composite</span><span class="tt-value">${(d.composite || 0).toFixed(3)}</span></div>
    <div class="tt-row"><span class="tt-label">Betweenness</span><span class="tt-value">${(d.betweenness || 0).toFixed(3)}</span></div>
    <div class="tt-row"><span class="tt-label">Closeness</span><span class="tt-value">${(d.closeness || 0).toFixed(3)}</span></div>
    <div class="tt-row"><span class="tt-label">Eigenvector</span><span class="tt-value">${(d.eigenvector || 0).toFixed(3)}</span></div>
    <div class="tt-row"><span class="tt-label">Affiliations</span><span class="tt-value">${(d.entities || []).length}</span></div>
    ${orgs ? `<div class="tt-orgs">${orgs}</div>` : ''}`;
}

// Default dimensions — fall back to these if container width isn't yet computed.
const DEFAULT_W = 1200;

// ── 1. AFFILIATION (BIPARTITE) GRAPH ────────────────────────────────────────
(function() {
  const container = document.getElementById('bipartiteGraph');
  const W = container.clientWidth || DEFAULT_W, H = 720;

  // Legend (one pill per entity type used)
  const legendEl = document.getElementById('entityLegend');
  const usedTypes = [...new Set(DATA.entities.map(e => e.type))];
  usedTypes.forEach(t => {
    const pill = document.createElement('div');
    pill.className = 'entity-pill';
    pill.innerHTML = `<span class="dot" style="background:${ENTITY_TYPE_COLORS[t]}"></span>${TYPE_LABELS[t] || t}`;
    pill.onclick = () => filterByType(t);
    legendEl.appendChild(pill);
  });

  const svg = d3.select(container).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  const defs = svg.append('defs');
  const glow = defs.append('filter').attr('id', 'glow');
  glow.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur');
  const merge = glow.append('feMerge');
  merge.append('feMergeNode').attr('in', 'blur');
  merge.append('feMergeNode').attr('in', 'SourceGraphic');

  const g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.3, 5]).on('zoom', e => g.attr('transform', e.transform)));

  // Build node lists
  const personMap = Object.fromEntries(DATA.persons.map(p => [p.id, p]));
  const entityMap = Object.fromEntries(DATA.entities.map(e => [e.id, e]));
  const nodes = [
    ...DATA.persons.map(p => ({...p, kind: 'person'})),
    ...DATA.entities.map(e => ({...e, kind: 'entity'})),
  ];
  const links = DATA.bipartiteEdges.map(d => ({...d}));

  // Pin entities radially
  const cx = W / 2, cy = H / 2;
  const entNodes = nodes.filter(n => n.kind === 'entity');
  entNodes.forEach((d, i) => {
    const a = (2 * Math.PI * i / entNodes.length) - Math.PI / 2;
    d.fx = cx + 280 * Math.cos(a);
    d.fy = cy + 280 * Math.sin(a);
  });

  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id)
      .distance(d => d.source.kind === 'entity' || d.target.kind === 'entity' ? 90 : 50)
      .strength(0.4))
    .force('charge', d3.forceManyBody().strength(d => d.kind === 'entity' ? -300 : -25))
    .force('center', d3.forceCenter(cx, cy).strength(0.05))
    .force('collision', d3.forceCollide().radius(d => d.kind === 'entity' ? 32 : 6))
    .alphaDecay(0.02);

  const link = g.append('g').selectAll('line')
    .data(links).join('line')
    .attr('stroke', d => {
      const ent = d.source.kind === 'entity' ? d.source : d.target;
      return ENTITY_TYPE_COLORS[ent.type] || '#444';
    })
    .attr('stroke-opacity', 0.18)
    .attr('stroke-width', d => 0.4 + (d.weight || 1) * 0.4);

  // Entity shapes
  const entityG = g.append('g').selectAll('g')
    .data(entNodes).join('g')
    .style('cursor', 'pointer')
    .on('mouseover', function(evt, d) {
      const memberCount = links.filter(l =>
        (typeof l.source === 'object' ? l.source.id : l.source) === d.id ||
        (typeof l.target === 'object' ? l.target.id : l.target) === d.id
      ).length;
      showTooltip(evt, `<div class="tt-name">${d.name}</div>
        <div class="tt-affil">${TYPE_LABELS[d.type] || d.type} · ${d.category || ''}</div>
        <div class="tt-row"><span class="tt-label">Members</span><span class="tt-value">${memberCount}</span></div>`);
    })
    .on('mouseout', hideTooltip)
    .on('click', (evt, d) => filterByEntity(d.id));

  entityG.each(function(d) {
    const sel = d3.select(this);
    const shape = ENTITY_TYPE_SHAPES[d.type] || 'circle';
    const baseSize = 16 + Math.sqrt(links.filter(l =>
      (typeof l.source === 'object' ? l.source.id : l.source) === d.id ||
      (typeof l.target === 'object' ? l.target.id : l.target) === d.id
    ).length) * 4;
    if (shape === 'circle') {
      sel.append('circle').attr('r', baseSize)
        .attr('fill', ENTITY_TYPE_COLORS[d.type])
        .attr('fill-opacity', 0.92)
        .attr('filter', 'url(#glow)');
    } else {
      sel.append('path').attr('d', symbolPath(shape, baseSize))
        .attr('fill', ENTITY_TYPE_COLORS[d.type])
        .attr('fill-opacity', 0.92)
        .attr('filter', 'url(#glow)');
    }
    sel.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', -baseSize - 6)
      .attr('fill', ENTITY_TYPE_COLORS[d.type])
      .attr('font-size', '10.5px')
      .attr('font-weight', 600)
      .attr('pointer-events', 'none')
      .text(d.name);
  });

  // Person nodes
  const personNodes = nodes.filter(n => n.kind === 'person');
  const personCircle = g.append('g').selectAll('circle')
    .data(personNodes).join('circle')
    .attr('r', d => 3 + Math.sqrt(d.entities ? d.entities.length : 1) * 1.6)
    .attr('fill', d => (d.entities && d.entities.length >= 4) ? '#d4a853' : '#7a8a9c')
    .attr('fill-opacity', 0.85)
    .attr('stroke', 'none')
    .style('cursor', 'pointer')
    .on('mouseover', function(evt, d) {
      d3.select(this).attr('stroke', '#fff').attr('stroke-width', 1.5);
      link.attr('stroke-opacity', l =>
        (l.source.id === d.id || l.target.id === d.id) ? 0.7 : 0.04);
      showTooltip(evt, personTooltipHTML(d));
    })
    .on('mouseout', function() {
      d3.select(this).attr('stroke', 'none');
      link.attr('stroke-opacity', 0.18);
      hideTooltip();
    });

  // Top-3 composite labels
  const top3 = [...personNodes].sort((a,b) => (b.composite||0) - (a.composite||0)).slice(0, 3);
  const topSet = new Set(top3.map(n => n.id));
  const topLabels = g.append('g').selectAll('text')
    .data(personNodes.filter(n => topSet.has(n.id))).join('text')
    .attr('text-anchor', 'middle')
    .attr('dy', -10)
    .attr('fill', '#ec6a4d')
    .attr('font-size', '11px')
    .attr('font-weight', 700)
    .attr('pointer-events', 'none')
    .text(d => d.name);

  sim.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    entityG.attr('transform', d => `translate(${d.x},${d.y})`);
    personCircle.attr('cx', d => d.x).attr('cy', d => d.y);
    topLabels.attr('x', d => d.x).attr('y', d => d.y);
  });

  personCircle.call(d3.drag()
    .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.1).restart(); d.fx = d.x; d.fy = d.y; })
    .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
  );

  function filterByEntity(entityId) {
    const memberIds = new Set();
    links.forEach(l => {
      const sid = typeof l.source === 'object' ? l.source.id : l.source;
      const tid = typeof l.target === 'object' ? l.target.id : l.target;
      if (sid === entityId || tid === entityId) {
        memberIds.add(sid); memberIds.add(tid);
      }
    });
    personCircle.attr('fill-opacity', d => memberIds.has(d.id) ? 1 : 0.07);
    link.attr('stroke-opacity', l => {
      const sid = typeof l.source === 'object' ? l.source.id : l.source;
      const tid = typeof l.target === 'object' ? l.target.id : l.target;
      return (sid === entityId || tid === entityId) ? 0.6 : 0.02;
    });
  }

  function filterByType(type) {
    const matches = new Set(DATA.entities.filter(e => e.type === type).map(e => e.id));
    const memberIds = new Set();
    links.forEach(l => {
      const sid = typeof l.source === 'object' ? l.source.id : l.source;
      const tid = typeof l.target === 'object' ? l.target.id : l.target;
      if (matches.has(sid) || matches.has(tid)) {
        memberIds.add(sid); memberIds.add(tid);
      }
    });
    personCircle.attr('fill-opacity', d => memberIds.has(d.id) ? 1 : 0.07);
    link.attr('stroke-opacity', l => {
      const sid = typeof l.source === 'object' ? l.source.id : l.source;
      const tid = typeof l.target === 'object' ? l.target.id : l.target;
      return (matches.has(sid) || matches.has(tid)) ? 0.6 : 0.02;
    });
  }

  window.resetBipartite = function() {
    personCircle.attr('fill-opacity', 0.85);
    link.attr('stroke-opacity', 0.18);
  };
})();

// ── 2. CENTRALITY GRAPH ──────────────────────────────────────────────────────
let currentMetric = 'composite';
let centralitySim, cNodes, cNodeCircles, cLabels, cSvgG, cLink;
let sirHighlight;  // declared early so drawSirCurve() can assign without TDZ error

(function() {
  const container = document.getElementById('centralityGraph');
  const W = container.clientWidth || DEFAULT_W, H = 700;
  const svg = d3.select(container).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  cSvgG = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.3, 6]).on('zoom', e => cSvgG.attr('transform', e.transform)));

  cNodes = DATA.persons.map(d => ({...d}));
  const cLinks = DATA.personEdges.filter(d => d.weight >= 1).map(d => ({...d}));

  centralitySim = d3.forceSimulation(cNodes)
    .force('link', d3.forceLink(cLinks).id(d => d.id).distance(70).strength(0.04))
    .force('charge', d3.forceManyBody().strength(-180).distanceMax(450))
    .force('center', d3.forceCenter(W/2, H/2).strength(0.05))
    .force('collision', d3.forceCollide().radius(10))
    .alphaDecay(0.01);

  cLink = cSvgG.append('g').selectAll('line')
    .data(cLinks).join('line')
    .attr('stroke', '#1e2a3a')
    .attr('stroke-opacity', 0.25)
    .attr('stroke-width', d => 0.3 + (d.weight || 1) * 0.07);

  cNodeCircles = cSvgG.append('g').selectAll('circle')
    .data(cNodes).join('circle')
    .attr('stroke', '#0a0e14')
    .attr('stroke-width', 0.5)
    .style('cursor', 'pointer')
    .on('mouseover', function(evt, d) {
      d3.select(this).attr('stroke', '#fff').attr('stroke-width', 2);
      cLink.attr('stroke-opacity', l =>
        (l.source.id === d.id || l.target.id === d.id) ? 0.6 : 0.05);
      showTooltip(evt, personTooltipHTML(d));
    })
    .on('mouseout', function() {
      d3.select(this).attr('stroke', '#0a0e14').attr('stroke-width', 0.5);
      cLink.attr('stroke-opacity', 0.25);
      hideTooltip();
    });

  cLabels = cSvgG.append('g').selectAll('text')
    .data(cNodes).join('text')
    .attr('text-anchor', 'middle')
    .attr('font-size', '10px')
    .attr('font-weight', 600)
    .attr('fill', '#fff')
    .attr('pointer-events', 'none')
    .attr('dy', -11)
    .attr('opacity', 0);

  centralitySim.on('tick', () => {
    cLink.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
         .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    cNodeCircles.attr('cx', d => d.x).attr('cy', d => d.y);
    cLabels.attr('x', d => d.x).attr('y', d => d.y);
  });

  cNodeCircles.call(d3.drag()
    .on('start', (e, d) => { if (!e.active) centralitySim.alphaTarget(0.1).restart(); d.fx = d.x; d.fy = d.y; })
    .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on('end', (e, d) => { if (!e.active) centralitySim.alphaTarget(0); d.fx = null; d.fy = null; })
  );

  updateCentralityColors();
})();

function updateCentralityColors() {
  const m = currentMetric;
  const vals = cNodes.map(d => d[m] || 0);
  const maxVal = Math.max(...vals) || 1;

  const sorted = [...cNodes].sort((a,b) => (b[m]||0) - (a[m]||0));
  const topSet = new Set(sorted.slice(0, 8).map(d => d.id));

  cNodeCircles
    .attr('r', d => 3 + ((d[m] || 0) / maxVal) * 14)
    .attr('fill', d => tempColor((d[m] || 0) / maxVal));

  cLabels
    .text(d => topSet.has(d.id) ? d.name : '')
    .attr('opacity', d => topSet.has(d.id) ? 1 : 0)
    .attr('fill', d => ((d[m]||0) / maxVal) > 0.5 ? '#fff' : '#c5cdd8');

  document.querySelectorAll('#centralitySelector button').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.metric === m);
  });
}

document.querySelectorAll('#centralitySelector button').forEach(btn => {
  btn.addEventListener('click', () => {
    currentMetric = btn.dataset.metric;
    updateCentralityColors();
  });
});

window.resetCentrality = () => centralitySim.alpha(0.5).restart();

// ── 3. RANKINGS TABLES ───────────────────────────────────────────────────────
(function() {
  const container = document.getElementById('tablesRow');
  const tables = [
    { title: 'Composite', desc: 'Best overall information broker', data: DATA.topComposite, color: '#ec6a4d' },
    { title: 'Betweenness', desc: 'Bridges between groups', data: DATA.topBetweenness, color: '#d4a853' },
    { title: 'Closeness', desc: 'Reaches everyone fast', data: DATA.topCloseness, color: '#4a9eed' },
    { title: 'Eigenvector', desc: 'Connected to the connected', data: DATA.topEigenvector, color: '#4ecb8d' },
  ];
  tables.forEach(t => {
    const maxScore = Math.max(...t.data.map(r => r.score));
    const rows = t.data.map((r, i) => {
      const pct = (r.score / maxScore * 100).toFixed(0);
      return `<tr>
        <td class="rank">${i + 1}</td>
        <td class="name">${r.name}</td>
        <td class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${pct}%;background:${t.color}"></div></div></td>
        <td class="score">${r.score.toFixed(3)}</td>
      </tr>`;
    }).join('');
    container.innerHTML += `
      <div class="ranking-table">
        <h3>${t.title}<span class="desc">${t.desc}</span></h3>
        <table>
          <thead><tr><th>#</th><th>Person</th><th></th><th>Score</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  });
})();

// ── 4. ENTITY HEATMAP ────────────────────────────────────────────────────────
(function() {
  const labels = DATA.entityNames;
  const labelTypes = DATA.entityTypes;
  const n = labels.length;
  const cellSize = 52;
  const margin = { top: 175, right: 30, bottom: 40, left: 235 };
  const W = margin.left + n * cellSize + margin.right;
  const H = margin.top + n * cellSize + margin.bottom;

  const svg = d3.select('#heatmapGraph').append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('width', W).attr('height', H)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', W + 'px').style('max-width', 'none');

  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const maxVal = Math.max(...DATA.heatmap.map(d => d.value));
  const x = d3.scaleBand().domain(labels).range([0, n * cellSize]).padding(0.06);
  const y = d3.scaleBand().domain(labels).range([0, n * cellSize]).padding(0.06);

  g.selectAll('rect.cell')
    .data(DATA.heatmap.filter(d => d.value > 0)).join('rect')
    .attr('class', 'cell')
    .attr('x', d => x(d.x)).attr('y', d => y(d.y))
    .attr('width', x.bandwidth()).attr('height', y.bandwidth())
    .attr('rx', 4)
    .attr('fill', d => tempColor(d.value / maxVal))
    .attr('opacity', d => 0.3 + 0.7 * (d.value / maxVal))
    .style('cursor', 'pointer')
    .on('mouseover', function(evt, d) {
      d3.select(this).attr('stroke', '#fff').attr('stroke-width', 2);
      showTooltip(evt, `<div class="tt-name">${d.y} &harr; ${d.x}</div>
        <div class="tt-row"><span class="tt-label">Shared people</span><span class="tt-value">${d.value}</span></div>`);
    })
    .on('mouseout', function() {
      d3.select(this).attr('stroke', 'none');
      hideTooltip();
    });

  g.selectAll('rect.empty')
    .data(DATA.heatmap.filter(d => d.value === 0 && d.x !== d.y)).join('rect')
    .attr('class', 'empty')
    .attr('x', d => x(d.x)).attr('y', d => y(d.y))
    .attr('width', x.bandwidth()).attr('height', y.bandwidth())
    .attr('rx', 4).attr('fill', '#0e1620')
    .attr('stroke', '#1a2434').attr('stroke-width', 0.5);

  g.selectAll('text.cellv')
    .data(DATA.heatmap.filter(d => d.value > 0)).join('text')
    .attr('class', 'cellv')
    .attr('x', d => x(d.x) + x.bandwidth()/2)
    .attr('y', d => y(d.y) + y.bandwidth()/2)
    .attr('text-anchor', 'middle').attr('dy', '0.35em')
    .attr('fill', d => d.value/maxVal > 0.5 ? '#fff' : '#8899aa')
    .attr('font-size', '14px').attr('font-weight', 600)
    .attr('pointer-events', 'none')
    .text(d => d.value);

  g.selectAll('text.xLabel')
    .data(labels).join('text')
    .attr('class', 'xLabel')
    .attr('x', d => x(d) + x.bandwidth()/2)
    .attr('y', -10).attr('text-anchor', 'end')
    .attr('transform', d => `rotate(-45, ${x(d) + x.bandwidth()/2}, -10)`)
    .attr('fill', d => ENTITY_TYPE_COLORS[labelTypes[d]] || '#8899aa')
    .attr('font-size', '12px').attr('font-weight', 500)
    .text(d => d);

  g.selectAll('text.yLabel')
    .data(labels).join('text')
    .attr('class', 'yLabel')
    .attr('x', -10).attr('y', d => y(d) + y.bandwidth()/2)
    .attr('text-anchor', 'end').attr('dy', '0.35em')
    .attr('fill', d => ENTITY_TYPE_COLORS[labelTypes[d]] || '#8899aa')
    .attr('font-size', '12px').attr('font-weight', 500)
    .text(d => d);
})();

// ── 5. STRUCTURAL HOLES ──────────────────────────────────────────────────────
(function() {
  const mc = document.getElementById('metricsRow');
  const fm = DATA.fullMetrics, rm = DATA.reducedMetrics;
  const metrics = [
    { label: 'Nodes', full: fm.nodes, reduced: rm.nodes },
    { label: 'Edges', full: fm.edges, reduced: rm.edges },
    { label: 'Components', full: fm.components, reduced: rm.components },
    { label: 'Density', full: fm.density, reduced: rm.density },
    { label: 'Clustering', full: fm.clustering, reduced: rm.clustering },
    { label: 'Largest CC', full: fm.largest_component, reduced: rm.largest_component },
  ];
  metrics.forEach(m => {
    const change = m.reduced - m.full;
    const pct = m.full !== 0 ? ((change / m.full) * 100).toFixed(1) : '0';
    const isUp = change > 0;
    mc.innerHTML += `
      <div class="metric-card">
        <div class="label">${m.label}</div>
        <div class="value">${m.full}</div>
        <div class="delta ${isUp ? 'up' : 'down'}">${isUp ? '+' : ''}${pct}% w/o brokers</div>
      </div>`;
  });

  function buildSubGraph(containerId, nodesData, edgesData) {
    const container = document.getElementById(containerId);
    const W = container.clientWidth || 580, H = 520;
    const svg = d3.select(container).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');
    const g = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.3, 5]).on('zoom', e => g.attr('transform', e.transform)));
    const nodes = nodesData.map(d => ({...d}));
    const links = edgesData.map(d => ({...d}));
    const maxBc = Math.max(...nodes.map(d => d.betweenness || 0), 0.001);
    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(60).strength(0.03))
      .force('charge', d3.forceManyBody().strength(-150).distanceMax(400))
      .force('center', d3.forceCenter(W/2, H/2).strength(0.06))
      .force('collision', d3.forceCollide().radius(7))
      .alphaDecay(0.012);
    const link = g.append('g').selectAll('line')
      .data(links).join('line')
      .attr('stroke', '#1e2a3a').attr('stroke-opacity', 0.18)
      .attr('stroke-width', d => 0.2 + (d.weight||1) * 0.05);
    const node = g.append('g').selectAll('circle')
      .data(nodes).join('circle')
      .attr('r', d => 3 + ((d.betweenness||0) / maxBc) * 9)
      .attr('fill', d => tempColor((d.betweenness||0) / maxBc))
      .attr('stroke', '#0a0e14').attr('stroke-width', 0.4)
      .style('cursor', 'pointer')
      .on('mouseover', function(evt, d) {
        d3.select(this).attr('stroke', '#fff').attr('stroke-width', 1.5);
        showTooltip(evt, `<div class="tt-name">${d.name || d.id}</div>
          <div class="tt-row"><span class="tt-label">Betweenness</span><span class="tt-value">${(d.betweenness||0).toFixed(3)}</span></div>`);
      })
      .on('mouseout', function() {
        d3.select(this).attr('stroke', '#0a0e14').attr('stroke-width', 0.4);
        hideTooltip();
      });
    const sortedNodes = [...nodes].sort((a,b) => (b.betweenness||0) - (a.betweenness||0));
    const topIds = new Set(sortedNodes.slice(0, 4).map(d => d.id));
    const lbl = g.append('g').selectAll('text')
      .data(nodes.filter(d => topIds.has(d.id))).join('text')
      .attr('text-anchor', 'middle').attr('dy', -10)
      .attr('fill', '#c5cdd8').attr('font-size', '9px')
      .attr('font-weight', 600).attr('pointer-events', 'none')
      .text(d => d.name || d.id);
    sim.on('tick', () => {
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      lbl.attr('x', d => d.x).attr('y', d => d.y);
    });
    node.call(d3.drag()
      .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.1).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
      .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
    );
  }

  buildSubGraph('fullGraph', DATA.persons, DATA.personEdges);
  buildSubGraph('reducedGraph', DATA.reducedNodes, DATA.reducedEdges);
})();

// ── 6. DIFFUSION SIMULATION ──────────────────────────────────────────────────
let diffSim, diffNodes, diffNodeCircles, diffLinkSel, currentScenarioKey = 'gossip';
let currentStep = 0, playInterval = null;

(function() {
  const container = document.getElementById('diffusionGraph');
  const W = container.clientWidth || 700, H = 540;
  const svg = d3.select(container).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');
  const g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.3, 5]).on('zoom', e => g.attr('transform', e.transform)));

  diffNodes = DATA.persons.map(d => ({...d}));
  // Filter to stronger ties so the diffusion graph isn't a hairball.
  // Person-person edges are normalized to [0,1]; >= 0.05 keeps the
  // structurally meaningful connections without saturating the canvas.
  const diffLinks = DATA.personEdges.filter(d => d.weight >= 0.05).map(d => ({...d}));

  diffSim = d3.forceSimulation(diffNodes)
    .force('link', d3.forceLink(diffLinks).id(d => d.id).distance(70).strength(0.05))
    .force('charge', d3.forceManyBody().strength(-180).distanceMax(450))
    .force('center', d3.forceCenter(W/2, H/2).strength(0.06))
    .force('collision', d3.forceCollide().radius(8))
    .alphaDecay(0.012);

  // Edges visible by default so you can read the network structure.
  // updateDiffusionEdges() recolors them on each step to highlight
  // active transmission paths.
  diffLinkSel = g.append('g').selectAll('line')
    .data(diffLinks).join('line')
    .attr('stroke', '#3a4a5e').attr('stroke-opacity', 0.45)
    .attr('stroke-width', d => 0.5 + (d.weight||1) * 1.2);

  diffNodeCircles = g.append('g').selectAll('circle')
    .data(diffNodes).join('circle')
    .attr('r', 6).attr('fill', '#5a6577')
    .attr('stroke', '#0a0e14').attr('stroke-width', 0.5)
    .style('cursor', 'pointer')
    .on('mouseover', function(evt, d) {
      d3.select(this).attr('stroke', '#fff').attr('stroke-width', 2);
      const ns = DATA.diffusion[currentScenarioKey].nodeStats[d.id] || {};
      showTooltip(evt, `<div class="tt-name">${d.name}</div>
        <div class="tt-row"><span class="tt-label">P(infected)</span><span class="tt-value">${(ns.prob_infected||0).toFixed(2)}</span></div>
        <div class="tt-row"><span class="tt-label">Mean inf. time</span><span class="tt-value">${ns.mean_inf_time != null ? ns.mean_inf_time.toFixed(1) : '—'}</span></div>`);
    })
    .on('mouseout', function() {
      d3.select(this).attr('stroke', '#0a0e14').attr('stroke-width', 0.5);
      hideTooltip();
    });

  diffSim.on('tick', () => {
    diffLinkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    diffNodeCircles.attr('cx', d => d.x).attr('cy', d => d.y);
  });

  diffNodeCircles.call(d3.drag()
    .on('start', (e, d) => { if (!e.active) diffSim.alphaTarget(0.1).restart(); d.fx = d.x; d.fy = d.y; })
    .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on('end', (e, d) => { if (!e.active) diffSim.alphaTarget(0); d.fx = null; d.fy = null; })
  );

  // Sync slider max to actual scenario length (HTML default is a placeholder)
  const initialMax = DATA.diffusion[currentScenarioKey].representative_run.sir_curve.length - 1;
  document.getElementById('stepSlider').max = initialMax;

  // SIR curve chart
  drawSirCurve();
  updateDiffusionStep(0);
})();

function nodeStateColor(state) {
  if (state === 'I') return '#ec6a4d';
  if (state === 'R') return '#9b7fe6';
  return '#5a6577';
}

function updateDiffusionStep(step) {
  const scenario = DATA.diffusion[currentScenarioKey];
  const maxIdx = scenario.representative_run.node_states.length - 1;
  step = Math.max(0, Math.min(step, maxIdx));
  currentStep = step;
  const stateMap = scenario.representative_run.node_states[step];
  diffNodeCircles
    .transition().duration(180)
    .attr('fill', d => nodeStateColor(stateMap[d.id]))
    .attr('r', d => stateMap[d.id] === 'I' ? 9 : 6);

  // Recolor edges based on endpoint states so transmission paths are legible.
  // Active transmission edges (Informed ↔ Susceptible) glow bright orange;
  // edges into informed/retained nodes are dimmer; the rest are muted so the
  // eye picks out the spread. Applied synchronously (no transition) because
  // the simulation tick handler clobbers transitioned attrs on this selection.
  if (diffLinkSel) {
    const edgeStyle = d => {
      const sId = typeof d.source === 'object' ? d.source.id : d.source;
      const tId = typeof d.target === 'object' ? d.target.id : d.target;
      const sS = stateMap[sId], tS = stateMap[tId];
      const base = 0.5 + (d.weight || 1) * 1.2;
      // Active transmission edge: I — S
      if ((sS === 'I' && tS === 'S') || (sS === 'S' && tS === 'I')) {
        return {color: '#ff8a4d', opacity: 0.85, width: base + 0.8};
      }
      // Informed cluster: I—I or I—R
      if (sS === 'I' || tS === 'I') {
        return {color: '#ec6a4d', opacity: 0.55, width: base};
      }
      // Retention spread: R—R or R—S
      if (sS === 'R' || tS === 'R') {
        return {color: '#7a6aa3', opacity: 0.35, width: base};
      }
      // Both still susceptible
      return {color: '#3a4a5e', opacity: 0.25, width: base};
    };
    diffLinkSel
      .attr('stroke', d => edgeStyle(d).color)
      .attr('stroke-opacity', d => edgeStyle(d).opacity)
      .attr('stroke-width', d => edgeStyle(d).width);
  }

  const counts = scenario.representative_run.sir_curve[step];
  document.getElementById('statS').textContent = counts.S;
  document.getElementById('statI').textContent = counts.I;
  document.getElementById('statR').textContent = counts.R;
  document.getElementById('stepLabel').textContent = `step ${step}`;
  document.getElementById('stepSlider').value = step;
  highlightSirCurveStep(step);
}

document.getElementById('stepSlider').addEventListener('input', e => {
  updateDiffusionStep(+e.target.value);
});

document.getElementById('playBtn').addEventListener('click', () => {
  const btn = document.getElementById('playBtn');
  if (playInterval) {
    clearInterval(playInterval); playInterval = null; btn.textContent = 'Play';
    return;
  }
  btn.textContent = 'Pause';
  // Source of truth is the data length, not the slider's max attribute
  const maxStep = DATA.diffusion[currentScenarioKey].representative_run.node_states.length - 1;
  if (currentStep >= maxStep) updateDiffusionStep(0);
  playInterval = setInterval(() => {
    if (currentStep >= maxStep) {
      clearInterval(playInterval); playInterval = null; btn.textContent = 'Play'; return;
    }
    updateDiffusionStep(currentStep + 1);
  }, 280);
});

document.querySelectorAll('#scenarioToggle button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#scenarioToggle button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentScenarioKey = btn.dataset.scenario;
    const maxSteps = DATA.diffusion[currentScenarioKey].representative_run.sir_curve.length - 1;
    document.getElementById('stepSlider').max = maxSteps;
    drawSirCurve();
    updateDiffusionStep(0);
  });
});

function drawSirCurve() {
  const container = document.getElementById('sirCurveChart');
  d3.select(container).selectAll('*').remove();
  const W = container.clientWidth || 500, H = 540;
  const margin = { top: 30, right: 25, bottom: 40, left: 50 };
  const innerW = W - margin.left - margin.right, innerH = H - margin.top - margin.bottom;
  const svg = d3.select(container).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const curve = DATA.diffusion[currentScenarioKey].sir_curve;
  const totalNodes = DATA.persons.length;
  const x = d3.scaleLinear().domain([0, curve.length-1]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, totalNodes]).range([innerH, 0]);
  // Axis
  g.append('g').attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(x).ticks(8))
    .selectAll('text').attr('fill', '#6b7a8d').attr('font-size', 11);
  g.append('g').call(d3.axisLeft(y).ticks(6))
    .selectAll('text').attr('fill', '#6b7a8d').attr('font-size', 11);
  g.selectAll('.domain, .tick line').attr('stroke', '#1e2a3a');
  g.append('text').attr('x', innerW/2).attr('y', innerH+32)
    .attr('text-anchor', 'middle').attr('fill', '#6b7a8d').attr('font-size', 11)
    .text('time step');
  g.append('text').attr('transform', `rotate(-90)`)
    .attr('x', -innerH/2).attr('y', -36)
    .attr('text-anchor', 'middle').attr('fill', '#6b7a8d').attr('font-size', 11)
    .text('# people');

  const line = (key) => d3.line().x(d => x(d.t)).y(d => y(d[key]));
  const area = (lower, upper) => d3.area()
    .x(d => x(d.t)).y0(d => y(d[lower])).y1(d => y(d[upper]));

  // Confidence bands
  const ICurve = curve.map(c => ({...c, I_lower: Math.max(0, c.I_mean - c.I_std), I_upper: c.I_mean + c.I_std}));
  const RCurve = curve.map(c => ({...c, R_lower: Math.max(0, c.R_mean - c.R_std), R_upper: c.R_mean + c.R_std}));

  g.append('path').datum(ICurve).attr('fill', '#ec6a4d').attr('opacity', 0.18)
    .attr('d', area('I_lower', 'I_upper'));
  g.append('path').datum(RCurve).attr('fill', '#9b7fe6').attr('opacity', 0.18)
    .attr('d', area('R_lower', 'R_upper'));

  // Mean lines
  g.append('path').datum(curve).attr('fill', 'none').attr('stroke', '#5a6577')
    .attr('stroke-width', 2).attr('d', d3.line().x(d=>x(d.t)).y(d=>y(d.S_mean)));
  g.append('path').datum(curve).attr('fill', 'none').attr('stroke', '#ec6a4d')
    .attr('stroke-width', 2.5).attr('d', d3.line().x(d=>x(d.t)).y(d=>y(d.I_mean)));
  g.append('path').datum(curve).attr('fill', 'none').attr('stroke', '#9b7fe6')
    .attr('stroke-width', 2).attr('d', d3.line().x(d=>x(d.t)).y(d=>y(d.R_mean)));

  // Step indicator
  sirHighlight = g.append('line')
    .attr('y1', 0).attr('y2', innerH)
    .attr('stroke', '#fff').attr('stroke-opacity', 0.4)
    .attr('stroke-dasharray', '3,3');

  // Legend
  const legend = g.append('g').attr('transform', `translate(${innerW - 130}, 10)`);
  const legendItems = [
    {label: 'Susceptible', color: '#5a6577'},
    {label: 'Informed (mean)', color: '#ec6a4d'},
    {label: 'Retained (mean)', color: '#9b7fe6'},
  ];
  legendItems.forEach((it, i) => {
    legend.append('rect').attr('x', 0).attr('y', i*18).attr('width', 14).attr('height', 4)
      .attr('fill', it.color).attr('rx', 2);
    legend.append('text').attr('x', 20).attr('y', i*18+5)
      .attr('fill', '#c5cdd8').attr('font-size', 11).text(it.label);
  });

  // Title
  svg.append('text').attr('x', W/2).attr('y', 18)
    .attr('text-anchor', 'middle').attr('fill', '#eef1f5')
    .attr('font-size', 13).attr('font-weight', 600)
    .text(`SIR Curve — mean over ${DATA.diffusion[currentScenarioKey].params.runs} runs`);
}

function highlightSirCurveStep(step) {
  if (!sirHighlight) return;
  const container = document.getElementById('sirCurveChart');
  const W = container.clientWidth || 500;
  const margin = { left: 50, right: 25 };
  const innerW = W - margin.left - margin.right;
  const curve = DATA.diffusion[currentScenarioKey].sir_curve;
  const x = d3.scaleLinear().domain([0, curve.length-1]).range([0, innerW]);
  sirHighlight.attr('x1', x(step)).attr('x2', x(step));
}

// ── 7. SEED OPTIMIZATION ─────────────────────────────────────────────────────
(function() {
  const container = document.getElementById('seedGraph');
  const W = container.clientWidth || 580, H = 540;
  const svg = d3.select(container).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');
  const g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.3, 5]).on('zoom', e => g.attr('transform', e.transform)));
  const nodes = DATA.persons.map(d => ({...d}));
  const links = DATA.personEdges.filter(d => d.weight >= 1).map(d => ({...d}));
  const seedSet = new Set(DATA.seeds.nodes);
  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(60).strength(0.04))
    .force('charge', d3.forceManyBody().strength(-160).distanceMax(450))
    .force('center', d3.forceCenter(W/2, H/2).strength(0.06))
    .force('collision', d3.forceCollide().radius(8))
    .alphaDecay(0.012);
  const link = g.append('g').selectAll('line')
    .data(links).join('line')
    .attr('stroke', '#1e2a3a').attr('stroke-opacity', 0.15)
    .attr('stroke-width', d => 0.2 + (d.weight||1) * 0.05);
  const node = g.append('g').selectAll('circle')
    .data(nodes).join('circle')
    .attr('r', d => seedSet.has(d.id) ? 14 : 5)
    .attr('fill', d => seedSet.has(d.id) ? '#ec6a4d' : '#5a6577')
    .attr('stroke', d => seedSet.has(d.id) ? '#fff' : '#0a0e14')
    .attr('stroke-width', d => seedSet.has(d.id) ? 2 : 0.4)
    .attr('opacity', d => seedSet.has(d.id) ? 1 : 0.6)
    .style('cursor', 'pointer')
    .on('mouseover', function(evt, d) {
      d3.select(this).attr('stroke', '#fff').attr('stroke-width', 2);
      const rank = DATA.seeds.nodes.indexOf(d.id);
      const seedNote = rank >= 0 ? `<div class="tt-row"><span class="tt-label">Seed rank</span><span class="tt-value">#${rank+1}</span></div>
        <div class="tt-row"><span class="tt-label">Marginal gain</span><span class="tt-value">+${DATA.seeds.marginalGains[rank].toFixed(2)}</span></div>` : '';
      showTooltip(evt, `<div class="tt-name">${d.name}</div>${seedNote}
        <div class="tt-row"><span class="tt-label">Composite</span><span class="tt-value">${(d.composite||0).toFixed(3)}</span></div>`);
    })
    .on('mouseout', function(_, d) {
      d3.select(this).attr('stroke', seedSet.has(d.id) ? '#fff' : '#0a0e14')
        .attr('stroke-width', seedSet.has(d.id) ? 2 : 0.4);
      hideTooltip();
    });
  const lbl = g.append('g').selectAll('text')
    .data(nodes.filter(d => seedSet.has(d.id))).join('text')
    .attr('text-anchor', 'middle').attr('dy', -18)
    .attr('fill', '#fff').attr('font-size', 11).attr('font-weight', 700)
    .attr('pointer-events', 'none').text(d => d.name);
  sim.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
    lbl.attr('x', d => d.x).attr('y', d => d.y);
  });
  node.call(d3.drag()
    .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.1).restart(); d.fx = d.x; d.fy = d.y; })
    .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
  );

  // Marginal gains chart
  const chart = document.getElementById('marginalChart');
  const cW = chart.clientWidth || 500, cH = 540;
  const margin = { top: 50, right: 20, bottom: 60, left: 220 };
  const innerW = cW - margin.left - margin.right;
  const innerH = cH - margin.top - margin.bottom;
  const csvg = d3.select(chart).append('svg')
    .attr('viewBox', `0 0 ${cW} ${cH}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');
  const cg = csvg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const seedNames = DATA.seeds.nodes.map(id => {
    const p = DATA.persons.find(p => p.id === id);
    return p ? p.name : id;
  });
  const gains = DATA.seeds.marginalGains;
  const xScale = d3.scaleLinear().domain([0, Math.max(...gains)]).range([0, innerW]);
  const yScale = d3.scaleBand().domain(seedNames).range([0, innerH]).padding(0.2);
  cg.append('g').attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(xScale).ticks(5))
    .selectAll('text').attr('fill', '#6b7a8d').attr('font-size', 11);
  cg.append('g').call(d3.axisLeft(yScale))
    .selectAll('text').attr('fill', '#c5cdd8').attr('font-size', 12).attr('font-weight', 500);
  cg.selectAll('.domain, .tick line').attr('stroke', '#1e2a3a');
  cg.selectAll('rect.bar')
    .data(gains).join('rect')
    .attr('class', 'bar')
    .attr('x', 0)
    .attr('y', (d, i) => yScale(seedNames[i]))
    .attr('width', d => xScale(d))
    .attr('height', yScale.bandwidth())
    .attr('fill', '#ec6a4d').attr('rx', 4);
  cg.selectAll('text.barv')
    .data(gains).join('text')
    .attr('class', 'barv')
    .attr('x', d => xScale(d) + 8)
    .attr('y', (d, i) => yScale(seedNames[i]) + yScale.bandwidth()/2)
    .attr('dy', '0.35em')
    .attr('fill', '#c5cdd8').attr('font-size', 11)
    .attr('font-variant-numeric', 'tabular-nums')
    .text(d => '+' + d.toFixed(2));
  csvg.append('text').attr('x', cW/2).attr('y', 22)
    .attr('text-anchor', 'middle').attr('fill', '#eef1f5')
    .attr('font-size', 14).attr('font-weight', 600)
    .text('Marginal Reach Gain per Seed');
  csvg.append('text').attr('x', cW/2).attr('y', 40)
    .attr('text-anchor', 'middle').attr('fill', '#6b7a8d')
    .attr('font-size', 11)
    .text(`Total expected reach: ${(DATA.seeds.expectedReach * 100).toFixed(0)}% of network`);
})();

// ── 8. SCENARIO COMPARISON ───────────────────────────────────────────────────
(function() {
  const container = document.getElementById('comparisonChart');
  const W = container.clientWidth || 1200, H = 500;
  const margin = { top: 50, right: 30, bottom: 50, left: 60 };
  const innerW = W - margin.left - margin.right, innerH = H - margin.top - margin.bottom;
  const svg = d3.select(container).append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const gossipCurve = DATA.diffusion.gossip.sir_curve;
  const annCurve = DATA.diffusion.announcement.sir_curve;
  const totalN = DATA.persons.length;
  const maxT = Math.max(gossipCurve.length, annCurve.length) - 1;
  const x = d3.scaleLinear().domain([0, maxT]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, totalN]).range([innerH, 0]);

  g.append('g').attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(x).ticks(10))
    .selectAll('text').attr('fill', '#6b7a8d');
  g.append('g').call(d3.axisLeft(y).ticks(6))
    .selectAll('text').attr('fill', '#6b7a8d');
  g.selectAll('.domain, .tick line').attr('stroke', '#1e2a3a');
  g.append('text').attr('x', innerW/2).attr('y', innerH+38)
    .attr('text-anchor', 'middle').attr('fill', '#6b7a8d').attr('font-size', 12)
    .text('time step');

  const lineFn = key => d3.line().x(d => x(d.t)).y(d => y(d[key]));

  // Cumulative reach = I + R
  const enrichG = gossipCurve.map(c => ({...c, reach: c.I_mean + c.R_mean}));
  const enrichA = annCurve.map(c => ({...c, reach: c.I_mean + c.R_mean}));

  // Gossip
  g.append('path').datum(enrichG).attr('fill', 'none')
    .attr('stroke', '#ec6a4d').attr('stroke-width', 3)
    .attr('d', lineFn('reach'));
  g.append('path').datum(gossipCurve).attr('fill', 'none')
    .attr('stroke', '#ec6a4d').attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,3').attr('opacity', 0.6)
    .attr('d', lineFn('I_mean'));

  // Announcement
  g.append('path').datum(enrichA).attr('fill', 'none')
    .attr('stroke', '#4a9eed').attr('stroke-width', 3)
    .attr('d', lineFn('reach'));
  g.append('path').datum(annCurve).attr('fill', 'none')
    .attr('stroke', '#4a9eed').attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,3').attr('opacity', 0.6)
    .attr('d', lineFn('I_mean'));

  // Title
  svg.append('text').attr('x', W/2).attr('y', 22)
    .attr('text-anchor', 'middle').attr('fill', '#eef1f5')
    .attr('font-size', 14).attr('font-weight', 600)
    .text('Cumulative Reach (solid) and Active Sharers (dashed)');

  // Legend
  const legend = svg.append('g').attr('transform', `translate(${margin.left + 10}, ${margin.top + 10})`);
  const items = [
    {label: 'Insider Gossip — cumulative reach', color: '#ec6a4d', dash: false},
    {label: 'Insider Gossip — actively sharing', color: '#ec6a4d', dash: true},
    {label: 'Public Announcement — cumulative reach', color: '#4a9eed', dash: false},
    {label: 'Public Announcement — actively sharing', color: '#4a9eed', dash: true},
  ];
  items.forEach((it, i) => {
    legend.append('line').attr('x1', 0).attr('x2', 26).attr('y1', i*22).attr('y2', i*22)
      .attr('stroke', it.color).attr('stroke-width', it.dash ? 1.5 : 3)
      .attr('stroke-dasharray', it.dash ? '4,3' : 'none');
    legend.append('text').attr('x', 34).attr('y', i*22+4)
      .attr('fill', '#c5cdd8').attr('font-size', 11).text(it.label);
  });

  // Final reach annotation
  const gossipFinal = enrichG[enrichG.length-1].reach;
  const annFinal = enrichA[enrichA.length-1].reach;
  g.append('text').attr('x', innerW - 6).attr('y', y(gossipFinal) - 4)
    .attr('text-anchor', 'end').attr('fill', '#ec6a4d')
    .attr('font-size', 12).attr('font-weight', 600)
    .text(`${gossipFinal.toFixed(0)}/${totalN} (${(gossipFinal/totalN*100).toFixed(0)}%)`);
  g.append('text').attr('x', innerW - 6).attr('y', y(annFinal) - 4)
    .attr('text-anchor', 'end').attr('fill', '#4a9eed')
    .attr('font-size', 12).attr('font-weight', 600)
    .text(`${annFinal.toFixed(0)}/${totalN} (${(annFinal/totalN*100).toFixed(0)}%)`);
})();
</script>
</body>
</html>
"""


def render_dashboard(graph_data: Dict, output_path: str | Path) -> Path:
    """Render the dashboard HTML to disk and return the path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(graph_data))
    output_path.write_text(html)
    return output_path
