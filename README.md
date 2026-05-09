# US, Italy & Japan Earthquake Network Analysis

Final project for Social Network Analysis (Master's level). Replicates and extends the
[Abe-Suzuki (2004)](https://doi.org/10.1209/epl/i2003-10108-1) earthquake network methodology
on USGS (United States), INGV (Italy), and JMA (Japan) earthquake catalogs, 1985–2025.

**Core idea:** consecutive earthquakes are connected in a directed weighted graph where nodes are
discretised 3D spatial cells (5 km or 10 km cubes) and edge weight counts how many times the
sequence moved from one cell to another. This turns 40 years of seismic catalogs into networks
whose topology encodes fault structure, stress transfer, and clustering patterns.

---

## Installation

### Option A — conda (recommended)

```bash
conda env create -f environment.yml   # first time
conda activate Earthquakes_Network

# To update an existing env after pulling changes:
conda env update -f environment.yml --prune
```

`fa2-modified` is the only package not on conda-forge; conda installs it via pip automatically as part of the environment.

### Option B — pip (virtualenv)

```bash
python -m venv venv_earthquakes
source venv_earthquakes/bin/activate   # Windows: venv_earthquakes\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Activate your environment (conda or pip, see above)
conda activate Earthquakes_Network          # conda
# source venv_earthquakes/bin/activate # pip

# Launch JupyterLab
jupyter lab

# Or run any script directly
python US_network_ABE.py
python ITALY_network_ABE.py
python JAPAN_network_ABE.py
python ITALY_preanalysis.py
python US_preanalysis.py
python JAPAN_preanalysis.py
python cross_catalog_comparison.py
python comparison_it_us_japan.py   # 3-catalog comparison (run after Japan notebooks)
python extras_temporal.py          # optional: temporal multilayer analysis (~15 min)

# Download Japan catalog (run once, ~30 min)
python download_japan_JMA.py

# Re-generate a notebook from its source script
python convert_to_notebook.py US_network_ABE.py             notebooks/US_network_ABE.ipynb
python convert_to_notebook.py ITALY_network_ABE.py          notebooks/ITALY_network_ABE.ipynb
python convert_to_notebook.py JAPAN_network_ABE.py          notebooks/JAPAN_network_ABE.ipynb
python convert_to_notebook.py ITALY_preanalysis.py          notebooks/ITALY_preanalysis.ipynb
python convert_to_notebook.py US_preanalysis.py             notebooks/US_preanalysis.ipynb
python convert_to_notebook.py JAPAN_preanalysis.py          notebooks/JAPAN_preanalysis.ipynb
python convert_to_notebook.py cross_catalog_comparison.py   notebooks/cross_catalog_comparison.ipynb
python convert_to_notebook.py comparison_it_us_japan.py     notebooks/comparison_it_us_japan.ipynb
python convert_to_notebook.py extras_temporal.py            notebooks/extras_temporal.ipynb
```

---

## Project Structure

```
Earthquakes_Network/
├── notebooks/                          # Deliverable notebooks (generated from scripts)
│   ├── US_network_ABE.ipynb             # US    — Abe-Suzuki full pipeline (28 sections)
│   ├── ITALY_network_ABE.ipynb          # Italy — Abe-Suzuki full pipeline (28 sections)
│   ├── JAPAN_network_ABE.ipynb          # Japan — Abe-Suzuki full pipeline (28 sections)
│   ├── ITALY_preanalysis.ipynb          # Italy — EDA + GR law + Omori law
│   ├── US_preanalysis.ipynb             # US    — EDA + GR law + Omori law
│   ├── JAPAN_preanalysis.ipynb          # Japan — EDA + GR law + Omori (Kobe, Tohoku)
│   ├── cross_catalog_comparison.ipynb   # Italy vs US side-by-side (15 sections)
│   ├── comparison_it_us_japan.ipynb     # Italy vs US vs Japan (16 sections)
│   └── extras_temporal.ipynb           # Temporal multilayer: 5-year windows (13 sections)
│
├── src/                                # Shared Python modules (both catalogs)
│   ├── network.py                      # discretize_space_3d, build_abe_suzuki_network
│   ├── metrics.py                      # estimate_gamma_mle, test_power_law
│   ├── viz.py                          # degree distribution plots (5 functions)
│   ├── seismology.py                   # fit_gr_law, omori_law, fit_omori
│   ├── nullmodels.py                   # build_null_graphs, compare_metrics, plot_degree_comparison
│   ├── centrality.py                   # 8 measures + geo maps + correlation heatmap
│   ├── assortativity.py                # attach_catalog_attrs, compute_assortativity
│   ├── robustness.py                   # simulate_robustness, plot_robustness_curves
│   ├── community.py                    # Louvain, Consensus, Spectral, InfoMap, NMI
│   ├── community_flow.py               # Markov chain over communities; entropy, stationary π
│   ├── spatial_nulls.py                # Gravity null model; structural excess w/λ
│   └── temporal.py                     # 5-year temporal windows; partition/hub/edge stability
│
├── data/
│   ├── USGS/                           # us_earthquakes_1985_2025.csv (M≥1.5, CONUS)
│   ├── INGV/                           # italy_earthquakes_1985_2025.csv (M≥1.5, Italy)
│   └── Japan/                          # japan_earthquakes_jma_1985_2025_m1_5.csv (M≥1.5, JMA)
│
├── results/                            # Generated CSVs (gitignored)
├── report/                             # Written deliverable
│
├── US_network_ABE.py                    # Source script → US_network_ABE.ipynb
├── ITALY_network_ABE.py                 # Source script → ITALY_network_ABE.ipynb
├── JAPAN_network_ABE.py                 # Source script → JAPAN_network_ABE.ipynb
├── ITALY_preanalysis.py                 # Source script → ITALY_preanalysis.ipynb
├── US_preanalysis.py                    # Source script → US_preanalysis.ipynb
├── JAPAN_preanalysis.py                 # Source script → JAPAN_preanalysis.ipynb
├── cross_catalog_comparison.py          # Source script → cross_catalog_comparison.ipynb
├── comparison_it_us_japan.py            # Source script → comparison_it_us_japan.ipynb
├── download_japan_JMA.py                # Downloads JMA catalog from ISC FDSN
├── extras_temporal.py                   # Source script → extras_temporal.ipynb
└── convert_to_notebook.py              # .py → .ipynb converter (section-header based)
```

**Notebook workflow:** `.py` scripts are the source of truth. Edit the `.py`, then regenerate
the `.ipynb` with `convert_to_notebook.py`. Never edit notebooks directly.

---

## Data Sources

| Catalog | Region | Period | Min Mag | Bounding Box |
|---------|--------|--------|---------|--------------|
| USGS    | CONUS  | 1985–2025 | M≥1.5 | lat [24.6, 50.0], lon [-125.0, -65.0] |
| INGV    | Italy & seas | 1985–2025 | M≥1.5 | lat [34, 48], lon [3, 22] |
| JMA (via ISC) | Japan & offshore | 1985–2025 | M≥1.5 | lat [24, 46.5], lon [122, 154] |

Large CSV files are gitignored. `data/*/data_info.txt` documents download parameters.
Negative depths are valid (numerical surface drift in USGS catalog) and are kept as-is.

---

## Analysis Steps — Status & Methods

### STEP 1 — Preliminary Data Analysis

**Method:** Descriptive statistics and visualisation of the raw catalog — event counts per year
(bar chart), magnitude and depth distributions over time (boxen/boxplots), scatter and hexbin
density plots of magnitude vs depth, and an interactive Plotly mapbox seismicity map.

**What it tells us:** Reveals catalog completeness thresholds (we use 1985 as the cut year),
detects anomalous periods (e.g. instrument upgrades, policy changes), and characterises the
physical regime — whether seismicity is dominantly shallow crustal or includes deep subduction
events. Italy shows a pronounced shallow clustering (< 30 km) consistent with continental
collision tectonics; the US shows a bimodal depth signature with a deep subduction component
in the Pacific Northwest.

| Task | Italy | US |
|------|-------|----|
| Event counts per year (bar chart) | ✅ `ITALY_preanalysis.ipynb` | ✅ `US_preanalysis.ipynb` |
| Magnitude distribution per year (boxenplot) | ✅ | ✅ |
| Depth distribution per year (boxplot) | ✅ | ✅ |
| Magnitude vs depth (scatter, hexbin) | ✅ | ✅ |
| Interactive seismicity map | ✅ M≥5, Plotly mapbox | ✅ M≥4 |

---

### STEP 2 — Gutenberg-Richter Law & Omori Law

**Gutenberg-Richter law:** The cumulative number of earthquakes above magnitude M follows
log₁₀ N(≥M) = a − bM. The b-value (~1.0 globally) measures the ratio of small to large
earthquakes; deviations indicate stress heterogeneity or catalog incompleteness. We fit it
at multiple max-magnitude thresholds to test sensitivity and identify the magnitude of
completeness Mc.

**Omori-Utsu law:** After a mainshock, aftershock rate decays as n(t) = K / (t + c)^p.
The p-value (~1.0 globally) measures how fast the aftershock sequence dies off; K scales
the productivity; c is a regularisation constant that prevents divergence at t=0. We fit
this for two sequences per catalog to compare decay dynamics between Italy and the US.

**What it tells us:** The b-value comparison between Italy (~1.0–1.1) and the US shows
whether one crust is more heterogeneous. The Omori p-value comparison between L'Aquila /
Amatrice (Italy) and Loma Prieta / Ridgecrest (US) reveals differences in post-mainshock
stress relaxation — higher p means faster decay, typical of warmer or more fluid-saturated
crust.

| Task | Italy | US |
|------|-------|----|
| Gutenberg-Richter b-value fit (sensitivity to max_mag) | ✅ `ITALY_preanalysis.ipynb` | ✅ `US_preanalysis.ipynb` |
| Omori-Utsu fit — Amatrice 2016 (M6.2) | ✅ | — |
| Omori-Utsu fit — L'Aquila 2009 (M6.3) | ✅ | — |
| Omori-Utsu fit — Loma Prieta 1989 (M6.9) | — | ✅ |
| Omori-Utsu fit — Ridgecrest 2019 (M7.1) | — | ✅ |

---

### STEP 3 — Abe-Suzuki Network Construction

**Method:** The catalog is sorted by time. Each earthquake is projected into a 3D grid of
cubic cells (5 km or 10 km side) using an equal-area CRS (EPSG:5070 for CONUS, EPSG:32632
for Italy). A directed edge is added from the cell of event i to the cell of event i+1;
self-loops represent consecutive events in the same cell. Edge weight = number of such
transitions. Node attributes store the mean latitude/longitude of all events in that cell.

**What it tells us:** The resulting graph encodes the spatial flow of seismic activity.
High-weight edges identify persistent migration paths between fault segments. Self-loops
indicate cells where aftershock sequences are spatially confined. The directed structure
preserves temporal causality — an edge A→B means A preceded B, not the reverse. Building
at two resolutions (5 km and 10 km) tests whether conclusions are cell-size dependent.

| Task | Italy | US |
|------|-------|----|
| Network at 5×5×5 km | ✅ `ITALY_network_ABE.ipynb` | ✅ `US_network_ABE.ipynb` |
| Network at 10×10×10 km | ✅ | ✅ |
| Self-loops (same-cell consecutive events) | ✅ | ✅ |
| Directed graph with edge weight = transition count | ✅ | ✅ |
| Node attributes: mean lat/lon | ✅ | ✅ |

---

### STEP 4 — Basic Network Properties

**Method:** Standard structural characterisation. The degree distribution P(k) is plotted
with linear binning, logarithmic binning (probability density), and as a complementary
CDF (CCDF). The power-law exponent γ is estimated via Maximum Likelihood Estimation (MLE)
following Clauset et al. (2009), using k_min = 10 as the tail threshold. The Clauset-Shalizi-Newman
(CSN 2009) likelihood ratio test compares the power-law fit against an exponential alternative
(R > 0, p < 0.05 rejects the exponential). Giant component size, average path length (exact
on the GCC), and average clustering coefficient are computed and compared against a same-size
Erdős–Rényi baseline.

**What it tells us:** A heavy-tailed degree distribution with γ ≈ 2–3 confirms scale-free
behaviour: a small number of cells concentrate most seismic transitions. The small-world
signature (C_real >> C_ER, L_real ≈ L_ER) means the network is locally clustered (fault
zones) yet globally well-connected (stress can propagate across the whole region in few hops).
These properties match Abe-Suzuki's original findings for Japanese seismicity.

| Task | Italy | US |
|------|-------|----|
| Adjacency matrix (sparsity plot) | ✅ | ✅ |
| Degree distribution: linear-binned, log-binned, CCDF | ✅ | ✅ |
| Power-law exponent γ via MLE (Clauset et al. 2009) | ✅ | ✅ |
| CSN (2009) power law vs exponential likelihood ratio test | ✅ | ✅ |
| Giant component (size, % of nodes) | ✅ | ✅ |
| Average path length & diameter | ✅ | ✅ |
| Clustering coefficient (vs Erdős–Rényi baseline) | ✅ | ✅ |

---

### STEP 5 — Null Model Comparison

**Method:** Four synthetic graphs are generated and overlaid on the real network's degree
distribution and structural metrics:
- **Erdős–Rényi G(n,m):** same N nodes and M edges, connected uniformly at random. Provides
  the pure-random baseline (Poisson degree distribution, no clustering, short paths).
- **Barabási–Albert:** preferential attachment with the same average degree. Generates a
  power-law distribution with γ ≈ 3 by construction.
- **Watts–Strogatz:** regular ring rewired with probability p = 0.1. Produces high clustering
  and short paths — the small-world null.
- **SBM (Stochastic Block Model):** block structure fitted from the Louvain community partition.
  Tests whether communities alone explain the degree distribution.

**What it tells us:** If the real network's degree distribution matches BA more than ER, it
supports scale-free growth (richer cells get more connections, consistent with fault maturation).
If clustering exceeds both ER and BA baselines, the small-world structure is not explained by
degree heterogeneity alone and reflects genuine spatial clustering of seismicity. The SBM
comparison checks whether the community structure is tight enough to replicate the heavy tail.

| Task | Italy | US |
|------|-------|----|
| Erdős–Rényi G(n,m) | ✅ `ITALY_network_ABE.ipynb` | ✅ `US_network_ABE.ipynb` |
| Barabási–Albert preferential attachment | ✅ | ✅ |
| Watts–Strogatz small-world | ✅ | ✅ |
| SBM fitted from Louvain communities | ✅ | ✅ |
| Overlay null-model degree distributions (log-binned) | ✅ | ✅ |
| Comparative metrics table: C, L, γ, ⟨k⟩ | ✅ | ✅ |

---

### STEP 6 — Centrality Measures

**Method:** Eight complementary centrality measures are computed on the directed weighted
10 km network. Each captures a different notion of "importance":

- **Degree:** raw transition count (in + out degree). Identifies the most seismically active
  cells — where earthquakes occur most frequently.
- **PageRank:** random-walk steady-state probability. "Stress sinks" that persistently receive
  seismic flow regardless of where sequences start.
- **Closeness:** inverse of average shortest-path distance to all other nodes. Cells that can
  broadcast seismic influence across the network fastest.
- **Betweenness (k=1000 sampled):** fraction of shortest paths passing through a node.
  "Fault bridges" that mediate stress transfer between otherwise disconnected clusters — removing
  them would most fragment the network.
- **Eigenvector:** score proportional to neighbours' scores (computed on undirected version for
  stability). The rich-club core: cells connected to other highly-active cells.
- **Katz:** counts all directed paths with exponential length decay. More robust than
  eigenvector on sparse directed graphs; captures indirect influence.
- **HITS Hub:** cells that point to high-authority destinations. "Seismic triggers" — cells
  whose activity tends to activate important fault zones.
- **HITS Authority:** cells that receive links from high-hub sources. The primary destinations
  of seismic propagation chains.

Results are visualised as: (1) a Spearman rank-correlation heatmap showing which measures
agree; (2) top-10 bar charts colour-coded by depth; (3) interactive Plotly mapbox with a
dropdown to switch between metrics (top-10 nodes on Italy/US map); (4) a convergence map
where colour = number of top-10 lists a node appears in.

**What it tells us:** High agreement between PageRank and HITS Authority (both > 0.8 Spearman)
confirms that stress-sink identity is robust. Betweenness outliers with low degree identify
structurally critical bridges that are not necessarily the most active cells — these are the
most seismically dangerous locations to monitor. The geo maps allow direct comparison with
known fault systems.

| Measure | Italy | US |
|---------|-------|----|
| Degree | ✅ | ✅ |
| PageRank | ✅ | ✅ |
| Closeness | ✅ | ✅ |
| Betweenness (k=1000) | ✅ | ✅ |
| Eigenvector | ✅ | ✅ |
| Katz | ✅ | ✅ |
| HITS Hub | ✅ | ✅ |
| HITS Authority | ✅ | ✅ |
| Spearman correlation heatmap | ✅ | ✅ |
| Top-10 bar chart (depth-coded) | ✅ | ✅ |
| Interactive geo map (dropdown per metric) | ✅ | ✅ |
| Multi-metric convergence map | ✅ | ✅ |

---

### STEP 7 — Homophily / Assortativity

**Method:** Newman's assortativity coefficient r measures the Pearson correlation of an
attribute across connected node pairs. r = +1 means nodes only connect to similar nodes;
r = −1 means they exclusively connect to dissimilar nodes; r ≈ 0 is neutral. Three variants
are computed:
- **Degree assortativity:** are high-degree cells connected to other high-degree cells?
- **Depth assortativity:** do deep seismic cells preferentially trigger other deep cells?
- **Magnitude assortativity:** do high-magnitude regions tend to activate other high-magnitude
  regions?

**What it tells us:** Scale-free networks are typically *disassortative* in degree (hubs connect
to periphery, r < 0) — a signature found in technological and biological networks. If the
earthquake network is disassortative in degree, it reinforces the scale-free interpretation.
Depth assortativity near zero would suggest seismic sequences freely cross depth horizons;
positive depth assortativity would indicate that crustal and mantle seismicity form separate
flow regimes. Magnitude assortativity tests whether large-magnitude regions cluster in the
network or are interspersed with small-magnitude activity.

| Task | Italy | US |
|------|-------|----|
| Degree-degree assortativity (Newman r) | ✅ `ITALY_network_ABE.ipynb` | ✅ `US_network_ABE.ipynb` |
| Depth assortativity | ✅ | ✅ |
| Magnitude assortativity | ✅ | ✅ |
| 2D histogram mixing pattern plots | ✅ | ✅ |

---

### STEP 8 — Robustness Analysis

**Method:** Nodes are removed iteratively in two orders — uniformly random (simulates
background noise or sensor failure) and targeted by highest degree first (simulates a
deliberate attack or the removal of the most active seismic zones). After each removal
step, the fraction of nodes in the giant connected component (GCC) is recorded. An
Erdős–Rényi graph of the same size is used as the fragility baseline.

**What it tells us:** Scale-free networks show a characteristic asymmetry: robust under
random removal (many small-degree nodes can be removed before the GCC collapses) but
fragile under targeted attack (removing a few hubs quickly fragments the network). This
translates seismologically to: the seismic communication network is resilient to most
small earthquakes disappearing, but the loss of a few critical seismogenic zones (highest
degree nodes) would cut the network into isolated segments — potential precursors to
quiescence followed by large events.

| Task | Italy | US |
|------|-------|----|
| Random node removal → GCC fraction decay | ✅ `ITALY_network_ABE.ipynb` | ✅ `US_network_ABE.ipynb` |
| Targeted removal (highest-degree first) | ✅ | ✅ |
| Erdős–Rényi baseline comparison | ✅ | ✅ |
| GCC fraction vs fraction removed plot | ✅ | ✅ |

---

### STEP 9 — Community Detection

**Method:** Four algorithms are applied to the undirected giant component, each identifying
a different notion of "community":

- **Louvain:** greedy modularity maximisation. Fast and widely used; finds communities that
  maximise internal edge density relative to a random-graph null. Single run with seed=42.
- **Consensus Louvain:** 100 independent Louvain runs are aggregated into a co-occurrence
  matrix C[i,j] = fraction of runs where nodes i and j are in the same community. Agglomerative
  clustering on (1 − C) produces a stable partition that removes the stochastic instability
  of single-run Louvain.
- **Spectral clustering:** the k smallest eigenvectors of the normalised Laplacian define an
  embedding where geometrically close nodes belong to the same community. k-means is then
  applied in this embedding space (Jordan-Weiss method). k is set to the Louvain community
  count for comparability.
- **InfoMap:** minimises the description length of a random walk on the graph (map equation).
  Communities are regions where the walk stays trapped; this is especially meaningful for
  weighted directed networks where flow dynamics matter.

All four partitions are compared via a pairwise **NMI (Normalised Mutual Information)
heatmap**: NMI = 1 means two methods produce identical partitions; NMI = 0 means complete
disagreement.

**What it tells us:** Communities in the earthquake network should correspond to tectonic
provinces, seismic zones, or fault systems (e.g. Apennines vs Sicily in Italy; San Andreas
vs Cascadia in the US). High NMI between InfoMap and Louvain (> 0.7) means the modular
structure is robust and not an artefact of the algorithm. Communities that are stable across
all four methods identify the most structurally real seismic zones. Spectral clustering
using the Laplacian captures the finest-grain spatial partitioning, often splitting zones
that Louvain merges.

| Method | Italy | US |
|--------|-------|----|
| Louvain (single run) | ✅ `ITALY_network_ABE.ipynb` | ✅ `US_network_ABE.ipynb` |
| Consensus Louvain (100 runs + agglomerative) | ✅ `src/community.py` | ✅ |
| Spectral clustering (k-way Laplacian embedding) | ✅ `src/community.py` | ✅ |
| InfoMap (flow-based, undirected) | ✅ `src/community.py` | ✅ |
| NMI comparison heatmap across all four methods | ✅ `src/community.py` | ✅ |
| Directed Louvain (Leiden, Leicht-Newman Q_d) | ✅ `src/community.py` | ✅ |

---

### Cross-Catalog Comparison (US vs Italy)

**Method:** Both 10 km networks are rebuilt from raw CSVs in a single self-contained notebook
(`cross_catalog_comparison.ipynb`). Every structural quantity is computed identically for both
catalogs and displayed side-by-side. Centrality CSVs from the individual notebooks are loaded
if available, avoiding recomputation.

**What it tells us:** The comparison is the core scientific contribution of extending
Abe-Suzuki beyond the original Japanese catalog. Key findings: despite different tectonic
regimes, log-binned GCC fits collapse to γ_Italy = 2.03 ≈ γ_US = 2.04, confirming the
universality of the generating mechanism. US γ(t) is flat at ≈ 2.1 across all 40 years
(clean stationarity confirmation); Italy shows higher early variance due to catalog
incompleteness, converging after 2010. Both networks have GCC fraction = 1.0; the US is
larger (N = 29,773, ⟨k⟩ = 24.7) and more clustered (C = 0.344) than Italy (N = 16,635,
⟨k⟩ = 14.2, C = 0.150). The most seismologically significant cross-catalog difference is
magnitude assortativity: Italy r = +0.14 (high-magnitude cells cluster together, reflecting
the compact Apennine fault corridor) vs US r ≈ +0.04 (near-zero, diluted by geographically
dispersed induced seismicity at low magnitude). Robustness collapse under targeted attack
occurs earlier in the US (f_c ≈ 0.15) than Italy (f_c ≈ 0.20), consistent with the US
having a heavier degree tail (lower γ). Hub depth reveals a structural difference: Italy
top-10 hubs sit at 5–6 km (crustal faulting); US hubs cluster near 0 km or negative,
identifying induced seismicity injection cells. Both catalogs show super-linear k_max
scaling (Italy slope 1.86, US slope 1.47) and diverging ⟨k²⟩, confirming genuine
scale-free behavior with no finite-size cutoff. Link prediction AUC: Katz 0.922 / PPR
0.912 (Italy) and PPR = Katz = 0.921 (US), with all six predictors above 0.81 in both
catalogs — framing seismic network topology as a forecasting tool.

| Task | Status |
|------|--------|
| Catalog summary statistics | ✅ `cross_catalog_comparison.ipynb` |
| Degree distribution overlay (log-binned, both catalogs) | ✅ γ_Italy=2.03, γ_US=2.04 |
| γ comparison — 5 km and 10 km, MLE | ✅ |
| Macroscopic metrics table (N, M, ⟨k⟩, C, L, GCC%) | ✅ |
| Centrality score distributions (violin plots per metric) | ✅ |
| Hub depth comparison (mean depth of top-10 per metric) | ✅ Italy ≈5–6 km, US ≈0 km (induced seismicity) |
| Robustness fragility curves overlay | ✅ f_c Italy≈0.20, US≈0.15 |
| Community structure (k, modularity, NMI stability) | ✅ ~12 Louvain communities in both |
| Assortativity r values (degree, depth, magnitude) | ✅ magnitude r: Italy +0.14, US +0.04 |
| Temporal evolution γ(t) — year-by-year, both catalogs | ✅ |
| Scaling laws: k_max ~ N^{1/(γ-1)}, ⟨k²⟩ divergence | ✅ slopes: Italy 1.86, US 1.47 |
| Link prediction AUC (Katz, PPR, CN, AA, RA, Jaccard) | ✅ best AUC: Katz 0.922 (IT), PPR/Katz 0.921 (US) |

---

### Additional Analyses

**Method and rationale for all implemented analyses:**

- **k-core decomposition:** assigns each node its core number — the largest k such that the
  node belongs to a subgraph where every node has degree ≥ k. The innermost core (highest k)
  identifies the most densely interconnected seismic cells, which correspond to the main
  seismogenic zone rather than peripheral aftershock clusters. `nx.core_number()` runs in
  O(M); geographic distribution of core numbers directly maps fault depth and activity.

- **Personalized PageRank from a mainshock:** standard PageRank distributes a uniform random
  walker across all nodes; personalized PageRank starts the walker exclusively from one seed
  node (the cell containing a major mainshock). The resulting scores measure how much seismic
  stress propagates to each cell specifically from that event — a network-theoretic aftershock
  forecast. Applied to L'Aquila 2009 (Italy) and Loma Prieta 1989 (US).

- **Configuration model null:** a random graph that preserves the exact degree sequence of
  the real network (Molloy-Reed construction). Unlike ER (random degree) or BA (power-law),
  the configuration model tests whether topology beyond degree sequence — clustering, path
  length, community structure — is explained by the degree distribution alone or requires
  additional spatial/physical constraints.

- **Condensation graph (SCC analysis):** Tarjan's algorithm finds strongly connected
  components (SCCs) — sets of nodes that can reach each other via directed paths. The
  condensation is the DAG of SCCs. Source SCCs (no incoming edges from other SCCs) are
  seismic regions that only trigger others; sink SCCs only receive. This separates the
  network into "drivers" and "receptors" of seismic flow, which maps directly onto fault
  source vs target zones.

| Task | Status |
|------|--------|
| k-core decomposition / core-periphery structure | ✅ both catalogs |
| Personalized PageRank from major mainshock | ✅ L'Aquila 2009 / Loma Prieta 1989 |
| Configuration model null (5th null model) | ✅ `src/nullmodels.py` |
| Condensation graph / SCC source vs sink analysis | ✅ both catalogs |
| Granovetter weak ties (low-weight inter-community bridges) | ✅ both catalogs |
| Directed Louvain (Leiden, Leicht-Newman modularity) | ✅ both catalogs |
| Directed vs undirected NMI comparison | ✅ 5×5 heatmap both catalogs |
| Signed network (±edges by magnitude change) | ✅ `src/signed.py` |
| Link prediction as seismic forecasting (AUC) | ✅ `src/link_prediction.py` |
| Force-directed layout (community vs geography panels) | ✅ both catalogs |
| kmax scaling + ⟨k²⟩ moment divergence | ✅ `cross_catalog_comparison.ipynb` |
| Community Markov flow (K×K chain; self-retention, entropy, π) | ✅ `src/community_flow.py` — both catalogs |
| Spatial interaction null model (gravity; structural excess w/λ) | ✅ `src/spatial_nulls.py` — both catalogs |
| Temporal multilayer (5-year windows; partition/hub/edge stability) | ✅ `src/temporal.py` — `extras_temporal.ipynb`; edge Jaccard ≈ 0.001 (near-complete renewal); Italy community NMI 0.104 (3× higher than US 0.032); hub Jaccard = 0 Italy, ≈ 0.007 US |

---

### Visualisation

| Task | Status |
|------|--------|
| Interactive 2D hub map — Plotly `scatter_map` (tile basemap) | ✅ both catalogs |
| Interactive 3D hub depth map — Plotly `scatter_3d` | ✅ both catalogs |
| Interactive community map — `scatter_map` + `bounds` (no Alaska/Hawaii) | ✅ both catalogs |
| Interactive centrality geo map (dropdown per metric) | ✅ both catalogs |
| Multi-metric convergence map | ✅ both catalogs |
| Export network to Gephi (`.gexf`) | ✅ both catalogs → `results/*.gexf` |

---

## Infrastructure

| Component | Status |
|-----------|--------|
| `src/` modules shared by both catalogs (15 modules) | ✅ |
| EPSG:5070 for US, EPSG:32632 for Italy (never swapped) | ✅ |
| `convert_to_notebook.py` section-header + `#\|` docblock pipeline | ✅ |
| Git repository with CHANGELOG | ✅ |
| `environment.yml` — conda env `Earthquakes_Network` (conda-forge + pip fa2-modified) | ✅ |
| `requirements.txt` — pip/venv fallback with minimum version pins | ✅ |
| Figure save: `SAVE_PDF=True`, `SAVE_JPG=True` flags via `src/plotutils.py` | ✅ |
| `results/` organised: `figures/`, `cache/`, `gephi/`, `data/` | ✅ |
| Gephi export (`nx.write_gexf`) | ✅ |
| All deprecation warnings fixed (Plotly mapbox→map, matplotlib colormaps) | ✅ |
| Plotly renderer `"notebook"` in all scripts — no cross-notebook interference | ✅ |
| Notebooks run end-to-end without errors (both catalogs) | ✅ |
| Report outline + results report in `report/` | ✅ |
| `report/results_analysis.md` — figure-by-figure scientific commentary | ✅ |
| Plot fixes: excess map zoom, PPR log-size, k-core log x-axis, centrality colorbar removed | ✅ |
| All figures verified after notebook rerun — results confirmed | ✅ |
| Preanalysis renderer fixed (`"iframe"` → `"notebook"`) — both preanalysis notebooks regenerated | ✅ |
| Plotly maps require Chrome/WebGL2 (confirmed working) | ✅ |
| `scatter_map` viewport fix: `bounds` param added to all geo functions — no Alaska/Hawaii drift | ✅ |
| Signed geo map colorscale: `range_color` from data percentile + marker border — nodes visible | ✅ |
| `report/results_analysis.md` — full US figure commentary added (Italy + US both complete) | ✅ |
| All geo maps migrated to `scatter_map` tile basemap — `scatter_geo` fully removed | ✅ |
| US bounds = lat [24.6,50] / lon [-125,-65]; Italy = lat [34,48] / lon [3,22]; zoom IT=3, US=2 | ✅ |
| `community_count_comparison`: log y-scale + bar labels (Louvain vs InfoMap now both readable) | ✅ |
| `centrality_violin_comparison`: `symlog` → `log` + 1e-10 clip (negative-label artifact removed) | ✅ |
| Temporal stability plots: auto-scaled y-axis per panel + bar labels (Hub/Edge Jaccard ≈0 visible) | ✅ |
| Plot fixes (session 4): stationary dist. colormap inversion; weak-ties colorbar numeric ticks; NMI + centrality correlation full matrices; gravity fit axis clipped to scatter range; distance decay + binned median overlay; run-length bins dynamic; PPR render order (yellow on top); adjacency matrix legends | ✅ |
| Plot fixes (session 5): stationary dist. redundant colorbar removed (two-color scheme instead); distance decay binned median → mean + 90th pct; assortativity panels clipped to 99th pct (depth outliers no longer dominate); gravity fit y=x line thicker + arrow annotation | ✅ |
