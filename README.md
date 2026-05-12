# Earthquake Network Analysis across Three Seismic Catalogs

**Replication and multi-catalog extension of the Abe–Suzuki (2004) earthquake network
methodology, applied to 40 years of seismicity records from the United States (USGS),
Italy (INGV), and Japan (JMA), 1985–2025.**

The study treats a seismic catalog as a directed weighted graph in which nodes are
discretised 3-D spatial cells and edges record the temporal succession of earthquakes
between them. This representation, introduced by Abe & Suzuki (2004) for Japanese
seismicity, encodes fault structure, stress-transfer pathways, and spatio-temporal
clustering in a single network object amenable to the full toolkit of complex-network
analysis. The present work extends the method to two additional tectonic regimes and
benchmarks it against seven alternative network-construction paradigms — spanning causal
triggering forests (Baiesi–Paczuski, Zaliapin–Ben-Zion, ETAS), visibility graphs
(Telesca–Lovallo NVG, Horizontal VG), and geometric or symbolic encodings (Recurrence
Network, Ordinal Transition Network). The Abe–Suzuki cell-transition model retains the
richest structural signal across all community-quality and centrality metrics, while the
alternative paradigms offer complementary perspectives on the temporal and dynamical
organisation of seismicity.

---

## Seismic Catalogs

| Catalog | Region | Period | Completeness threshold | Bounding box |
|---------|--------|--------|----------------------|--------------|
| USGS    | Conterminous United States | 1985–2025 | M ≥ 1.5 | lat [24.6, 50.0], lon [−125.0, −65.0] |
| INGV    | Italy and surrounding seas  | 1985–2025 | M ≥ 1.5 | lat [34, 48], lon [3, 22] |
| JMA (via ISC) | Japan and offshore regions | 1985–2025 | M ≥ 1.5 | lat [24, 46.5], lon [122, 154] |

Raw catalog files are not versioned. `data/*/data_info.txt` documents the exact download
parameters for each agency. The Japan catalog is retrieved from the ISC FDSN web service
with contributor filter `JMA` (`python download_japan_JMA.py`, approximately 30 minutes).

---

## Analysis Pipeline

### Preliminary Seismological Analysis

Each catalog is characterised independently before network construction. Event counts,
magnitude distributions, and depth profiles are examined as functions of time to verify
catalog stationarity and identify the completeness threshold (1985 for all three catalogs).
Interactive seismicity maps provide spatial context.

Two classical seismological laws are fitted for each catalog:

**Gutenberg–Richter law.** The cumulative frequency–magnitude relation
log₁₀ N(≥M) = a − bM is fitted by maximum likelihood at multiple upper-magnitude cutoffs
to assess sensitivity and estimate the magnitude of completeness M_c. The *b*-value (~1.0
globally) measures the relative proportion of large to small earthquakes; deviations from
unity signal stress heterogeneity or catalog incompleteness.

**Omori–Utsu law.** Aftershock productivity following a mainshock decays as
n(t) = K (t + c)^{−p}. The *p*-value (~1.0 globally) governs the speed of stress
relaxation; elevated values are associated with warmer or fluid-saturated crust. Fits are
performed for two well-recorded sequences per catalog: Amatrice 2016 (M 6.2) and
L'Aquila 2009 (M 6.3) for Italy; Loma Prieta 1989 (M 6.9) and Ridgecrest 2019 (M 7.1)
for the US; Kobe 1995 (M 6.9) and Tōhoku 2011 (M 9.0) for Japan.

---

### Network Construction — Abe–Suzuki Method

The catalog is sorted chronologically. Each event is projected into a 3-D grid of cubic
cells (side length 5 km or 10 km) using an equal-area coordinate reference system
(EPSG:5070 for the US, EPSG:32632 for Italy, EPSG:32654 for Japan). A directed edge is
drawn from the cell of event *i* to the cell of event *i*+1; self-loops represent
consecutive events within the same cell. Edge weight equals the number of such transitions.
Building at two spatial resolutions tests whether topological conclusions depend on the
discretisation scale.

*Reference:* Abe S. & Suzuki N. (2004). Scale-free network of earthquakes. *Europhysics
Letters*, 65(4), 581–586. https://doi.org/10.1209/epl/i2003-10108-1

---

### Degree Distribution and Power-Law Scaling

The degree distribution P(k) is estimated with linear binning, logarithmic binning
(probability density), and as a complementary cumulative distribution function (CCDF).
The power-law exponent γ is estimated by maximum likelihood following Clauset, Shalizi &
Newman (2009), with k_min = 10 as the lower tail threshold. The likelihood ratio test of
Clauset et al. compares the power-law fit against an exponential alternative; R > 0 with
p < 0.05 rejects the exponential. Super-linear k_max scaling and divergence of the second
moment ⟨k²⟩ are computed as additional scale-free diagnostics.

*Reference:* Clauset A., Shalizi C. R. & Newman M. E. J. (2009). Power-law distributions
in empirical data. *SIAM Review*, 51(4), 661–703.

---

### Null Model Comparison

Four synthetic benchmark graphs are constructed and compared against the empirical network
on degree distribution, clustering coefficient, average path length, and modularity:

- **Erdős–Rényi G(n, m)** — uniformly random, same node and edge counts; establishes the
  baseline expectation under the null hypothesis of no structure.
- **Barabási–Albert** preferential attachment — generates a power-law degree distribution
  with γ ≈ 3 by construction; tests whether scale-free topology alone accounts for the
  observed clustering.
- **Watts–Strogatz** small-world — ring lattice rewired with probability p = 0.1; the
  reference model for high clustering combined with short path lengths.
- **Configuration model** (Molloy–Reed) — preserves the exact empirical degree sequence;
  isolates topological properties that cannot be explained by the degree distribution alone,
  attributing the remainder to spatial and physical constraints.
- **Stochastic Block Model** fitted from the Louvain community partition — tests whether
  the detected community structure is sufficient to reproduce the heavy tail.

---

### Centrality Analysis

Eight complementary centrality measures are computed on the directed, weighted 10 km
network. Each quantifies a distinct notion of importance within the stress-transfer
topology:

| Measure | Seismological interpretation |
|---------|------------------------------|
| Degree | Seismic activity rate of a cell — the most visited fault segments |
| PageRank | Steady-state stress-sink: cells that persistently receive seismic flow |
| Closeness | Broadcast speed: cells from which influence propagates most rapidly across the network |
| Betweenness | Fault bridges: cells that mediate stress transfer between otherwise disconnected clusters |
| Eigenvector | Rich-club core: cells embedded in the most densely active neighbourhood |
| Katz | Indirect influence along all directed paths, with exponential length decay |
| HITS Hub score | Seismic triggers: cells whose activity tends to activate high-authority destinations |
| HITS Authority score | Primary destinations of seismic propagation chains |

Spearman rank correlations across measures identify which notions of centrality converge;
geographic maps allow direct comparison with known fault systems.

---

### Community Detection

Four algorithms partition the network into communities — groups of cells with dense
internal connectivity relative to the rest of the network — each encoding a different
notion of cohesion:

- **Louvain** greedy modularity maximisation (Blondel et al. 2008).
- **Consensus Louvain** — 100 independent runs aggregated into a co-occurrence matrix;
  agglomerative clustering on the complement removes the stochastic instability of a
  single run.
- **Spectral clustering** via the *k* smallest eigenvectors of the normalised Laplacian
  (Jordan & Weiss 2002); *k* is set to the Louvain community count for comparability.
- **InfoMap** (Rosvall & Bergstrom 2008) — minimises the description length of a random
  walk; communities are regions where the walk remains trapped. Particularly informative
  for weighted directed networks where flow dynamics matter.

Partition agreement is quantified by Normalised Mutual Information (NMI); a 4×4 pairwise
heatmap identifies the most structurally robust community boundaries. Directed community
detection is performed separately using the Leiden algorithm with the Leicht–Newman
directed modularity Q_d.

*References:*  
Blondel V. D. et al. (2008). Fast unfolding of communities in large networks. *Journal of
Statistical Mechanics*, P10008.  
Rosvall M. & Bergstrom C. T. (2008). Maps of random walks on complex networks reveal
community structure. *PNAS*, 105(4), 1118–1123.

---

### Assortativity

Newman's assortativity coefficient *r* (the Pearson correlation of a node attribute across
connected pairs) is computed for three node attributes: degree, mean cell depth, and mean
cell magnitude. Degree disassortativity (*r* < 0) is a hallmark of scale-free networks;
depth and magnitude assortativity reveal whether seismicity respects crustal depth horizons
and whether high-magnitude fault zones cluster in the network or are interspersed with
low-magnitude activity.

---

### Robustness

Nodes are removed iteratively under two strategies — uniform random removal (analogous to
catalog incompleteness or sensor failure) and targeted removal in descending degree order
(analogous to the suppression of the most active seismogenic zones). The fraction of nodes
retained in the giant connected component is recorded after each step. Scale-free networks
exhibit a characteristic asymmetry: robust under random removal but fragile under targeted
attack. The critical fraction f_c at which the giant component collapses provides a single
comparable fragility index across catalogs.

---

### Extended Analyses

**k-core decomposition.** The coreness of a node is the largest *k* such that it belongs
to a subgraph in which every node has degree ≥ *k*. The innermost core identifies the
densest seismogenic zone; the shell distribution maps the periphery of aftershock activity
relative to the mainshock core.

**Personalized PageRank (PPR) from a mainshock seed.** Standard PageRank distributes a
uniform random walker over all nodes. PPR starts the walker exclusively from the cell
containing a major mainshock, producing a network-theoretic estimate of the stress
field radiated by that event. Applied to the L'Aquila 2009 (M 6.3) and Loma Prieta 1989
(M 6.9) mainshocks.

**Signed network analysis.** Edges are labelled positive or negative according to whether
the magnitude of consecutive events increases or decreases. Heider structural balance
(the tendency of triangles to be positive-positive-positive or positive-negative-negative)
is tested against a random baseline. The analysis identifies spatial chains of
accelerating or decelerating seismicity.

**Granovetter weak ties.** Inter-community edges with low transition weight are identified
as weak ties in the sense of Granovetter (1973): they are structurally critical bridges
that, if severed, would disconnect tectonic provinces.

**Condensation graph (SCC analysis).** Tarjan's algorithm decomposes the directed network
into strongly connected components (SCCs). The condensation — the directed acyclic graph
of SCCs — separates seismic regions into source zones (that exclusively trigger others)
and sink zones (that exclusively receive).

**Community Markov flow.** The K × K transition matrix F between Louvain communities,
row-normalised to form a Markov chain, characterises inter-community seismic flow. The
self-retention probability T_ii, Shannon entropy H_i of the exit distribution, and the
stationary distribution π (computed by power iteration) quantify the persistence and
directionality of flow between tectonic provinces.

*Reference:* Rosvall M. & Bergstrom C. T. (2008), op. cit.

**Spatial interaction null model.** A gravity model λ_ij = C · A_i^α · A_j^β · d_ij^{−η}
is fitted by log-linear OLS to the directed edge weights, where A_i is node activity
(degree) and d_ij is inter-cell distance. The structural excess w_ij / λ_ij identifies
transitions that are significantly stronger than distance and activity alone can explain,
highlighting edges with a genuine tectonic or fluid-driven origin.

*References:* Wilson A. G. (1971). A family of spatial interaction models.
*Environment and Planning A*, 3, 1–32. Krings G. et al. (2009). Urban gravity: a model
for inter-city telecommunication flows. *Journal of Statistical Mechanics*, L07003.

**Link prediction as seismic forecasting.** The network is trained on events up to 2022
and tested on 2022–2025. Six topological similarity indices — Common Neighbours (CN),
Adamic–Adar (AA), Resource Allocation (RA), Jaccard coefficient, Katz index, and
Personalised PageRank — are evaluated by AUC-ROC. High AUC framing network topology as
a seismic forecasting tool motivates the connection between static network structure and
future seismic activity.

**Temporal multilayer analysis.** The catalog is divided into eight non-overlapping 5-year
windows. Per-window networks are characterised by γ, ⟨k⟩, C, and L. Partition stability
(NMI between consecutive Louvain partitions, restricted to shared nodes), hub persistence
(Jaccard overlap of top-20 PageRank cells), and edge turnover (Jaccard overlap of edge
sets) quantify the temporal evolution of the network's modular and centrality structure.

---

## Alternative Network Models — Italy

Seven alternative construction paradigms are implemented for the INGV catalog, each
encoding seismicity from a different theoretical perspective. All use the same raw catalog
as the ABE baseline, enabling direct structural comparison.

### Paradigm 1 — Causal / Triggering

Each earthquake is an individual node. A directed edge connects every event to at most one
parent, yielding a directed forest. The three models in this paradigm differ in how the
parent is selected.

**BP — Baiesi–Paczuski (2004).** Each event *j* is linked to the preceding event *i* that
minimises the metric η_ij = t_ij · r_ij^{d_f} · 10^{−b m_i}, where t_ij is the
inter-event time, r_ij the inter-event distance, d_f the fractal dimension of seismicity,
and m_i the magnitude of the putative parent. This produces a single spanning tree.

*Reference:* Baiesi M. & Paczuski M. (2004). Scale-free networks of earthquakes and
aftershocks. *Physical Review E*, 69, 066106.

**ZBZ — Zaliapin & Ben-Zion (2008).** The same η metric is computed, but a Gaussian
Mixture Model fitted to the bimodal distribution of log₁₀ η separates background events
(roots) from clustered events (children). The result is a forest of multiple trees, with
the background fraction (~20–40 %) providing an estimate of the rate of spontaneous
(non-triggered) seismicity.

*Reference:* Zaliapin I. & Ben-Zion Y. (2008). Nonclassical earthquake statistics: from
theory to practice. *Pure and Applied Geophysics*, 165, 1–21.

**ETAS — Epidemic Type Aftershock Sequence (Ogata 1988).** Each event *j* is assigned to
the parent *i* that maximises the triggering kernel κ_ij = κ₀ · 10^{α m_i} · (t_ij +
c)^{−p} · (r_ij² + d²)^{−q}. Parameters are calibrated on Italian seismicity (Console
et al. 2003). Background events are those for which the background rate μ exceeds all
κ_ij values.

*References:* Ogata Y. (1988). Statistical models for earthquake occurrences and residual
analysis for point processes. *Journal of the American Statistical Association*, 83,
9–27. Console R. et al. (2003). Refining earthquake clustering models. *Journal of
Geophysical Research*, 108(B10).

### Paradigm 2 — Visibility / Temporal

The magnitude time series is mapped to a network without reference to spatial coordinates.
Both models in this paradigm preserve only the temporal ordering and magnitude of events.

**TL — Telesca–Lovallo Natural Visibility Graph (2012).** Two events are connected if and
only if all intermediate events in the time series are below the straight line joining
their (time, magnitude) coordinates. Hub nodes correspond to the largest mainshocks.

*Reference:* Telesca L. & Lovallo M. (2012). Analysis of seismic sequences by using the
method of visibility graph. *Europhysics Letters*, 97, 50002.

**HVG — Horizontal Visibility Graph (Luque et al. 2009).** A stricter variant: event *i*
sees event *j* only if no intermediate event has magnitude exceeding min(m_i, m_j). The
resulting graph is sparser and has a theoretically known degree distribution for i.i.d.
series (geometric, P(k) = (1/3)(2/3)^{k−2}), providing an analytic null baseline.

*Reference:* Luque B. et al. (2009). Horizontal visibility graphs: exact results for
random time series. *Physical Review E*, 80, 046103.

### Paradigm 3 — Geometric / Symbolic

**RN — Recurrence Network (Donner et al. 2010).** Events are embedded in a normalised
5-dimensional feature space (time, longitude, latitude, depth, magnitude). Two events are
connected if their Euclidean distance is below a threshold ε, auto-selected to yield mean
degree ⟨k⟩ = 20. The topology reflects recurrent dynamical states in the phase space of
the seismic process.

*Reference:* Donner R. V. et al. (2010). Recurrence networks — a novel paradigm for
nonlinear time series analysis. *New Journal of Physics*, 12, 033025.

**OTN — Ordinal Transition Network (Bandt & Pompe 2002; Pessa & Ribeiro 2019).** Windows
of *d* = 4 consecutive magnitudes are mapped to their rank permutation (one of at most
d! = 24 ordinal patterns), and a directed edge is drawn between successive patterns. The
resulting network has at most 24 nodes; its topology characterises the sequential structure
of the magnitude series. The permutation entropy H_PE = 0.9875 (z = −34.8 relative to a
shuffled i.i.d. null) confirms significant non-randomness in the magnitude ordering of
Italian seismicity.

*References:* Bandt C. & Pompe B. (2002). Permutation entropy: a natural complexity
measure for time series. *Physical Review Letters*, 88, 174102. Pessa A. A. B. & Ribeiro
H. V. (2019). Characterizing stochastic time series with ordinal networks. *Physical
Review E*, 100, 042304.

---

## Key Findings

**Universal scale-free topology.** Despite distinct tectonic regimes, the log-binned
degree distributions of the Italian and US giant components yield near-identical power-law
exponents (γ_Italy = 2.03, γ_US = 2.04). Both networks satisfy the diagnostic criteria
for scale-free behaviour: super-linear k_max scaling (slopes 1.86 and 1.47, respectively)
and diverging second moment ⟨k²⟩. These results are consistent with Abe & Suzuki's
original finding for Japan and support the universality of the generating mechanism across
tectonic settings.

**Small-world structure.** Both networks exhibit clustering coefficients far above the
Erdős–Rényi baseline (Italy: C = 0.150, 167× ER; US: C = 0.344, 430× ER) with short
average path lengths (Italy: L = 3.31; US: L = 2.96), confirming the small-world
property. The US network is more clustered and more fragile under targeted hub removal
(critical fraction f_c ≈ 0.15 vs. 0.20 for Italy), consistent with its heavier degree
tail.

**Induced seismicity dominates the US backbone.** The highest-degree cell in the US
network (degree = 0.364) corresponds to The Geysers geothermal field in northern
California — an industrial fluid-injection site, not a tectonic fault. US top-10 hub
depths cluster near 0 km or below (an artefact of injection-well records), contrasting
sharply with Italy's top hubs at 5–6 km depth along the central Apennines. This structural
contrast between a tectonically dominated network (Italy) and one partly controlled by
induced seismicity (US) is the most original contribution of the multi-catalog extension.

**Magnitude assortativity contrast.** Italy shows positive magnitude assortativity
(r = +0.14): high-magnitude cells cluster together in the network, reflecting the compact
geometry of the Apennine fault corridor. The US value is near zero (r ≈ +0.04), diluted
by geographically dispersed induced seismicity at low magnitude. Degree assortativity is
negative in both catalogs (Italy: r = −0.095; US: r = −0.135), confirming the
disassortative mixing typical of scale-free networks.

**Temporal stationarity.** The US power-law exponent is stable across all 5-year windows
(γ = 1.74–1.78, flat). Italy shows a decreasing trend (γ = 1.82–2.21), interpreted as
improving catalog completeness over time. Edge Jaccard overlap between consecutive windows
is near zero in both catalogs (~0.001), indicating near-complete network renewal every
five years. Despite this, Italy's community partition NMI between consecutive windows
(mean = 0.104) is three times higher than the US value (0.032), suggesting that tectonic
fault zones reconstitute the same community structure even as individual edges turn over.

**Link prediction.** Using network topology trained on events up to 2022 to predict edges
in 2022–2025, Katz index and Personalised PageRank achieve AUC = 0.922 and 0.921 for
Italy and the US, respectively, with all six predictors exceeding AUC = 0.81. This
establishes a quantitative connection between static network structure and future seismic
activity.

---

## Installation

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate Earthquakes_Network
```

To update an existing environment after pulling changes:

```bash
conda env update -f environment.yml --prune
```

### Option B — pip

```bash
python -m venv venv_earthquakes
source venv_earthquakes/bin/activate   # Windows: venv_earthquakes\Scripts\activate
pip install -r requirements.txt
```

---

## Reproducing the Analysis

```bash
# Preliminary seismological analysis
python ITALY_preanalysis.py
python US_preanalysis.py
python JAPAN_preanalysis.py

# Main network pipeline (Abe–Suzuki)
python ITALY_network_ABE.py
python US_network_ABE.py
python JAPAN_network_ABE.py

# Alternative network models (Italy)
python ITALY_network_BP.py
python ITALY_network_TL.py
python ITALY_network_ZBZ.py
python ITALY_network_HVG.py
python ITALY_network_RN.py
python ITALY_network_OTN.py
python ITALY_network_ETAS.py

# Cross-catalog comparisons
python cross_catalog_comparison.py     # Italy vs US
python comparison_it_us_japan.py       # Italy vs US vs Japan

# Optional: temporal multilayer analysis (~15 min)
python extras_temporal.py

# Download Japan catalog (run once, ~30 min)
python download_japan_JMA.py
```

Each `.py` script can also be converted to a self-contained Jupyter notebook:

```bash
python convert_to_notebook.py ITALY_network_ABE.py notebooks/ITALY_network_ABE.ipynb
```

Interactive Plotly maps require a browser with WebGL2 support (Chrome recommended).

---

## Repository Structure

```
.
├── ITALY_network_ABE.ipynb
├── ITALY_network_BP.ipynb
├── ITALY_network_TL.ipynb
├── ITALY_network_ZBZ.ipynb
├── ITALY_network_HVG.ipynb
├── ITALY_network_RN.ipynb
├── ITALY_network_OTN.ipynb
├── ITALY_network_ETAS.ipynb
├── ITALY_preanalysis.ipynb
├── US_network_ABE.ipynb
├── US_preanalysis.ipynb
├── JAPAN_network_ABE.ipynb
├── JAPAN_preanalysis.ipynb
├── cross_catalog_comparison.ipynb
├── comparison_it_us_japan.ipynb
├── extras_temporal.ipynb
│
├── src/                              # Shared analysis modules
│   ├── network.py                    # Network builders (ABE + 7 alternative models)
│   ├── metrics.py                    # Power-law MLE, degree balance tests
│   ├── viz.py                        # Degree distribution plots
│   ├── seismology.py                 # Gutenberg–Richter, Omori–Utsu fitting
│   ├── nullmodels.py                 # ER, BA, WS, SBM, configuration model
│   ├── centrality.py                 # Eight centrality measures + geographic maps
│   ├── assortativity.py              # Newman assortativity (degree, depth, magnitude)
│   ├── robustness.py                 # Random and targeted node removal
│   ├── community.py                  # Louvain, Consensus, Spectral, InfoMap, NMI
│   ├── community_flow.py             # Markov chain over communities
│   ├── spatial_nulls.py              # Gravity null model, structural excess
│   ├── temporal.py                   # Temporal multilayer, partition/hub/edge stability
│   ├── signed.py                     # Signed network, structural balance
│   └── link_prediction.py            # AUC-ROC link prediction
│
├── data/
│   ├── USGS/                         # us_earthquakes_1985_2025.csv
│   ├── INGV/                         # italy_earthquakes_1985_2025.csv
│   └── JMA/                          # japan_earthquakes_jma_1985_2025_m1_5.csv
│
├── results/
│   ├── figures/                      # Saved figures (PDF, JPG, HTML)
│   ├── cache/                        # Pickled network objects
│   ├── gephi/                        # GEXF exports for Gephi
│   └── data/                         # Derived CSVs (metrics, community flow, etc.)
│
├── environment.yml
└── requirements.txt
```

---

## References

Abe S. & Suzuki N. (2004). Scale-free network of earthquakes. *Europhysics Letters*,
65(4), 581–586.

Baiesi M. & Paczuski M. (2004). Scale-free networks of earthquakes and aftershocks.
*Physical Review E*, 69, 066106.

Bandt C. & Pompe B. (2002). Permutation entropy: a natural complexity measure for time
series. *Physical Review Letters*, 88, 174102.

Blondel V. D., Guillaume J.-L., Lambiotte R. & Lefebvre E. (2008). Fast unfolding of
communities in large networks. *Journal of Statistical Mechanics*, P10008.

Clauset A., Shalizi C. R. & Newman M. E. J. (2009). Power-law distributions in empirical
data. *SIAM Review*, 51(4), 661–703.

Console R., Murru M. & Lombardi A. M. (2003). Refining earthquake clustering models.
*Journal of Geophysical Research*, 108(B10), 2468.

Donner R. V., Zou Y., Donges J. F., Marwan N. & Kurths J. (2010). Recurrence networks —
a novel paradigm for nonlinear time series analysis. *New Journal of Physics*, 12, 033025.

Granovetter M. S. (1973). The strength of weak ties. *American Journal of Sociology*,
78(6), 1360–1380.

Krings G., Calabrese F., Ratti C. & Blondel V. D. (2009). Urban gravity: a model for
inter-city telecommunication flows. *Journal of Statistical Mechanics*, L07003.

Luque B., Lacasa L., Ballesteros F. & Luque J. (2009). Horizontal visibility graphs:
exact results for random time series. *Physical Review E*, 80, 046103.

Newman M. E. J. (2002). Assortative mixing in networks. *Physical Review Letters*, 89,
208701.

Ogata Y. (1988). Statistical models for earthquake occurrences and residual analysis for
point processes. *Journal of the American Statistical Association*, 83, 9–27.

Pessa A. A. B. & Ribeiro H. V. (2019). Characterizing stochastic time series with ordinal
networks. *Physical Review E*, 100, 042304.

Rosvall M. & Bergstrom C. T. (2008). Maps of random walks on complex networks reveal
community structure. *PNAS*, 105(4), 1118–1123.

Telesca L. & Lovallo M. (2012). Analysis of seismic sequences by using the method of
visibility graph. *Europhysics Letters*, 97, 50002.

Wilson A. G. (1971). A family of spatial interaction models and their derivation.
*Environment and Planning A*, 3, 1–32.

Zaliapin I. & Ben-Zion Y. (2008). Nonclassical earthquake statistics: from theory to
practice. *Pure and Applied Geophysics*, 165, 1–21.
