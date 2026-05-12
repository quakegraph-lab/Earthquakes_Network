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

### Preferential Attachment

The Barabási–Albert model (1999) predicts that a network grows by preferential attachment:
a new edge incident on node *i* is drawn with probability proportional to its current
degree k_i — the *rich-get-richer* mechanism that generates a power-law degree distribution
with γ ≈ 3. Empirical verification requires measuring the *attachment kernel* π(k) —
the per-unit-time rate at which a node of degree *k* gains new edges — from the observed
edge-addition sequence. If π(k) ∝ k^α, then α = 1 recovers linear preferential attachment;
α < 1 indicates sub-linear growth (fitness or geographical constraints modulating the
attachment process); α > 1 indicates super-linear growth, associated with extreme hub
reinforcement and condensation.

The measurement follows the empirical estimator of Jeong, Néda & Barabási (2003). The
event sequence is replayed in chronological order. At each time step a directed edge is
added from the cell of event *i* to the cell of event *i*+1, and the degree of both
endpoints at the moment of attachment is recorded. The accumulated degree increments are
binned by pre-attachment degree *k* to form π(k). A power-law fit on log-log axes yields
the attachment exponent α; the best-fit curve and the α = 1 reference line (linear BA
prediction) are plotted for comparison. In the Abe–Suzuki cell-transition model nodes
(spatial cells) are fixed — not newly created — so the estimator is adapted to measure
how the rate at which an already-present cell attracts new transitions scales with its
accumulated degree. This adaptation is exact for multi-graph growth processes where the
degree of an existing node changes over time in response to new events.

The seismological interpretation of α is direct: α < 1 implies that seismic activity
distributes across fault zones in a manner tempered by spatial capacity or rupture-area
limits (sub-linear growth); α ≈ 1 recovers the BA universality class and supports the
pure preferential-attachment origin of the power-law exponent; α > 1 is consistent with
runaway aftershock sequences in which already-active cells continue to dominate (super-
linear reinforcement), a pattern expected transiently following M > 7 mainshocks such as
Tōhoku 2011.

*Reference:* Jeong H., Néda Z. & Barabási A.-L. (2003). Measuring preferential attachment
in evolving networks. *Europhysics Letters*, 61(4), 567–572.

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

Thirteen complementary centrality measures are computed on the directed, weighted 10 km
network. Each quantifies a distinct notion of importance within the stress-transfer
topology:

| Measure | Seismological interpretation |
|---------|------------------------------|
| In-Degree | *Susceptibility*: how often a cell is triggered by predecessor cells |
| Out-Degree | *Productivity*: how many distinct cells a cell triggers (aftershock output) |
| Degree | Total seismic activity rate — the most visited fault segments |
| PageRank | Steady-state stress-sink: cells that persistently receive seismic flow from well-connected predecessors |
| Harmonic | Topological reach via sum of inverse distances; handles disconnected nodes gracefully (closeness = 0 for unreachable nodes) |
| Closeness | Broadcast speed: cells from which influence propagates most rapidly across the network |
| Betweenness | Fault bridges: cells that mediate stress transfer between otherwise disconnected clusters |
| Eigenvector | Rich-club core: cells embedded in the most densely active neighbourhood |
| Katz | Indirect influence along all directed paths, with exponential length decay |
| HITS Hub score | Seismic triggers: cells whose activity tends to activate high-authority destinations |
| HITS Authority score | Primary destinations of seismic propagation chains |
| Clustering coefficient | Local fault-junction density: fraction of a cell's neighbours that are mutually connected |
| Triangle count | Raw feedback-loop count per cell; zero in a pure aftershock tree |

Spearman rank correlations across all 13 measures identify which notions of centrality
converge (high ρ, functionally redundant) vs diverge (low ρ, structurally orthogonal);
geographic maps allow direct comparison with known fault systems.

A **Bianconi–Barabási fitness** analysis follows the centrality section. The empirical
fitness estimator β̂_i = ln(k_i(T)) / ln(T/t_i) isolates the intrinsic seismogenic
potential of each cell from the first-mover advantage of preferential attachment. The
Lorenz condensation curve and Gini coefficient diagnose whether one fault zone has
captured a disproportionate share of the network's degree — the network-theoretic
signature of Bose-Einstein condensation (Bianconi & Barabási 2001).

The observed fitness distribution ρ(β̂) is compared against three theoretical regimes.
Under the *equal-fitness* (pure BA) regime all nodes share the same fitness, and ρ(β̂)
collapses to a spike at β̂ = 0.5. Under a *uniform-fitness* distribution — nodes drawn
independently from a flat prior — ρ(β̂) is approximately uniform on [0, γ−1]. In the
*Bose-Einstein condensation* limit a single node (the fittest fault zone) accumulates a
macroscopic fraction of all degree, appearing as an extreme outlier well beyond the bulk
of the distribution. The empirical ρ(β̂) is plotted alongside a kernel density estimate,
a vertical marker at β̂ = 0.5, and a shaded band spanning the uniform-fitness support;
an automated verdict classifies the network into the closest theoretical regime.

---

### Community Detection

Seven algorithms partition the network into communities — groups of cells with dense
internal connectivity relative to the rest of the network — each encoding a different
notion of cohesion:

- **Louvain** greedy modularity maximisation (Blondel et al. 2008), implemented via the
  Leiden algorithm (Traag, Waltman & van Eck 2019) with `leidenalg` / `igraph` for
  guaranteed well-connected communities.
- **Consensus Louvain** — 100 independent runs aggregated into a co-occurrence matrix;
  agglomerative clustering on the complement removes the stochastic instability of a
  single run.
- **Spectral clustering** via the *k* smallest eigenvectors of the normalised Laplacian
  (Jordan & Weiss 2002); *k* is set to the Louvain community count for comparability.
- **InfoMap** (Rosvall & Bergstrom 2008) — minimises the description length of a random
  walk; communities are regions where the walk remains trapped. Particularly informative
  for weighted directed networks where flow dynamics matter.
- **HDBSCAN-Spectral** (Campello et al. 2013) — density-based clustering applied in the
  normalised Laplacian spectral embedding. Unlike the four methods above, HDBSCAN requires
  no pre-specification of the number of communities: clusters emerge wherever the node
  density in embedding space exceeds a local threshold, as formalised by the mutual
  reachability distance. Noise points are assigned to the nearest cluster centroid.
- **HDBSCAN-Geographic** — the same HDBSCAN algorithm applied to projected node
  coordinates (kilometres), with no graph structure whatsoever. This provides a purely
  spatial baseline: communities are geographic density concentrations of seismic cells,
  independent of how those cells are connected.
- **BigCLAM** (Yang & Leskovec 2013) — the only method in the suite that produces
  *overlapping* communities. Each node holds a non-negative membership vector
  **F**_u ∈ ℝ^K, and the probability of a link between u and v is modelled as
  P(A_{uv}=1) = 1 − exp(−**F**_u · **F**_v). Parameters are estimated by coordinate
  ascent on the log-likelihood. The hard partition (argmax of **F**) enters the NMI
  comparison; the full **F** matrix reveals which cells simultaneously belong to
  multiple fault systems. Implemented from scratch following the equations in Yang &
  Leskovec (2013) and the educational reference implementation of Romijnders (2017).

The NMI between HDBSCAN-Spectral and HDBSCAN-Geographic is of particular scientific
interest: high agreement would imply that network community structure is largely an
artefact of geographic clustering; divergence — the more informative outcome — indicates
that the network encodes fault connectivity patterns that spatial proximity alone cannot
reproduce.

Partition agreement across all seven methods is quantified by a 7×7 Normalised Mutual
Information heatmap. Directed community detection is performed separately using the Leiden
algorithm with the Leicht–Newman directed modularity Q_d.

**Partition quality scoring.** All seven partitions are compared on a common set of nine
quality metrics: modularity Q (Newman–Girvan), conductance φ, coverage, normalised cut
N_cut, map equation description length L (Rosvall & Bergstrom 2008), degree-corrected SBM
log-likelihood (Karrer & Newman 2011), Surprise (Aldecoa & Marín 2013), geographic
compactness (mean haversine distance to community centroid, km), and depth coherence (mean
within-community standard deviation of focal depth, km). Results are presented as a
z-score-normalised heatmap, with metrics where lower values are better sign-reversed so
that brighter cells uniformly indicate better partition quality. This multi-criterion
comparison is more informative than any single measure and directly addresses whether the
detected communities correspond to coherent seismogenic zones.

*References:*  
Aldecoa R. & Marín I. (2013). Exploring the limits of community detection strategies in
complex networks. *Scientific Reports*, 3, 2216.  
Blondel V. D. et al. (2008). Fast unfolding of communities in large networks. *Journal of
Statistical Mechanics*, P10008.  
Campello R. J. G. B., Moulavi D. & Sander J. (2013). Density-based clustering based on
hierarchical density estimates. *PAKDD*, LNAI 7819, 160–172.  
Karrer B. & Newman M. E. J. (2011). Stochastic blockmodels and community structure in
networks. *Physical Review E*, 83, 016107.  
McInnes L., Healy J. & Astels S. (2017). hdbscan: Hierarchical density based clustering.
*Journal of Open Source Software*, 2(11), 205.  
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

Three complementary diagnostics extend the scalar *r* to a richer characterisation of
mixing structure:

**Average nearest-neighbour degree k_nn(k).** Rather than a single scalar, the function
k̄_nn(k) — the average degree of neighbours of a degree-*k* node — traces the mixing
profile across the full degree spectrum. On log-log axes, the slope μ of a power-law fit
k̄_nn(k) ∝ k^μ is the *degree-mixing exponent* (Pastor-Satorras, Vázquez & Vespignani
2001). A negative μ confirms disassortativity not merely on average but monotonically
across all degree classes; its magnitude characterises how steeply the tendency to avoid
hubs grows with degree. The exponent μ is reported alongside Newman's *r* in the summary
table.

Two finite-size cutoff thresholds are annotated on the k̄_nn(k) plot to disambiguate
structural from genuine disassortativity (Boguñá, Pastor-Satorras & Vespignani 2004). The
*structural cutoff* k_str = √N marks the degree beyond which, purely by the finite-size
constraint on multi-edges, hubs are forced to share neighbours — producing spurious
disassortativity that is an artefact of the sampling regime rather than a property of the
generating process. The *natural cutoff* k_nat = N^{1/(γ−1)}, derived from the power-law
exponent γ estimated by MLE, gives the finite-size truncation point of the degree
distribution itself: in an infinite network with the same γ, the tail would extend beyond
k_nat. Disassortativity observed above k_str should be interpreted with caution; below
k_str, the mixing exponent μ reflects the genuine tendency of mid-degree nodes to connect
to the periphery.

**Directed degree mixing.** Because the network is directed, four distinct mixing channels
exist: out→out (do cells that trigger many events preferentially connect to other prolific
triggers?), out→in, in→out, and in→in. These four panels, visualised as 2-D histograms in
log-log degree space, address a question with direct seismological content: whether
high-productivity fault cells activate other high-productivity cells or act as one-to-many
hubs feeding seismic activity into a passive periphery (Foster et al. 2010).

**Rich-club coefficient φ_norm(k).** The raw rich-club coefficient φ(k) measures the edge
density among all nodes with degree above threshold *k*. Normalised by the expectation
under the configuration model (obtained by averaging over 50 degree-preserving random
rewirings), φ_norm(k) > 1 indicates rich-club ordering — hubs are more densely
interconnected than degree alone predicts. Values φ_norm(k) < 1 across all *k* confirm
the absence of a rich club, consistent with the hub-periphery topology expected of
aftershock-driven networks (Colizza et al. 2006).

**Depth E-I index.** The E-I index (Krackhardt & Stern 1988) partitions edges into
External (crossing depth-layer boundaries) and Internal (within the same depth layer):
E-I = (E − I) / (E + I) ∈ [−1, 1]. Depth layers are defined as shallow (≤ 15 km),
intermediate (15–35 km), and deep (> 35 km), corresponding to the upper crust, lower
crust, and upper mantle seismogenic zones. E-I < 0 (homophily) indicates that seismic
cells predominantly trigger other cells at the same depth — evidence for depth-stratified
fault systems with limited cross-horizon energy transfer. E-I > 0 (heterophily) indicates
dominant cross-depth triggering, consistent with throughgoing fault systems or deep-to-
shallow stress transfer (e.g. deep subduction triggering shallow crustal seismicity in the
Calabrian arc or Cascadia subduction zone). The E-I index is reported alongside Newman's
*r*, the k_nn exponent μ, and the rich-club verdict in the same summary table.

*References:*  
Bianconi G. & Barabási A.-L. (2001). Bose-Einstein condensation in complex networks.
*Physical Review Letters*, 86, 5632–5635.  
Colizza V., Flammini A., Serrano M. A. & Vespignani A. (2006). Detecting rich-club
ordering in complex networks. *Nature Physics*, 2, 110–115.  
Foster J. G., Foster D. V., Grassberger P. & Paczuski M. (2010). Edge direction and the
structure of networks. *PNAS*, 107(24), 10815–10820.  
Krackhardt D. & Stern R. N. (1988). Informal networks and organizational crises: an
experimental simulation. *Social Psychology Quarterly*, 51(2), 123–140.  
Newman M. E. J. (2002). Assortative mixing in networks. *Physical Review Letters*, 89,
208701.  
Pastor-Satorras R., Vázquez A. & Vespignani A. (2001). Dynamical and correlation
properties of the Internet. *Physical Review Letters*, 87, 258701.

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
│   ├── centrality.py                 # 13 centrality measures + BB fitness + geographic maps
│   ├── assortativity.py              # Newman r, k_nn μ, directed mixing, rich-club, E-I index
│   ├── robustness.py                 # Random and targeted node removal
│   ├── community.py                  # Seven detection methods, partition quality scoring, NMI
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

Aldecoa R. & Marín I. (2013). Exploring the limits of community detection strategies in
complex networks. *Scientific Reports*, 3, 2216.

Baiesi M. & Paczuski M. (2004). Scale-free networks of earthquakes and aftershocks.
*Physical Review E*, 69, 066106.

Bandt C. & Pompe B. (2002). Permutation entropy: a natural complexity measure for time
series. *Physical Review Letters*, 88, 174102.

Bianconi G. & Barabási A.-L. (2001). Bose-Einstein condensation in complex networks.
*Physical Review Letters*, 86, 5632–5635.

Blondel V. D., Guillaume J.-L., Lambiotte R. & Lefebvre E. (2008). Fast unfolding of
communities in large networks. *Journal of Statistical Mechanics*, P10008.

Boguñá M., Pastor-Satorras R. & Vespignani A. (2004). Cut-offs and finite size effects in
scale-free networks. *European Physical Journal B*, 38(2), 205–209.

Clauset A., Shalizi C. R. & Newman M. E. J. (2009). Power-law distributions in empirical
data. *SIAM Review*, 51(4), 661–703.

Console R., Murru M. & Lombardi A. M. (2003). Refining earthquake clustering models.
*Journal of Geophysical Research*, 108(B10), 2468.

Colizza V., Flammini A., Serrano M. A. & Vespignani A. (2006). Detecting rich-club
ordering in complex networks. *Nature Physics*, 2, 110–115.

Donner R. V., Zou Y., Donges J. F., Marwan N. & Kurths J. (2010). Recurrence networks —
a novel paradigm for nonlinear time series analysis. *New Journal of Physics*, 12, 033025.

Foster J. G., Foster D. V., Grassberger P. & Paczuski M. (2010). Edge direction and the
structure of networks. *PNAS*, 107(24), 10815–10820.

Granovetter M. S. (1973). The strength of weak ties. *American Journal of Sociology*,
78(6), 1360–1380.

Jeong H., Néda Z. & Barabási A.-L. (2003). Measuring preferential attachment in evolving
networks. *Europhysics Letters*, 61(4), 567–572.

Krings G., Calabrese F., Ratti C. & Blondel V. D. (2009). Urban gravity: a model for
inter-city telecommunication flows. *Journal of Statistical Mechanics*, L07003.

Luque B., Lacasa L., Ballesteros F. & Luque J. (2009). Horizontal visibility graphs:
exact results for random time series. *Physical Review E*, 80, 046103.

Newman M. E. J. (2002). Assortative mixing in networks. *Physical Review Letters*, 89,
208701.

Ogata Y. (1988). Statistical models for earthquake occurrences and residual analysis for
point processes. *Journal of the American Statistical Association*, 83, 9–27.

Pastor-Satorras R., Vázquez A. & Vespignani A. (2001). Dynamical and correlation
properties of the Internet. *Physical Review Letters*, 87, 258701.

Pessa A. A. B. & Ribeiro H. V. (2019). Characterizing stochastic time series with ordinal
networks. *Physical Review E*, 100, 042304.

Romijnders R. (2017). bigclam — educational implementation of BigCLAM.
https://github.com/RobRomijnders/bigclam

Rosvall M. & Bergstrom C. T. (2008). Maps of random walks on complex networks reveal
community structure. *PNAS*, 105(4), 1118–1123.

Telesca L. & Lovallo M. (2012). Analysis of seismic sequences by using the method of
visibility graph. *Europhysics Letters*, 97, 50002.

Traag V. A., Waltman L. & van Eck N. J. (2019). From Louvain to Leiden: guaranteeing
well-connected communities. *Scientific Reports*, 9, 5233.

Wilson A. G. (1971). A family of spatial interaction models and their derivation.
*Environment and Planning A*, 3, 1–32.

Yang J. & Leskovec J. (2013). Overlapping community detection at scale: a nonnegative
matrix factorization approach. *ACM WSDM*, 587–596.

Zaliapin I. & Ben-Zion Y. (2008). Nonclassical earthquake statistics: from theory to
practice. *Pure and Applied Geophysics*, 165, 1–21.
