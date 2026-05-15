# Earthquake Network Analysis — Italy INGV Catalog (Abe–Suzuki Method)

**Replication and analysis of the Abe–Suzuki (2004) earthquake network methodology
applied to 40 years of seismicity records from the Italian National Institute of
Geophysics and Volcanology (INGV), 1985–2025.**

The study treats the seismic catalog as a directed weighted graph in which nodes are
discretised 3-D spatial cells (10×10×10 km) and edges record the temporal succession of
earthquakes between them. This representation, introduced by Abe & Suzuki (2004) for
Japanese seismicity, encodes fault structure, stress-transfer pathways, and
spatio-temporal clustering in a single network object amenable to the full toolkit of
complex-network analysis.

---

## Seismic Catalog

| Catalog | Region | Period | Completeness threshold | Bounding box |
|---------|--------|--------|----------------------|--------------|
| INGV | Italy and surrounding seas | 1985–2025 | M ≥ 1.5 | lat [34, 48], lon [3, 22] |

The network is built on the filtered subset **M > 2.0, 2005–2015**, restricted to the
Italian mainland polygon (excluding North Africa and the Adriatic Sea perimeter). Raw
catalog files are not versioned. `data/INGV/data_info.txt` documents the exact download
parameters.

---

## Analysis Pipeline

### Network Construction — Abe–Suzuki Method

The catalog is sorted chronologically. Each event is projected into a 3-D grid of cubic
cells (side length 10 km) using the UTM Zone 32N coordinate reference system
(EPSG:32632). A directed edge is drawn from the cell of event *i* to the cell of event
*i*+1; self-loops represent consecutive events within the same cell. Edge weight equals
the number of such transitions.

*Reference:* Abe S. & Suzuki N. (2004). Scale-free network of earthquakes. *Europhysics
Letters*, 65(4), 581–586. https://doi.org/10.1209/epl/i2003-10108-1

---

### Geographic Visualisation

Two interactive Plotly maps are produced after construction:

**Hub Map — 2D.** The top 2% of cells by total degree are mapped onto an Italy basemap.
Marker size and colour both encode degree, using the `plasma` colorscale. These are the
most frequently visited fault segments in the earthquake sequence.

**Node Map — Degree & Depth.** Every active network node (cells with degree > 0) is
placed on the same basemap. Marker **size** encodes total degree; marker **colour**
encodes cell-centre depth (km, `plasma_r` colorscale, clipped at 200 km). Together, size
and colour reveal that most network hubs are shallow (seismogenic crust, 5–20 km), while
deep cells are relatively isolated — consistent with the Chiarabba *et al.* (2005)
Italy seismicity model.

---

### Degree Balance Verification

By construction, every interior node satisfies k_i^in = k_i^out: the cell that receives
the *n*-th event also sends the *(n+1)*-th. Only the first and last events in the catalog
produce a single unbalanced node each. A non-empty list of unbalanced nodes signals a
data-loading or sorting error.

---

### Degree Distribution and Power-Law Scaling

The degree distribution P(k) is estimated in four complementary representations:

- **Linear-scale histogram** (truncated at k = 50) to reveal the low-degree bulk.
- **Log-log scatter** of in-degree and out-degree separately, confirming the symmetry
  expected from degree balance.
- **Log-binned probability density** — degrees grouped into exponentially spaced bins
  and normalised by bin width, as recommended by Clauset *et al.* (2009) for visual
  assessment of the power-law tail.
- **Complementary CDF** P(K ≥ k), which avoids binning artefacts entirely and is
  monotone non-increasing by construction.

The power-law exponent γ is estimated by maximum likelihood (Clauset *et al.* 2009):

$$\hat{\gamma} = 1 + n\left[\sum_{i=1}^{n}\ln\frac{k_i}{k_{\min}}\right]^{-1}$$

with k_min = 10. The Clauset–Shalizi–Newman likelihood-ratio test compares the power-law
fit against an exponential alternative; R > 0 with p < 0.05 rejects the exponential. All
three distribution plots overlay the MLE-fitted curve for honest comparison.

*Reference:* Clauset A., Shalizi C. R. & Newman M. E. J. (2009). Power-law distributions
in empirical data. *SIAM Review*, 51(4), 661–703.

---

### Macroscopic Network Metrics

The small-world signature (Watts & Strogatz 1998) requires simultaneously
C_real ≫ C_ER (high local clustering) and L_real ≈ L_ER (short average path length),
where C_ER ≈ ⟨k⟩ / N and L_ER ≈ ln N / ln ⟨k⟩. Computed metrics include:

- **Giant component fraction** — values > 0.9 indicate a well-connected seismic system.
- **Average clustering coefficient** and Erdős–Rényi baseline.
- **Average shortest path length** and diameter on the undirected giant component.
- **Adjacency matrix sparsity** — visualised as a spy plot confirming the ultra-sparse
  structure characteristic of real-world networks.

---

### Centrality Analysis

Five complementary centrality measures capture distinct structural roles in the directed
seismic cell network:

| Measure | Seismological interpretation |
|---------|------------------------------|
| Degree | Total activity rate — the most visited fault segments |
| PageRank | Steady-state stress sinks: cells that persistently receive seismic flow from well-connected predecessors |
| Closeness | Broadcast speed — cells from which stress signals propagate most rapidly |
| Betweenness | Fault bridges — cells that mediate stress transfer between otherwise disconnected clusters |
| Clustering coefficient | Fault-junction density — fraction of a cell's neighbours that are mutually connected |

Spearman rank correlations across all five measures identify functionally redundant
(ρ > 0.9) vs structurally orthogonal (ρ < 0.3) pairs.

Two interactive geographic maps project the top-10 cells per metric onto the Italy
basemap. A **convergence map** colours each cell by the number of metrics for which it
ranks in the top 10, identifying structurally robust hubs.

---

### Community Detection

Five algorithms partition the network into communities, each encoding a different notion
of cohesion. The primary NMI comparison is a 5×5 matrix across all five methods; a
6th directed method is compared separately.

- **InfoMap** (Rosvall & Bergstrom 2008) — run on the *directed* giant component.
  Minimises the map equation description length of a random walk; communities are regions
  where the walk remains trapped. The conceptually correct primary method for a directed
  Markov-chain network.
- **Louvain** — Leiden algorithm (Traag *et al.* 2019) with resolution γ = 0.5
  (RB configuration vertex partition) on the undirected symmetrised giant component.
  Structural communities correspond to cells that frequently co-occur in the same
  aftershock cluster.
- **Consensus Louvain** (Lancichinetti & Fortunato 2012) — 100 Louvain runs aggregated
  into a co-occurrence matrix; final Louvain on the co-occurrence graph eliminates
  single-run stochastic instability.
- **Spectral clustering** — k-means on the k smallest eigenvectors of the normalised
  Laplacian (Jordan & Weiss 2002); k is set to the InfoMap community count.
- **HDBSCAN-Geographic** — density-based clustering on projected node coordinates (km)
  with no graph structure. Pure spatial baseline: communities are geographic density
  concentrations of seismic cells, independent of connectivity.

**Directed community detection.** An additional partition is produced by the Leiden
algorithm using the Leicht–Newman directed modularity Q_d, which groups cells that
*send and receive* seismic activity within the same community. Its NMI against the
undirected methods quantifies how much directionality reorganises community structure.

**Partition quality scoring.** All six partitions (five undirected/flow + directed Louvain)
are compared on nine quality metrics: modularity Q, conductance φ, coverage, normalised
cut N_cut, map equation description length L, degree-corrected SBM log-likelihood,
Surprise, geographic compactness (mean haversine distance to centroid, km), and depth
coherence (mean within-community depth standard deviation, km). Results are presented as
a z-score-normalised heatmap.

**Adjacency matrix community ordering.** Nodes are reordered by InfoMap community, then
by degree descending within each community, to reveal the block structure hidden in the
raw matrix. A companion block-density matrix shows the fraction of possible edges
realised between each pair of communities.

*References:*  
Blondel V. D. et al. (2008). Fast unfolding of communities in large networks. *Journal of
Statistical Mechanics*, P10008.  
Rosvall M. & Bergstrom C. T. (2008). Maps of random walks on complex networks reveal
community structure. *PNAS*, 105(4), 1118–1123.  
Traag V. A., Waltman L. & van Eck N. J. (2019). From Louvain to Leiden: guaranteeing
well-connected communities. *Scientific Reports*, 9, 5233.

---

### Assortativity

Newman's assortativity coefficient *r* is computed for three node attributes: degree,
mean cell depth, and mean cell magnitude. Four complementary diagnostics extend the
scalar summary:

**Average nearest-neighbour degree k_nn(k).** The slope μ of k̄_nn(k) ∝ k^μ on log-log
axes is the *degree-mixing exponent* (Pastor-Satorras *et al.* 2001). Two finite-size
cutoff thresholds are annotated: the structural cutoff k_str = √N (above which spurious
disassortativity arises from multi-edge constraints) and the natural cutoff
k_nat = N^{1/(γ−1)} (finite-size truncation of the degree distribution).

**Directed degree mixing.** Four mixing channels — out→out, out→in, in→out, in→in —
visualised as 2-D histograms in log-log degree space, addressing whether high-productivity
fault cells activate other high-productivity cells or feed into a passive periphery
(Foster *et al.* 2010).

**Rich-club coefficient φ_norm(k).** Raw φ(k) normalised by the configuration-model
expectation (50 degree-preserving rewirings). φ_norm < 1 across all *k* confirms the
absence of a rich club, consistent with hub-periphery aftershock-tree topology.

**Depth E-I index** (Krackhardt & Stern 1988). Depth layers: shallow (≤ 15 km),
intermediate (15–35 km), deep (> 35 km). E-I < 0 indicates seismic cells predominantly
trigger other cells in the same depth layer; E-I > 0 indicates dominant cross-depth
triggering.

*References:*  
Colizza V., Flammini A., Serrano M. A. & Vespignani A. (2006). Detecting rich-club
ordering in complex networks. *Nature Physics*, 2, 110–115.  
Foster J. G., Foster D. V., Grassberger P. & Paczuski M. (2010). Edge direction and the
structure of networks. *PNAS*, 107(24), 10815–10820.  
Krackhardt D. & Stern R. N. (1988). Informal networks and organizational crises.
*Social Psychology Quarterly*, 51(2), 123–140.  
Newman M. E. J. (2002). Assortative mixing in networks. *Physical Review Letters*, 89,
208701.  
Pastor-Satorras R., Vázquez A. & Vespignani A. (2001). Dynamical and correlation
properties of the Internet. *Physical Review Letters*, 87, 258701.

---

### Robustness

Nodes are removed iteratively under two strategies — uniform random removal and targeted
removal in descending degree order — and the fraction of nodes retained in the giant
connected component is recorded at each step. Both strategies are applied to the empirical
network and to a size-matched Erdős–Rényi baseline. Scale-free networks exhibit a
characteristic asymmetry: robust under random removal but fragile under targeted attack.
The critical fraction f_c at which the giant component collapses is the key fragility index.

*Reference:* Albert R., Jeong H. & Barabási A.-L. (2000). Error and attack tolerance of
complex networks. *Nature*, 406, 378–382.

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
# Run the full pipeline
python network_10km.py

# Convert to a self-contained Jupyter notebook
python convert_to_notebook.py network_10km.py notebooks/network_10km.ipynb
```

Interactive Plotly maps require a browser with WebGL2 support (Chrome recommended).

---

## Repository Structure

```
.
├── network_10km.py                   # Main analysis script (source of truth)
├── notebooks/
│   └── network_10km.ipynb            # Generated notebook (do not edit directly)
│
├── src/                              # Shared analysis modules
│   ├── network.py                    # Network builder (Abe–Suzuki)
│   ├── metrics.py                    # Power-law MLE, degree balance tests
│   ├── viz.py                        # Degree distribution plots
│   ├── centrality.py                 # 5 centrality measures + geographic maps
│   ├── assortativity.py              # Newman r, k_nn μ, directed mixing, rich-club, E-I index
│   ├── robustness.py                 # Random and targeted node removal
│   └── community.py                  # Six detection methods, partition quality scoring, NMI
│
├── data/
│   └── INGV/                         # italy_earthquakes_1985_2025.csv
│
├── results/
│   ├── figures/italy/abe/            # Saved figures (PDF, JPG, HTML)
│   ├── cache/italy_G10km.pkl         # Pickled network object
│   ├── gephi/italy_G10km.gexf        # GEXF export for Gephi
│   └── data/italy_eq_network_metrics.csv
│
├── convert_to_notebook.py            # Script → notebook converter
├── environment.yml
└── requirements.txt
```

---

## References

Abe S. & Suzuki N. (2004). Scale-free network of earthquakes. *Europhysics Letters*,
65(4), 581–586.

Albert R., Jeong H. & Barabási A.-L. (2000). Error and attack tolerance of complex
networks. *Nature*, 406, 378–382.

Blondel V. D., Guillaume J.-L., Lambiotte R. & Lefebvre E. (2008). Fast unfolding of
communities in large networks. *Journal of Statistical Mechanics*, P10008.

Chiarabba C. et al. (2005). The 2004 seismicity sequence in Italy: evidence for a new
Apennines stress regime. *Geophysical Research Letters*, 32, L23306.

Clauset A., Shalizi C. R. & Newman M. E. J. (2009). Power-law distributions in empirical
data. *SIAM Review*, 51(4), 661–703.

Colizza V., Flammini A., Serrano M. A. & Vespignani A. (2006). Detecting rich-club
ordering in complex networks. *Nature Physics*, 2, 110–115.

Foster J. G., Foster D. V., Grassberger P. & Paczuski M. (2010). Edge direction and the
structure of networks. *PNAS*, 107(24), 10815–10820.

Jordan M. I. & Weiss Y. (2002). On spectral clustering: analysis and an algorithm.
*Advances in Neural Information Processing Systems*, 14.

Krackhardt D. & Stern R. N. (1988). Informal networks and organizational crises: an
experimental simulation. *Social Psychology Quarterly*, 51(2), 123–140.

Lancichinetti A. & Fortunato S. (2012). Consensus clustering in complex networks.
*Scientific Reports*, 2, 336.

Leicht E. A. & Newman M. E. J. (2008). Community structure in directed networks.
*Physical Review Letters*, 100, 118703.

Newman M. E. J. (2002). Assortative mixing in networks. *Physical Review Letters*, 89,
208701.

Pastor-Satorras R., Vázquez A. & Vespignani A. (2001). Dynamical and correlation
properties of the Internet. *Physical Review Letters*, 87, 258701.

Rosvall M. & Bergstrom C. T. (2008). Maps of random walks on complex networks reveal
community structure. *PNAS*, 105(4), 1118–1123.

Traag V. A., Waltman L. & van Eck N. J. (2019). From Louvain to Leiden: guaranteeing
well-connected communities. *Scientific Reports*, 9, 5233.

Watts D. J. & Strogatz S. H. (1998). Collective dynamics of 'small-world' networks.
*Nature*, 393, 440–442.
