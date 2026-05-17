"""
Community detection suite for the Abe-Suzuki earthquake network.

Five methods, all returning the same {node: community_id} dict so they can be
passed to any downstream function interchangeably:

  Louvain              — modularity optimisation via leidenalg/igraph (Leiden
                         algorithm); strictly better than the NetworkX implementation
  Consensus Louvain    — 100-run co-occurrence → consensus matrix → Louvain;
                         removes partition instability inherent to single-run Louvain
  Spectral             — k-way spectral clustering on the normalised Laplacian
                         (Jordan-Weiss); k taken from Louvain community count
  InfoMap              — flow-based compression (directed, weighted); identifies
                         communities as regions where random walkers stay trapped
  HDBSCAN-Geographic   — density-based clustering on projected (x, y) node
                         coordinates; communities = spatial density concentrations
                         independent of network topology

NMI utilities compare any pair of partitions; the heatmap shows pairwise
agreement across all methods.

Partition scoring: ``score_partition`` computes nine quality metrics for a
single partition (modularity Q, conductance, coverage, Ncut, map equation,
DC-SBM log-likelihood, Surprise, geographic compactness, depth coherence).
``compare_partitions`` scores all methods and returns a tidy DataFrame;
``plot_partition_scores`` renders a z-score-normalised heatmap ranked by Q.
"""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)

_PALETTE = px.colors.qualitative.Bold


# ── Internal helper ───────────────────────────────────────────────────────────

def _to_igraph(G: nx.Graph) -> tuple:
    """
    Convert a NetworkX graph to an igraph Graph for use with leidenalg.

    Returns (ig.Graph, node_list) where node_list[i] is the NetworkX node
    corresponding to igraph vertex i.  Edge weights are transferred if the
    ``"weight"`` attribute is present; otherwise all weights default to 1.0.
    """
    import igraph as ig

    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}
    directed = G.is_directed()

    g = ig.Graph(directed=directed)
    g.add_vertices(len(nodes))

    edges = [(node_to_int[u], node_to_int[v]) for u, v in G.edges()]
    weights = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    g.add_edges(edges)
    g.es["weight"] = weights

    return g, nodes


# ── Type alias ───────────────────────────────────────────────────────────────
Partition = dict[str, int]


# ── Method 1: single-run Louvain ─────────────────────────────────────────────

def run_louvain(G: nx.Graph, seed: int = 42, resolution: float = 1.0) -> Partition:
    """
    Louvain community detection via the Leiden algorithm (leidenalg / igraph).

    Maximises the Reichardt–Bornholdt modularity with resolution parameter
    :math:`\\gamma`:

    .. math::

        Q_{\\gamma} = \\frac{1}{2m}\\sum_{ij}\\left[A_{ij}
            - \\gamma\\frac{k_i k_j}{2m}\\right]\\delta(c_i, c_j).

    At :math:`\\gamma = 1` this is identical to Newman–Girvan modularity.
    Smaller :math:`\\gamma` (e.g. 0.5) merges more nodes into fewer, larger
    communities; larger :math:`\\gamma > 1` splits into finer communities.

    The optimisation uses the Leiden algorithm (Traag *et al.* 2019), which
    guarantees internally well-connected communities.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph (self-loops removed).
    seed : int
        Random seed for the Leiden optimiser.
    resolution : float
        Resolution parameter γ (default 1.0 = standard modularity).
        Lower values → fewer, larger communities.

    Returns
    -------
    Partition
        ``{node_id: community_int}`` mapping (0-indexed).

    References
    ----------
    Traag V. A., Waltman L. & van Eck N. J. (2019). From Louvain to Leiden:
    guaranteeing well-connected communities. *Scientific Reports*, 9, 5233.

    Reichardt J. & Bornholdt S. (2006). Statistical mechanics of community
    detection. *Physical Review E*, 74, 016110.
    """
    import leidenalg

    g, nodes = _to_igraph(G)
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights="weight", seed=seed, resolution_parameter=resolution,
    )
    log.info("Louvain (Leiden, γ=%.2f): %d communities, Q=%.4f",
             resolution, len(part), part.modularity)
    return {nodes[i]: cid for cid, members in enumerate(part) for i in members}


# ── Method 2: consensus Louvain ───────────────────────────────────────────────

def run_consensus_louvain(
    G: nx.Graph,
    n_runs: int = 100,
    seed: int = 42,
    max_nodes: int = 50_000,
    resolution: float = 1.0,
) -> Partition:
    """
    Consensus Louvain (Lancichinetti & Fortunato 2012): run Louvain ``n_runs``
    times, build the consensus matrix, then run Louvain once on that matrix.

    Step 1 — co-occurrence: C[i,j] = fraction of runs where i and j land in
    the same community (C ∈ [0, 1], diagonal excluded).
    Step 2 — consensus graph: treat C as a weighted undirected graph; edges
    with weight 0 are dropped (pairs never co-assigned are not connected).
    Step 3 — final partition: run Louvain once on the consensus graph.  The
    edge weights now encode community co-assignment probability, so the Louvain
    objective naturally groups nodes that consistently cluster together.

    For graphs with more than ``max_nodes`` nodes the N×N co-occurrence matrix
    would be prohibitively large. In that case the function falls back to a
    single Louvain run.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph (self-loops removed).
    n_runs : int
        Number of independent Louvain runs (≥ 50 recommended).
    seed : int
        Base seed; each run uses ``seed + run_index``.
    max_nodes : int
        Maximum graph size for full consensus. Larger graphs fall back to
        single Louvain. Default 50_000.

    Returns
    -------
    Partition
        ``{node_id: community_int}`` mapping.

    References
    ----------
    Lancichinetti A. & Fortunato S. (2012). Consensus clustering in complex
    networks. *Scientific Reports*, 2, 336.
    """
    import igraph as ig
    import leidenalg

    nodes = list(G.nodes())
    n = len(nodes)
    g, _ = _to_igraph(G)

    if n > max_nodes:
        mem_gib = n * n * 4 / 1024**3
        log.warning(
            "Consensus Louvain skipped: N=%d > max_nodes=%d "
            "(co-occurrence matrix would require %.0f GiB). "
            "Falling back to single Louvain run.",
            n, max_nodes, mem_gib,
        )
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights="weight", seed=seed, resolution_parameter=resolution,
        )
        return {nodes[i]: cid for cid, members in enumerate(part) for i in members}

    # Step 1: build co-occurrence matrix
    co = np.zeros((n, n), dtype=np.float32)
    log.info("Consensus Louvain: %d runs (γ=%.2f)...", n_runs, resolution)
    for r in range(n_runs):
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights="weight", seed=seed + r, resolution_parameter=resolution,
        )
        for members in part:
            ids = list(members)
            for i in ids:
                co[i, ids] += 1.0
    co /= n_runs
    np.fill_diagonal(co, 0.0)

    # Step 2: build consensus graph (upper triangle only, drop zero edges)
    rows, cols = np.where(co > 0)
    mask = rows < cols
    edges   = list(zip(rows[mask].tolist(), cols[mask].tolist()))
    weights = co[rows[mask], cols[mask]].tolist()
    g_cons = ig.Graph(n=n, edges=edges, directed=False)
    g_cons.es["weight"] = weights
    log.info("  consensus graph: %d edges", len(edges))

    # Step 3: run Louvain on the consensus graph (resolution=1 here — the
    # consensus weights already encode the desired coarseness)
    part_final = leidenalg.find_partition(
        g_cons, leidenalg.ModularityVertexPartition, weights="weight", seed=seed,
    )
    log.info("  consensus partition: %d communities, Q=%.4f",
             len(part_final), part_final.modularity)
    return {nodes[i]: cid for cid, members in enumerate(part_final) for i in members}


# ── Method 3: spectral clustering ────────────────────────────────────────────

def run_spectral(G: nx.Graph, k: int, seed: int = 42) -> Partition:
    """
    k-way spectral clustering using the k smallest eigenvectors of the
    symmetric normalised Laplacian (Jordan-Weiss embedding → k-means).

    Parameters
    ----------
    G : nx.Graph
        Undirected graph (self-loops removed).
    k : int
        Number of clusters (use Louvain community count for comparability).
    seed : int
        Random seed for k-means initialisation.

    Returns
    -------
    Partition
        ``{node_id: community_int}`` mapping.

    Notes
    -----
    Uses ``scipy.sparse.linalg.eigsh`` for the k smallest eigenvalues, which
    is far faster than dense decomposition on large graphs.
    """
    from scipy.sparse.linalg import eigsh

    nodes = list(G.nodes())
    log.info("Spectral clustering: k=%d, n=%d nodes...", k, len(nodes))

    L = nx.normalized_laplacian_matrix(G, nodelist=nodes).astype(float)
    # k+1 eigenvectors; discard the trivial zero eigenvector (smallest)
    _, vecs = eigsh(L, k=k + 1, which="SM")
    embedding = normalize(vecs[:, 1:], norm="l2")   # rows = nodes, cols = eigenvecs

    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    labels = km.fit_predict(embedding)
    return {nodes[i]: int(labels[i]) for i in range(len(nodes))}


# ── Method 4: InfoMap ─────────────────────────────────────────────────────────

def run_infomap(G: nx.Graph, directed: bool = False, seed: int = 42) -> Partition:
    """
    Flow-based community detection via the InfoMap algorithm.

    InfoMap finds communities as regions where a random walker stays trapped,
    minimising the map equation (description length of the walk). Especially
    suited to directed and weighted networks.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Graph to partition. Edge weights (``"weight"`` attribute) are used
        if present. Self-loops are stripped before running.
    directed : bool
        Pass ``True`` when G is a DiGraph and you want directed-flow InfoMap.
    seed : int
        Random seed for InfoMap's internal stochastic search.

    Returns
    -------
    Partition
        ``{node_id: community_int}`` mapping (1-indexed module IDs from InfoMap
        are converted to 0-indexed integers).

    Raises
    ------
    ImportError
        If the ``infomap`` package is not installed
        (``pip install infomap``).
    """
    try:
        import infomap as im_pkg
    except ImportError as exc:
        raise ImportError("pip install infomap") from exc

    G_nsl = G.copy()
    G_nsl.remove_edges_from(nx.selfloop_edges(G_nsl))

    nodes = list(G_nsl.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}

    im = im_pkg.Infomap(directed=directed, silent=True, seed=seed)
    for u, v, data in G_nsl.edges(data=True):
        w = float(data.get("weight", 1.0))
        im.add_link(node_to_int[u], node_to_int[v], w)

    im.run()

    module_ids = {node_to_int[n]: 0 for n in nodes}
    for node in im.nodes:
        module_ids[node.node_id] = node.module_id

    # Remap module IDs to 0-indexed contiguous integers
    unique = sorted(set(module_ids.values()))
    remap = {m: i for i, m in enumerate(unique)}
    log.info("InfoMap: %d modules discovered", len(unique))
    return {nodes[i]: remap[module_ids[i]] for i in range(len(nodes))}


# ── Method 5: HDBSCAN on geographic coordinates ──────────────────────────────

def run_hdbscan_geo(
    G: nx.Graph,
    min_cluster_size: int = 10,
) -> Partition:
    """
    Density-based community detection in projected geographic space.

    Each node is mapped to a 2-D point :math:`(x_i, y_i)` in kilometres via a
    mean-latitude equirectangular projection:

    .. math::

        x_i = \\lambda_i \\cdot 111.0\\cos(\\bar{\\phi}), \\quad
        y_i = \\phi_i \\cdot 111.0,

    where :math:`\\phi_i`, :math:`\\lambda_i` are node latitude and longitude
    (degrees) and :math:`\\bar{\\phi}` is the mean latitude of all nodes.
    HDBSCAN is then run on these coordinates using the mutual reachability
    distance (see :func:`run_hdbscan_spectral` for the formula).

    This variant contains no graph-structural information whatsoever: it
    partitions the seismicity purely by spatial density.  Comparing its
    partition with that of :func:`run_hdbscan_spectral` via NMI directly
    quantifies how much of the network's community structure is explained by
    geographic proximity versus genuine topological organisation.  A low NMI
    between the two HDBSCAN variants is the more seismologically interesting
    outcome: it implies that the network encodes fault connectivity patterns
    that a purely spatial analysis would miss.

    Noise points (label −1) are assigned to the nearest cluster centroid in
    the projected coordinate plane.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph whose nodes carry ``lat`` and ``lon`` attributes.
    min_cluster_size : int
        HDBSCAN minimum cluster size (same recommendation as
        :func:`run_hdbscan_spectral`).

    Returns
    -------
    Partition
        ``{node_id: community_int}`` mapping (noise nodes reassigned to nearest
        centroid; no −1 labels in the output).

    References
    ----------
    Campello R. J. G. B., Moulavi D. & Sander J. (2013). Density-based clustering
    based on hierarchical density estimates. *PAKDD*, LNAI 7819, 160–172.

    McInnes L., Healy J. & Astels S. (2017). hdbscan: Hierarchical density based
    clustering. *Journal of Open Source Software*, 2(11), 205.
    """
    try:
        import hdbscan as hdbscan_pkg
    except ImportError as exc:
        raise ImportError("pip install hdbscan") from exc

    nodes = list(G.nodes())
    n = len(nodes)

    lats = np.array([G.nodes[v].get("lat", float("nan")) for v in nodes])
    lons = np.array([G.nodes[v].get("lon", float("nan")) for v in nodes])

    mean_lat = np.nanmean(lats)
    xy = np.column_stack([
        lats * 111.0,
        lons * 111.0 * np.cos(np.radians(mean_lat)),
    ])

    valid = ~np.isnan(xy).any(axis=1)
    if not valid.all():
        log.warning(
            "HDBSCAN-Geo: %d nodes lack lat/lon attributes; assigned to cluster 0",
            int((~valid).sum()),
        )

    labels = np.zeros(n, dtype=int)
    if valid.sum() >= min_cluster_size:
        clusterer = hdbscan_pkg.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=1)
        sub_labels = clusterer.fit_predict(xy[valid])

        unique_labels = sorted(set(sub_labels) - {-1})
        if not unique_labels:
            sub_labels = np.zeros(int(valid.sum()), dtype=int)
        elif -1 in set(sub_labels):
            centroids = np.array([xy[valid][sub_labels == c].mean(axis=0) for c in unique_labels])
            noise_mask = sub_labels == -1
            dists = np.linalg.norm(
                xy[valid][noise_mask, None, :] - centroids[None, :, :], axis=2
            )
            sub_labels[noise_mask] = np.array(unique_labels)[dists.argmin(axis=1)]

        labels[valid] = sub_labels

    n_clusters = len(set(labels))
    log.info("HDBSCAN-Geo: %d geographic clusters discovered", n_clusters)
    return {nodes[i]: int(labels[i]) for i in range(n)}


# ── Partition quality scoring ─────────────────────────────────────────────────

def score_partition(
    G: nx.Graph,
    partition: "Partition",
    cell_size_km: float = 10.0,
) -> dict[str, float]:
    """
    Compute nine quality metrics for a single community partition.

    The metrics span three categories:

    **Graph-structural**

    * *Modularity Q* — Newman-Girvan modularity via leidenalg:

      .. math::

          Q = \\frac{1}{2m}\\sum_{uv}\\left[A_{uv} - \\frac{k_u k_v}{2m}\\right]
              \\delta(c_u, c_v)

      Higher is better; Q ∈ (−1, 1].

    * *Conductance φ* — weighted mean over communities:

      .. math::

          \\phi = \\frac{1}{|\\mathcal{C}|}\\sum_{C}
                  \\frac{\\text{cut}(C)}{\\min(\\text{vol}(C),\\,\\text{vol}(V\\setminus C))}

      Lower is better; φ = 0 means perfect separation.

    * *Coverage* — fraction of total edge weight that falls inside communities:

      .. math::

          \\text{cov} = \\frac{\\sum_{(u,v)\\in E,\\,c_u=c_v} w_{uv}}{\\sum_{(u,v)\\in E} w_{uv}}

      Higher is better; cov ∈ [0, 1].

    * *Normalised Cut (Ncut)* —

      .. math::

          N_{\\mathrm{cut}} = \\sum_{C}
                \\frac{\\text{cut}(C)}{\\text{vol}(C)}

      Lower is better; Ncut → 0 as communities become fully isolated.

    **Information-theoretic**

    * *Map equation L* — description length of an infinite random walk when
      compressed with the community codebook (Rosvall & Bergstrom 2008):

      .. math::

          L = q_{\\curvearrowleft}H(\\mathcal{Q})
              + \\sum_{C} p_{\\circlearrowleft}^{C}H(\\mathcal{P}^C)

      where :math:`q_{\\curvearrowleft}` is the rate of inter-module movement,
      :math:`H(\\mathcal{Q})` its entropy, and :math:`H(\\mathcal{P}^C)` the
      entropy within module :math:`C`.  Lower is better.  This approximation
      uses teleportation-free random-walk stationary probabilities.

    * *DC-SBM log-likelihood* — degree-corrected stochastic block model
      (Karrer & Newman 2011):

      .. math::

          \\ell_{\\mathrm{DC}} =
              \\sum_{r,s} e_{rs}\\ln\\!\\frac{e_{rs}}{\\kappa_r \\kappa_s}

      where :math:`e_{rs}` is the number of edges from block :math:`r` to
      block :math:`s`, and :math:`\\kappa_r = \\sum_{u\\in r} k_u` is the
      degree sum.  Higher is better (less negative).

    * *Surprise* — log-probability of the observed intra-community edge count
      under a hypergeometric null (Aldecoa & Marín 2013), accessed via
      leidenalg's ``SurpriseVertexPartition``.  Higher is better.

    **Seismological**

    * *Geographic compactness (km)* — mean haversine distance from each node
      to its community centroid.  Lower is better; compact communities trace
      single fault segments, while large values indicate dispersed, possibly
      spurious groupings.

    * *Depth coherence (km)* — mean within-community standard deviation of
      focal depth (``node[2] * cell_size_km``).  Lower is better; coherent
      depth bands correspond to distinct seismogenic layers.

    Parameters
    ----------
    G : nx.Graph
        The network (directed or undirected; edge weights used where relevant).
    partition : Partition
        ``{node: community_id}`` dict covering all nodes in *G*.
    cell_size_km : float
        Grid resolution used to convert discrete cell coordinates to depth in km.

    Returns
    -------
    dict[str, float]
        Keys: ``Q``, ``conductance``, ``coverage``, ``ncut``,
        ``map_L``, ``dcsbm_ll``, ``surprise``,
        ``geo_compactness_km``, ``depth_coherence_km``.

    References
    ----------
    Newman M.E.J. & Girvan M. (2004). Finding and evaluating community
    structure in networks. *Phys. Rev. E* 69, 026113.

    Rosvall M. & Bergstrom C.T. (2008). Maps of random walks on complex
    networks reveal community structure. *PNAS* 105, 1118–1123.

    Karrer B. & Newman M.E.J. (2011). Stochastic blockmodels and community
    structure in networks. *Phys. Rev. E* 83, 016107.

    Aldecoa R. & Marín I. (2013). Exploring the limits of community detection
    strategies in complex networks. *Scientific Reports* 3, 2216.

    Yang J. & Leskovec J. (2013). Overlapping community detection at scale.
    *ACM WSDM*, 587–596.
    """
    import math
    import leidenalg
    import igraph as ig

    nodes = list(G.nodes())
    N = len(nodes)
    communities = sorted(set(partition.values()))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    comm_of = np.array([partition[n] for n in nodes], dtype=np.int32)

    # Community membership lists
    comm_nodes: dict[int, list[int]] = {c: [] for c in communities}
    for i, n in enumerate(nodes):
        comm_nodes[partition[n]].append(i)

    # Edge weight helpers
    G_und = G.to_undirected() if G.is_directed() else G
    total_w = sum(d.get("weight", 1.0) for _, _, d in G_und.edges(data=True))

    # ── Modularity Q ─────────────────────────────────────────────────────────
    g, ig_nodes = _to_igraph(G_und)
    ig_to_nx = {i: n for i, n in enumerate(ig_nodes)}
    membership = [int(partition[ig_to_nx[i]]) for i in range(len(ig_nodes))]
    # Remap to 0-based contiguous integers (leidenalg requirement)
    label_map = {old: new for new, old in enumerate(sorted(set(membership)))}
    membership_0 = [label_map[m] for m in membership]
    part_obj = leidenalg.RBConfigurationVertexPartition(
        g, initial_membership=membership_0, weights="weight"
    )
    Q = part_obj.modularity

    # ── Cut / vol helpers ─────────────────────────────────────────────────────
    cut_w: dict[int, float] = {c: 0.0 for c in communities}
    vol_w: dict[int, float] = {c: 0.0 for c in communities}
    intra_w: float = 0.0

    for u, v, d in G_und.edges(data=True):
        w = d.get("weight", 1.0)
        cu, cv = partition[u], partition[v]
        vol_w[cu] += w
        vol_w[cv] += w
        if cu == cv:
            intra_w += w
        else:
            cut_w[cu] += w
            cut_w[cv] += w

    vol_total = sum(vol_w.values())

    # ── Coverage ─────────────────────────────────────────────────────────────
    coverage = intra_w / total_w if total_w > 0 else 0.0

    # ── Conductance φ ────────────────────────────────────────────────────────
    phi_vals = []
    for c in communities:
        cut = cut_w[c]
        vol = vol_w[c]
        vol_out = vol_total - vol
        denom = min(vol, vol_out)
        if denom > 0:
            phi_vals.append(cut / denom)
    conductance = float(np.mean(phi_vals)) if phi_vals else 0.0

    # ── Normalised Cut ────────────────────────────────────────────────────────
    ncut = sum(cut_w[c] / vol_w[c] for c in communities if vol_w[c] > 0)

    # ── Map equation L ───────────────────────────────────────────────────────
    # Stationary distribution via degree (undirected teleportation-free approx)
    degree_w = dict(G_und.degree(weight="weight"))
    total_deg = sum(degree_w.values())
    pi = {n: degree_w[n] / total_deg for n in nodes} if total_deg > 0 else {n: 1 / N for n in nodes}

    # Module visit rates
    q_exit = 0.0
    p_in: dict[int, float] = {c: 0.0 for c in communities}
    for u, v, d in G_und.edges(data=True):
        w = d.get("weight", 1.0)
        cu, cv = partition[u], partition[v]
        if cu != cv:
            flow_uv = pi[u] * w / (degree_w[u] if degree_w[u] > 0 else 1)
            flow_vu = pi[v] * w / (degree_w[v] if degree_w[v] > 0 else 1)
            q_exit += flow_uv + flow_vu

    p_stay: dict[int, float] = {}
    for c in communities:
        p_stay[c] = sum(pi[nodes[i]] for i in comm_nodes[c])

    q_c: dict[int, float] = {}
    for c in communities:
        cut_out = 0.0
        for i in comm_nodes[c]:
            n = nodes[i]
            for nbr, dd in G_und[n].items():
                if partition[nbr] != c:
                    w = dd.get("weight", 1.0)
                    cut_out += pi[n] * w / (degree_w[n] if degree_w[n] > 0 else 1)
        q_c[c] = cut_out

    q_total = sum(q_c.values())

    def _ent(probs):
        s = sum(probs)
        if s <= 0:
            return 0.0
        return -sum((p / s) * math.log2(p / s) for p in probs if p > 0)

    H_Q = _ent(list(q_c.values()))
    H_P = {}
    for c in communities:
        terms = [pi[nodes[i]] for i in comm_nodes[c]] + [q_c[c]]
        H_P[c] = _ent(terms)

    denom_map = q_total + sum(p_stay[c] for c in communities)
    if denom_map > 0:
        map_L = (
            q_total * H_Q
            + sum((p_stay[c] + q_c[c]) * H_P[c] for c in communities)
        ) / denom_map
    else:
        map_L = 0.0

    # ── DC-SBM log-likelihood ─────────────────────────────────────────────────
    e_rs: dict[tuple, float] = {}
    kappa: dict[int, float] = {c: 0.0 for c in communities}
    for u, v, d in G_und.edges(data=True):
        w = d.get("weight", 1.0)
        cu, cv = partition[u], partition[v]
        key = (min(cu, cv), max(cu, cv))
        e_rs[key] = e_rs.get(key, 0.0) + w
        kappa[cu] += w
        kappa[cv] += w

    dcsbm_ll = 0.0
    for (r, s), e in e_rs.items():
        kr_ks = kappa[r] * kappa[s]
        if kr_ks > 0 and e > 0:
            dcsbm_ll += e * math.log(e / kr_ks)

    # ── Surprise ─────────────────────────────────────────────────────────────
    try:
        g_surp, surp_nodes = _to_igraph(G_und)
        surp_label_map = {old: new for new, old in enumerate(sorted(set(membership)))}
        surp_membership = [
            surp_label_map[int(partition[surp_nodes[i]])]
            for i in range(len(surp_nodes))
        ]
        surp_part = leidenalg.SurpriseVertexPartition(
            g_surp, initial_membership=surp_membership
        )
        surprise = surp_part.quality()
    except Exception:
        surprise = float("nan")

    # ── Geographic compactness (km) ───────────────────────────────────────────
    from math import radians, sin, cos, sqrt, atan2

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    geo_dists = []
    for c, idxs in comm_nodes.items():
        lats = [G_und.nodes[nodes[i]].get("lat") for i in idxs]
        lons = [G_und.nodes[nodes[i]].get("lon") for i in idxs]
        if any(x is None for x in lats + lons):
            continue
        clat = float(np.mean(lats))
        clon = float(np.mean(lons))
        for lat, lon in zip(lats, lons):
            geo_dists.append(haversine_km(lat, lon, clat, clon))

    geo_compactness_km = float(np.mean(geo_dists)) if geo_dists else float("nan")

    # ── Depth coherence (km) ─────────────────────────────────────────────────
    depth_stds = []
    for c, idxs in comm_nodes.items():
        depths = []
        for i in idxs:
            n = nodes[i]
            attr_d = G_und.nodes[n].get("depth_km")
            if attr_d is not None:
                depths.append(float(attr_d))
            elif isinstance(n, (tuple, list)) and len(n) >= 3:
                depths.append(float(n[2]) * cell_size_km)
        if len(depths) >= 2:
            depth_stds.append(float(np.std(depths)))

    depth_coherence_km = float(np.mean(depth_stds)) if depth_stds else float("nan")

    return {
        "Q": float(Q),
        "conductance": conductance,
        "coverage": float(coverage),
        "ncut": float(ncut),
        "map_L": float(map_L),
        "dcsbm_ll": float(dcsbm_ll),
        "surprise": float(surprise),
        "geo_compactness_km": geo_compactness_km,
        "depth_coherence_km": depth_coherence_km,
    }


def compare_partitions(
    G: nx.Graph,
    partitions: dict[str, "Partition"],
    cell_size_km: float = 10.0,
) -> pd.DataFrame:
    """
    Score every partition in *partitions* and return a tidy comparison table.

    Parameters
    ----------
    G : nx.Graph
        The network all partitions were computed on.
    partitions : dict[str, Partition]
        ``{method_name: {node: community_id}}`` mapping.
    cell_size_km : float
        Passed through to :func:`score_partition`.

    Returns
    -------
    pd.DataFrame
        Rows = methods, columns = metric names.  Methods are sorted by
        modularity Q descending.
    """
    records = {}
    for name, part in partitions.items():
        log.info("Scoring partition: %s", name)
        records[name] = score_partition(G, part, cell_size_km=cell_size_km)
    df = pd.DataFrame(records).T
    df.index.name = "method"
    return df.sort_values("Q", ascending=False)


def plot_partition_scores(
    df: pd.DataFrame,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Heatmap of partition quality scores, ranked by modularity.

    Each column is z-score normalised so that metrics with very different
    scales (e.g. Surprise in nats vs conductance ∈ [0,1]) are visually
    comparable.  For metrics where *lower is better* (conductance, Ncut,
    map_L, geo_compactness_km, depth_coherence_km) the sign is flipped
    before normalisation so that brighter cells always mean *better*.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compare_partitions`.
    title : str
        Figure title suffix.
    save : bool
        If True, saves PDF + JPG via :func:`savefig`.
    """
    lower_is_better = {"conductance", "ncut", "map_L", "geo_compactness_km", "depth_coherence_km"}

    display = df.copy()
    for col in display.columns:
        if col in lower_is_better:
            display[col] = -display[col]

    # z-score normalise column-wise
    normed = (display - display.mean()) / (display.std() + 1e-12)

    pretty_cols = {
        "Q": "Modularity Q",
        "conductance": "Conductance φ ↓",
        "coverage": "Coverage",
        "ncut": "Ncut ↓",
        "map_L": "Map eq. L ↓",
        "dcsbm_ll": "DC-SBM LL",
        "surprise": "Surprise",
        "geo_compactness_km": "Geo compact. ↓",
        "depth_coherence_km": "Depth coherence ↓",
    }
    normed = normed.rename(columns=pretty_cols)

    n_rows, n_cols = normed.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.2), max(4, n_rows * 0.8)))
    sns.heatmap(
        normed,
        annot=df.rename(columns=pretty_cols).round(3),
        fmt="g",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "z-score (brighter = better)"},
    )
    ax.set_title(f"Partition quality scores{' — ' + title if title else ''}", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Method")
    plt.tight_layout()
    if save:
        savefig(f"community_partition_scores_{_slug(title)}")
    plt.show()


# ── NMI comparison ────────────────────────────────────────────────────────────

def compute_nmi_matrix(partitions: dict[str, Partition]) -> pd.DataFrame:
    """
    Compute pairwise Normalised Mutual Information between all partitions.

    Parameters
    ----------
    partitions : dict[str, Partition]
        Keys are method names; values are ``{node: community_id}`` dicts.
        All partitions must share the same node set.

    Returns
    -------
    pd.DataFrame
        Symmetric NMI matrix (methods × methods), values in [0, 1].
    """
    methods = list(partitions.keys())
    nodes = sorted(next(iter(partitions.values())).keys())

    arr = {m: np.array([partitions[m][n] for n in nodes]) for m in methods}
    nmi = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            nmi.loc[a, b] = normalized_mutual_info_score(arr[a], arr[b])
    return nmi


def plot_nmi_heatmap(nmi: pd.DataFrame, title: str = "", save: bool = True) -> None:
    """
    Heatmap of pairwise NMI values across community-detection methods.

    Values close to 1 mean the two methods produce nearly identical partitions;
    values near 0 mean they disagree completely.

    Parameters
    ----------
    nmi : pd.DataFrame
        Output of :func:`compute_nmi_matrix`.
    title : str
        Figure title suffix.
    """
    n_methods = len(nmi)
    sz = max(5, n_methods)
    fig, ax = plt.subplots(figsize=(sz, sz - 1))
    sns.heatmap(
        nmi,
        annot=True,
        fmt=".3f",
        cmap="YlGn",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "NMI"},
        ax=ax,
    )
    ax.set_title(f"Community Method Agreement (NMI): {title}", fontsize=12, pad=10)
    plt.tight_layout()
    if save:
        savefig(f"nmi_heatmap_{_slug(title)}")
    plt.show()


# ── Geographical community map ────────────────────────────────────────────────

def plot_community_geo(
    G: nx.Graph,
    community_map: Partition,
    title: str = "",
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 0,
    bounds: dict | None = None,
    min_community_size: int = 50,
    method_name: str = "",
    height: int = 600,
    width: int = 1100,
    save: bool = True,
) -> None:
    """
    Tile-map (scatter_map) coloured by community assignment.

    Only communities with at least ``min_community_size`` nodes are shown to
    avoid noise from singleton/tiny communities.

    Parameters
    ----------
    G : nx.Graph
        Graph whose nodes carry ``lat``/``lon`` attributes.
    community_map : Partition
        ``{node_id: community_int}`` mapping.
    title : str
        Figure title suffix (catalog + cell size).
    center_lat, center_lon, zoom : float
        Initial map view.
    bounds : dict or None
        Optional ``dict(west=, east=, south=, north=)`` viewport constraint.
    min_community_size : int
        Minimum number of nodes for a community to be rendered.
    method_name : str
        Method label shown in the figure title.
    """
    rows = []
    for n in G.nodes():
        if "lat" not in G.nodes[n]:
            continue
        rows.append({
            "cell_id":   n,
            "community": str(community_map.get(n, -1)),
            "lat":       G.nodes[n]["lat"],
            "lon":       G.nodes[n]["lon"],
            "degree":    G.degree(n),
        })
    df = pd.DataFrame(rows)

    counts = df["community"].value_counts()
    large = counts[counts >= min_community_size].index
    df = df[df["community"].isin(large)].copy()
    n_shown = df["community"].nunique()

    # Use log1p(degree) so low-degree background nodes are still visible
    df = df.assign(size_val=np.log1p(df["degree"]).clip(lower=0.5))

    fig = px.scatter_map(
        df,
        lat="lat", lon="lon",
        color="community",
        size="size_val", size_max=18,
        color_discrete_sequence=_PALETTE,
        hover_name="community",
        hover_data={"lat": ":.3f", "lon": ":.3f", "degree": True, "size_val": False},
        map_style="carto-positron",
        title=(
            f"Seismic Communities — {method_name} "
            f"({n_shown} communities ≥ {min_community_size} nodes) — {title}"
        ),
    )
    fig.update_traces(marker=dict(opacity=0.7))
    map_cfg: dict = {"center": {"lat": center_lat, "lon": center_lon}, "zoom": zoom}
    if bounds is not None:
        map_cfg["bounds"] = bounds
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        width=width, height=height,
        showlegend=True,
        map=map_cfg,
    )
    if save:
        save_plotly(fig, f"community_geo_{_slug(method_name)}_{_slug(title)}")
    fig.show()


# ── Directed Louvain (Leiden algorithm on DiGraph) ───────────────────────────

def run_directed_louvain(
    G: nx.DiGraph,
    seed: int = 42,
) -> Partition:
    """
    Directed modularity optimisation via the Leiden algorithm.

    Uses the Leicht-Newman (2008) directed modularity Q_d, which accounts
    for the asymmetry between in-degree and out-degree when deciding whether
    two nodes belong to the same community.  Implemented via ``leidenalg``
    with ``ModularityVertexPartition`` on a directed igraph object.

    Requires: ``pip install leidenalg python-igraph``

    Parameters
    ----------
    G : nx.DiGraph
        Directed weighted earthquake network (self-loops included or not).
    seed : int
        Random seed for Leiden's internal optimiser.

    Returns
    -------
    Partition
        ``{node_id: community_int}`` mapping (0-indexed).

    Notes
    -----
    The Leiden algorithm is a corrected version of Louvain that guarantees
    well-connected communities; it subsumes Louvain as a special case.
    """
    import leidenalg

    g, nodes = _to_igraph(G)
    part = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, weights="weight", seed=seed,
    )
    log.info("Directed Louvain (Leiden): %d communities, Q=%.4f",
             len(part), part.modularity)
    return {nodes[i]: cid for cid, members in enumerate(part) for i in members}

