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

from src.network import (
    build_abe_suzuki_network,
    discretize_space_3d
)

log = logging.getLogger(__name__)

_PALETTE = px.colors.qualitative.Bold




# ── Type alias ───────────────────────────────────────────────────────────────
Partition = dict[str, int]


def _to_igraph(G: nx.Graph):
    """
    Convert NetworkX → igraph with correct edge-weight alignment.
    """
    import igraph as ig

    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}

    g = ig.Graph(directed=G.is_directed())
    g.add_vertices(len(nodes))

    edges = []
    weights = []

    for u, v, d in G.edges(data=True):
        edges.append((node_to_int[u], node_to_int[v]))
        weights.append(float(d.get("weight", 1.0)))

    g.add_edges(edges)

    if len(weights) > 0:
        g.es["weight"] = weights

    return g, nodes


def _run_leiden(
    G: nx.Graph,
    seed: int = 42,
    resolution: float = 1.0,
    n_iterations: int = -1,
    partition_type: str = "RB",   # default: dimensionless modularity
) -> Partition:
    """
    Leiden optimisation for hybrid weighted earthquake network.

    partition_type:
        - "RB"  (default) — Reichardt-Bornholdt modularity; dimensionless
                resolution γ, weight-scale-invariant. Matches the standard
                "Louvain γ" parameter in the literature. Use this for any
                graph with non-trivial edge-weight scale (e.g. the hybrid
                network with weights spanning many decades).
        - "CPM" — Constant Potts Model; γ is an *absolute* density threshold
                in edge-weight units. Only sensible when edge weights are
                normalised to ~O(1); on a wide-range weight distribution the
                resolution becomes scale-dependent and γ ≈ 1 either over- or
                under-merges depending on the median weight.
    """
    import leidenalg

    g, nodes = _to_igraph(G)

    weights = "weight" if "weight" in g.es.attributes() else None

    if partition_type == "CPM":
        part = leidenalg.find_partition(
            g,
            leidenalg.CPMVertexPartition,
            weights=weights,
            seed=seed,
            resolution_parameter=resolution,
            n_iterations=n_iterations,
        )

    else:  # fallback to RB (your original)
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            seed=seed,
            resolution_parameter=resolution,
            n_iterations=n_iterations,
        )

    return {nodes[i]: cid for cid, members in enumerate(part) for i in members}


def run_louvain_hybrid(
    G: nx.Graph,
    seed: int = 42,
    resolution: float = 1.0,
    partition_type: str = "RB",
) -> Partition:
    """
    Community detection on undirected hybrid network.

    Uses Reichardt-Bornholdt modularity by default (``partition_type="RB"``)
    — dimensionless γ, weight-scale-invariant. Pass ``partition_type="CPM"``
    only if you have a normalised-weight graph and know that γ is a meaningful
    absolute density threshold for your data.
    """
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    return _run_leiden(G_und, seed=seed, resolution=resolution,
                       partition_type=partition_type)


def run_louvain_directed_hybrid(
    G: nx.DiGraph,
    seed: int = 42,
    resolution: float = 1.0,
    partition_type: str = "RB",
) -> Partition:
    """
    Directed community detection for hybrid network. See
    :func:`run_louvain_hybrid` for the ``partition_type`` discussion.
    """
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))

    return _run_leiden(G, seed=seed, resolution=resolution,
                       partition_type=partition_type)




def run_louvain_consensus_hybrid(
    G: nx.Graph,
    n_runs: int = 20,
    resolution: float = 1.0,
    directed: bool = False,
    threshold: float = 0.5,
    max_iter: int = 1,
    sample_pairs: bool = False,
    max_pairs_per_comm: int = 500,
) -> dict:
    """
    Consensus community detection (Lancichinetti & Fortunato 2012) for
    hybrid weighted network: run Louvain ``n_runs`` times → build the
    co-occurrence graph H where edge (u,v) has weight = fraction of runs
    in which u, v ended up in the same community → run Louvain once on H,
    keeping only co-occurrence edges with weight ≥ ``threshold``.

    Parameters
    ----------
    n_runs : int
        Number of Louvain runs to average over.
    resolution : float
        γ for each individual Louvain run. MUST match the γ used for any
        plain Louvain you are comparing against — different γ give partitions
        at different granularity scales (low NMI without method disagreement).
    threshold : float
        Lancichinetti-Fortunato cutoff. Keep co-occurrence edges with
        normalised weight ≥ threshold. 0.5 is standard.
    max_iter : int
        Number of iterative consensus rounds. **Default 1** — the standard
        algorithm is a single round (run on G, build H, run on H). Iterating
        replaces G with H repeatedly, which (a) loses directedness because H
        is undirected by construction and (b) compounds shrinkage, ending
        with many micro-components. Set >1 only if you have a specific reason.
    sample_pairs : bool
        Subsample members of large communities before counting co-occurrence.
        **Default False** — sampling drops genuine co-occurrences below the
        threshold for any community larger than ``max_pairs_per_comm`` (each
        node has p = max_pairs_per_comm/|C| of being sampled, so a pair has
        p² of both being in the same run's sample; for |C|=1000 and
        max=500 that's 25% per run, well below the 50% threshold even with
        perfect within-community stability).
    max_pairs_per_comm : int
        Only used when ``sample_pairs=True``; cap on members sampled per
        community per run.
    """

    import random
    from collections import defaultdict

    def run_partition(graph):
        partitions = []
        for seed in range(n_runs):

            if directed:
                p = run_louvain_directed_hybrid(
                    graph, seed=seed, resolution=resolution
                )
            else:
                p = run_louvain_hybrid(
                    graph, seed=seed, resolution=resolution
                )

            partitions.append(p)

        return partitions

    def build_consensus_graph(partitions):
        edge_weights = defaultdict(int)

        for part in partitions:

            comms = defaultdict(list)
            for n, c in part.items():
                comms[c].append(n)

            for members in comms.values():

                # subsample large communities
                if sample_pairs and len(members) > max_pairs_per_comm:
                    members = random.sample(members, max_pairs_per_comm)

                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = members[i], members[j]
                        if a > b:
                            a, b = b, a
                        edge_weights[(a, b)] += 1

        # build consensus graph
        H = nx.Graph()

        for (a, b), w in edge_weights.items():
            w = w / n_runs  # normalize to [0,1]

            if w >= threshold:
                H.add_edge(a, b, weight=w)

        return H

    current_graph = G.copy()

    for _ in range(max_iter):

        partitions = run_partition(current_graph)
        new_graph = build_consensus_graph(partitions)

        # stopping conditions
        if new_graph.number_of_edges() == 0:
            break

        if new_graph.number_of_edges() == current_graph.number_of_edges():
            break

        current_graph = new_graph

    # final partition
    if directed:
        return run_louvain_directed_hybrid(
            current_graph, resolution=resolution
        )
    else:
        return run_louvain_hybrid(
            current_graph, resolution=resolution
        )




def run_infomap_hybrid(
    G: nx.Graph,
    directed: bool = True,
    seed: int = 42,
) -> dict:
    """
    InfoMap community detection adapted for hybrid weighted earthquake network.

    Works with:
    - exponential interaction weights
    - directed or undirected graphs
    - sparse thresholded networks
    """

    try:
        from infomap import Infomap
    except ImportError:
        raise ImportError("pip install infomap")

    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))

    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}

    # Pass directed/silent/seed via the constructor — the attribute-set form
    # (``im.directed = True`` after init) is non-standard and silently no-ops
    # on current infomap versions, which would run undirected flow on a
    # directed graph (different community boundaries entirely).
    im = Infomap(directed=directed, silent=True, seed=seed)

    # Add weighted edges (hybrid interaction strengths)
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))

        # safety: Infomap expects strictly positive weights
        if w <= 0:
            continue

        im.add_link(node_to_int[u], node_to_int[v], w)

    im.run()

    partition = {}

    # robust extraction (API-safe)
    for node in im.tree:
        if node.is_leaf:
            partition[nodes[node.node_id]] = node.module_id

    return partition


# ====================================================================================
# Density-based geographic clustering (spatial null baseline)
# ====================================================================================


def run_hdbscan_geo_hybrid(
    G: nx.Graph,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    target_crs: str = "epsg:32632",
) -> dict:
    """
    HDBSCAN on the geographic coordinates of the cells — a *spatial null
    baseline* for community detection. Ignores the network entirely and
    clusters cells purely by (projected) (x, y) position.

    Useful as the "is the network adding signal beyond spatial proximity?"
    contrast against the graph-aware methods (Louvain, InfoMap, MM-SBM):
    high NMI with HDBSCAN-geo means the network methods are mostly
    rediscovering geography; moderate NMI means the network captures
    structure beyond spatial clustering.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Network. Each node must have ``lat`` and ``lon`` attributes (assigned
        by ``_assign_node_coords`` in ``src/network_custom.py``).
    min_cluster_size : int, default 10
        HDBSCAN minimum cluster size. Matches the ``min_community_size=10``
        filter convention used elsewhere in the project.
    min_samples : int or None
        HDBSCAN ``min_samples``. ``None`` defaults to ``min_cluster_size`` —
        a relatively conservative setting that yields fewer noise points.
    target_crs : str, default ``"epsg:32632"``
        Metric CRS used to project lat/lon into kilometres before clustering.
        Italy: UTM Zone 32N. Mirrors the project's standard projection.

    Returns
    -------
    partition : dict
        ``{node_id: cluster_id}``. HDBSCAN noise points (label ``-1``) are
        re-labelled as unique singleton clusters (one cluster id per noise
        node), so the ``≥ 10`` cell filter applied downstream cleanly drops
        them from NMI without lumping them into a single artificial
        "noise community".

    References
    ----------
    Campello, R.J.G.B., Moulavi, D. & Sander, J. (2013).
    *Density-Based Clustering Based on Hierarchical Density Estimates.*
    PAKDD.

    McInnes, L. & Healy, J. (2017). *Accelerated Hierarchical Density
    Based Clustering.* IEEE ICDMW.
    """
    import hdbscan
    from pyproj import Transformer

    nodes = list(G.nodes())
    coords_ll = np.asarray(
        [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in nodes],
        dtype=np.float64,
    )

    # Project (lat, lon) → metric (x, y) in km so distances are physical.
    fwd = Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
    xs_m, ys_m = fwd.transform(coords_ll[:, 1], coords_ll[:, 0])
    xy_km = np.column_stack([xs_m, ys_m]) / 1000.0

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples if min_samples is not None else min_cluster_size,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(xy_km)

    # Re-label noise (-1) as unique singletons so the ≥ 10-cell filter excludes
    # them rather than treating all noise as one giant "community".
    next_singleton = int(labels.max()) + 1 if (labels >= 0).any() else 0
    partition: dict = {}
    for n, lbl in zip(nodes, labels):
        if int(lbl) == -1:
            partition[n] = next_singleton
            next_singleton += 1
        else:
            partition[n] = int(lbl)
    return partition


# ====================================================================================

# ── Geographical community map ────────────────────────────────────────────────

# ====================================================================================


def plot_community_geo_hybrid(
    G: nx.Graph,
    community_map: Partition,
    title: str = "",
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 5,
    bounds: dict | None = None,
    min_community_size: int = 50,
    method_name: str = "",
    height: int = 600,
    width: int = 1100,
    save: bool = True,
) -> None:
    """
    Geographic visualization of communities in the hybrid earthquake network.

    Node size reflects weighted degree (interaction strength).
    Only sufficiently large communities are shown.
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
            "strength":  G.degree(n, weight="weight"),  # ← FIXED
        })

    df = pd.DataFrame(rows)
    df = df[df["community"] != "-1"]   # to remove nodes excluded from consensus louvain

    # filter small communities
    counts = df["community"].value_counts()
    large = counts[counts >= min_community_size].index
    df = df[df["community"].isin(large)].copy()

    n_shown = df["community"].nunique()

    if len(df) == 0:
        print("No communities large enough to display.")
        return

    # log-scale size (robust for hybrid weights)
    df["size_val"] = np.log1p(df["strength"]).clip(lower=0.5)

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="community",
        size="size_val",
        size_max=18,
        color_discrete_sequence=_PALETTE,
        hover_name="community",
        hover_data={
            "lat": ":.3f",
            "lon": ":.3f",
            "strength": ":.3e",   # scientific notation (important!)
            "size_val": False,
        },
        mapbox_style="carto-positron",
        title=(
            f"Seismic Communities (Hybrid) — {method_name} "
            f"({n_shown} communities ≥ {min_community_size}) — {title}"
        ),
    )

    fig.update_traces(marker=dict(opacity=0.7))

    map_cfg = {
        "center": {"lat": center_lat, "lon": center_lon},
        "zoom": zoom,
    }

    if bounds is not None:
        map_cfg["bounds"] = bounds

    fig.update_layout(
        mapbox=map_cfg,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        width=width,
        height=height,
        showlegend=True,
    )

    if save:
        save_plotly(fig, f"community_geo_hybrid_{_slug(method_name)}_{_slug(title)}")

    fig.show()





# ====================================================================================

# ──────────────────────────────── NMI ────────────────────────────────────────────────

# ====================================================================================




def align_partitions(part1: dict, part2: dict):
    """
    Align two partitions on common nodes (stable ordering).
    """
    common_nodes = sorted(set(part1.keys()) & set(part2.keys()))

    labels1 = np.array([part1[n] for n in common_nodes])
    labels2 = np.array([part2[n] for n in common_nodes])

    return labels1, labels2



def compute_nmi(part1: dict, part2: dict) -> float:
    labels1, labels2 = align_partitions(part1, part2)
    return normalized_mutual_info_score(labels1, labels2)




def compute_nmi_matrix(partitions: dict) -> pd.DataFrame:
    """
    Compute symmetric NMI similarity matrix between methods.
    """
    methods = list(partitions.keys())
    n = len(methods)

    mat = np.zeros((n, n))

    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            nmi = compute_nmi(partitions[methods[i]], partitions[methods[j]])
            mat[i, j] = nmi
            mat[j, i] = nmi

    return pd.DataFrame(mat, index=methods, columns=methods)



def plot_nmi_heatmap(nmi_df: pd.DataFrame):
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        nmi_df,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="YlGn",
        square=True,
        linewidths=0.5
    )

    plt.title("NMI between Community Detection Methods")
    plt.tight_layout()
    plt.show()


# ====================================================================================
# Mixed-Membership SBM via variational EM (Airoldi et al. 2008)
# ====================================================================================


def run_mmsbm_custom_hybrid(
    G: nx.Graph,
    K: int = 10,
    n_iter: int = 50,
    alpha: float | None = None,
    weight_mode: str = "poisson",
    tol: float = 1e-4,
    seed: int = 42,
    init: str = "louvain",
    verbose: bool = True,
) -> tuple[np.ndarray, Partition]:
    """
    Mixed-Membership Stochastic Blockmodel via variational EM
    (Airoldi, Blei, Fienberg & Xing 2008).

    Each node :math:`p` carries a Dirichlet-distributed membership vector
    :math:`\\pi_p \\in \\Delta^{K-1}`, and each ordered pair :math:`(p, q)`
    has latent block assignments :math:`z_{p \\to q}, z_{p \\leftarrow q}
    \\sim \\mathrm{Mult}(\\pi)`. The edge :math:`A_{pq}` is drawn from
    :math:`\\mathrm{Bernoulli}(B[z_{p \\to q}, z_{p \\leftarrow q}])`
    (``weight_mode='binary'``) or :math:`\\mathrm{Poisson}(B[z, z'])`
    on :math:`\\log(1 + w_{pq})`-transformed edge weights (``weight_mode='poisson'``,
    default). The log1p transform compresses the ~6-decade hybrid weight range
    into a numerically stable ~0–11 range while preserving relative ordering.

    Variational EM follows the original paper (eqs. 3, 5, 7) with full
    materialisation of the per-pair multinomials :math:`\\phi_{p \\to q}`
    and :math:`\\phi_{p \\leftarrow q}` as :math:`(N, N, K)` tensors —
    memory cost :math:`\\sim N^2 K` floats. For the hybrid 30 km Italy
    giant (:math:`N \\approx 1{,}800`) at :math:`K=10` this is :math:`\\sim`
    260 MB, runnable in a notebook. For larger networks, consider the
    stochastic-variational variant of Gopalan & Blei (2013).

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Network. Edge weights used iff ``weight_mode='poisson'``.
    K : int
        Number of blocks. For cross-validation with graph-tool's
        ``OverlapBlockState``, set this to graph-tool's auto-selected
        block count (read from its output CSV).
    n_iter : int
        Maximum variational EM iterations.
    alpha : float or None
        Dirichlet concentration hyperparameter. Default ``1/K`` encourages
        sparse memberships (most mass on one or two blocks).
    weight_mode : {'binary', 'poisson'}, default ``'poisson'``
        * ``'binary'`` — Bernoulli edge likelihood on the 0/1 adjacency.
          Loses the hybrid's continuous weight information but is the
          canonical Airoldi 2008 formulation.
        * ``'poisson'`` — Poisson likelihood on ``log1p(weight)``. Preserves
          relative weight ordering across the hybrid's ~6-decade range
          (min ≈ 0.02, max ≈ 4.8e4) by compressing to ~0–11, which keeps
          Poisson rates numerically stable. This matches the prof's algorithm
          slide (MM-SBM: weighted=YES) for the hybrid network.
    tol : float
        Convergence threshold (currently advisory only — fixed
        ``n_iter`` is used).
    seed : int
        RNG seed for initialisation.
    verbose : bool
        Print iteration summary (entropy of mean π, block matrix extrema).

    Returns
    -------
    pi : np.ndarray, shape (N, K)
        Expected membership probabilities under the variational posterior:
        :math:`\\hat\\pi_{p, k} = \\gamma_{p, k} / \\sum_k \\gamma_{p, k}`.
    hard_partition : dict
        ``{node_id: int}`` mapping (argmax of :math:`\\hat\\pi`) — suitable
        for NMI comparison against single-membership methods.

    References
    ----------
    Airoldi E.M., Blei D.M., Fienberg S.E. & Xing E.P. (2008). Mixed
    Membership Stochastic Blockmodels. *Journal of Machine Learning
    Research*, 9, 1981-2014.

    Gopalan P. & Blei D.M. (2013). Efficient discovery of overlapping
    communities in massive networks. *PNAS*, 110(36), 14534-14539.
    """
    from scipy.special import digamma

    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    N = len(nodes)
    node2idx = {n: i for i, n in enumerate(nodes)}

    if alpha is None:
        alpha = 1.0 / K
    if weight_mode not in ("binary", "poisson"):
        raise ValueError(f"weight_mode must be 'binary' or 'poisson', got {weight_mode!r}")

    # ── Adjacency matrix (directed; A[i,j] = edge i → j) ─────────────────────
    A = np.zeros((N, N), dtype=np.float32)
    for u, v, d in G.edges(data=True):
        i, j = node2idx[u], node2idx[v]
        if weight_mode == "binary":
            A[i, j] = 1.0
        else:
            A[i, j] = float(np.log1p(d.get("weight", 1.0)))

    if verbose:
        mem_mb = (N * N * K * 4 * 2) / 1024**2  # phi_send + phi_recv
        log.info("MMSB-EM: N=%d, K=%d, weight_mode=%s, est. memory %.0f MB",
                 N, K, weight_mode, mem_mb)
        log.info("  adjacency: %d edges, A.sum()=%.3g, A.mean()=%.4f",
                 int((A > 0).sum()), float(A.sum()), float(A.mean()))

    # ── Initial hard labels for symmetry breaking ────────────────────────────
    # Pure-random γ init traps the EM in a degenerate fixed point: when B is
    # uniform across blocks, the E-step cannot differentiate r values, so the
    # M-step keeps B uniform forever. We seed with a hard partition from
    # spectral / Louvain / random, then soften it.
    if init == "spectral":
        # K-means on top-K eigenvectors of the symmetrized adjacency
        from sklearn.cluster import KMeans
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        A_sym = (A + A.T) / 2.0
        # Add self-loops to ensure connectivity for the eigensolver
        np.fill_diagonal(A_sym, A_sym.sum(axis=1) / max(N, 1) + 1e-6)
        try:
            k_eig = min(K, N - 1)
            _, vecs = eigsh(csr_matrix(A_sym.astype(np.float64)), k=k_eig, which="LA")
            init_labels = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(vecs)
        except Exception as e:
            log.warning("spectral init failed (%s), falling back to random", e)
            init_labels = rng.integers(0, K, size=N)
    elif init == "louvain":
        try:
            import leidenalg, igraph as ig
            G_und = G.to_undirected()
            G_und.remove_edges_from(nx.selfloop_edges(G_und))
            g_ig = ig.Graph(directed=False)
            g_ig.add_vertices(N)
            g_ig.add_edges([(node2idx[u], node2idx[v]) for u, v in G_und.edges()])
            part = leidenalg.find_partition(
                g_ig, leidenalg.RBConfigurationVertexPartition, seed=seed
            )
            init_labels = np.array(part.membership)
            # If Louvain found more/fewer than K, remap via K-means on indicator embedding
            if init_labels.max() + 1 != K:
                from sklearn.cluster import KMeans
                one_hot = np.eye(init_labels.max() + 1)[init_labels]
                init_labels = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(one_hot)
        except Exception as e:
            log.warning("louvain init failed (%s), falling back to random", e)
            init_labels = rng.integers(0, K, size=N)
    else:  # "random"
        init_labels = rng.integers(0, K, size=N)

    # ── Variational parameters ───────────────────────────────────────────────
    # γ_p: high concentration on the initial label, small mass on others. Soft
    # enough that the EM can move nodes between blocks; hard enough that B is
    # immediately differentiated across blocks.
    gamma = np.full((N, K), alpha, dtype=np.float32)
    gamma[np.arange(N), init_labels] += 10.0
    gamma += rng.random((N, K)).astype(np.float32) * 0.1

    # φ[i, j, k] = ξ_{i→j, k} = per-pair multinomial — i's block when interacting
    # with j (the *sender* indicator for endpoint i). The receiver indicator for
    # the same pair is φ[j, i, k] = ξ_{j→i, k} — same tensor, transposed. There
    # is no separate "receive" tensor in Airoldi's formulation.
    # Initialise φ from the hard labels too (peaked at init_labels[i]).
    phi = np.full((N, N, K), 0.01, dtype=np.float32)
    for k in range(K):
        mask = (init_labels == k)
        phi[mask, :, k] = 0.9
    phi /= phi.sum(axis=2, keepdims=True)

    # Block interaction matrix — diagonal-dominant init so that block-pair
    # likelihoods differ from the start (constant-B init traps the EM in a
    # symmetric degenerate fixed point where all blocks are interchangeable).
    mean_density = float(A.mean()) if weight_mode == "binary" else max(float(A.mean()), 1e-3)
    B = np.full((K, K), mean_density * 0.5, dtype=np.float64)
    np.fill_diagonal(B, mean_density * 3.0)
    B *= 1.0 + rng.uniform(-0.1, 0.1, size=(K, K))
    if weight_mode == "binary":
        B = np.clip(B, 1e-3, 1.0 - 1e-3)
    else:
        B = np.maximum(B, 1e-3)

    # ── EM loop ──────────────────────────────────────────────────────────────
    for it in range(n_iter):
        Elog_pi = digamma(gamma) - digamma(gamma.sum(axis=1, keepdims=True))  # (N, K)

        # E-step: update φ[i, j, r] = ξ_{i→j, r}.
        # The receiver multinomial for the same pair is φ[j, i, s] = ξ_{j→i, s}.
        # `einsum('jis,rs->ijr', phi, logB)` computes Σ_s ξ_{j→i,s} · logB[r,s].
        if weight_mode == "binary":
            logB   = np.log(B + 1e-10)
            log1mB = np.log(1.0 - B + 1e-10)
            log_phi = (
                Elog_pi[:, None, :]
                + A[:, :, None]         * np.einsum('jis,rs->ijr', phi, logB)
                + (1.0 - A)[:, :, None] * np.einsum('jis,rs->ijr', phi, log1mB)
            )
        else:  # poisson
            logB = np.log(B + 1e-10)
            log_phi = (
                Elog_pi[:, None, :]
                + A[:, :, None] * np.einsum('jis,rs->ijr', phi, logB)
                - np.einsum('jis,rs->ijr', phi, B)
            )

        log_phi -= log_phi.max(axis=2, keepdims=True)  # numerical stability
        phi = np.exp(log_phi).astype(np.float32)
        phi /= phi.sum(axis=2, keepdims=True)

        # ── M-step ───────────────────────────────────────────────────────────
        # γ_{p,k} = α + Σ_q φ[p, q, k]   — Airoldi eq. 7
        gamma = (alpha + phi.sum(axis=1)).astype(np.float32)

        # B[r, s] = Σ_{ij} A[i,j] · φ[i,j,r] · φ[j,i,s] / Σ_{ij} φ[i,j,r] · φ[j,i,s]
        # — Airoldi eq. 6 (Bernoulli) / analogous for Poisson rate
        phi_T = phi.transpose(1, 0, 2)  # phi_T[i, j, s] = phi[j, i, s] = ξ_{j→i, s}
        numer = np.einsum('ijr,ijs->rs', phi * A[..., None], phi_T)
        denom = np.einsum('ijr,ijs->rs', phi, phi_T)
        B = (numer + 1e-10) / (denom + 1e-10)
        if weight_mode == "binary":
            B = np.clip(B, 1e-6, 1.0 - 1e-6)
        else:
            B = np.maximum(B, 1e-6)

        if verbose and (it % 5 == 0 or it == n_iter - 1):
            pi_tmp = gamma / gamma.sum(axis=1, keepdims=True)
            mean_entropy = float(
                -np.sum(pi_tmp * np.log(pi_tmp + 1e-10), axis=1).mean()
            )
            log.info(
                "  iter %3d/%d  mean π entropy=%.3f  B range=[%.3g, %.3g]",
                it + 1, n_iter, mean_entropy, float(B.min()), float(B.max()),
            )

    # ── Final: π = expected membership, hard partition = argmax ─────────────
    pi = gamma / gamma.sum(axis=1, keepdims=True)
    hard_partition = {nodes[i]: int(np.argmax(pi[i])) for i in range(N)}
    if verbose:
        log.info("MMSB-EM done: %d unique argmax blocks (of K=%d possible)",
                 len(set(hard_partition.values())), K)
    return pi, hard_partition