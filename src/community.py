"""
Community detection suite for the Abe-Suzuki earthquake network.

Four methods, all returning the same {node: community_id} dict so they can be
passed to any downstream function interchangeably:

  Louvain           — fast modularity optimisation (undirected, unweighted)
  Consensus Louvain — 100-run co-occurrence → agglomerative clustering;
                      removes partition instability inherent to single-run Louvain
  Spectral          — k-way spectral clustering on the normalised Laplacian
                      (Jordan-Weiss); k taken from Louvain community count
  InfoMap           — flow-based compression (directed, weighted); identifies
                      communities as regions where random walkers stay trapped

NMI utilities compare any pair of partitions; the heatmap shows pairwise
agreement across all four methods.
"""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)

_PALETTE = px.colors.qualitative.Bold


# ── Type alias ───────────────────────────────────────────────────────────────
Partition = dict[str, int]


# ── Method 1: single-run Louvain ─────────────────────────────────────────────

def run_louvain(G: nx.Graph, seed: int = 42) -> Partition:
    """
    Run one pass of the NetworkX Louvain algorithm.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph (self-loops removed).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Partition
        ``{node_id: community_int}`` mapping.
    """
    communities = nx.community.louvain_communities(G, seed=seed)
    return {node: cid for cid, nodes in enumerate(communities) for node in nodes}


# ── Method 2: consensus Louvain ───────────────────────────────────────────────

def run_consensus_louvain(
    G: nx.Graph,
    n_runs: int = 100,
    seed: int = 42,
    max_nodes: int = 50_000,
) -> Partition:
    """
    Consensus Louvain: run Louvain n_runs times, build a co-occurrence matrix,
    then use agglomerative clustering to find a stable partition.

    The co-occurrence matrix C[i,j] is the fraction of runs where nodes i and j
    land in the same community. Agglomerative clustering on (1-C) with
    ``n_clusters`` equal to the median number of Louvain communities removes
    the stochasticity that makes single-run Louvain unreliable.

    For graphs with more than ``max_nodes`` nodes the N×N co-occurrence matrix
    would be prohibitively large (e.g. 173 GiB for N=215k). In that case the
    function falls back to a single Louvain run.

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
    """
    nodes = list(G.nodes())
    n = len(nodes)

    if n > max_nodes:
        mem_gib = n * n * 4 / 1024**3
        log.warning(
            "Consensus Louvain skipped: N=%d > max_nodes=%d "
            "(co-occurrence matrix would require %.0f GiB). "
            "Falling back to single Louvain run.",
            n, max_nodes, mem_gib,
        )
        comms = nx.community.louvain_communities(G, seed=seed)
        return {v: i for i, members in enumerate(comms) for v in members}

    idx = {node: i for i, node in enumerate(nodes)}
    co = np.zeros((n, n), dtype=np.float32)
    k_counts = []

    log.info("Consensus Louvain: %d runs...", n_runs)
    for r in range(n_runs):
        comms = nx.community.louvain_communities(G, seed=seed + r)
        k_counts.append(len(comms))
        for members in comms:
            ids = [idx[v] for v in members]
            for i in ids:
                co[i, ids] += 1.0

    co /= n_runs                              # fraction of runs in same community
    k_consensus = int(np.median(k_counts))
    log.info("  median k=%d across runs", k_consensus)

    dist = 1.0 - co
    agg = AgglomerativeClustering(
        n_clusters=k_consensus,
        metric="precomputed",
        linkage="average",
    )
    labels = agg.fit_predict(dist)
    return {nodes[i]: int(labels[i]) for i in range(n)}


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
    fig, ax = plt.subplots(figsize=(6, 5))
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

    fig = px.scatter_map(
        df,
        lat="lat", lon="lon",
        color="community",
        size="degree", size_max=18,
        color_discrete_sequence=_PALETTE,
        hover_name="community",
        hover_data={"lat": ":.3f", "lon": ":.3f", "degree": True},
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
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError("pip install leidenalg python-igraph") from exc

    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}

    g = ig.Graph(directed=True)
    g.add_vertices(len(nodes))
    g.vs["name"] = nodes

    edges = [(node_to_int[u], node_to_int[v]) for u, v in G.edges()]
    weights = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    g.add_edges(edges)
    g.es["weight"] = weights

    part = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=seed,
    )
    log.info("Directed Louvain (Leiden): %d communities, Q=%.4f",
             len(part), part.modularity)
    return {nodes[i]: cid for cid, members in enumerate(part) for i in members}


# ── Granovetter weak ties ─────────────────────────────────────────────────────

def analyze_weak_ties(
    G: nx.Graph,
    community_map: Partition,
    n_bins: int = 8,
) -> pd.DataFrame:
    """
    Test Granovetter's weak-tie hypothesis on the earthquake network.

    Edges are ranked by weight and split into ``n_bins`` quantile bins.
    For each bin, the *bridge fraction* is computed: the proportion of edges
    that connect nodes in **different** Louvain communities.

    Granovetter's prediction: weak edges (low weight) have a higher bridge
    fraction than strong edges (high weight), which mostly connect nodes
    within the same community.

    Seismological interpretation: rare long-distance seismic transitions
    (weight = 1–2, crossing community / fault-zone boundaries) are the
    topological bridges that make the network small-world.  Frequent
    transitions (high weight) stay within the same seismogenic zone.

    Parameters
    ----------
    G : nx.Graph
        Undirected earthquake network (GCC, no self-loops).
    community_map : Partition
        ``{node_id: community_int}`` from any community-detection method.
    n_bins : int
        Number of weight quantile bins.

    Returns
    -------
    pd.DataFrame
        Columns: ``weight_bin``, ``weight_min``, ``weight_max``,
        ``n_edges``, ``n_bridges``, ``bridge_fraction``.
    """
    weights = np.array([d.get("weight", 1.0) for _, _, d in G.edges(data=True)])
    is_bridge = np.array([
        community_map.get(u, -1) != community_map.get(v, -2)
        for u, v in G.edges()
    ])

    w_min, w_max = weights.min(), weights.max()

    # Use log-spaced edges when the weight range spans > 1 order of magnitude
    # (typical for Abe-Suzuki networks: most edges weight=1, few weight=100s).
    # Quantile bins collapse to 1 bin in this case because the median == 1.
    if w_max / max(w_min, 1) >= 10:
        edges = np.unique(np.round(
            np.logspace(np.log10(w_min), np.log10(w_max + 1), n_bins + 1)
        ).astype(int))
    else:
        edges = np.unique(np.quantile(weights, np.linspace(0, 1, n_bins + 1)))

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (weights >= lo) & (weights < hi)
        if lo == edges[-2]:          # include upper boundary in last bin
            mask = (weights >= lo) & (weights <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        br = int(is_bridge[mask].sum())
        rows.append({
            "weight_bin":      f"[{lo:.0f}, {hi:.0f}]",
            "weight_min":      float(lo),
            "weight_max":      float(hi),
            "n_edges":         n,
            "n_bridges":       br,
            "bridge_fraction": round(br / n, 4),
        })
    return pd.DataFrame(rows)


def plot_weak_ties(df_wt: pd.DataFrame, title: str = "", save: bool = True) -> None:
    """
    Bar chart of bridge fraction per weight bin.

    Bars are coloured from blue (weak) to red (strong) to emphasise the
    expected left-to-right decline under Granovetter's hypothesis.

    Parameters
    ----------
    df_wt : pd.DataFrame
        Output of :func:`analyze_weak_ties`.
    title : str
        Figure title suffix.
    """
    n = len(df_wt)
    cmap = plt.cm.RdYlBu_r
    colors = cmap(np.linspace(0.1, 0.9, n))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(n), df_wt["bridge_fraction"],
        color=colors, edgecolor="k", linewidth=0.6,
    )
    for bar, row in zip(bars, df_wt.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{row.bridge_fraction:.2f}\n(n={row.n_edges})",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(n))
    ax.set_xticklabels(df_wt["weight_bin"], rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Edge weight bin (transition count)", fontsize=12)
    ax.set_ylabel("Bridge fraction\n(proportion crossing community boundary)", fontsize=11)
    ax.set_title(f"Granovetter's Weak-Tie Test — {title}", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.axhline(df_wt["bridge_fraction"].mean(), color="gray", ls="--", lw=1.2,
               label=f"Mean = {df_wt['bridge_fraction'].mean():.2f}")
    ax.legend(fontsize=10)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["Weak ties\n(0.00)", "0.25", "0.50", "0.75", "Strong ties\n(1.00)"])
    plt.tight_layout()
    if save:
        savefig(f"weak_ties_{_slug(title)}")
    plt.show()


# ── Condensation graph (SCC analysis) ────────────────────────────────────────

def analyze_condensation(
    G: nx.DiGraph,
    min_scc_size: int = 5,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """
    Compute the condensation graph of a directed network.

    Each strongly connected component (SCC) is collapsed to a single node.
    The condensation is a DAG whose structure reveals the macroscopic flow
    of seismic activity:

    * **Source SCCs** — no incoming edges from other SCCs; regions that
      consistently trigger others without being triggered back.
    * **Sink SCCs**   — no outgoing edges to other SCCs; terminal receptors
      of seismic sequences.
    * **Transit SCCs** — both incoming and outgoing; intermediate relay zones.

    Parameters
    ----------
    G : nx.DiGraph
        Directed earthquake network (with or without self-loops).
    min_scc_size : int
        SCCs smaller than this are grouped into a ``"trivial"`` category
        (most SCCs in sparse directed graphs are singletons).

    Returns
    -------
    C : nx.DiGraph
        The condensation graph (nodes = SCCs, edges = inter-SCC transitions).
        Each node carries: ``size`` (number of original nodes),
        ``role`` (``"source"`` / ``"sink"`` / ``"transit"`` / ``"isolated"``),
        ``mean_lat``, ``mean_lon``.
    df : pd.DataFrame
        One row per SCC with columns: ``scc_id``, ``size``, ``role``,
        ``mean_lat``, ``mean_lon``, ``in_degree``, ``out_degree``.
    """
    sccs = list(nx.strongly_connected_components(G))
    log.info("SCCs: %d total  (largest: %d nodes)",
             len(sccs), max(len(s) for s in sccs))

    C = nx.condensation(G, scc=sccs)

    rows = []
    for scc_id in C.nodes():
        members = list(C.nodes[scc_id]["members"])
        size    = len(members)
        in_d    = C.in_degree(scc_id)
        out_d   = C.out_degree(scc_id)

        if in_d == 0 and out_d == 0:
            role = "isolated"
        elif in_d == 0:
            role = "source"
        elif out_d == 0:
            role = "sink"
        else:
            role = "transit"

        lats = [G.nodes[n].get("lat") for n in members if "lat" in G.nodes[n]]
        lons = [G.nodes[n].get("lon") for n in members if "lon" in G.nodes[n]]

        C.nodes[scc_id]["size"]     = size
        C.nodes[scc_id]["role"]     = role
        C.nodes[scc_id]["mean_lat"] = float(np.mean(lats)) if lats else float("nan")
        C.nodes[scc_id]["mean_lon"] = float(np.mean(lons)) if lons else float("nan")

        rows.append({
            "scc_id":    scc_id,
            "size":      size,
            "role":      role,
            "mean_lat":  C.nodes[scc_id]["mean_lat"],
            "mean_lon":  C.nodes[scc_id]["mean_lon"],
            "in_degree":  in_d,
            "out_degree": out_d,
        })

    df = pd.DataFrame(rows)
    log.info("Roles — source: %d  sink: %d  transit: %d  isolated: %d",
             (df["role"] == "source").sum(),   (df["role"] == "sink").sum(),
             (df["role"] == "transit").sum(),  (df["role"] == "isolated").sum())
    return C, df


def plot_condensation_geo(
    df_scc: pd.DataFrame,
    min_size: int = 2,
    title: str = "",
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 0,
    bounds: dict | None = None,
    height: int = 600,
    width: int = 1100,
    save: bool = True,
) -> None:
    """
    Interactive geo map of SCCs coloured by role (source / sink / transit).

    Only SCCs with at least ``min_size`` members are shown; singleton SCCs
    (which dominate in sparse directed graphs) are hidden to reduce clutter.

    Parameters
    ----------
    df_scc : pd.DataFrame
        Output ``df`` from :func:`analyze_condensation`.
    min_size : int
        Minimum SCC size to plot.
    title : str
        Figure title suffix.
    center_lat, center_lon, zoom : float
        Initial map view.
    """
    df = df_scc[df_scc["size"] >= min_size].dropna(subset=["mean_lat", "mean_lon"])

    color_map = {"source": "#2a9d8f", "sink": "#e63946",
                 "transit": "#f4a261", "isolated": "#aaa"}
    df = df.copy()
    df["color"] = df["role"].map(color_map)

    fig = px.scatter_map(
        df,
        lat="mean_lat", lon="mean_lon",
        color="role",
        size="size", size_max=25,
        color_discrete_map=color_map,
        hover_name="scc_id",
        hover_data={"size": True, "role": True,
                    "in_degree": True, "out_degree": True},
        map_style="carto-positron",
        title=(f"Condensation Graph — SCC Roles: {title}\n"
               f"Green=source (triggers others)  Red=sink (receives only)  "
               f"Orange=transit"),
    )
    map_cfg: dict = {"center": {"lat": center_lat, "lon": center_lon}, "zoom": zoom}
    if bounds is not None:
        map_cfg["bounds"] = bounds
    fig.update_layout(margin={"r": 0, "t": 60, "l": 0, "b": 0}, width=width, height=height, map=map_cfg)
    if save:
        save_plotly(fig, f"condensation_geo_{_slug(title)}")
    fig.show()
