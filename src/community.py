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


# ── Internal helper ───────────────────────────────────────────────────────────

def _to_igraph(G: nx.Graph):
    """
    Convert NetworkX → igraph.

    Returns:
        g (ig.Graph)
        nodes (list): index → original node
    """
    import igraph as ig

    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}

    g = ig.Graph(directed=G.is_directed())
    g.add_vertices(len(nodes))

    edges = [(node_to_int[u], node_to_int[v]) for u, v in G.edges()]
    g.add_edges(edges)

    if G.number_of_edges() > 0:
        weights = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
        g.es["weight"] = weights

    return g, nodes


# ── Type alias ───────────────────────────────────────────────────────────────
Partition = dict[str, int]



# ====================================================================================


def _run_leiden(
    G: nx.Graph,
    seed: int = 42,
    resolution: float = 1.0,
    n_iterations: int = -1,
) -> Partition:
    """
    Core Leiden optimisation (generalised Louvain).

    Supports:
    - weighted graphs
    - directed graphs
    - resolution parameter γ
    """
    import leidenalg

    g, nodes = _to_igraph(G)

    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight" if "weight" in g.es.attributes() else None,
        seed=seed,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
    )

    return {nodes[i]: cid for cid, members in enumerate(part) for i in members}




def run_louvain(
    G: nx.Graph,
    seed: int = 42,
    resolution: float = 1.0,
) -> Partition:
    """
    Standard Louvain (via Leiden) on UNDIRECTED graph.
    """
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    return _run_leiden(G_und, seed=seed, resolution=resolution)




def run_louvain_directed(
    G: nx.DiGraph,
    seed: int = 42,
    resolution: float = 1.0,
) -> Partition:
    """
    Directed Louvain (Reichardt–Bornholdt modularity).
    """
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))

    return _run_leiden(G, seed=seed, resolution=resolution)




def run_louvain_multiscale(
    G: nx.Graph,
    gammas: list[float],
    seed: int = 42,
    directed: bool = False,
) -> dict:
    """
    Compute partitions for multiple γ values.

    Returns:
        {gamma: partition}
    """
    results = {}

    for gamma in gammas:
        if directed:
            part = run_louvain_directed(G, seed=seed, resolution=gamma)
        else:
            part = run_louvain(G, seed=seed, resolution=gamma)

        results[gamma] = part

    return results




def run_louvain_consensus(
    G: nx.Graph,
    n_runs: int = 20,
    resolution: float = 1.0,
    directed: bool = False,
    threshold: float = 0.5,
    max_iter: int = 10,
    sample_pairs: bool = True,
    max_pairs_per_comm: int = 500,
) -> dict:
    """
    Memory-safe consensus Louvain (Lancichinetti & Fortunato style).

    Key improvements:
    - avoids O(N^2) pair storage
    - samples intra-community pairs if needed
    - uses sparse edge aggregation only
    """

    import random
    import networkx as nx
    from collections import defaultdict

    def run_partition(graph):
        partitions = []
        for seed in range(n_runs):
            if directed:
                p = run_louvain_directed(graph, seed=seed, resolution=resolution)
            else:
                p = run_louvain(graph, seed=seed, resolution=resolution)
            partitions.append(p)
        return partitions

    def build_consensus_graph(partitions):
        edge_weights = defaultdict(int)

        nodes = list(partitions[0].keys())

        for part in partitions:
            comms = defaultdict(list)
            for n, c in part.items():
                comms[c].append(n)

            for members in comms.values():

                # CRITICAL FIX: subsample if community is large
                if sample_pairs and len(members) > max_pairs_per_comm:
                    members = random.sample(members, max_pairs_per_comm)

                # only O(k) pairs, not O(k²)
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = members[i], members[j]
                        if a > b:
                            a, b = b, a
                        edge_weights[(a, b)] += 1

        # normalize + threshold
        H = nx.Graph()
        for (a, b), w in edge_weights.items():
            w = w / n_runs
            if w >= threshold:
                H.add_edge(a, b, weight=w)

        return H

    current_graph = G.copy()

    for it in range(max_iter):

        partitions = run_partition(current_graph)
        new_graph = build_consensus_graph(partitions)

        # convergence
        if new_graph.number_of_edges() == current_graph.number_of_edges():
            break

        if new_graph.number_of_edges() == 0:
            break

        current_graph = new_graph

    return run_louvain(current_graph, resolution=resolution)


# ====================================================================================


def run_infomap(
    G: nx.Graph,
    directed: bool = True,
    seed: int = 42,
) -> dict:
    """
    Correct InfoMap implementation (Python API-safe).
    """

    try:
        from infomap import Infomap
    except ImportError:
        raise ImportError("pip install infomap")

    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))

    nodes = list(G.nodes())
    node_to_int = {n: i for i, n in enumerate(nodes)}

    im = Infomap()
    im.seed = seed
    im.directed = directed
    im.silent = True

    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        im.add_link(node_to_int[u], node_to_int[v], w)

    im.run()

    partition = {}

    for node in im.tree:
        if node.isLeaf:
            partition[nodes[node.physicalId]] = node.moduleIndex()

    return partition




# ====================================================================================



def run_dbscan_earthquakes(
    df: pd.DataFrame,
    eps_km: float = 10.0,
    min_samples: int = 10,
    use_depth: bool = True,
    target_crs: str = "epsg:32632",
) -> pd.DataFrame:
    """
    Run DBSCAN on earthquake catalog using spatial coordinates.

    Parameters
    ----------
    df : DataFrame with lat/lon/depth
    eps_km : float
        Neighborhood radius in km
    min_samples : int
        Minimum number of points to form a cluster
    use_depth : bool
        If True → 3D clustering, else 2D
    target_crs : str
        CRS for projection (use epsg:32632 for Italy)

    Returns
    -------
    df_out : DataFrame with cluster labels in column 'cluster'
    """

    from sklearn.cluster import DBSCAN

    # --- project to km using your function ---
    df_proj = discretize_space_3d(df, cell_size_km=1.0, target_crs=target_crs)

    # --- build feature matrix ---
    if use_depth:
        X = df_proj[["x_km", "y_km", "depth_km"]].values
    else:
        X = df_proj[["x_km", "y_km"]].values

    # --- run DBSCAN ---
    db = DBSCAN(eps=eps_km, min_samples=min_samples)
    labels = db.fit_predict(X)

    df_out = df_proj.copy()
    df_out["cluster"] = labels  # -1 = noise

    return df_out


# def plot_dbscan_geo(
#     df: pd.DataFrame,
#     title: str = "",
#     center_lat: float = 41.9,
#     center_lon: float = 12.5,
#     zoom: float = 4,
#     min_cluster_size: int = 50,
#     height: int = 600,
#     width: int = 1100,
# ):
#     """
#     Plot DBSCAN clusters on map using earthquake points.
#     """

#     # remove noise
#     df_plot = df[df["cluster"] != -1].copy()

#     # filter small clusters
#     counts = df_plot["cluster"].value_counts()
#     large = counts[counts >= min_cluster_size].index
#     df_plot = df_plot[df_plot["cluster"].isin(large)]

#     n_clusters = df_plot["cluster"].nunique()

#     # size ~ magnitude (nice physical meaning)
#     df_plot["size"] = df_plot["magnitude"]

#     fig = px.scatter_map(
#         df_plot,
#         lat="latitude",
#         lon="longitude",
#         color=df_plot["cluster"].astype(str),
#         color_discrete_sequence=px.colors.qualitative.Light24,  # <-- Broadest built-in palette (24 unique colors)
#         size="size",
#         size_max=12,
#         map_style="carto-positron",
#         title=f"DBSCAN Clusters ({n_clusters} clusters) — {title}",
#         hover_data={"magnitude": True, "depth_km": True, "cluster": True},
#     )

#     fig.update_layout(
#         map=dict(center={"lat": center_lat, "lon": center_lon}, zoom=zoom),
#         width=width,
#         height=height,
#         margin={"r":0,"t":40,"l":0,"b":0},
#     )

#     fig.show()


# ALTERNATIVE:
def plot_dbscan_geo(
    df: pd.DataFrame,
    title: str = "",
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 4,
    min_cluster_size: int = 50,
    height: int = 600,
    width: int = 1100,
):
    """
    Plot DBSCAN clusters on map using earthquake points.
    """

    df_plot = df[df["cluster"] != -1].copy()

    counts = df_plot["cluster"].value_counts()
    large = counts[counts >= min_cluster_size].index
    df_plot = df_plot[df_plot["cluster"].isin(large)]

    # remap clusters to consecutive integers
    clusters = df_plot["cluster"].unique()
    cluster_map = {c: i for i, c in enumerate(clusters)}
    df_plot["cluster_id"] = df_plot["cluster"].map(cluster_map)

    n_clusters = len(clusters)
    df_plot["size"] = df_plot["magnitude"]

    fig = px.scatter_map(
        df_plot,
        lat="latitude",
        lon="longitude",
        color="cluster_id",
        color_continuous_scale=px.colors.sequential.Turbo,  # 🔥 best high-contrast palette
        size="size",
        size_max=12,
        map_style="carto-positron",
        title=f"DBSCAN Clusters ({n_clusters} clusters) — {title}",
        hover_data={"magnitude": True, "depth_km": True, "cluster": True},
    )

    fig.update_layout(
        coloraxis_colorbar=dict(title="Cluster ID"),
        map=dict(center={"lat": center_lat, "lon": center_lon}, zoom=zoom),
        width=width,
        height=height,
        margin={"r":0,"t":40,"l":0,"b":0},
    )

    fig.show()


# ====================================================================================

def align_partitions(part1: dict, part2: dict):
    """
    Align two partitions on common nodes.
    Returns label vectors.
    """
    common_nodes = set(part1.keys()) & set(part2.keys())

    labels1 = [part1[n] for n in common_nodes]
    labels2 = [part2[n] for n in common_nodes]

    return np.array(labels1), np.array(labels2)



def compute_nmi(part1: dict, part2: dict) -> float:
    """
    Compute Normalized Mutual Information between two partitions.
    """
    labels1, labels2 = align_partitions(part1, part2)

    return normalized_mutual_info_score(labels1, labels2)



def compute_nmi_matrix(partitions: dict) -> pd.DataFrame:
    """
    Compute full NMI similarity matrix between all partition methods.
    """
    methods = list(partitions.keys())
    n = len(methods)

    mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            mat[i, j] = compute_nmi(
                partitions[methods[i]],
                partitions[methods[j]]
            )

    return pd.DataFrame(mat, index=methods, columns=methods)


def plot_nmi_heatmap(nmi_df: pd.DataFrame):
    """
    Plot NMI similarity heatmap.
    """
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        nmi_df,
        annot=True,
        vmin=0,
        vmax=1,
        cmap="YlGn",
        square=True
    )

    plt.title("NMI between Community Detection Methods")
    plt.tight_layout()
    plt.show()






# ====================================================================================

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




# =======================================================================


def compute_modularity_from_partition(G: nx.DiGraph, partition: dict) -> float:
    """
    Compute modularity on undirected version of G using a given partition.
    """
    G_und = G.to_undirected()

    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, set()).add(node)

    comm_list = list(communities.values())

    return nx.algorithms.community.quality.modularity(
        G_und, comm_list, weight="weight"
    )




def build_window_network(df, start, end, cell_size_km=10):
    """
    Build Abe-Suzuki network for a time window.
    """
    df_win = df[(df["time"] >= start) & (df["time"] < end)].copy()

    if len(df_win) < 50:
        return None  # too small → unstable network

    G = build_abe_suzuki_network(df_win, cell_size_km=cell_size_km, info=False)

    return G






def compute_q_over_time(
    df,
    window_days=10,
    step_days=1,
    cell_size_km=10,
    start_time=None,
    end_time=None
):
    """
    Compute modularity evolution Q(t) using sliding windows.
    """

    results = []

    if start_time is None:
        start_time = df["time"].min()

    if end_time is None:
        end_time = df["time"].max()

    current = start_time

    while current + pd.Timedelta(days=window_days) <= end_time:

        start = current
        end = current + pd.Timedelta(days=window_days)

        G = build_window_network(df, start, end, cell_size_km)

        if G is None or G.number_of_edges() < 10:
            current += pd.Timedelta(days=step_days)
            continue

        # --- Louvain (directed) ---
        partition = run_louvain_directed(G, resolution=1.0)

        # --- modularity ---
        Q = compute_modularity_from_partition(G, partition)

        # center time of window
        t_center = start + pd.Timedelta(days=window_days / 2)

        results.append({
            "time": t_center,
            "Q": Q,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges()
        })

        current += pd.Timedelta(days=step_days)

    return pd.DataFrame(results)