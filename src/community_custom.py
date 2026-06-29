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
from src.network_custom import build_abe_suzuki_network_custom_hybrid

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
    partition_type: str = "CPM",   # NEW
) -> Partition:
    """
    Leiden optimisation for hybrid weighted earthquake network.

    partition_type:
        - "CPM" (recommended for hybrid network)
        - "RB"  (classic modularity)
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
) -> Partition:
    """
    Community detection on undirected hybrid network.
    """
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    return _run_leiden(G_und, seed=seed, resolution=resolution, partition_type="CPM")


def run_louvain_directed_hybrid(
    G: nx.DiGraph,
    seed: int = 42,
    resolution: float = 1.0,
) -> Partition:
    """
    Directed community detection for hybrid network.
    """
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))

    return _run_leiden(G, seed=seed, resolution=resolution, partition_type="CPM")




def run_louvain_consensus_hybrid(
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
    Consensus community detection for hybrid weighted network.

    Uses hybrid Louvain (Leiden CPM) and builds a co-assignment graph.
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

    im = Infomap()
    im.seed = seed
    im.directed = directed
    im.silent = True

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




def run_bigclam_hybrid(
    G: nx.Graph,
    k: int,
    n_iter: int = 100,
    lr: float = 0.005,
    seed: int = 42,
    use_weights: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    BigCLAM adapted to the hybrid Abe-Suzuki network.

    IMPORTANT CHANGE vs original:
    - The hybrid network is weighted + directed
    - BigCLAM assumes undirected binary adjacency

    Therefore we:
    - symmetrize the graph
    - optionally binarize weights (default: YES, ignore weights)

    Returns
    -------
    F : (N, K) membership matrix
    partition : hard assignment (argmax)
    """

    import numpy as np
    import networkx as nx

    nodes = list(G.nodes())
    N = len(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # ── 1. Convert to UNDIRECTED ─────────────────────────────────────────────
    G0 = G.to_undirected()
    G0.remove_edges_from(nx.selfloop_edges(G0))

    # ── 2. Optionally binarize weights ────────────────────────────────────────
    # BigCLAM is defined for adjacency, not weights
    if use_weights:
        # soft hack: threshold small weights
        # (keeps stronger seismic links only)
        threshold = np.percentile(
            [d.get("weight", 1.0) for _, _, d in G0.edges(data=True)], 50
        )
        G0 = nx.Graph(
            (u, v, 1.0)
            for u, v, d in G0.edges(data=True)
            if d.get("weight", 1.0) >= threshold
        )
    else:
        G0 = nx.Graph((u, v) for u, v in G0.edges())

    # ── 3. Build neighbor list ───────────────────────────────────────────────
    nb_lists = [
        np.array([node_to_idx[v] for v in G0.neighbors(nodes[u])], dtype=np.int32)
        for u in range(N)
    ]

    # ── 4. Initialize F ───────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    F = rng.random((N, k)).astype(np.float32) + 1e-3

    log.info("BigCLAM (hybrid): N=%d, K=%d, iters=%d", N, k, n_iter)

    # ── 5. Coordinate ascent ─────────────────────────────────────────────────
    for epoch in range(n_iter):
        S = F.sum(axis=0)

        for u in range(N):
            nb_idx = nb_lists[u]
            if len(nb_idx) == 0:
                continue

            F_nb = F[nb_idx]

            dots = F_nb @ F[u]
            dots = np.clip(dots, 1e-6, 30.0)

            exp_neg = np.exp(-dots)
            sigm = exp_neg / (1.0 - exp_neg + 1e-10)

            grad_nb = (F_nb * sigm[:, None]).sum(axis=0)
            grad_nnb = -(S - F_nb.sum(axis=0) - F[u])

            F[u] += lr * (grad_nb + grad_nnb)
            F[u] = np.maximum(1e-3, F[u])

        if (epoch + 1) % 10 == 0:
            log.info("  epoch %3d/%d", epoch + 1, n_iter)

    # ── 6. Hard partition for comparison ─────────────────────────────────────
    partition = {nodes[i]: int(np.argmax(F[i])) for i in range(N)}

    log.info("BigCLAM done: %d communities", len(set(partition.values())))

    return F, partition



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





# ---------------------------------------------------------------------------------------------
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


def build_window_network_hybrid(
    df,
    start,
    end,
    cell_size_km=10,
    alpha=1.0,
    r0=50.0,
    tau_days=0.5,
):
    """
    Build HYBRID Abe–Suzuki network for a time window.
    """

    df_win = df[(df["time"] >= start) & (df["time"] < end)].copy()

    if len(df_win) < 50:
        return None

    G = build_abe_suzuki_network_custom_hybrid(
        df_win,
        cell_size_km=cell_size_km,
        spatial_threshold_km=300.0,
        time_threshold_sec=24 * 3600,
        alpha=alpha,
        tau=tau_days * 86400.0,
        r0=r0,
        info=False
    )

    return G





def compute_q_over_time_hybrid(
    df,
    window_days=10,
    step_days=1,
    cell_size_km=10,
    start_time=None,
    end_time=None,
    alpha=1.0,
    r0=50.0,
    tau_days=0.5,
    resolution=1.0,
):
    """
    Modularity evolution Q(t) for HYBRID network.
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

        G = build_window_network_hybrid(
            df, start, end,
            cell_size_km=cell_size_km,
            alpha=alpha,
            r0=r0,
            tau_days=tau_days
        )

        if G is None or G.number_of_edges() < 10:
            current += pd.Timedelta(days=step_days)
            continue

        # ── Louvain (HYBRID) ──
        partition = run_louvain_directed_hybrid(
            G,
            resolution=resolution
        )

        # ── Modularity ──
        Q = compute_modularity_from_partition(G, partition)

        t_center = start + pd.Timedelta(days=window_days / 2)

        results.append({
            "time": t_center,
            "Q": Q,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges()
        })

        current += pd.Timedelta(days=step_days)

    return pd.DataFrame(results)