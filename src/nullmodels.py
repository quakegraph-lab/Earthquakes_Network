"""
Null graph models for benchmarking the Abe-Suzuki earthquake network.

All functions operate on undirected graphs without self-loops, consistent with
the macroscopic analysis in the network notebooks.

Models
------
ER     – Erdős–Rényi G(n, m): random wiring, Poisson degree distribution.
BA     – Barabási–Albert preferential attachment: power-law degree distribution.
WS     – Watts–Strogatz: regular lattice + rewiring; high C, short L (small-world).
SBM    – Stochastic Block Model: edge probabilities fitted from Louvain partition.
Config – Configuration model: preserves exact degree sequence (Molloy-Reed).

References
----------
Erdős, P., & Rényi, A. (1959). On random graphs. Publicationes Mathematicae.
Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks.
  Science, 286(5439), 509–512.
Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world'
  networks. Nature, 393, 440–442.
Holland, P. W., Laskey, K. B., & Leinhardt, S. (1983). Stochastic blockmodels.
  Social Networks, 5(2), 109–137.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)

# ── Consistent visual identity across all null-model plots ───────────────────
_MODELS = {
    "Real":   dict(color="#e63946", ls="-",          lw=2.5, label="Earthquake Network",    marker="o"),
    "ER":     dict(color="#457b9d", ls="--",          lw=1.8, label="Erdős–Rényi",           marker=None),
    "BA":     dict(color="#2a9d8f", ls="-.",          lw=1.8, label="Barabási–Albert",       marker=None),
    "WS":     dict(color="#f4a261", ls=":",           lw=1.8, label="Watts–Strogatz",        marker=None),
    "SBM":    dict(color="#9b5de5", ls=(0,(3,1,1,1)),lw=1.8, label="Stoch. Block Model",    marker=None),
    "Config": dict(color="#e9c46a", ls=(0,(5,2)),    lw=1.8, label="Configuration Model",   marker=None),
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_binned_pk(
    degrees: list[float],
    n_bins: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Log-binned degree distribution: returns (bin_centers, prob_density)."""
    arr = np.array([d for d in degrees if d > 0], dtype=float)
    if len(arr) < 2:
        return np.array([]), np.array([])
    bins    = np.logspace(np.log10(arr.min()), np.log10(arr.max()), n_bins)
    counts, edges = np.histogram(arr, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths  = np.diff(edges)
    P_k     = counts / (len(arr) * widths)
    mask    = P_k > 0
    return centers[mask], P_k[mask]


def _approx_avg_path_length(
    G: nx.Graph,
    n_samples: int = 150,
    seed: int = 42,
) -> float:
    """
    Approximate average shortest path length via BFS from random source nodes.

    Uses exact computation for n ≤ 500; BFS sampling otherwise.
    """
    n = G.number_of_nodes()
    if n == 0:
        return float("nan")
    if n <= 500:
        try:
            return nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            return float("nan")

    rng   = np.random.default_rng(seed)
    nodes = list(G.nodes())
    sample = rng.choice(nodes, size=min(n_samples, n), replace=False)

    total, count = 0.0, 0
    for src in sample:
        lengths = nx.single_source_shortest_path_length(G, src)
        for d in lengths.values():
            if d > 0:
                total += d
                count += 1
    return total / count if count > 0 else float("nan")


def _mle_gamma(degrees: list[float], k_min: float = 10.0) -> float:
    from src.metrics import estimate_gamma_mle
    return estimate_gamma_mle(degrees, k_min)


# ── SBM builder ───────────────────────────────────────────────────────────────

def _build_sbm(
    G: nx.Graph,
    community_map: dict,
    seed: int = 42,
) -> nx.Graph:
    """
    Fit block edge probabilities from a Louvain community partition and
    generate an SBM random graph.

    Communities with fewer than 2 nodes are merged into a single residual
    block so the SBM preserves the total node count.

    Parameters
    ----------
    G : nx.Graph
        Reference undirected graph (no self-loops).
    community_map : dict
        Mapping {node_id: community_int} from Louvain detection.
    seed : int
        RNG seed passed to ``nx.stochastic_block_model``.

    Returns
    -------
    nx.Graph
        Undirected SBM graph with the same number of nodes as G.
    """
    from collections import Counter

    # Assign every node in G to a block (singletons → block -1 "other")
    comm_sizes = Counter(community_map.get(n) for n in G.nodes()
                         if n in community_map)
    valid      = {c for c, cnt in comm_sizes.items() if cnt >= 2}
    OTHER      = -1

    def block(node):
        c = community_map.get(node, OTHER)
        return c if c in valid else OTHER

    # Canonical block ordering (OTHER last for readability)
    all_blocks = sorted(valid) + ([OTHER] if any(block(n) == OTHER for n in G.nodes()) else [])
    b_idx      = {b: i for i, b in enumerate(all_blocks)}
    n_blocks   = len(all_blocks)

    ordered_nodes  = list(G.nodes())
    node_block_idx = [b_idx[block(n)] for n in ordered_nodes]
    sizes = [sum(1 for bi in node_block_idx if bi == i) for i in range(n_blocks)]

    # Count undirected edges between/within blocks
    edge_mat = np.zeros((n_blocks, n_blocks))
    for u, v in G.edges():
        i = b_idx[block(u)]
        j = b_idx[block(v)]
        edge_mat[i][j] += 1
        if i != j:
            edge_mat[j][i] += 1   # keep matrix symmetric for off-diagonal

    # Convert to probability (cap at 1.0 for numerical safety)
    prob_mat = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            if i == j:
                denom = sizes[i] * (sizes[i] - 1) / 2.0
            else:
                denom = sizes[i] * sizes[j]
            if denom > 0:
                prob_mat[i][j] = min(1.0, edge_mat[i][j] / denom)

    log.info(
        "SBM: %d blocks, sizes min=%d max=%d, p_in=%.4f p_out_mean=%.6f",
        n_blocks, min(sizes), max(sizes),
        float(np.diag(prob_mat).mean()),
        float(prob_mat[~np.eye(n_blocks, dtype=bool)].mean()),
    )
    return nx.stochastic_block_model(sizes, prob_mat.tolist(), seed=seed)


# ── Configuration model ───────────────────────────────────────────────────────

def _build_config_model(G: nx.Graph, seed: int = 42) -> nx.Graph:
    """
    Molloy-Reed configuration model: random graph with the exact degree
    sequence of G. Multi-edges and self-loops are removed after construction,
    so the resulting degree sequence may differ slightly from G's.
    """
    degree_sequence = [d for _, d in G.degree()]
    G_cfg = nx.configuration_model(degree_sequence, seed=seed)
    G_cfg = nx.Graph(G_cfg)                        # collapse multi-edges
    G_cfg.remove_edges_from(nx.selfloop_edges(G_cfg))
    return G_cfg


# ── Public API ────────────────────────────────────────────────────────────────

def build_null_graphs(
    G: nx.Graph,
    community_map: Optional[dict] = None,
    include_config: bool = False,
    seed: int = 42,
) -> dict[str, nx.Graph]:
    """
    Generate ER, BA, WS, and optionally SBM null graphs matching G's size.

    Parameters
    ----------
    G : nx.Graph
        Undirected reference graph without self-loops (typically the giant
        component of the earthquake network).
    community_map : dict, optional
        Mapping ``{node: community_int}`` from Louvain detection. If provided,
        an SBM null graph whose block structure mirrors the Louvain partition
        is added to the output. Omit to skip SBM.
    include_config : bool
        If True, add a configuration-model null that preserves the exact
        degree sequence (Molloy-Reed). Default False to keep backward compat.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, nx.Graph]
        Keys ``"ER"``, ``"BA"``, ``"WS"``, and optionally ``"SBM"``.

    Notes
    -----
    Parameter matching strategy:

    * **ER** – G(n, m): exact same node count N and edge count M as G.
    * **BA** – m_attach = max(1, round(avg_degree / 2)); produces ~same M.
    * **WS** – k = nearest even integer to avg_degree, p = 0.1 (standard
      small-world rewiring probability).
    * **SBM** – block edge probabilities fitted from the Louvain partition
      via :func:`_build_sbm`.
    """
    n     = G.number_of_nodes()
    m     = G.number_of_edges()
    avg_d = 2 * m / n if n > 0 else 2.0

    log.info("Building null models: n=%d  m=%d  avg_deg=%.2f", n, m, avg_d)
    graphs: dict[str, nx.Graph] = {}

    # Erdős–Rényi
    graphs["ER"] = nx.gnm_random_graph(n, m, seed=seed)
    log.info("ER:  %d nodes  %d edges", graphs["ER"].number_of_nodes(),
             graphs["ER"].number_of_edges())

    # Barabási–Albert
    m_ba = max(1, round(avg_d / 2))
    graphs["BA"] = nx.barabasi_albert_graph(n, m_ba, seed=seed)
    log.info("BA:  m_attach=%d  →  %d edges", m_ba, graphs["BA"].number_of_edges())

    # Watts–Strogatz
    k_ws = max(2, round(avg_d))
    if k_ws % 2 != 0:
        k_ws += 1
    k_ws = min(k_ws, n - 1)
    graphs["WS"] = nx.watts_strogatz_graph(n, k_ws, p=0.1, seed=seed)
    log.info("WS:  k=%d  p=0.1  →  %d edges", k_ws, graphs["WS"].number_of_edges())

    # SBM (optional)
    if community_map is not None:
        try:
            graphs["SBM"] = _build_sbm(G, community_map, seed=seed)
            log.info("SBM: %d nodes  %d edges",
                     graphs["SBM"].number_of_nodes(),
                     graphs["SBM"].number_of_edges())
        except Exception as exc:
            log.warning("SBM skipped: %s", exc)

    # Configuration model (optional)
    if include_config:
        try:
            graphs["Config"] = _build_config_model(G, seed=seed)
            log.info("Config: %d nodes  %d edges",
                     graphs["Config"].number_of_nodes(),
                     graphs["Config"].number_of_edges())
        except Exception as exc:
            log.warning("Config model skipped: %s", exc)

    return graphs


def compare_metrics(
    G_real: nx.Graph,
    null_graphs: dict[str, nx.Graph],
    n_path_samples: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute comparative structural metrics for the real graph and all null models.

    Metrics: N (nodes), M (edges), mean degree ⟨k⟩, clustering coefficient C,
    approximate average path length L̄, and MLE power-law exponent γ.

    Parameters
    ----------
    G_real : nx.Graph
        Reference earthquake network (undirected, no self-loops).
    null_graphs : dict[str, nx.Graph]
        Output of :func:`build_null_graphs`.
    n_path_samples : int
        Number of BFS source nodes for approximate path length. Exact
        computation is used automatically when N ≤ 500.
    seed : int
        RNG seed for path-length sampling.

    Returns
    -------
    pd.DataFrame
        Rows indexed by model name; columns: N, M, mean_k, C, L, gamma_MLE.

    Notes
    -----
    Small-world signature: C_real ≫ C_ER while L_real ≈ L_ER.
    Scale-free signature: γ_real ≈ γ_BA < γ_ER ≈ γ_WS (ER/WS have thin tails).
    """
    all_graphs = {"Real": G_real, **null_graphs}
    rows = []

    for name, G in all_graphs.items():
        log.info("Metrics for %s ...", name)
        degs = [d for _, d in G.degree()]
        rows.append({
            "Model":      name,
            "N":          G.number_of_nodes(),
            "M":          G.number_of_edges(),
            "mean_k":     round(float(np.mean(degs)), 2),
            "C":          round(nx.average_clustering(G), 5),
            "L (approx)": round(_approx_avg_path_length(G, n_path_samples, seed), 3),
            "γ (MLE)":    round(_mle_gamma([d for d in degs if d > 0], k_min=10), 3),
        })

    return pd.DataFrame(rows).set_index("Model")


def plot_degree_comparison(
    G_real: nx.Graph,
    null_graphs: dict[str, nx.Graph],
    title: str,
    k_min_fit: float = 10,
    n_bins: int = 25,
    save: bool = True,
) -> None:
    """
    Overlay log-binned degree distributions for the real graph and all null models.

    A power-law fit line is overlaid on the real network's tail to indicate γ.

    Parameters
    ----------
    G_real : nx.Graph
        Reference earthquake network.
    null_graphs : dict[str, nx.Graph]
        Output of :func:`build_null_graphs`.
    title : str
        Figure title suffix.
    k_min_fit : float
        Minimum degree for the real-network power-law fit overlay.
    n_bins : int
        Number of logarithmic bins.
    """
    all_graphs = {"Real": G_real, **null_graphs}

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, G in all_graphs.items():
        style = _MODELS.get(name)
        if style is None:
            continue
        degs = [d for _, d in G.degree()]
        k_v, P_v = _log_binned_pk(degs, n_bins)
        if len(k_v) == 0:
            continue
        ax.plot(
            k_v, P_v,
            color=style["color"], linestyle=style["ls"], linewidth=style["lw"],
            marker=style["marker"], markersize=4 if style["marker"] else 0,
            alpha=0.9, label=style["label"],
        )

    # Overlay power-law fit on the real network tail
    real_degs = [d for _, d in G_real.degree() if d > 0]
    k_r, P_r  = _log_binned_pk(real_degs, n_bins)
    mask = k_r >= k_min_fit
    if mask.sum() > 2:
        slope, intercept, *_ = linregress(np.log10(k_r[mask]), np.log10(P_r[mask]))
        gamma = -slope
        ax.plot(
            k_r[mask], 10**intercept * k_r[mask]**(-gamma),
            "--", color=_MODELS["Real"]["color"], linewidth=1.5, alpha=0.5,
            label=rf"Power-law fit: $\gamma = {gamma:.2f}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree $k$", fontsize=13)
    ax.set_ylabel("Probability Density $P(k)$", fontsize=13)
    ax.set_title(f"Degree Distribution vs Null Models: {title}", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    if save:
        savefig(f"null_model_degree_comparison_{_slug(title)}")
    plt.show()
