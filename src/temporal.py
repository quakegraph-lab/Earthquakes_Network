"""
Temporal / multilayer network analysis for the Abe-Suzuki earthquake network.

Builds a sequence of earthquake networks over fixed time windows and tracks
how topology, communities, and hubs evolve across windows.

Key outputs
-----------
- Per-window network metrics: γ, C, L, ⟨k⟩, node count, edge count
- Partition stability: NMI between Louvain partitions of consecutive windows
- Hub persistence: Jaccard similarity of top-N PageRank sets across windows
- Edge turnover: Jaccard similarity of edge sets across consecutive windows

References
----------
Holme, P., & Saramäki, J. (2012). Temporal networks.
  Physics Reports, 519(3), 97–125.
"""

import logging
from typing import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from src.plotutils import savefig, _slug

log = logging.getLogger(__name__)


# ── Network construction ──────────────────────────────────────────────────────

def build_temporal_networks(
    df: pd.DataFrame,
    window_years: int = 5,
    cell_size_km: int = 10,
    target_crs: str = "epsg:5070",
    min_events: int = 50,
) -> list[tuple[str, nx.DiGraph]]:
    """
    Build one Abe-Suzuki network per fixed-width time window.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog with a ``time`` column (timezone-aware datetime).
    window_years : int
        Width of each time window in years.
    cell_size_km : int
        Cell size in km (passed to :func:`src.network.build_abe_suzuki_network`).
    target_crs : str
        CRS string for coordinate projection (EPSG:5070 for US, EPSG:32632 for Italy).
    min_events : int
        Windows with fewer events are skipped.

    Returns
    -------
    list[tuple[str, nx.DiGraph]]
        Ordered list of ``(label, G)`` pairs, label = "YYYY–YYYY".
    """
    from src.network import build_abe_suzuki_network

    years = df["time"].dt.year
    y_min = int(years.min())
    y_max = int(years.max())

    windows = []
    y = y_min
    while y + window_years - 1 <= y_max:
        windows.append((y, y + window_years - 1))
        y += window_years

    results = []
    for y_start, y_end in windows:
        mask = (years >= y_start) & (years <= y_end)
        df_win = df[mask].sort_values("time").reset_index(drop=True)
        if len(df_win) < min_events:
            log.info("Window %d–%d: only %d events – skipped", y_start, y_end, len(df_win))
            continue
        label = f"{y_start}–{y_end}"
        log.info("Building network for %s  (%d events)...", label, len(df_win))
        G = build_abe_suzuki_network(df_win, cell_size_km=cell_size_km, target_crs=target_crs)
        results.append((label, G))
        log.info("  → %d nodes  %d edges", G.number_of_nodes(), G.number_of_edges())

    return results


# ── Metric computation ────────────────────────────────────────────────────────

def _approx_avg_path(G_und: nx.Graph, n_samples: int = 100, seed: int = 42) -> float:
    """Average shortest path length via BFS from a random sample of sources."""
    rng = np.random.default_rng(seed)
    nodes = list(G_und.nodes())
    if len(nodes) <= 1:
        return float("nan")
    sources = rng.choice(nodes, size=min(n_samples, len(nodes)), replace=False)
    total, count = 0.0, 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(G_und, s)
        for _, l in lengths.items():
            if l > 0:
                total += l
                count += 1
    return total / count if count > 0 else float("nan")


def compute_temporal_metrics(
    temporal_graphs: list[tuple[str, nx.DiGraph]],
    k_min_gamma: int = 5,
    n_path_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute per-window network metrics.

    Parameters
    ----------
    temporal_graphs : list[tuple[str, nx.DiGraph]]
        Output of :func:`build_temporal_networks`.
    k_min_gamma : int
        Minimum degree for MLE γ estimation.
    n_path_samples : int
        Number of BFS samples for approximate average path length.

    Returns
    -------
    pd.DataFrame
        Columns: ``window``, ``n_nodes``, ``n_edges``, ``mean_degree``,
        ``gamma``, ``clustering``, ``avg_path_length``.
    """
    from src.metrics import estimate_gamma_mle

    rows = []
    for label, G in temporal_graphs:
        # Giant component (undirected)
        wcc = list(nx.weakly_connected_components(G))
        if not wcc:
            continue
        G_giant = G.subgraph(max(wcc, key=len)).copy()
        G_und = G_giant.to_undirected()
        G_und.remove_edges_from(nx.selfloop_edges(G_und))

        n = G_giant.number_of_nodes()
        m = G_giant.number_of_edges()
        degs = [d for _, d in G.degree(weight="weight") if d > 0]
        mean_k = float(np.mean(degs)) if degs else float("nan")

        # γ
        degs_ge = [d for d in degs if d >= k_min_gamma]
        gamma = estimate_gamma_mle(degs_ge, k_min=k_min_gamma) if len(degs_ge) >= 10 else float("nan")

        # Clustering and path length
        C = nx.average_clustering(G_und) if G_und.number_of_nodes() > 1 else float("nan")
        L = _approx_avg_path(G_und, n_samples=n_path_samples, seed=seed)

        rows.append({
            "window":          label,
            "n_nodes":         n,
            "n_edges":         m,
            "mean_degree":     round(mean_k, 3),
            "gamma":           round(gamma, 4),
            "clustering":      round(C, 4),
            "avg_path_length": round(L, 4),
        })
        log.info("%s  n=%d  m=%d  γ=%.3f  C=%.4f  L=%.3f", label, n, m, gamma, C, L)

    return pd.DataFrame(rows)


def compute_partition_stability(
    temporal_graphs: list[tuple[str, nx.DiGraph]],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute Louvain partition NMI between consecutive time windows.

    The comparison is restricted to nodes present in both consecutive windows;
    this makes NMI robust to the entry and exit of active cells.

    Parameters
    ----------
    temporal_graphs : list[tuple[str, nx.DiGraph]]
        Output of :func:`build_temporal_networks`.

    Returns
    -------
    pd.DataFrame
        Columns: ``transition``, ``nmi``, ``shared_nodes``.
    """
    partitions = {}
    for label, G in temporal_graphs:
        wcc = list(nx.weakly_connected_components(G))
        if not wcc:
            continue
        G_giant = G.subgraph(max(wcc, key=len)).copy()
        G_und = G_giant.to_undirected()
        G_und.remove_edges_from(nx.selfloop_edges(G_und))
        comms = nx.community.louvain_communities(G_und, seed=seed)
        partitions[label] = {node: cid for cid, members in enumerate(comms) for node in members}

    labels = [lbl for lbl, _ in temporal_graphs if lbl in partitions]
    rows = []
    for i in range(len(labels) - 1):
        a, b = labels[i], labels[i + 1]
        pa, pb = partitions[a], partitions[b]
        shared = sorted(set(pa) & set(pb))
        if len(shared) < 10:
            nmi = float("nan")
        else:
            va = np.array([pa[n] for n in shared])
            vb = np.array([pb[n] for n in shared])
            nmi = float(normalized_mutual_info_score(va, vb))
        rows.append({
            "transition":   f"{a} → {b}",
            "nmi":          round(nmi, 4),
            "shared_nodes": len(shared),
        })
        log.info("%s → %s  NMI=%.4f  shared=%d", a, b, nmi, len(shared))

    return pd.DataFrame(rows)


def compute_hub_persistence(
    temporal_graphs: list[tuple[str, nx.DiGraph]],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Jaccard similarity of top-N PageRank hubs across consecutive windows.

    Parameters
    ----------
    temporal_graphs : list[tuple[str, nx.DiGraph]]
        Output of :func:`build_temporal_networks`.
    top_n : int
        Number of top PageRank nodes to consider.

    Returns
    -------
    pd.DataFrame
        Columns: ``transition``, ``jaccard``, ``n_common``.
    """
    hub_sets: dict[str, set] = {}
    for label, G in temporal_graphs:
        pr = nx.pagerank(G, alpha=0.85, weight="weight")
        top = set(sorted(pr, key=pr.__getitem__, reverse=True)[:top_n])
        hub_sets[label] = top

    labels = [lbl for lbl, _ in temporal_graphs if lbl in hub_sets]
    rows = []
    for i in range(len(labels) - 1):
        a, b = labels[i], labels[i + 1]
        ha, hb = hub_sets[a], hub_sets[b]
        inter = len(ha & hb)
        jaccard = inter / len(ha | hb) if ha | hb else float("nan")
        rows.append({
            "transition": f"{a} → {b}",
            "jaccard":    round(jaccard, 4),
            "n_common":   inter,
        })
    return pd.DataFrame(rows)


def compute_edge_turnover(
    temporal_graphs: list[tuple[str, nx.DiGraph]],
) -> pd.DataFrame:
    """
    Jaccard similarity of edge sets across consecutive time windows.

    High Jaccard → stable seismic corridors.  Low Jaccard → rapid reorganisation.

    Parameters
    ----------
    temporal_graphs : list[tuple[str, nx.DiGraph]]
        Output of :func:`build_temporal_networks`.

    Returns
    -------
    pd.DataFrame
        Columns: ``transition``, ``jaccard``, ``n_common_edges``.
    """
    edge_sets: dict[str, set] = {}
    for label, G in temporal_graphs:
        edge_sets[label] = set(G.edges())

    labels = [lbl for lbl, _ in temporal_graphs if lbl in edge_sets]
    rows = []
    for i in range(len(labels) - 1):
        a, b = labels[i], labels[i + 1]
        ea, eb = edge_sets[a], edge_sets[b]
        inter = len(ea & eb)
        union = len(ea | eb)
        rows.append({
            "transition":     f"{a} → {b}",
            "jaccard":        round(inter / union, 4) if union > 0 else float("nan"),
            "n_common_edges": inter,
        })
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_temporal_metrics(
    df: pd.DataFrame,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Four-panel time-series of per-window topology metrics.

    Panels: (a) γ, (b) clustering coefficient C, (c) average path length L,
    (d) mean degree ⟨k⟩.  Reference lines for Italy/US overall values are not
    drawn automatically – add them in the notebook where values are known.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_temporal_metrics`.
    title : str
        Figure suptitle suffix.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    x = range(len(df))
    xtick_labels = df["window"].tolist()

    for ax, col, ylabel, color in [
        (axes[0, 0], "gamma",           "Power-law exponent γ",          "#e63946"),
        (axes[0, 1], "clustering",      "Clustering coefficient C",      "#2a9d8f"),
        (axes[1, 0], "avg_path_length", "Avg path length L",             "#f4a261"),
        (axes[1, 1], "mean_degree",     "Mean degree ⟨k⟩",              "#457b9d"),
    ]:
        vals = df[col].values
        ax.plot(x, vals, "o-", color=color, lw=2, ms=6)
        ax.fill_between(x, vals, alpha=0.15, color=color)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, ls="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, fontsize=8)

    fig.suptitle(f"Temporal Network Topology – {title}", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        savefig(f"temporal_metrics_{_slug(title)}")
    plt.show()


def plot_temporal_stability(
    df_stability: pd.DataFrame,
    df_hub: pd.DataFrame,
    df_edge: pd.DataFrame,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Three-panel bar chart of partition NMI, hub Jaccard, and edge Jaccard.

    Parameters
    ----------
    df_stability : pd.DataFrame
        Output of :func:`compute_partition_stability`.
    df_hub : pd.DataFrame
        Output of :func:`compute_hub_persistence`.
    df_edge : pd.DataFrame
        Output of :func:`compute_edge_turnover`.
    title : str
        Figure suptitle suffix.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, df, col, ylabel, color, label in [
        (axes[0], df_stability, "nmi",     "NMI(C_t, C_{t+1})",            "#9b5de5", "Community NMI"),
        (axes[1], df_hub,       "jaccard", "Jaccard (top-20 PageRank hubs)", "#e63946", "Hub Jaccard"),
        (axes[2], df_edge,      "jaccard", "Jaccard (edge set)",             "#457b9d", "Edge Jaccard"),
    ]:
        vals = df[col].values
        x = range(len(vals))
        bars = ax.bar(x, vals, color=color, alpha=0.75, edgecolor="k", linewidth=0.5)
        mean_val = float(np.nanmean(vals))
        ax.axhline(mean_val, color="gray", ls="--", lw=1.2,
                   label=f"Mean={mean_val:.3f}")
        # Auto-scale y so near-zero panels are not empty
        vmax = max(float(np.nanmax(vals)) if len(vals) else 0, 1e-4)
        ax.set_ylim(0, vmax * 1.4)
        # Label each bar with its value
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + vmax * 0.03,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df["transition"].tolist(), rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", ls="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Network Stability Across Windows – {title}", fontsize=13, y=1.01)
    plt.tight_layout()
    if save:
        savefig(f"temporal_stability_{_slug(title)}")
    plt.show()


def plot_temporal_comparison(
    df_italy: pd.DataFrame,
    df_us: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Overlay plot of a single metric for Italy vs US across windows.

    Parameters
    ----------
    df_italy, df_us : pd.DataFrame
        Outputs of :func:`compute_temporal_metrics` for each catalog.
    metric : str
        Column name to plot (e.g., ``"gamma"``, ``"clustering"``).
    ylabel : str
        Y-axis label.
    title : str
        Figure title suffix.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for df, color, label in [
        (df_italy, "#e63946", "Italy"),
        (df_us,    "#457b9d", "US"),
    ]:
        x = range(len(df))
        vals = df[metric].values
        ax.plot(x, vals, "o-", color=color, lw=2.2, ms=7, label=label)
        ax.fill_between(x, vals, alpha=0.12, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(df["window"].tolist(), rotation=30, ha="right", fontsize=9)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Time window", fontsize=11)
    ax.set_title(f"{ylabel} Evolution – Italy vs US – {title}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        savefig(f"temporal_comparison_{_slug(metric)}_{_slug(title)}")
    plt.show()
