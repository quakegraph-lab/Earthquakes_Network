"""
Community-level Markov flow analysis for the Abe-Suzuki earthquake network.

Models seismicity as a coarse-grained Markov chain: each Louvain community
becomes a state, and edge weights provide transition probabilities.  Three
quantities characterise each community state:

  Self-retention T[i,i] – fraction of flow staying within the community.
  Shannon entropy H_i    – diversity of outgoing flow (bits).
  Stationary distribution π – long-run fraction of time spent in each state.

References
----------
Rosvall, M., & Bergstrom, C. T. (2008). Maps of random walks on complex
  networks reveal community structure. PNAS, 105(4), 1118–1123.
"""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from src.plotutils import savefig, _slug

log = logging.getLogger(__name__)

Partition = dict[str, int]


# ── Matrix construction ───────────────────────────────────────────────────────

def build_community_flow_matrix(
    G: nx.DiGraph,
    community_map: Partition,
) -> pd.DataFrame:
    """
    Build a K×K community-level transition count matrix.

    Entry (i, j) = total edge weight flowing from community i to j.
    Self-loops (intra-community) appear on the diagonal.  Nodes absent from
    ``community_map`` (e.g., nodes outside the GCC) are silently skipped.

    Parameters
    ----------
    G : nx.DiGraph
        Directed earthquake network with ``weight`` edge attribute.
    community_map : Partition
        ``{node_id: community_int}`` mapping (from any detection method).

    Returns
    -------
    pd.DataFrame
        K×K DataFrame, index = source community, columns = target community.
    """
    communities = sorted(set(community_map.values()))
    K = len(communities)
    c_idx = {c: i for i, c in enumerate(communities)}
    C = np.zeros((K, K), dtype=np.float64)

    for u, v, data in G.edges(data=True):
        cu = community_map.get(u)
        cv = community_map.get(v)
        if cu is None or cv is None:
            continue
        w = float(data.get("weight", 1.0))
        C[c_idx[cu], c_idx[cv]] += w

    log.info("Community flow matrix: %d×%d  total flow=%.0f", K, K, C.sum())
    return pd.DataFrame(C, index=communities, columns=communities)


def compute_markov_chain(count_df: pd.DataFrame) -> np.ndarray:
    """
    Normalise a count matrix to a row-stochastic transition matrix.

    Rows with zero total flow are replaced by a uniform distribution so that
    isolated communities remain valid Markov states.

    Parameters
    ----------
    count_df : pd.DataFrame
        Output of :func:`build_community_flow_matrix`.

    Returns
    -------
    np.ndarray
        Row-stochastic matrix of shape (K, K).
    """
    C = count_df.values.astype(np.float64)
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return C / row_sums


def compute_stationary_distribution(
    T: np.ndarray,
    max_iter: int = 20_000,
    tol: float = 1e-14,
) -> np.ndarray:
    """
    Stationary distribution of a row-stochastic matrix via power iteration.

    Solves π = π T until convergence.  Guaranteed to converge for ergodic
    chains; for reducible chains returns the distribution reachable from the
    uniform start.

    Parameters
    ----------
    T : np.ndarray
        Row-stochastic matrix of shape (K, K).
    max_iter : int
        Maximum iterations before early stop.
    tol : float
        Convergence threshold (max absolute change in any π component).

    Returns
    -------
    np.ndarray
        Stationary distribution π of length K (sums to 1).
    """
    K = T.shape[0]
    pi = np.ones(K, dtype=np.float64) / K
    for _ in range(max_iter):
        pi_new = pi @ T
        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new
    return pi / pi.sum()


# ── Statistics ────────────────────────────────────────────────────────────────

def community_flow_stats(
    T: np.ndarray,
    count_df: pd.DataFrame,
    stat_dist: np.ndarray,
) -> pd.DataFrame:
    """
    Per-community statistics derived from the Markov chain.

    Parameters
    ----------
    T : np.ndarray
        Row-stochastic transition matrix (K, K).
    count_df : pd.DataFrame
        Raw count matrix (for total flow volume per community).
    stat_dist : np.ndarray
        Stationary distribution from :func:`compute_stationary_distribution`.

    Returns
    -------
    pd.DataFrame
        Columns: ``community``, ``self_retention``, ``entropy``,
        ``dominant_target``, ``stationary``, ``total_flow``.
        Sorted by stationary probability (descending).
    """
    communities = list(count_df.index)
    rows = []
    for i, c in enumerate(communities):
        row = T[i]
        self_ret = float(row[i])
        # Shannon entropy in bits; 0 * log2(0) := 0
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = float(-np.nansum(row * np.log2(np.where(row > 0, row, 1.0))))
        # Primary receiver: highest probability *excluding self*
        off_diag = row.copy()
        off_diag[i] = -1.0
        dom = int(communities[int(np.argmax(off_diag))])
        rows.append({
            "community":      c,
            "self_retention": round(self_ret, 4),
            "entropy":        round(entropy, 4),
            "dominant_target": dom,
            "stationary":     round(float(stat_dist[i]), 6),
            "total_flow":     int(count_df.iloc[i].sum()),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("stationary", ascending=False)
        .reset_index(drop=True)
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_flow_heatmap(
    T: np.ndarray,
    count_df: pd.DataFrame,
    title: str = "",
    max_communities: int = 25,
    save: bool = True,
) -> None:
    """
    Heatmap of the community-to-community Markov transition matrix.

    Rows/columns are ordered by total outgoing flow.  When K > ``max_communities``
    only the largest communities are shown.

    Parameters
    ----------
    T : np.ndarray
        Row-stochastic transition matrix (K, K).
    count_df : pd.DataFrame
        Raw count matrix (provides flow-based ordering).
    title : str
        Figure title suffix (catalog + cell size).
    max_communities : int
        Maximum communities to display.
    """
    communities = list(count_df.index)
    total_flow = count_df.values.sum(axis=1)
    order = np.argsort(-total_flow)[:max_communities]
    labels = [str(communities[i]) for i in order]
    T_sub = T[np.ix_(order, order)]

    n = len(labels)
    fig_size = (max(6, min(14, n)), max(5, min(12, n)))
    fig, ax = plt.subplots(figsize=fig_size)
    sns.heatmap(
        T_sub,
        xticklabels=labels,
        yticklabels=labels,
        annot=(n <= 15),
        fmt=".2f" if n <= 15 else "",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.4 if n <= 20 else 0,
        square=True,
        cbar_kws={"label": "Transition probability"},
        ax=ax,
    )
    ax.set_xlabel("Target community", fontsize=11)
    ax.set_ylabel("Source community", fontsize=11)
    ax.set_title(f"Community Markov Flow – {title}", fontsize=13, pad=10)
    plt.tight_layout()
    if save:
        savefig(f"community_flow_heatmap_{_slug(title)}")
    plt.show()


def plot_flow_entropy(
    stats_df: pd.DataFrame,
    K: int,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Bar chart of Shannon entropy per community, coloured by self-retention.

    High entropy: seismicity diffuses across many communities.
    Low entropy: activity is confined (self-retention or narrow corridor).

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output of :func:`community_flow_stats`.
    K : int
        Total number of communities (sets max-entropy reference line).
    title : str
        Figure title suffix.
    """
    df = stats_df.sort_values("entropy", ascending=False).reset_index(drop=True)
    max_h = np.log2(K)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.5), 5))
    colors = plt.cm.RdYlBu_r(df["self_retention"].values)
    ax.bar(range(len(df)), df["entropy"], color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(max_h, color="gray", ls="--", lw=1.2,
               label=f"Max entropy log₂({K}) = {max_h:.2f} bits")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"C{c}" for c in df["community"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Community", fontsize=11)
    ax.set_ylabel("Outflow entropy (bits)", fontsize=11)
    ax.set_title(f"Community Outflow Entropy – {title}", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max_h * 1.2)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    sm = plt.cm.ScalarMappable(
        cmap="RdYlBu_r", norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Self-retention", pad=0.01, shrink=0.7)
    plt.tight_layout()
    if save:
        savefig(f"community_flow_entropy_{_slug(title)}")
    plt.show()


def plot_stationary_distribution(
    stats_df: pd.DataFrame,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Bar chart of the stationary distribution across communities.

    The community with the highest stationary probability is the dominant
    long-run attractor: a random seismic walk spends most time there.
    The dashed line marks the uniform baseline (1/K).

    Parameters
    ----------
    stats_df : pd.DataFrame
        Output of :func:`community_flow_stats`, sorted by stationary desc.
    title : str
        Figure title suffix.
    """
    df = stats_df.sort_values("stationary", ascending=False).reset_index(drop=True)
    baseline = 1.0 / len(df)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.5), 5))
    # Two-color scheme: above baseline = dominant (teal), at/below = minor (light blue)
    colors = ["#2a9d8f" if v > baseline else "#a8dadc" for v in df["stationary"]]
    ax.bar(range(len(df)), df["stationary"], color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(baseline, color="gray", ls="--", lw=1.2,
               label=f"Uniform baseline 1/{len(df)} = {baseline:.3f}")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"C{c}" for c in df["community"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Community", fontsize=11)
    ax.set_ylabel("Stationary probability π", fontsize=11)
    ax.set_title(f"Stationary Distribution of Seismic Flow – {title}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        savefig(f"community_flow_stationary_{_slug(title)}")
    plt.show()
