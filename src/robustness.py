"""
Network robustness analysis for the Abe-Suzuki earthquake network.

Simulates two failure scenarios on the undirected giant component and
tracks how the giant connected component (GCC) shrinks:

* **Random failure** – nodes removed uniformly at random (models
  random equipment failure or catalog incompleteness).
* **Targeted attack** – nodes removed in decreasing order of initial
  degree (models deliberate removal of the most active seismic cells).

Theory prediction for scale-free networks (Le14):
  * Robust under random failure: GCC persists until most nodes are removed.
  * Fragile under targeted attack: GCC collapses at a low removal fraction
    because a few hubs carry the connectivity.
  * Erdős–Rényi graphs are fragile under both scenarios (no hubs to protect
    or to attack).

The comparison of both curves against an ER baseline with identical N and M
directly illustrates the scale-free robustness signature.

References
----------
Albert, R., Jeong, H., & Barabási, A.-L. (2000). Error and attack tolerance
  of complex networks. Nature, 406, 378–382.
"""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)

_STYLE = {
    "Random (Real)":    dict(color="#e63946", ls="-",  lw=2.2, label="Random failure – Earthquake net"),
    "Targeted (Real)":  dict(color="#c1121f", ls="--", lw=2.2, label="Targeted attack – Earthquake net"),
    "Random (ER)":      dict(color="#457b9d", ls="-",  lw=1.5, label="Random failure – ER baseline", alpha=0.7),
    "Targeted (ER)":    dict(color="#1d3557", ls="--", lw=1.5, label="Targeted attack – ER baseline", alpha=0.7),
}


def simulate_robustness(
    G: nx.Graph,
    strategy: str = "targeted",
    n_checkpoints: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate node removal and record giant component fraction at checkpoints.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph without self-loops (typically the giant component
        of the earthquake network).
    strategy : {"targeted", "random"}
        * ``"targeted"`` – static highest-degree-first (initial degree order).
        * ``"random"``   – uniformly random order.
    n_checkpoints : int
        Number of evenly-spaced points at which to record GCC size.
        Actual x-values will be ``np.linspace(0, 1, n_checkpoints + 1)``.
    seed : int
        Used only when ``strategy = "random"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``fraction_removed`` (float in [0, 1]) and
        ``gcc_fraction`` (GCC size / original N).

    Notes
    -----
    **Static targeting**: the removal order is fixed to the initial degree
    sequence. This is the standard literature convention (Albert et al. 2000)
    and avoids O(N²) recomputation after each removal.
    """
    n_total = G.number_of_nodes()
    if n_total == 0:
        return pd.DataFrame({"fraction_removed": [0.0], "gcc_fraction": [0.0]})

    # Compute removal order
    if strategy == "targeted":
        order = [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)]
    elif strategy == "random":
        rng   = np.random.default_rng(seed)
        order = list(G.nodes())
        rng.shuffle(order)
    else:
        raise ValueError(f"strategy must be 'targeted' or 'random', got {strategy!r}")

    # Checkpoint indices (evenly spaced across removal sequence)
    checkpoints = np.unique(
        np.round(np.linspace(0, n_total, n_checkpoints + 1)).astype(int)
    )

    H = G.copy()
    records = []
    removed = 0

    for i, ckpt in enumerate(checkpoints):
        # Remove nodes up to this checkpoint
        while removed < ckpt and order:
            H.remove_node(order[removed])
            removed += 1

        if H.number_of_nodes() == 0:
            gcc = 0
        else:
            gcc = max((len(c) for c in nx.connected_components(H)), default=0)

        records.append({
            "fraction_removed": removed / n_total,
            "gcc_fraction":     gcc / n_total,
        })

    log.info("Robustness (%s): %d checkpoints recorded.", strategy, len(records))
    return pd.DataFrame(records)


def plot_robustness_curves(
    results: dict[str, pd.DataFrame],
    title: str = "",
    save: bool = True,
) -> None:
    """
    Plot GCC fraction vs fraction of nodes removed for multiple scenarios.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Keys matching ``_STYLE`` (e.g. ``"Random (Real)"``,
        ``"Targeted (Real)"``). Values are DataFrames from
        :func:`simulate_robustness`.
    title : str
        Figure title suffix.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, df in results.items():
        style = _STYLE.get(label, dict(color="gray", ls="-", lw=1.5, label=label))
        ax.plot(
            df["fraction_removed"],
            df["gcc_fraction"],
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            alpha=style.get("alpha", 1.0),
            label=style["label"],
        )

    ax.set_xlabel("Fraction of nodes removed", fontsize=12)
    ax.set_ylabel("GCC size / original N", fontsize=12)
    ax.set_title(f"Network Robustness: {title}" if title else "Network Robustness", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save:
        savefig(f"robustness_curves_{_slug(title)}")
    plt.show()
