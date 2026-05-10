"""
Assortativity analysis for the Abe-Suzuki earthquake network.

Computes structural (degree-degree) and attribute (depth, magnitude)
assortativity, then visualises the mixing patterns via scatter plots.

Seismological interpretation
----------------------------
* Degree assortativity r < 0 (disassortative) is the hallmark of scale-free
  networks: hubs connect to low-degree peripheral cells rather than to other
  hubs. This is directly related to the star-like topology of aftershock trees.

* Depth assortativity r > 0 (assortative) would indicate that deep events
  preferentially trigger other deep events — a signature of distinct
  seismogenic depth horizons (e.g. crustal vs subduction-zone seismicity).

* Magnitude assortativity r > 0 would indicate that high-magnitude regions
  cluster together in the temporal sequence — possible evidence of
  mainshock–aftershock structuring.

References
----------
Newman, M. E. J. (2002). Assortative mixing in networks.
  Physical Review Letters, 89(20), 208701.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)


def attach_catalog_attrs(
    G: nx.DiGraph,
    df_net: pd.DataFrame,
    cell_size_km: float = 10.0,
    target_crs: str = "epsg:5070",
) -> None:
    """
    Attach mean magnitude and mean depth to each node of G.

    Re-discretises the catalog (fast: no edge construction) to obtain
    per-cell statistics, then sets ``mean_magnitude`` and ``mean_depth``
    as node attributes. Mutates G in place.

    Parameters
    ----------
    G : nx.DiGraph
        Earthquake network; nodes are cell_id strings.
    df_net : pd.DataFrame
        Full earthquake catalog with columns ``magnitude`` and ``depth_km``.
    cell_size_km : float
        Must match the resolution used to build G.
    target_crs : str
        Projection CRS (``"epsg:5070"`` for US, ``"epsg:32632"`` for Italy).
    """
    from src.network import discretize_space_3d  # noqa: PLC0415

    log.info("Re-discretising catalog for node attribute attachment...")
    df_grid = discretize_space_3d(df_net, cell_size_km=cell_size_km,
                                  target_crs=target_crs)

    agg_cols = {}
    if "magnitude" in df_grid.columns:
        agg_cols["mean_magnitude"] = ("magnitude", "mean")
    if "depth_km" in df_grid.columns:
        agg_cols["mean_depth"] = ("depth_km", "mean")

    if not agg_cols:
        log.warning("No magnitude or depth_km column found in catalog.")
        return

    cell_stats = df_grid.groupby("cell_id").agg(**agg_cols)

    n_attached = 0
    for node in G.nodes():
        if node in cell_stats.index:
            for col in cell_stats.columns:
                G.nodes[node][col] = float(cell_stats.at[node, col])
            n_attached += 1

    log.info("Attached attributes to %d / %d nodes.", n_attached, G.number_of_nodes())


def compute_assortativity(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute degree-degree and attribute assortativity coefficients.

    Uses the *undirected* version of G (removing self-loops) for all
    measures, so that Newman's standard formula applies.

    Parameters
    ----------
    G : nx.DiGraph
        Network; may have ``mean_magnitude`` and ``mean_depth`` node attrs
        (set by :func:`attach_catalog_attrs`).

    Returns
    -------
    pd.DataFrame
        Index = attribute name, columns = ``r`` (coefficient) and
        ``interpretation``.
    """
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    rows: list[dict] = []

    # ── 1. Structural degree-degree assortativity ────────────────────────────
    r_deg = nx.degree_assortativity_coefficient(G_und)
    rows.append({
        "attribute":     "degree",
        "r":             round(r_deg, 4),
        "interpretation": (
            "disassortative (hubs→periphery, typical scale-free)"
            if r_deg < -0.05 else
            "assortative (hubs→hubs)" if r_deg > 0.05
            else "neutral"
        ),
    })

    # ── 2. Attribute assortativity ───────────────────────────────────────────
    for attr, phys_label in [
        ("mean_depth",     "depth (km)"),
        ("mean_magnitude", "magnitude"),
    ]:
        # Build subgraph with only nodes that have the attribute
        nodes_ok = [n for n in G_und.nodes() if attr in G_und.nodes[n]]
        if len(nodes_ok) < 10:
            log.warning("Skipping %s: too few nodes have the attribute.", attr)
            continue
        H = G_und.subgraph(nodes_ok)
        if H.number_of_edges() < 5:
            log.warning("Skipping %s: too few edges after filtering.", attr)
            continue
        try:
            r_attr = nx.numeric_assortativity_coefficient(H, attr)
        except Exception as exc:
            log.warning("Assortativity for %s failed: %s", attr, exc)
            r_attr = float("nan")
        rows.append({
            "attribute":     phys_label,
            "r":             round(r_attr, 4),
            "interpretation": (
                f"assortative: {phys_label} clusters in time"
                if r_attr > 0.05 else
                f"disassortative: {phys_label} alternates in time"
                if r_attr < -0.05 else
                "neutral"
            ),
        })

    return pd.DataFrame(rows).set_index("attribute")


def plot_assortativity(
    G: nx.DiGraph,
    title: str = "",
    n_edge_samples: int = 5000,
    seed: int = 42,
    save: bool = True,
) -> None:
    """
    Visualise mixing patterns as edge-source vs edge-target scatter plots.

    Three panels: degree mixing, depth mixing, magnitude mixing.
    Large graphs are subsampled to ``n_edge_samples`` edges for speed.

    Parameters
    ----------
    G : nx.DiGraph
        Network with ``mean_depth`` and ``mean_magnitude`` node attributes.
    title : str
        Figure title suffix.
    n_edge_samples : int
        Max edges to plot per panel.
    seed : int
        RNG seed for edge subsampling.
    """
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    rng   = np.random.default_rng(seed)
    edges = list(G_und.edges())
    if len(edges) > n_edge_samples:
        idx   = rng.choice(len(edges), size=n_edge_samples, replace=False)
        edges = [edges[i] for i in idx]

    panels = [
        ("degree",         "Degree $k$",          lambda n: G_und.degree(n)),
        ("mean_depth",     "Mean Depth (km)",      lambda n: G_und.nodes[n].get("mean_depth", None)),
        ("mean_magnitude", "Mean Magnitude",       lambda n: G_und.nodes[n].get("mean_magnitude", None)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (attr, label, getter) in zip(axes, panels):
        src_vals, tgt_vals = [], []
        for u, v in edges:
            sv, tv = getter(u), getter(v)
            if sv is not None and tv is not None:
                src_vals.append(sv)
                tgt_vals.append(tv)

        if not src_vals:
            ax.set_visible(False)
            continue

        src = np.array(src_vals, dtype=float)
        tgt = np.array(tgt_vals, dtype=float)

        # 2-D histogram for readability
        _, _, _, mesh = ax.hist2d(src, tgt, bins=40, cmap="Blues",
                                  norm=plt.matplotlib.colors.LogNorm())
        fig.colorbar(mesh, ax=ax, label="Count (log)")
        # Regression line
        m, b = np.polyfit(src, tgt, 1)
        x_line = np.linspace(src.min(), src.max(), 100)
        ax.plot(x_line, m * x_line + b, "r-", linewidth=1.5, alpha=0.8)

        # Clip view to 99th percentile — outliers otherwise dominate axis range
        ax.set_xlim(src.min(), np.percentile(src, 99))
        ax.set_ylim(tgt.min(), np.percentile(tgt, 99))

        r, _ = pearsonr(src, tgt)
        ax.set_xlabel(f"Source node — {label}", fontsize=10)
        ax.set_ylabel(f"Target node — {label}", fontsize=10)
        ax.set_title(f"{label} mixing\n$r = {r:.3f}$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(f"Assortativity Mixing Patterns: {title}", fontsize=13)
    plt.tight_layout()
    if save:
        savefig(f"assortativity_{_slug(title)}")
    plt.show()
