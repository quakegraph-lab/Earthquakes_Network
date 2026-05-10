"""
Centrality computation and comparison for the Abe-Suzuki earthquake network.

Computes all 8 measures in a single function, returns a unified DataFrame,
and provides two diagnostic visualisations:
  1. Spearman rank-correlation heatmap across measures.
  2. Multi-panel top-N cell bar chart per measure.

Measures and seismological interpretations
------------------------------------------
Degree       — most seismically active cells (highest transition count).
PageRank     — "stress sinks": cells that persistently receive seismic flow.
Closeness    — cells that can spread seismic influence fastest across the network.
Betweenness  — "bridges": cells on shortest paths between fault clusters.
Eigenvector  — cells embedded in the high-activity core (rich-club).
Katz         — like eigenvector but counts ALL paths (with exponential decay),
               more robust for directed/sparse graphs.
HITS Hub     — cells that trigger important seismic zones (high out-connections
               to high-authority cells).
HITS Auth    — cells that are the primary destinations of seismic propagation.
"""

import logging
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)

_METRICS = [
    "Degree", "PageRank", "Closeness", "Betweenness",
    "Eigenvector", "Katz", "HITS_Hub", "HITS_Auth",
]
_LABELS = {
    "Degree":      "Degree\n(active cells)",
    "PageRank":    "PageRank\n(stress sinks)",
    "Closeness":   "Closeness\n(global spread)",
    "Betweenness": "Betweenness\n(fault bridges)",
    "Eigenvector": "Eigenvector\n(rich-club core)",
    "Katz":        "Katz\n(all-path influence)",
    "HITS_Hub":    "HITS Hub\n(seismic triggers)",
    "HITS_Auth":   "HITS Authority\n(seismic destinations)",
}


def compute_all_centralities(
    G: nx.DiGraph,
    k_betweenness: int = 1000,
    cell_size_km: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute all 8 centrality measures for the earthquake network.

    Parameters
    ----------
    G : nx.DiGraph
        Directed weighted earthquake network (may have self-loops).
    k_betweenness : int
        Pivot nodes for betweenness approximation (exact if k ≥ N).
    cell_size_km : float
        Cell edge length used to recover physical depth:
        ``depth_km = cell_z * cell_size_km``.
    seed : int
        Random seed for betweenness sampling.

    Returns
    -------
    pd.DataFrame
        One row per node that has geographic coordinates. Columns:
        ``cell_id``, ``lat``, ``lon``, ``depth_km``,
        ``Degree``, ``PageRank``, ``Closeness``, ``Betweenness``,
        ``Eigenvector``, ``Katz``, ``HITS_Hub``, ``HITS_Auth``.

    Notes
    -----
    * Eigenvector is computed on the *undirected* version of G for
      numerical stability (power iteration converges more reliably).
    * Katz uses ``alpha = 0.85 / max_degree``, which is always below
      ``1 / lambda_max``, guaranteeing convergence.
    * HITS operates on G with self-loops removed; hub and authority
      scores are returned as separate columns.
    """
    t_total = time.time()
    n = G.number_of_nodes()

    # ── Undirected version (eigenvector only) ────────────────────────────────
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    # ── G without self-loops (HITS) ──────────────────────────────────────────
    G_nsl = G.copy()
    G_nsl.remove_edges_from(nx.selfloop_edges(G_nsl))

    # ── 1. Degree ────────────────────────────────────────────────────────────
    log.info("Degree centrality...")
    t0 = time.time()
    deg_cent = nx.degree_centrality(G)
    log.info("  %.1fs", time.time() - t0)

    # ── 2. PageRank ──────────────────────────────────────────────────────────
    log.info("PageRank...")
    t0 = time.time()
    pr_cent = nx.pagerank(G, weight="weight")
    log.info("  %.1fs", time.time() - t0)

    # ── 3. Closeness ─────────────────────────────────────────────────────────
    log.info("Closeness centrality...")
    t0 = time.time()
    close_cent = nx.closeness_centrality(G)
    log.info("  %.1fs", time.time() - t0)

    # ── 4. Betweenness (sampled) ─────────────────────────────────────────────
    log.info("Betweenness centrality (k=%d)...", k_betweenness)
    t0 = time.time()
    bet_cent = nx.betweenness_centrality(G, k=min(k_betweenness, n), seed=seed)
    log.info("  %.1fs", time.time() - t0)

    # ── 5. Eigenvector (undirected, numpy fallback) ──────────────────────────
    log.info("Eigenvector centrality (undirected)...")
    t0 = time.time()
    try:
        eig_cent = nx.eigenvector_centrality(
            G_und, weight="weight", max_iter=500, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        log.warning("  eigenvector_centrality did not converge, falling back to numpy")
        eig_cent = nx.eigenvector_centrality_numpy(G_und, weight="weight")
    log.info("  %.1fs", time.time() - t0)

    # ── 6. Katz ──────────────────────────────────────────────────────────────
    log.info("Katz centrality...")
    t0 = time.time()
    max_deg   = max((G.degree(n) for n in G.nodes()), default=1)
    alpha_katz = 0.85 / max_deg          # always < 1/lambda_max (safe bound)
    try:
        katz_cent = nx.katz_centrality(
            G, alpha=alpha_katz, weight="weight",
            normalized=True, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        log.warning("  Katz did not converge — using numpy solver")
        katz_cent = nx.katz_centrality_numpy(G, alpha=alpha_katz, weight="weight")
    log.info("  %.1fs  alpha=%.2e", time.time() - t0, alpha_katz)

    # ── 7 & 8. HITS hub + authority ──────────────────────────────────────────
    log.info("HITS (hub + authority)...")
    t0 = time.time()
    try:
        hits_hub, hits_auth = nx.hits(G_nsl, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        log.warning("  HITS did not converge — setting scores to 0")
        zeros     = {n: 0.0 for n in G.nodes()}
        hits_hub  = zeros.copy()
        hits_auth = zeros.copy()
    log.info("  %.1fs", time.time() - t0)

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    rows = [
        {
            "cell_id":     node,
            "lat":         G.nodes[node]["lat"],
            "lon":         G.nodes[node]["lon"],
            "depth_km":    float(node.split("_")[2]) * cell_size_km,
            "Degree":      deg_cent.get(node, 0.0),
            "PageRank":    pr_cent.get(node, 0.0),
            "Closeness":   close_cent.get(node, 0.0),
            "Betweenness": bet_cent.get(node, 0.0),
            "Eigenvector": eig_cent.get(node, 0.0),
            "Katz":        katz_cent.get(node, 0.0),
            "HITS_Hub":    hits_hub.get(node, 0.0),
            "HITS_Auth":   hits_auth.get(node, 0.0),
        }
        for node in G.nodes()
        if "lat" in G.nodes[node] and "lon" in G.nodes[node]
    ]

    df = pd.DataFrame(rows)
    log.info("Centrality complete: %d nodes, %.1fs total", len(df), time.time() - t_total)
    return df


def plot_centrality_correlation(df: pd.DataFrame, title: str = "", save: bool = True) -> None:
    """
    Spearman rank-correlation heatmap of all 8 centrality measures.

    Spearman is used instead of Pearson because centrality distributions
    are heavy-tailed; rank correlations are more interpretable.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_all_centralities`.
    title : str
        Figure title suffix.
    """
    available = [m for m in _METRICS if m in df.columns]
    corr = df[available].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
        ax=ax,
    )
    labels = [_LABELS.get(m, m).replace("\n", " ") for m in available]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    ax.set_title(f"Centrality Measure Correlation (Spearman): {title}", fontsize=13, pad=12)
    plt.tight_layout()
    if save:
        savefig(f"centrality_correlation_{_slug(title)}")
    plt.show()


def plot_top_n_cells(
    df: pd.DataFrame,
    top_n: int = 10,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Multi-panel horizontal bar chart: top N cells for each centrality measure.

    Layout: 4 columns × 2 rows (8 panels).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_all_centralities`.
    top_n : int
        Number of top cells to display per measure.
    title : str
        Figure title suffix.
    """
    available = [m for m in _METRICS if m in df.columns]
    n_cols = 4
    n_rows = int(np.ceil(len(available) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for i, metric in enumerate(available):
        ax = axes[i]
        top = df.nlargest(top_n, metric)[["cell_id", metric]].copy()
        top = top.sort_values(metric, ascending=True)  # ascending so highest is at top

        ax.barh(range(len(top)), top[metric], color="#1a3a6b", edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["cell_id"], fontsize=7)
        ax.set_xlabel("Centrality value", fontsize=9)
        ax.set_title(_LABELS.get(metric, metric), fontsize=10, pad=4)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused panels
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Top {top_n} Cells per Centrality Measure: {title}",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        savefig(f"centrality_top_n_cells_{_slug(title)}")
    plt.show()


def plot_geo_top_n_interactive(
    df: pd.DataFrame,
    top_n: int = 10,
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
    Interactive Plotly mapbox: dropdown to switch between all 8 centrality
    metrics, showing the top-N nodes for the selected metric.

    Markers are coloured by depth (plasma scale) and sized by rank
    (rank 1 = largest). Hover shows cell_id, rank, metric value, depth.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_all_centralities`.
    top_n : int
        Number of top nodes to display per metric.
    title : str
        Figure title suffix (catalog name, cell size, etc.).
    center_lat, center_lon : float
        Map centre coordinates.
    zoom : float
        Initial Plotly mapbox zoom level.
    """
    available = [m for m in _METRICS if m in df.columns]
    depth_min = float(df["depth_km"].min())
    depth_max = float(df["depth_km"].max())

    # One trace per metric; only the first is visible initially.
    traces: list[go.BaseTraceType] = []
    for i, metric in enumerate(available):
        top = df.nlargest(top_n, metric).copy().reset_index(drop=True)
        top["rank"] = top.index + 1
        # Rank 1 gets the largest marker; scale linearly down to ~8 px.
        size_max, size_min = 28, 8
        top["marker_size"] = (
            size_max - (top["rank"] - 1) * (size_max - size_min) / max(top_n - 1, 1)
        )

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Rank: %{customdata[1]}<br>"
            f"{_LABELS.get(metric, metric).replace('<br>', ' ')}: %{{customdata[2]:.5f}}<br>"
            "Depth: %{customdata[3]:.0f} km<br>"
            "Lat: %{lat:.3f} | Lon: %{lon:.3f}<extra></extra>"
        )
        custom = list(
            zip(top["cell_id"], top["rank"], top[metric], top["depth_km"])
        )

        traces.append(
            go.Scattermap(
                lat=top["lat"],
                lon=top["lon"],
                mode="markers",
                marker=go.scattermap.Marker(
                    size=top["marker_size"].tolist(),
                    color=top["depth_km"].tolist(),
                    colorscale="plasma",
                    cmin=depth_min,
                    cmax=depth_max,
                    showscale=(i == 0),
                    colorbar=dict(title="Depth (km)", thickness=14) if i == 0 else {},
                ),
                hovertemplate=hover,
                customdata=custom,
                visible=(i == 0),
                name=_LABELS.get(metric, metric).replace("\n", " "),
            )
        )

    # Build dropdown buttons — toggle one trace at a time.
    buttons = []
    for i, metric in enumerate(available):
        vis = [j == i for j in range(len(available))]
        buttons.append(
            dict(
                method="update",
                label=_LABELS.get(metric, metric).replace("\n", " "),
                args=[{"visible": vis}],
            )
        )

    map_cfg = dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=zoom)
    if bounds is not None:
        map_cfg["bounds"] = bounds

    fig = go.Figure(traces)
    fig.update_layout(
        map=map_cfg,
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                showactive=True,
                bgcolor="white",
                bordercolor="#ccc",
            )
        ],
        margin=dict(r=0, t=50, l=0, b=0),
        width=width, height=height,
        title=f"Top {top_n} Seismic Cells by Centrality Metric — {title}",
        legend_title="Metric",
    )
    if save:
        save_plotly(fig, f"centrality_geo_top_n_{_slug(title)}")
    fig.show()


def plot_geo_centrality_overlap(
    df: pd.DataFrame,
    top_n: int = 10,
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
    Composite map: nodes that appear in the top-N for multiple metrics.

    Colour encodes how many of the 8 centrality top-N lists a node appears
    in (1 = single metric, 8 = all metrics agree). Nodes appearing across
    many metrics are the true structural hubs of the seismic network.

    Hover shows every centrality value, so the map doubles as a data table.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_all_centralities`.
    top_n : int
        Top-N threshold used for each metric.
    title : str
        Figure title suffix.
    center_lat, center_lon : float
        Map centre coordinates.
    zoom : float
        Initial Plotly mapbox zoom level.
    """
    available = [m for m in _METRICS if m in df.columns]

    counts: dict[str, int] = {}
    for metric in available:
        for cell_id in df.nlargest(top_n, metric)["cell_id"]:
            counts[cell_id] = counts.get(cell_id, 0) + 1

    overlap_ids = set(counts)
    df_ov = df[df["cell_id"].isin(overlap_ids)].copy()
    df_ov["n_metrics"] = df_ov["cell_id"].map(counts)
    df_ov = df_ov.sort_values("n_metrics", ascending=False)

    hover_extra = {m: ":.5f" for m in available if m in df_ov.columns}

    fig = px.scatter_map(
        df_ov,
        lat="lat",
        lon="lon",
        color="n_metrics",
        size="n_metrics",
        size_max=22,
        color_continuous_scale="YlOrRd",
        range_color=[1, len(available)],
        hover_name="cell_id",
        hover_data={"depth_km": True, "n_metrics": True, **hover_extra},
        map_style="carto-positron",
        title=(
            f"Centrality Convergence: Nodes in Multiple Top-{top_n} Rankings — {title}"
        ),
    )
    map_cfg: dict = {"center": {"lat": center_lat, "lon": center_lon}, "zoom": zoom}
    if bounds is not None:
        map_cfg["bounds"] = bounds
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        width=width, height=height,
        coloraxis_colorbar=dict(
            title="# Metrics<br>top-10",
            tickvals=list(range(1, len(available) + 1)),
        ),
        map=map_cfg,
    )
    if save:
        save_plotly(fig, f"centrality_geo_overlap_{_slug(title)}")
    fig.show()
