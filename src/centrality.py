"""
Centrality computation and comparison for the Abe-Suzuki earthquake network.

Computes 13 measures in a single function, returns a unified DataFrame,
and provides two diagnostic visualisations:
  1. Spearman rank-correlation heatmap across measures.
  2. Multi-panel top-N cell bar chart per measure.

Measures and seismological interpretations
------------------------------------------
In_Degree    — susceptibility: how often a cell is triggered by others.
Out_Degree   — productivity: how many distinct cells a cell triggers.
Degree       — total activity (in + out), most seismically active cells.
PageRank     — "stress sinks": cells that persistently receive seismic flow.
Harmonic     — topological reach via sum of inverse distances; handles
               disconnected nodes gracefully (closeness is 0 for unreachable nodes).
Closeness    — cells that can spread seismic influence fastest across the network.
Betweenness  — "bridges": cells on shortest paths between fault clusters.
Eigenvector  — cells embedded in the high-activity core (rich-club).
Katz         — like eigenvector but counts ALL paths (with exponential decay),
               more robust for directed/sparse graphs.
HITS Hub     — cells that trigger important seismic zones (high out-connections
               to high-authority cells).
HITS Auth    — cells that are the primary destinations of seismic propagation.
Clustering   — local clustering coefficient: fraction of a cell's neighbours
               that are also mutually connected; high at fault junctions.
Triangles    — raw triangle count per node (undirected); zero in a perfect tree,
               high at fault intersections and dense aftershock clusters.

Bianconi-Barabasi fitness
-------------------------
compute_bb_fitness estimates the growth-rate exponent beta_i for each cell
from its final degree and birth time (first recorded event in that cell).
plot_bb_fitness shows the fitness distribution, growth diagram, and Lorenz
condensation curve. plot_bb_fitness_geo maps fitness geographically.
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
    "In_Degree", "Out_Degree", "Degree", "PageRank",
    "Harmonic", "Closeness", "Betweenness", "Eigenvector",
    "Katz", "HITS_Hub", "HITS_Auth", "Clustering", "Triangles",
]
_LABELS = {
    "In_Degree":   "In-Degree\n(susceptibility)",
    "Out_Degree":  "Out-Degree\n(productivity)",
    "Degree":      "Degree\n(active cells)",
    "PageRank":    "PageRank\n(stress sinks)",
    "Harmonic":    "Harmonic\n(topological reach)",
    "Closeness":   "Closeness\n(global spread)",
    "Betweenness": "Betweenness\n(fault bridges)",
    "Eigenvector": "Eigenvector\n(rich-club core)",
    "Katz":        "Katz\n(all-path influence)",
    "HITS_Hub":    "HITS Hub\n(seismic triggers)",
    "HITS_Auth":   "HITS Authority\n(seismic destinations)",
    "Clustering":  "Clustering\n(local density)",
    "Triangles":   "Triangles\n(fault junctions)",
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

    # ── 1. In-degree / Out-degree / Total degree ─────────────────────────────
    log.info("Degree centralities (in, out, total)...")
    t0 = time.time()
    in_deg_cent  = nx.in_degree_centrality(G)
    out_deg_cent = nx.out_degree_centrality(G)
    deg_cent     = nx.degree_centrality(G)
    log.info("  %.1fs", time.time() - t0)

    # ── 2. PageRank ──────────────────────────────────────────────────────────
    log.info("PageRank...")
    t0 = time.time()
    pr_cent = nx.pagerank(G, weight="weight")
    log.info("  %.1fs", time.time() - t0)

    # ── 3. Harmonic ──────────────────────────────────────────────────────────
    log.info("Harmonic centrality...")
    t0 = time.time()
    harm_cent = nx.harmonic_centrality(G)
    log.info("  %.1fs", time.time() - t0)

    # ── 4. Closeness ─────────────────────────────────────────────────────────
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

    # ── 9. Clustering coefficient (undirected, weighted) ─────────────────────
    log.info("Clustering coefficient...")
    t0 = time.time()
    clust_cent = nx.clustering(G_und, weight="weight")
    log.info("  %.1fs", time.time() - t0)

    # ── 10. Triangle count (undirected) ──────────────────────────────────────
    log.info("Triangle count...")
    t0 = time.time()
    tri_count = nx.triangles(G_und)
    log.info("  %.1fs", time.time() - t0)

    # ── 11 & 12. HITS hub + authority ────────────────────────────────────────
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
    def _depth(node):
        d = G.nodes[node].get("depth_km")
        if d is not None:
            return float(d)
        if isinstance(node, (tuple, list)):
            return float(node[2]) * cell_size_km
        try:
            return float(str(node).split("_")[2]) * cell_size_km
        except (IndexError, ValueError):
            return 0.0

    rows = [
        {
            "cell_id":     node,
            "lat":         G.nodes[node]["lat"],
            "lon":         G.nodes[node]["lon"],
            "depth_km":    _depth(node),
            "In_Degree":   in_deg_cent.get(node, 0.0),
            "Out_Degree":  out_deg_cent.get(node, 0.0),
            "Degree":      deg_cent.get(node, 0.0),
            "PageRank":    pr_cent.get(node, 0.0),
            "Harmonic":    harm_cent.get(node, 0.0),
            "Closeness":   close_cent.get(node, 0.0),
            "Betweenness": bet_cent.get(node, 0.0),
            "Eigenvector": eig_cent.get(node, 0.0),
            "Katz":        katz_cent.get(node, 0.0),
            "HITS_Hub":    hits_hub.get(node, 0.0),
            "HITS_Auth":   hits_auth.get(node, 0.0),
            "Clustering":  clust_cent.get(node, 0.0),
            "Triangles":   float(tri_count.get(node, 0)),
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

    sz = max(7, len(available) * 0.8)
    fig, ax = plt.subplots(figsize=(sz + 1, sz))
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


# ── Bianconi–Barabási fitness ─────────────────────────────────────────────────

def compute_bb_fitness(
    G: nx.DiGraph,
    df: pd.DataFrame,
    cell_size_km: float = 10.0,
    target_crs: str = "epsg:5070",
) -> pd.DataFrame:
    """
    Estimate Bianconi–Barabási fitness for each spatial cell.

    In the Bianconi–Barabási (2001) model, new nodes attach to existing nodes
    with probability proportional to both degree *and* an intrinsic fitness
    :math:`\\eta_i`:

    .. math::

        \\pi_i \\propto \\eta_i\\, k_i.

    Under this rule the degree of node :math:`i`, born at time :math:`t_i`,
    grows as

    .. math::

        k_i(t) \\approx m\\left(\\frac{t}{t_i}\\right)^{\\beta_i},
        \\qquad \\beta_i = \\frac{\\eta_i}{C},

    where :math:`C = \\int \\eta\\,\\rho(\\eta)\\,\\beta(\\eta)\\,d\\eta` is a
    self-consistency constant and :math:`m` is the number of edges added per
    new node.  The empirical estimate

    .. math::

        \\hat{\\beta}_i = \\frac{\\ln k_i(T)}{\\ln(T / t_i)}

    follows directly from the power-law growth equation.
    :math:`\\hat{\\beta}_i` is proportional to :math:`\\eta_i` up to the common
    factor :math:`C`; it therefore provides a relative ranking of cell fitness.

    **Seismological interpretation.** A cell with high :math:`\\hat{\\beta}`
    acquired connections rapidly after its first recorded event — it is an
    *intrinsically productive* fault zone whose seismogenic rate is not merely
    a consequence of being old (first-mover advantage) but of a high intrinsic
    activity level.  **Bose-Einstein condensation** occurs when one cell
    captures a finite fraction of all edges; this is the network-theoretic
    signature of a dominant seismogenic zone (e.g. The Geysers geothermal
    field in the US catalog, or the central Apennines corridor in Italy).

    Parameters
    ----------
    G : nx.DiGraph
        The Abe–Suzuki cell-transition network (10 km resolution recommended).
    df : pd.DataFrame
        Raw earthquake catalog with columns ``time``, ``latitude``,
        ``longitude``, ``depth_km``.  Need not be pre-sorted.
    cell_size_km : float
        Grid resolution; must match the resolution used to build *G*.
    target_crs : str
        Projection CRS; must match the one used to build *G*.

    Returns
    -------
    pd.DataFrame
        One row per cell with columns ``cell_id``, ``lat``, ``lon``,
        ``k_final``, ``t_birth_days``, ``fitness_beta``.
        Cells with :math:`k \\leq 1` or born in the last 5 % of the catalog
        are excluded (undefined logarithm or insufficient growth time).

    References
    ----------
    Bianconi G. & Barabási A.-L. (2001). Competition and multiscaling in
    evolving networks. *Europhysics Letters*, 54(4), 436–442.

    Bianconi G. & Barabási A.-L. (2001). Bose-Einstein condensation in
    complex networks. *Physical Review Letters*, 86(24), 5632–5635.
    """
    from src.network import discretize_space_3d  # noqa: PLC0415

    df_s = df.sort_values("time").reset_index(drop=True)
    df_grid = discretize_space_3d(df_s, cell_size_km=cell_size_km, target_crs=target_crs)

    times = pd.to_datetime(df_s["time"])
    t_start = times.iloc[0]
    t_end   = times.iloc[-1]
    T_days  = (t_end - t_start).total_seconds() / 86400.0

    df_grid = df_grid.copy()
    df_grid["t_days"] = (times - t_start).dt.total_seconds() / 86400.0

    first_t = df_grid.groupby("cell_id")["t_days"].min()

    degrees = dict(G.degree())

    rows = []
    for node in G.nodes():
        if node not in first_t.index:
            continue
        ti = float(first_t[node])
        k  = degrees.get(node, 0)
        # skip: no growth data, too young, or trivial degree
        if k <= 1 or ti <= 0 or ti >= 0.95 * T_days:
            continue
        ratio = T_days / ti
        if ratio <= 1.0:
            continue
        beta = np.log(k) / np.log(ratio)
        rows.append({
            "cell_id":      node,
            "lat":          G.nodes[node].get("lat"),
            "lon":          G.nodes[node].get("lon"),
            "k_final":      k,
            "t_birth_days": ti,
            "fitness_beta": float(beta),
        })

    df_fit = pd.DataFrame(rows)
    log.info(
        "BB fitness: %d cells; β range [%.3f, %.3f]; T=%.0f days",
        len(df_fit),
        df_fit["fitness_beta"].min() if len(df_fit) else 0,
        df_fit["fitness_beta"].max() if len(df_fit) else 0,
        T_days,
    )
    return df_fit


def compute_bb_fitness_events(
    G: nx.Graph,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate Bianconi–Barabási fitness for each event in an event-level network.

    Equivalent to :func:`compute_bb_fitness` but for networks where nodes are
    individual earthquakes (BP, ZBZ, ETAS, TL, HVG) rather than spatial cells.
    Node *i* in *G* corresponds to row *i* of *df* (0-indexed, catalog sorted
    by time).  The fitness estimator is

    .. math::

        \\hat{\\beta}_i = \\frac{\\ln k_i(T)}{\\ln(T / t_i)},

    where :math:`t_i` is the time of event *i* and :math:`T` is the catalog
    duration.

    Parameters
    ----------
    G : nx.Graph
        Event-level network (directed or undirected). Node IDs are integers
        0…N-1 matching the time-sorted DataFrame row order.
    df : pd.DataFrame
        Earthquake catalog **sorted by time**, with columns ``time``,
        ``latitude``, ``longitude``.

    Returns
    -------
    pd.DataFrame
        One row per node with columns ``cell_id`` (node index), ``lat``,
        ``lon``, ``k_final``, ``t_birth_days``, ``fitness_beta``.
        Nodes with k ≤ 1, t_i = 0 (first event), or born in the last 5 %
        of the catalog are excluded.

    References
    ----------
    Bianconi G. & Barabási A.-L. (2001). Competition and multiscaling in
    evolving networks. *Europhysics Letters*, 54(4), 436–442.
    """
    df_s = df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df_s["time"])
    t_start = times.iloc[0]
    t_end   = times.iloc[-1]
    T_days  = (t_end - t_start).total_seconds() / 86400.0
    t_days  = (times - t_start).dt.total_seconds() / 86400.0

    degrees = dict(G.degree())
    rows = []
    for node in G.nodes():
        if node >= len(df_s):
            continue
        ti = float(t_days.iloc[node])
        k  = degrees.get(node, 0)
        if k <= 1 or ti <= 0 or ti >= 0.95 * T_days:
            continue
        ratio = T_days / ti
        if ratio <= 1.0:
            continue
        beta = np.log(k) / np.log(ratio)
        rows.append({
            "cell_id":      node,
            "lat":          float(df_s["latitude"].iloc[node]),
            "lon":          float(df_s["longitude"].iloc[node]),
            "k_final":      k,
            "t_birth_days": ti,
            "fitness_beta": float(beta),
        })

    df_fit = pd.DataFrame(rows)
    log.info(
        "BB fitness (events): %d nodes; β range [%.3f, %.3f]; T=%.0f days",
        len(df_fit),
        df_fit["fitness_beta"].min() if len(df_fit) else 0.0,
        df_fit["fitness_beta"].max() if len(df_fit) else 0.0,
        T_days,
    )
    return df_fit


def plot_bb_fitness(
    df_fit: pd.DataFrame,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Three-panel diagnostic for Bianconi–Barabási fitness.

    **Panel 1 — Fitness distribution ρ(β).**  Histogram of :math:`\\hat{\\beta}`
    values with KDE overlay.  A broad distribution indicates heterogeneous
    seismogenic potential; a narrow peak near :math:`\\beta \\approx 0` signals
    near-uniform fitness (standard Barabási–Albert behaviour).

    **Panel 2 — Growth diagram.**  :math:`\\ln k_i` vs :math:`\\ln(T/t_i)`,
    coloured by :math:`\\hat{\\beta}`.  Under pure preferential attachment
    (uniform fitness) all points lie on a single line of slope :math:`\\bar{\\beta}`;
    scatter above the line identifies high-fitness outliers.

    **Panel 3 — Condensation Lorenz curve.**  Cells sorted by
    :math:`\\hat{\\beta}` descending; cumulative share of total degree (y-axis)
    vs cumulative fraction of cells (x-axis).  Perfect equality = diagonal
    (standard BA); a convex curve bowing toward the top-left corner indicates
    degree concentration in the high-fitness minority — the network-theoretic
    signature of Bose-Einstein condensation.

    Parameters
    ----------
    df_fit : pd.DataFrame
        Output of :func:`compute_bb_fitness`.
    title : str
        Figure title suffix.

    References
    ----------
    Bianconi G. & Barabási A.-L. (2001). Bose-Einstein condensation in
    complex networks. *Physical Review Letters*, 86, 5632–5635.
    """
    if df_fit.empty:
        log.warning("plot_bb_fitness: empty DataFrame, skipping.")
        return

    beta  = df_fit["fitness_beta"].values
    k     = df_fit["k_final"].values
    ti    = df_fit["t_birth_days"].values
    T_max = ti.max() / (1 - 0.05)   # approximate T from max birth time

    log_k    = np.log(k)
    log_ratio = np.log(T_max / ti)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: fitness distribution ────────────────────────────────────────
    ax = axes[0]
    ax.hist(beta, bins=40, color="steelblue", alpha=0.7, edgecolor="white",
            linewidth=0.4, density=True, label="ρ(β) histogram")
    from scipy.stats import gaussian_kde
    kde_x = np.linspace(beta.min(), beta.max(), 300)
    kde_y = gaussian_kde(beta)(kde_x)
    ax.plot(kde_x, kde_y, "r-", linewidth=1.8, label="KDE")
    ax.axvline(float(np.median(beta)), color="gray", linestyle="--",
               linewidth=1.2, label=f"Median β = {np.median(beta):.3f}")
    ax.set_xlabel("Fitness β̂", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Fitness distribution ρ(β)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # ── Panel 2: growth diagram ───────────────────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(log_ratio, log_k, c=beta, cmap="plasma", s=8, alpha=0.6)
    # reference line: pure BA (slope = mean beta)
    mean_beta = float(np.mean(beta))
    x_line = np.array([log_ratio.min(), log_ratio.max()])
    ax.plot(x_line, mean_beta * x_line, "w--", linewidth=1.5, alpha=0.8,
            label=f"BA reference (β̄ = {mean_beta:.3f})")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Fitness β̂", fontsize=9)
    ax.set_xlabel("ln(T / t_i)  [network age ratio]", fontsize=11)
    ax.set_ylabel("ln(k_i)  [log degree]", fontsize=11)
    ax.set_title("Growth diagram", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # ── Panel 3: Lorenz / condensation curve ─────────────────────────────────
    ax = axes[2]
    sort_idx  = np.argsort(beta)[::-1]   # descending fitness
    k_sorted  = k[sort_idx]
    cum_k     = np.cumsum(k_sorted) / k_sorted.sum()
    cum_cells = np.arange(1, len(k_sorted) + 1) / len(k_sorted)
    ax.plot(cum_cells, cum_k, color="tomato", linewidth=2,
            label="Empirical Lorenz curve")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.6, label="Perfect equality (BA)")
    # Gini coefficient as annotation
    gini = 1 - 2 * np.trapz(cum_k, cum_cells)
    ax.text(0.05, 0.92, f"Gini = {gini:.3f}", transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xlabel("Cumulative fraction of cells\n(sorted by β̂ desc.)", fontsize=11)
    ax.set_ylabel("Cumulative degree share", fontsize=11)
    ax.set_title("Condensation Lorenz curve", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        f"Bianconi–Barabási fitness — {title}\n"
        f"n={len(df_fit)} cells, β̄={mean_beta:.3f}, "
        f"top-1% hold {100*cum_k[max(0,int(0.01*len(cum_k))-1)]:.1f}% of degree",
        fontsize=12,
    )
    plt.tight_layout()
    if save:
        savefig(f"bb_fitness_{_slug(title)}")
    plt.show()


def plot_bb_fitness_theory(
    df_fit: pd.DataFrame,
    gamma: float,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Compare the observed β̂ distribution against three Bianconi-Barabási regimes.

    The BB model predicts different fitness distributions depending on ρ(η):

    * **Equal fitness** ρ(η) = δ(η − 1) reproduces Barabási-Albert with γ = 3.
      All cells have the same growth exponent β = ½ (vertical dashed line).

    * **Uniform fitness** ρ(η) = U[0, 1] produces a degree distribution
      :math:`p_k \\propto k^{-(1+C)}/\\ln k` with :math:`C \\approx 1.255`
      (Bianconi & Barabási 2001).  The β̂ values are approximately uniformly
      distributed on :math:`[0,\\,(\\gamma-1)]` (shown as a shaded band).

    * **Bose-Einstein condensation**: one cell captures a finite fraction of
      all edges.  Empirically this appears as an isolated large-β̂ outlier far
      above the bulk distribution.

    The observed γ from the degree distribution is used to estimate C = 1/(γ-1)
    and the upper bound of the uniform-fitness prediction β_max = γ − 1.

    Parameters
    ----------
    df_fit : pd.DataFrame
        Output of :func:`compute_bb_fitness`.
    gamma : float
        Observed power-law exponent of the degree distribution (MLE estimate).
    title : str
        Figure title suffix.
    save : bool
        Whether to save the figure.

    References
    ----------
    Bianconi G. & Barabási A.-L. (2001). Competition and multiscaling in
    evolving networks. *Europhysics Letters* 54, 436–442.

    Bianconi G. & Barabási A.-L. (2001). Bose-Einstein condensation in
    complex networks. *Physical Review Letters* 86, 5632–5635.
    """
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    if df_fit.empty:
        log.warning("plot_bb_fitness_theory: empty DataFrame, skipping.")
        return

    beta = df_fit["fitness_beta"].values

    # Theoretical parameters
    beta_BA   = 0.5             # equal-fitness (BA) prediction: k(T) ~ (T/t)^{1/2}
    beta_max  = float(gamma - 1.0) if gamma > 1.0 else 1.0  # uniform-fitness upper bound
    C_uniform = 1.255           # self-consistency constant for U[0,1] fitness

    # Condensation: top-1 % β̂ relative to median
    beta_99 = float(np.percentile(beta, 99))
    beta_med = float(np.median(beta))
    condensation_ratio = beta_99 / beta_med if beta_med > 0 else np.nan

    fig, ax = plt.subplots(figsize=(9, 5))

    # Observed histogram + KDE
    ax.hist(beta, bins=50, density=True, color="steelblue", alpha=0.45,
            edgecolor="white", linewidth=0.4, label="Observed ρ(β̂)")
    kde_x = np.linspace(beta.min(), max(beta.max(), beta_max * 1.1), 400)
    kde_y = gaussian_kde(beta)(kde_x)
    ax.plot(kde_x, kde_y, color="steelblue", linewidth=2.0, label="Observed KDE")

    # Equal-fitness prediction (BA regime)
    ax.axvline(beta_BA, color="green", linewidth=2.0, linestyle="--",
               label=rf"Equal fitness (BA): $\beta = {beta_BA:.2f}$, $\gamma = 3$")

    # Uniform-fitness prediction band [0, beta_max]
    if beta_max > 0:
        density_unif = 1.0 / beta_max  # uniform density on [0, beta_max]
        ax.fill_betweenx([0, density_unif * 1.05], 0, beta_max,
                         color="tomato", alpha=0.15, label=None)
        ax.hlines(density_unif, 0, beta_max, colors="tomato", linewidths=1.8,
                  linestyles="-",
                  label=rf"Uniform fitness: $U[0,\,{beta_max:.2f}]$, "
                        rf"$\gamma={gamma:.2f}$")
        ax.axvline(beta_max, color="tomato", linewidth=1.2, linestyle=":",
                   alpha=0.7)

    # Condensation indicator
    ax.axvline(beta_99, color="purple", linewidth=1.4, linestyle="-.",
               label=rf"99th pct β̂ = {beta_99:.2f}  (ratio to median: {condensation_ratio:.1f}×)")

    # Verdict
    dist_to_BA    = abs(float(np.median(beta)) - beta_BA)
    spread_rel    = float(beta.std()) / beta_BA if beta_BA > 0 else np.nan
    if condensation_ratio > 5.0:
        verdict = "possible Bose-Einstein condensation (extreme outlier)"
    elif dist_to_BA < 0.15 and spread_rel < 0.4:
        verdict = "consistent with equal-fitness / BA regime"
    elif float(np.median(beta)) < beta_max * 0.9:
        verdict = "consistent with heterogeneous fitness (uniform/mixed regime)"
    else:
        verdict = "intermediate regime"

    ax.set_xlabel(r"Fitness $\hat{\beta}_i$", fontsize=12)
    ax.set_ylabel(r"Density $\rho(\hat{\beta})$", fontsize=12)
    ax.set_title(
        f"BB fitness regime analysis — {title}\n"
        f"γ = {gamma:.2f}, median β̂ = {beta_med:.3f} → {verdict}",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    if save:
        savefig(f"bb_fitness_theory_{_slug(title)}")
    plt.show()


def plot_bb_fitness_geo(
    df_fit: pd.DataFrame,
    title: str = "",
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 0,
    bounds: dict | None = None,
    height: int = 600,
    width: int = 900,
    save: bool = True,
) -> None:
    """
    Interactive geographic map of Bianconi–Barabási fitness β̂.

    Marker colour encodes fitness :math:`\\hat{\\beta}`; marker size encodes
    the final degree :math:`k_i`.  High-fitness cells (warm colours, large
    markers) are intrinsically productive fault zones whose seismogenic rate
    cannot be attributed to seniority alone.

    Parameters
    ----------
    df_fit : pd.DataFrame
        Output of :func:`compute_bb_fitness`.
    """
    if df_fit.empty:
        log.warning("plot_bb_fitness_geo: empty DataFrame, skipping.")
        return

    df_plot = df_fit.dropna(subset=["lat", "lon"]).copy()
    df_plot = df_plot.sort_values("fitness_beta", ascending=True)

    fig = px.scatter_map(
        df_plot,
        lat="lat",
        lon="lon",
        color="fitness_beta",
        size="k_final",
        size_max=20,
        color_continuous_scale="plasma",
        hover_name="cell_id",
        hover_data={"k_final": True, "t_birth_days": ":.0f", "fitness_beta": ":.4f"},
        map_style="carto-positron",
        title=f"Bianconi–Barabási Fitness β̂ — {title}",
    )
    map_cfg: dict = {"center": {"lat": center_lat, "lon": center_lon}, "zoom": zoom}
    if bounds is not None:
        map_cfg["bounds"] = bounds
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        width=width, height=height,
        coloraxis_colorbar=dict(title="Fitness β̂"),
        map=map_cfg,
    )
    if save:
        save_plotly(fig, f"bb_fitness_geo_{_slug(title)}")
    fig.show()
