"""
Centrality computation for the hybrid Abe–Suzuki earthquake network.

The network is weighted and thresholded:
- edges represent *effective interaction strength*
- weights combine magnitude, temporal decay, spatial decay, and filtering

We focus on five robust, interpretable measures:

--------------------------------------------------
CORE MEASURES (used in analysis)
--------------------------------------------------

Degree       – number of interacting neighbours (unweighted degree),
               proxy for the connectivity of a cell.

PageRank     – stationary flow of seismic influence through the network,
               identifying persistent "stress sinks" (in-flow / authority side).

CheiRank     – PageRank on the transposed network (A^T, i.e. G.reverse()),
               identifying persistent "stress sources" (out-flow / hub side).
               Same algorithm as PageRank; the PageRank–CheiRank pair forms a
               2D ranking of a directed network (Zhirov et al. 2010).

Closeness    – how quickly a cell can be reached by all others via shortest
               paths (in-closeness), proxy for global accessibility as a sink.

Betweenness  – how often a cell lies on shortest paths,
               identifies structural bridges between fault systems.

Note on weighting: Closeness and Betweenness are computed UNWEIGHTED (pure
topology). Edge weight here is an interaction *strength* (high = strong link),
not a distance, so feeding it to a shortest-path algorithm would invert its
meaning (it would treat strong links as long detours). A correct weighted
variant would require distance = 1/weight; we keep the unweighted topology
instead. PageRank and CheiRank do use the weights (degree is the unweighted count).

Clustering   – local density of weighted interactions,
               identifies coherent seismic neighborhoods / fault junctions.
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

from src.plotutils import savefig, save_plotly, _slug, pres_title

log = logging.getLogger(__name__)

_METRICS = [
    "Degree",
    "PageRank",
    "CheiRank",
    "Closeness",
    "Betweenness",
    "Clustering",
]
_LABELS = {
    "Degree":        "Degree",
    "PageRank":      "PageRank\n(stress sinks)",
    "CheiRank":      "CheiRank\n(stress sources)",
    "Closeness":     "Closeness in",
    "Betweenness":   "Betweenness",
    "Clustering":    "Clustering",
}


_DEFAULT_MEASURES = frozenset({
    "Degree", "PageRank", "CheiRank",
    "Closeness", "Betweenness", "Clustering",
})


def compute_all_centralities_hybrid(
    G: nx.DiGraph,
    k_betweenness: int = 1000,
    seed: int = 42,
    measures: set | list | None = None,
) -> pd.DataFrame:
    """
    Centrality measures for hybrid weighted earthquake network.
    """

    _DEFAULT = set(_DEFAULT_MEASURES)
    _req = set(measures) if measures is not None else _DEFAULT

    t_total = time.time()
    n = G.number_of_nodes()

    # --- Undirected projection for some measures ---
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    # --- Transposed network (A^T) for out-flow / source-side measures ---
    # G.reverse() flips every edge i->j into j->i; its adjacency is A^T.
    # Built once and shared by CheiRank.
    if _req & {"CheiRank"}:
        G_rev = G.reverse(copy=False)

    # --- 1. Degree (unweighted) ---
    if "Degree" in _req:
        log.info("Degree...")
        t0 = time.time()
        deg_cent = dict(G.degree())
        log.info("  %.1fs", time.time() - t0)

    # --- 2. PageRank (weighted flow, in-flow / sink side) ---
    if "PageRank" in _req:
        log.info("PageRank...")
        t0 = time.time()
        pr_cent = nx.pagerank(G, weight="weight")
        log.info("  %.1fs", time.time() - t0)

    # --- 2b. CheiRank = PageRank on the transposed network (out-flow / source) ---
    if "CheiRank" in _req:
        log.info("CheiRank (PageRank on A^T)...")
        t0 = time.time()
        chei_cent = nx.pagerank(G_rev, weight="weight")
        log.info("  %.1fs", time.time() - t0)

    # --- 3. Closeness, unweighted topology (in-closeness / accessibility as sink) ---
    if "Closeness" in _req:
        log.info("Closeness centrality (in)...")
        t0 = time.time()
        close_cent = nx.closeness_centrality(G)
        log.info("  %.1fs", time.time() - t0)

    # --- 4. Betweenness ---
    if "Betweenness" in _req:
        log.info("Betweenness centrality (k=%d)...", k_betweenness)
        t0 = time.time()
        bet_cent = nx.betweenness_centrality(
            G, k=min(k_betweenness, n), seed=seed
        )
        log.info("  %.1fs", time.time() - t0)

    # --- 5. Clustering (weighted local structure) ---
    if "Clustering" in _req:
        log.info("Weighted clustering...")
        t0 = time.time()
        clust_cent = nx.clustering(G_und, weight="weight")
        log.info("  %.1fs", time.time() - t0)

    # --- Assemble dataframe ---
    rows = []

    for node in G.nodes():
        if "lat" not in G.nodes[node] or "lon" not in G.nodes[node]:
            continue

        row = {
            "cell_id": node,
            "lat": G.nodes[node]["lat"],
            "lon": G.nodes[node]["lon"],
            "depth_km": G.nodes[node].get("depth_km", 0.0),
        }

        if "Degree" in _req:
            row["Degree"] = deg_cent.get(node, 0.0)

        if "PageRank" in _req:
            row["PageRank"] = pr_cent.get(node, 0.0)

        if "CheiRank" in _req:
            row["CheiRank"] = chei_cent.get(node, 0.0)

        if "Closeness" in _req:
            row["Closeness"] = close_cent.get(node, 0.0)

        if "Betweenness" in _req:
            row["Betweenness"] = bet_cent.get(node, 0.0)

        if "Clustering" in _req:
            row["Clustering"] = clust_cent.get(node, 0.0)

        rows.append(row)

    df = pd.DataFrame(rows)

    log.info(
        "Centrality complete: %d nodes, %d measures, %.1fs total",
        len(df), len(_req), time.time() - t_total
    )

    return df



def plot_centrality_correlation_hybrid(
    df: pd.DataFrame,
    title: str = "",
    save: bool = True
) -> None:
    """
    Spearman rank-correlation heatmap of hybrid centrality measures.

    Uses Spearman correlation because centrality distributions are
    heavy-tailed and often non-linear.

    In the hybrid Abe–Suzuki network, centralities reflect:
    - weighted interaction strength (not simple connectivity)
    - thresholded seismic influence structure
    """

    # Only hybrid-compatible metrics
    available = [
        m for m in _METRICS
        if m in df.columns and df[m].nunique() > 1
    ]

    if len(available) < 2:
        print("Not enough centrality measures for correlation plot.")
        return

    corr = df[available].corr(method="spearman")

    sz = max(7, len(available) * 0.8)
    fig, ax = plt.subplots(figsize=(sz + 1, sz))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
        ax=ax,
    )

    # cleaner labels (hybrid-consistent)
    labels = [
        _LABELS.get(m, m).replace("\n", " ")
        for m in available
    ]

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)

    # minimal title: the colorbar already reads "Spearman ρ" and the slide header
    # carries the catalog, so a short title fits the narrow square-heatmap band
    ax.set_title("Centrality correlation", pad=12)

    plt.tight_layout()

    if save:
        savefig(f"centrality_correlation_hybrid_{_slug(title)}")

    plt.show()







def plot_pagerank_cheirank_2d(
    df: pd.DataFrame,
    title: str = "",
    top_n_labels: int = 6,
    save: bool = True,
) -> None:
    """
    2D PageRank–CheiRank ranking plot for a directed network.

    Each cell is placed by its PageRank (in-flow / sink importance, x-axis)
    and CheiRank (out-flow / source importance, y-axis), both on log scales.
    The diagonal P = C marks flow balance; points are coloured by the
    asymmetry log10(PageRank / CheiRank): red = net sink (receives more
    influence than it emits), blue = net source. This is the standard 2D
    directed-network ranking of Zhirov, Zhirov & Shepelyansky (2010).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``PageRank`` and ``CheiRank`` columns (and ``cell_id``
        for labelling the most asymmetric cells).
    title : str
        Plot title suffix.
    top_n_labels : int
        Number of most sink-biased and most source-biased cells to annotate.
    save : bool
        Save the figure via ``savefig`` before showing.
    """
    if not {"PageRank", "CheiRank"} <= set(df.columns):
        print("Need both PageRank and CheiRank columns for the 2D plot.")
        return

    d = df[(df["PageRank"] > 0) & (df["CheiRank"] > 0)].copy()
    if d.empty:
        print("No cells with positive PageRank and CheiRank.")
        return

    d["asymmetry"] = np.log10(d["PageRank"] / d["CheiRank"])
    vmax = float(np.abs(d["asymmetry"]).quantile(0.98)) or 1.0

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        d["PageRank"], d["CheiRank"],
        c=d["asymmetry"], cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        s=22, alpha=0.8, edgecolors="none",
    )

    # diagonal P = C (perfect sink/source balance)
    lo = min(d["PageRank"].min(), d["CheiRank"].min())
    hi = max(d["PageRank"].max(), d["CheiRank"].max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=1, zorder=0,
            label="PageRank = CheiRank (balanced)")

    # annotate most asymmetric cells on both sides
    if "cell_id" in d.columns and top_n_labels > 0:
        extremes = pd.concat([
            d.nlargest(top_n_labels, "asymmetry"),   # net sinks
            d.nsmallest(top_n_labels, "asymmetry"),  # net sources
        ])
        for _, r in extremes.iterrows():
            ax.annotate(str(r["cell_id"]), (r["PageRank"], r["CheiRank"]),
                        fontsize=10, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PageRank  (in-flow / stress sink)")
    ax.set_ylabel("CheiRank  (out-flow / stress source)")
    ax.set_title(f"PageRank–CheiRank 2D ranking: {title}", pad=12)
    ax.legend(loc="lower right")

    cbar = fig.colorbar(sc, ax=ax, label=r"$\log_{10}(\mathrm{PageRank}/\mathrm{CheiRank})$")
    cbar.ax.text(0.5, 1.02, "sink", transform=cbar.ax.transAxes,
                 ha="center", va="bottom", fontsize=12)
    cbar.ax.text(0.5, -0.02, "source", transform=cbar.ax.transAxes,
                 ha="center", va="top", fontsize=12)

    plt.tight_layout()
    if save:
        savefig(f"pagerank_cheirank_2d_hybrid_{_slug(title)}")
    plt.show()


def plot_geo_top_n_interactive_hybrid(
    df: pd.DataFrame,
    top_n: int = 10,
    title: str = "",
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 5,
    bounds: dict | None = None,
    height: int = 600,
    width: int = 1100,
    save: bool = True,
    renderer: str | None = None,
) -> None:
    """
    Interactive Mapbox visualization of top-N hybrid centrality nodes.

    Each dropdown selection shows the most central cells according to:
    - weighted degree (interaction strength)
    - PageRank (flow of seismic influence)
    - closeness (global accessibility)
    - betweenness (fault bridges)
    - clustering (local interaction density)

    Pass ``renderer="png"`` (or ``"svg"`` / ``"pdf"``) to render a static image
    instead of the live figure (avoids exhausting the browser WebGL context cap).
    """

    # Only hybrid-valid metrics
    available = [
        m for m in _METRICS
        if m in df.columns and df[m].nunique() > 1
    ]

    if len(available) == 0:
        print("No valid centrality metrics found in dataframe.")
        return

    depth_min = float(df["depth_km"].min())
    depth_max = float(df["depth_km"].max())

    traces = []

    for i, metric in enumerate(available):

        top = df.nlargest(top_n, metric).copy().reset_index(drop=True)
        top["rank"] = top.index + 1

        # marker size: rank 1 = largest
        size_max, size_min = 28, 8
        top["marker_size"] = (
            size_max
            - (top["rank"] - 1) * (size_max - size_min) / max(top_n - 1, 1)
        )

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Rank: %{customdata[1]}<br>"
            f"{_LABELS.get(metric, metric).replace('\n', ' ')}: %{{customdata[2]:.5f}}<br>"
            "Depth: %{customdata[3]:.0f} km<br>"
            "Lat: %{lat:.3f} | Lon: %{lon:.3f}<extra></extra>"
        )

        custom = list(
            zip(top["cell_id"], top["rank"], top[metric], top["depth_km"])
        )

        traces.append(
            go.Scattermapbox(
                lat=top["lat"],
                lon=top["lon"],
                mode="markers",
                marker=dict(
                    size=top["marker_size"].tolist(),
                    color=top["depth_km"].tolist(),
                    colorscale="plasma",
                    cmin=depth_min,
                    cmax=depth_max,
                    showscale=(i == 0),
                    colorbar=dict(title="Depth (km)") if i == 0 else None,
                ),
                hovertemplate=hover,
                customdata=custom,
                visible=(i == 0),
                name=_LABELS.get(metric, metric).replace("\n", " "),
            )
        )

    # Dropdown menu
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

    fig = go.Figure(traces)

    map_cfg = dict(
        style="carto-positron",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=zoom,
    )

    if bounds is not None:
        map_cfg["bounds"] = bounds

    fig.update_layout(
        mapbox=map_cfg,
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
        width=width,
        height=height,
        title=pres_title(
            f"Top {top_n} central cells – {title}",
            "ranked by hybrid weighted centrality (dropdown = metric)"),
        legend_title="Metric",
    )

    if save:
        save_plotly(fig, f"centrality_geo_top_n_hybrid_{_slug(title)}")

    fig.show(renderer)








def plot_geo_centrality_overlap_hybrid(
    df: pd.DataFrame,
    top_n: int = 10,
    title: str = "",
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 5,
    bounds: dict | None = None,
    height: int = 600,
    width: int = 1100,
    save: bool = True,
    renderer: str | None = None,
) -> None:
    """
    Composite map showing nodes that appear in the top-N lists
    of multiple hybrid centrality measures.

    Interpretation:
    - 1 metric → locally important node
    - many metrics → structurally and dynamically dominant seismic hub

    In the hybrid network, overlap reflects agreement across:
    weighted interaction strength, flow (PageRank),
    accessibility (closeness), bridges (betweenness),
    and local interaction density (clustering).

    Pass ``renderer="png"`` (or ``"svg"`` / ``"pdf"``) to render a static image
    instead of the live figure (avoids exhausting the browser WebGL context cap).
    """

    # Only hybrid-compatible metrics
    available = [
        m for m in _METRICS
        if m in df.columns and df[m].nunique() > 1
    ]

    if len(available) < 2:
        print("Not enough metrics for overlap analysis.")
        return

    # --- Count how many top-N lists each node appears in ---
    counts = {}

    for metric in available:
        top_nodes = df.nlargest(top_n, metric)["cell_id"]

        for cell_id in top_nodes:
            counts[cell_id] = counts.get(cell_id, 0) + 1

    overlap_ids = set(counts.keys())

    df_ov = df[df["cell_id"].isin(overlap_ids)].copy()
    df_ov["n_metrics"] = df_ov["cell_id"].map(counts)

    df_ov = df_ov.sort_values("n_metrics", ascending=False)

    # hover only on available hybrid metrics
    hover_extra = {
        m: ":.5f"
        for m in available
        if m in df_ov.columns
    }

    fig = px.scatter_mapbox(
        df_ov,
        lat="lat",
        lon="lon",
        color="n_metrics",
        size="n_metrics",
        size_max=22,
        color_continuous_scale="YlOrRd",
        range_color=[1, len(available)],
        hover_name="cell_id",
        hover_data={
            "depth_km": True,
            "n_metrics": True,
            **hover_extra,
        },
        mapbox_style="carto-positron",
        title=pres_title(
            f"Centrality convergence: top-{top_n} overlap – {title}",
            "cells appearing in multiple centrality top-N lists"),
    )

    map_cfg = {
        "center": {"lat": center_lat, "lon": center_lon},
        "zoom": zoom,
    }

    if bounds is not None:
        map_cfg["bounds"] = bounds

    fig.update_layout(
        mapbox=map_cfg,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        width=width,
        height=height,
        coloraxis_colorbar=dict(
            title="# Metrics<br>top-N",
            tickvals=list(range(1, len(available) + 1)),
        ),
    )

    if save:
        save_plotly(fig, f"centrality_geo_overlap_hybrid_{_slug(title)}")

    fig.show(renderer)