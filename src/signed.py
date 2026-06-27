"""
Signed Abe-Suzuki earthquake network.

Extends the standard (weighted, directed) Abe-Suzuki network by labelling each
edge with a sign based on the direction of magnitude change between consecutive
earthquakes:

  +1  (positive / escalating) : magnitude increases  A → B
  -1  (negative / decaying)   : magnitude decreases  A → B
   0  (neutral)               : same magnitude

When multiple transitions exist between the same pair of cells the majority
sign is stored; ties go to +1 (escalation wins).

Analyses provided
-----------------
  plot_signed_degree        – in/out degree split by sign
  compute_structural_balance – fraction of triangles satisfying Heider balance
  analyze_chains             – length distribution of escalating / decaying runs
  plot_signed_geo_map        – map of net sign per node (net imbalance)

Seismological interpretations
------------------------------
  Positive chains (sequences of escalating magnitudes) should precede
  mainshocks if stress loading is gradual.  Negative chains should dominate
  aftershock sequences.  High structural balance would imply the network
  evolves toward a bipolar "loading vs releasing" state.

References
----------
Heider, F. (1946). Attitudes and cognitive organisation. Journal of Psychology,
  21, 107–112.
Cartwright, D., & Harary, F. (1956). Structural balance: a generalization of
  Heider's theory. Psychological Review, 63(5), 277–293.
"""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)


# ── Network construction ──────────────────────────────────────────────────────

def build_signed_network(df: pd.DataFrame, target_crs: str, cell_size_km: float = 10.0) -> nx.DiGraph:
    """
    Build a signed directed Abe-Suzuki network.

    Each consecutive pair of events contributes an edge with sign determined
    by the magnitude change.  Edge attributes stored:

    * ``weight``      – number of transitions between the two cells
    * ``pos_count``   – transitions with magnitude increase
    * ``neg_count``   – transitions with magnitude decrease
    * ``sign``        – majority sign (+1 / -1); ties → +1
    * ``net_sign``    – (pos_count - neg_count) / weight  ∈ [−1, +1]

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog sorted by time; must contain columns
        ``latitude``, ``longitude``, ``depth_km``, ``magnitude``.
    target_crs : str
        Equal-area CRS for the projection (e.g. ``"epsg:32632"``).
    cell_size_km : float
        Cubic cell edge length in km.

    Returns
    -------
    nx.DiGraph
        Signed directed network with the attributes described above.
    """
    from src.network import discretize_space_3d

    df_disc = discretize_space_3d(df, cell_size_km=cell_size_km, target_crs=target_crs)

    G = nx.DiGraph()

    # Node attributes: mean lat/lon
    for cell_id, grp in df_disc.groupby("cell_id"):
        G.add_node(cell_id,
                   lat=float(grp["latitude"].mean()),
                   lon=float(grp["longitude"].mean()))

    # Edge construction with sign tracking
    cells = df_disc["cell_id"].tolist()
    mags  = df_disc["magnitude"].tolist()
    edge_data: dict[tuple, dict] = {}

    for i in range(len(cells) - 1):
        src, dst = cells[i], cells[i + 1]
        sign = int(np.sign(mags[i + 1] - mags[i]))  # +1, 0, -1
        key = (src, dst)
        if key not in edge_data:
            edge_data[key] = {"weight": 0, "pos_count": 0, "neg_count": 0}
        edge_data[key]["weight"] += 1
        if sign >= 0:
            edge_data[key]["pos_count"] += 1
        else:
            edge_data[key]["neg_count"] += 1

    for (src, dst), d in edge_data.items():
        majority = +1 if d["pos_count"] >= d["neg_count"] else -1
        net = (d["pos_count"] - d["neg_count"]) / d["weight"]
        G.add_edge(src, dst,
                   weight=d["weight"],
                   pos_count=d["pos_count"],
                   neg_count=d["neg_count"],
                   sign=majority,
                   net_sign=net)

    log.info("Signed network: %d nodes, %d edges "
             "(+: %d, -: %d)",
             G.number_of_nodes(), G.number_of_edges(),
             sum(1 for _, _, d in G.edges(data=True) if d["sign"] > 0),
             sum(1 for _, _, d in G.edges(data=True) if d["sign"] < 0))
    return G


# ── Analysis functions ────────────────────────────────────────────────────────

def plot_signed_degree(G: nx.DiGraph, title: str = "", save: bool = True) -> None:
    """
    Four-panel degree distributions split by sign and direction.

    Panels: positive out-degree, negative out-degree,
            positive in-degree, negative in-degree.
    Log-log scale with power-law fit on the tail (k ≥ 5).
    """
    from scipy.stats import linregress

    def _degrees(G: nx.DiGraph, direction: str, sign: int) -> list[int]:
        result = []
        for n in G.nodes():
            if direction == "out":
                result.append(sum(
                    1 for _, _, d in G.out_edges(n, data=True) if d["sign"] == sign
                ))
            else:
                result.append(sum(
                    1 for _, _, d in G.in_edges(n, data=True) if d["sign"] == sign
                ))
        return [x for x in result if x > 0]

    panels = [
        ("out", +1, "Positive out-degree\n(escalating triggers)",  "steelblue"),
        ("out", -1, "Negative out-degree\n(decaying triggers)",    "crimson"),
        ("in",  +1, "Positive in-degree\n(escalating receptions)", "teal"),
        ("in",  -1, "Negative in-degree\n(decaying receptions)",   "orangered"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (direction, sign, label, color) in zip(axes, panels):
        degs = _degrees(G, direction, sign)
        if not degs:
            ax.set_visible(False)
            continue
        arr = np.array(degs, dtype=float)
        bins = np.logspace(np.log10(arr.min()), np.log10(arr.max()), 20)
        counts, edges = np.histogram(arr, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        Pk = counts / (len(arr) * widths)
        valid = Pk > 0
        ax.scatter(centers[valid], Pk[valid], color=color, s=40, alpha=0.8)
        mask = centers[valid] >= 5
        if mask.sum() > 2:
            slope, intercept, *_ = linregress(
                np.log10(centers[valid][mask]),
                np.log10(Pk[valid][mask])
            )
            fit = 10**intercept * centers[valid][mask]**slope
            ax.plot(centers[valid][mask], fit, "k--", lw=1.5,
                    label=rf"$\gamma={-slope:.2f}$")
            ax.legend(fontsize=9)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Degree k", fontsize=10)
        ax.set_ylabel("P(k)", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.3)

    fig.suptitle(f"Signed Degree Distributions – {title}", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        savefig(f"signed_degree_{_slug(title)}")
    plt.show()


def compute_structural_balance(G: nx.DiGraph) -> dict:
    """
    Compute the fraction of triangles satisfying Heider structural balance.

    A triangle (A, B, C) is balanced if the product of its three edge signs
    is +1 (i.e., all positive, or exactly two negatives).  The analysis is
    performed on the undirected projection of G.

    Returns
    -------
    dict with keys: ``n_triangles``, ``n_balanced``, ``balance_ratio``,
    ``frustration`` (1 − balance_ratio).
    """
    G_und = nx.Graph()
    for u, v, d in G.edges(data=True):
        if G_und.has_edge(u, v):
            # Keep majority sign if both directions exist
            existing = G_und[u][v]["sign"]
            G_und[u][v]["sign"] = d["sign"] if d["weight"] >= G_und[u][v].get("weight", 0) else existing
        else:
            G_und.add_edge(u, v, sign=d["sign"], weight=d["weight"])

    n_balanced = 0
    n_total    = 0

    # Enumerate triangles explicitly
    for u in G_und.nodes():
        nbrs = set(G_und.neighbors(u))
        for v in nbrs:
            for w in (nbrs & set(G_und.neighbors(v))):
                if u < v < w:
                    s_uv = G_und[u][v].get("sign", 0)
                    s_vw = G_und[v][w].get("sign", 0)
                    s_uw = G_und[u][w].get("sign", 0)
                    product = s_uv * s_vw * s_uw
                    n_total += 1
                    if product > 0:
                        n_balanced += 1

    balance_ratio = n_balanced / n_total if n_total > 0 else float("nan")
    log.info("Balance: %d / %d triangles balanced (%.1f%%)",
             n_balanced, n_total, balance_ratio * 100 if n_total > 0 else 0)
    return {
        "n_triangles":   n_total,
        "n_balanced":    n_balanced,
        "balance_ratio": round(balance_ratio, 4),
        "frustration":   round(1 - balance_ratio, 4) if n_total > 0 else float("nan"),
    }


def analyze_chains(df: pd.DataFrame, max_chain: int | None = None) -> pd.DataFrame:
    """
    Compute the length distribution of escalating (+) and decaying (−) runs
    in the raw magnitude time series.

    A run of length k means k consecutive magnitude increases (or decreases)
    in a row.  Short runs dominate random processes; long escalating runs are
    consistent with stress loading toward a mainshock.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog sorted by time with a ``magnitude`` column.
    max_chain : int or None
        Maximum run length to track explicitly; longer runs are grouped into
        the ``max_chain`` bucket.  If ``None`` (default), auto-detected as the
        longest run actually present in the data so no empty bins are created.

    Returns
    -------
    pd.DataFrame
        Columns: ``length``, ``n_escalating``, ``n_decaying``.
    """
    mags = df.sort_values("time")["magnitude"].values
    signs = np.sign(np.diff(mags))   # +1, 0, -1

    raw_pos: dict[int, int] = {}
    raw_neg: dict[int, int] = {}

    i = 0
    while i < len(signs):
        s = signs[i]
        if s == 0:
            i += 1
            continue
        length = 1
        while i + length < len(signs) and signs[i + length] == s:
            length += 1
        if s > 0:
            raw_pos[length] = raw_pos.get(length, 0) + 1
        else:
            raw_neg[length] = raw_neg.get(length, 0) + 1
        i += length

    all_lengths = set(raw_pos) | set(raw_neg)
    if not all_lengths:
        return pd.DataFrame({"length": [], "n_escalating": [], "n_decaying": []})

    if max_chain is None:
        max_chain = max(all_lengths)

    pos_counts = {k: 0 for k in range(1, max_chain + 1)}
    neg_counts = {k: 0 for k in range(1, max_chain + 1)}
    for length, cnt in raw_pos.items():
        pos_counts[min(length, max_chain)] += cnt
    for length, cnt in raw_neg.items():
        neg_counts[min(length, max_chain)] += cnt

    return pd.DataFrame({
        "length":        list(pos_counts.keys()),
        "n_escalating":  list(pos_counts.values()),
        "n_decaying":    list(neg_counts.values()),
    })


def plot_chains(df_chains: pd.DataFrame, title: str = "", save: bool = True) -> None:
    """
    Bar chart of escalating vs decaying chain length distribution.

    Log-scale y-axis to show the tail clearly.

    Parameters
    ----------
    df_chains : pd.DataFrame
        Output of :func:`analyze_chains`.
    title : str
        Figure title suffix.
    """
    x = np.arange(len(df_chains))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, df_chains["n_escalating"], w,
           color="steelblue", alpha=0.85, edgecolor="k", label="Escalating (+)")
    ax.bar(x + w/2, df_chains["n_decaying"], w,
           color="crimson", alpha=0.85, edgecolor="k", label="Decaying (−)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(l) if l < df_chains["length"].max() else f"≥{l}"
         for l in df_chains["length"]]
    )
    ax.set_xlabel("Run length (consecutive same-sign transitions)", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_title(f"Magnitude Run-Length Distribution – {title}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        savefig(f"signed_chains_{_slug(title)}")
    plt.show()


def plot_signed_geo_map(
    G: nx.DiGraph,
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
    Interactive Plotly mapbox where each node is coloured by its net-sign
    imbalance: mean net_sign over all edges incident to the node.

    Red nodes (net_sign > 0) are net escalating cells – they tend to trigger
    or receive events with higher magnitude than themselves.
    Blue nodes (net_sign < 0) are net decaying cells – typical aftershock zones.

    Parameters
    ----------
    G : nx.DiGraph
        Output of :func:`build_signed_network`.
    title : str
        Figure title suffix.
    center_lat, center_lon, zoom : float
        Initial map view.
    bounds : dict, optional
        Explicit viewport bounds ``{"west": ..., "east": ..., "south": ...,
        "north": ...}`` – overrides zoom/center for static exports.
    """
    rows = []
    for n in G.nodes():
        if "lat" not in G.nodes[n]:
            continue
        incident = (
            [d["net_sign"] for _, _, d in G.out_edges(n, data=True)]
            + [d["net_sign"] for _, _, d in G.in_edges(n, data=True)]
        )
        if not incident:
            continue
        rows.append({
            "cell_id":   n,
            "lat":       G.nodes[n]["lat"],
            "lon":       G.nodes[n]["lon"],
            "net_sign":  float(np.mean(incident)),
            "degree":    G.degree(n),
        })
    df = pd.DataFrame(rows)

    # Use actual data range so near-zero nodes don't collapse to white.
    vmax = max(float(df["net_sign"].abs().quantile(0.95)), 0.05)

    fig = px.scatter_map(
        df, lat="lat", lon="lon",
        color="net_sign", size="degree", size_max=18,
        color_continuous_scale="RdBu",
        range_color=[-vmax, vmax],
        hover_name="cell_id",
        hover_data={"net_sign": ":.3f", "degree": True},
        map_style="carto-darkmatter",
        title=f"Signed Network – Net Escalation Index per Cell: {title}",
    )
    map_cfg: dict = {"center": {"lat": center_lat, "lon": center_lon}, "zoom": zoom}
    if bounds is not None:
        map_cfg["bounds"] = bounds
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        width=width, height=height,
        coloraxis_colorbar=dict(title="Net sign<br>(+: escalating, −: decaying)"),
        map=map_cfg,
    )
    if save:
        save_plotly(fig, f"signed_geo_map_{_slug(title)}")
    fig.show()
