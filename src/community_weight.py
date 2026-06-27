"""
Magnitude-aware weighting and ranking of seismic communities.

The community-detection step assigns every spatial cell (network node) to a
community.  By default a community's "importance" is just its node count, which
treats a large swarm of tiny tremors the same as a compact cluster that hosted a
mainshock.  This module re-weights communities by folding in the **magnitude of
the events** they contain, so attention concentrates on the big, energetic
communities rather than the many small ones.

Pipeline
--------
1. :func:`aggregate_community_stats` – map events → cells → communities and
   compute per-community size, event count, mean/max magnitude and total
   Gutenberg-Richter energy.
2. One of three weighting schemes (each a separate function, all returning the
   stats frame with a new ``weight`` column):

   * :func:`weight_gr_energy`   – total radiated energy, ``Σ 10^{1.5 M}``.
   * :func:`weight_count_mag`   – ``n_events · mean(M)``.
   * :func:`weight_size_exp`    – ``n_cells · 10^{α · mean(M)}``.

3. :func:`rank_top_k` – sort by the weight and keep the top-K communities.
4. :func:`plot_weight_bars` / :func:`plot_weighted_community_geo` – visualise.

References
----------
Gutenberg, B., & Richter, C. F. (1944). Frequency of earthquakes in
California. BSSA, 34(4), 185-188.  (Energy scaling ``log E = 1.5 M + 4.8``;
the additive constant is irrelevant to the ranking and is dropped here.)
"""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)

Partition = dict


def aggregate_community_stats(
    df_grid: pd.DataFrame,
    community_map: Partition,
    mag_col: str = "magnitude",
    cell_col: str = "cell_id",
) -> pd.DataFrame:
    """
    Aggregate per-event magnitudes up to the community level.

    Each event is mapped to its cell's community via ``community_map``; events
    in cells outside the partition (e.g. not in the giant component) are
    dropped.  Gutenberg-Richter energy uses ``E ∝ 10^{1.5 M}`` (the additive
    constant cancels in any ranking, so it is omitted).

    Parameters
    ----------
    df_grid : pd.DataFrame
        Event table that has been through ``discretize_space_3d`` so it carries
        ``cell_col`` and ``mag_col``.  Must use the **same** ``cell_size``/
        ``target_crs`` as the network whose communities are passed in, so the
        cell ids align with the node ids.
    community_map : dict
        ``{cell_id: community_int}`` partition.
    mag_col, cell_col : str
        Column names for magnitude and cell id.

    Returns
    -------
    pd.DataFrame
        One row per community with columns ``community``, ``n_cells``,
        ``n_events``, ``mean_mag``, ``max_mag``, ``energy_sum``.
    """
    comm = df_grid[cell_col].map(community_map)
    sub = df_grid.assign(community=comm).dropna(subset=["community"])
    sub = sub.assign(
        community=sub["community"].astype(int),
        _energy=10.0 ** (1.5 * sub[mag_col]),
    )

    grp = sub.groupby("community")
    stats = pd.DataFrame({
        "n_events":   grp.size(),
        "mean_mag":   grp[mag_col].mean(),
        "max_mag":    grp[mag_col].max(),
        "energy_sum": grp["_energy"].sum(),
    })

    n_cells = pd.Series(community_map).value_counts().rename("n_cells")
    stats = stats.join(n_cells, how="left")
    stats["n_cells"] = stats["n_cells"].fillna(0).astype(int)

    return stats.reset_index()[
        ["community", "n_cells", "n_events", "mean_mag", "max_mag", "energy_sum"]
    ]


# ── Three weighting schemes (each returns a new frame with a `weight` column) ──

def weight_gr_energy(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Weight = total Gutenberg-Richter energy ``Σ 10^{1.5 M}`` of the community.

    Physically motivated: a single strong event dominates the released energy,
    so communities that hosted a mainshock float to the top regardless of how
    many micro-tremors the catalog logged elsewhere.
    """
    return stats.assign(weight=stats["energy_sum"])


def weight_count_mag(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Weight = ``n_events · mean(M)``.

    A simple, readable blend of how busy a community is and how strong its
    events are on average; less skewed by a single outlier than the energy sum.
    """
    return stats.assign(weight=stats["n_events"] * stats["mean_mag"])


def weight_size_exp(stats: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    """
    Weight = ``n_cells · 10^{α · mean(M)}``.

    Mirrors the soft-network edge weighting: the spatial extent (cell count) is
    scaled by an exponential of the mean magnitude.  ``alpha`` dials how
    strongly magnitude dominates over size (``alpha → 0`` recovers pure size).
    """
    return stats.assign(weight=stats["n_cells"] * 10.0 ** (alpha * stats["mean_mag"]))


def rank_top_k(
    stats: pd.DataFrame,
    k: int = 10,
    weight_col: str = "weight",
    min_cells: int = 1,
    min_events: int = 1,
) -> pd.DataFrame:
    """
    Sort communities by ``weight_col`` (descending) and keep the top ``k``.

    Communities below the size floors are dropped *before* ranking. This guards
    the ``size × 10^(α·M̄)`` scheme in particular, where a singleton community
    holding one strong event has ``mean_mag`` equal to that event's magnitude
    and would otherwise outrank large clusters, contrary to the goal of
    focusing on the larger communities.

    Parameters
    ----------
    stats : pd.DataFrame
        Output of a weighting function (must contain ``weight_col``,
        ``n_cells``, ``n_events``).
    k : int
        Number of top communities to keep.
    weight_col : str
        Column to rank by.
    min_cells, min_events : int
        Minimum spatial extent / event count for a community to be eligible.

    Returns
    -------
    pd.DataFrame
        Top-``k`` communities with a 1-based ``rank`` column.
    """
    eligible = stats[(stats["n_cells"] >= min_cells) & (stats["n_events"] >= min_events)]
    out = eligible.sort_values(weight_col, ascending=False).head(k).reset_index(drop=True)
    return out.assign(rank=np.arange(1, len(out) + 1))


def plot_weight_bars(
    ranked: pd.DataFrame,
    weight_col: str = "weight",
    title: str = "",
    weight_label: str = "Community weight",
    save: bool = True,
    save_name: str | None = None,
) -> None:
    """
    Horizontal bar chart of the top-K communities by weight.

    Bars are coloured by weight (``plasma``) with a ``ScalarMappable`` colorbar,
    per the project plot-style standard.  Bar labels show ``n_cells`` /
    ``n_events`` / mean magnitude so size and strength are both legible.
    """
    df = ranked.iloc[::-1]  # largest at top
    norm = Normalize(vmin=df[weight_col].min(), vmax=df[weight_col].max())
    cmap = plt.get_cmap("plasma")
    colors = cmap(norm(df[weight_col].values))

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(df) + 1)))
    ypos = np.arange(len(df))
    ax.barh(ypos, df[weight_col].values, color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"C{int(c)}" for c in df["community"]])
    ax.set_xlabel(weight_label)
    ax.set_ylabel("Community")
    ax.set_title(f"Top {len(df)} communities by weight – {title}")

    for y, (_, r) in zip(ypos, df.iterrows()):
        ax.text(
            r[weight_col], y,
            f"  {int(r['n_cells'])} cells · {int(r['n_events'])} ev · M̄={r['mean_mag']:.2f}",
            va="center", ha="left", fontsize=8,
        )
    ax.margins(x=0.25)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=weight_label)
    plt.tight_layout()
    if save:
        savefig(save_name or f"community_weight_bars_{_slug(title)}")
    plt.show()


def plot_weighted_community_geo(
    G: nx.Graph,
    community_map: Partition,
    ranked: pd.DataFrame,
    title: str = "",
    weight_label: str = "Community weight",
    method_name: str = "",
    palette: list[str] | None = None,
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 0,
    bounds: dict | None = None,
    height: int = 700,
    width: int = 770,
    cell_event_counts: dict | None = None,
    save: bool = True,
    save_name: str | None = None,
) -> None:
    """
    Map only the top-K communities, one **discrete colour per community**.

    Nodes whose community is not in ``ranked`` are dropped, so the map shows the
    big/energetic communities only. Colour denotes community identity (legend
    ordered by rank, biggest weight first).

    Marker size encodes **per-cell activity** when ``cell_event_counts`` is
    provided (recommended): dense seismic cells render large; sparse peripheral
    cells fade into background even within "top weighted" communities. This
    decouples colour (community membership) from size (intrinsic cell activity)
    and avoids the misleading "community footprint as uniform tiles" rendering
    you get when all cells of a community share one size value. If
    ``cell_event_counts`` is ``None``, size falls back to the community's
    aggregate weight (legacy behaviour – every cell of a community gets the
    same marker size).
    """
    weight_of = dict(zip(ranked["community"], ranked["weight"]))
    rank_of = dict(zip(ranked["community"], ranked["rank"]))
    keep = set(ranked["community"])

    rows = []
    for n in G.nodes():
        if "lat" not in G.nodes[n]:
            continue
        c = community_map.get(n)
        if c is None or c not in keep:
            continue
        rows.append({
            "cell_id":   n,
            "community": str(int(c)),
            "rank":      int(rank_of[c]),
            "lat":       G.nodes[n]["lat"],
            "lon":       G.nodes[n]["lon"],
            "weight":    float(weight_of[c]),
            "n_events":  int(cell_event_counts.get(n, 1)) if cell_event_counts else 1,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("No nodes to plot for top-K communities (%s)", title)
        return

    # Legend ordered by rank (rank 1 = highest weight first).
    order = [str(int(c)) for c in ranked.sort_values("rank")["community"]]
    palette = palette or px.colors.qualitative.Dark24

    use_per_cell_size = cell_event_counts is not None
    size_col = "n_events" if use_per_cell_size else "weight"
    size_label = "events / cell" if use_per_cell_size else weight_label

    fig = px.scatter_map(
        df,
        lat="lat", lon="lon",
        color="community",
        size=size_col, size_max=22,
        color_discrete_sequence=palette,
        category_orders={"community": order},
        map_style="carto-positron",
        hover_name="community",
        hover_data={
            "rank": True,
            "weight": ":.3e",
            "n_events": True,
            "lat": ":.3f",
            "lon": ":.3f",
        },
        title=(
            f"Top {len(ranked)} weighted communities – {method_name} "
            f"(weighting: {weight_label}, size ∝ {size_label}) – {title}"
        ),
    )
    fig.update_traces(marker=dict(opacity=0.75))
    map_cfg: dict = {"center": {"lat": center_lat, "lon": center_lon}, "zoom": zoom}
    if bounds is not None:
        map_cfg["bounds"] = bounds
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        width=width, height=height,
        legend_title_text="community (by rank)",
        map=map_cfg,
    )
    if save:
        save_plotly(fig, save_name or f"community_weighted_geo_{_slug(method_name)}_{_slug(title)}")
    fig.show()
