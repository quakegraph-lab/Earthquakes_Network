"""
Network construction functions for the Abe-Suzuki earthquake network.

References
----------
Abe, S., & Suzuki, N. (2004). Scale-free network of earthquakes.
Europhysics Letters, 65(4), 581-586.
"""

import logging
import time
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer

log = logging.getLogger(__name__)


def discretize_space_3d(
    df: pd.DataFrame,
    cell_size_km: float,
    target_crs: str = "epsg:5070",
    info=True
) -> pd.DataFrame:
    """
    Project geographic coordinates to metric space and assign each earthquake
    to a cubic grid cell.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog with columns ``latitude``, ``longitude``,
        ``depth_km``.
    cell_size_km : float
        Edge length of each cubic cell in kilometres.
    target_crs : str
        Projected CRS for metric conversion. Default ``"epsg:5070"`` (NAD83 /
        CONUS Albers Equal Area) is correct for the US catalog. Use
        ``"epsg:32632"`` (UTM Zone 32N) for the Italy catalog.

    Returns
    -------
    pd.DataFrame
        New DataFrame (original unchanged) with added columns ``cell_x``,
        ``cell_y``, ``cell_z``, ``cell_id`` (string key ``"cx_cy_cz"``),
        and ``x_km`` / ``y_km`` (projected metric coordinates, unshifted).

    Notes
    -----
    Negative depths (surface-drift artefacts) are kept; they map to
    ``cell_z = -1``.
    Horizontal origin is shifted so that (cell_x, cell_y) ≥ 0 everywhere;
    depth is *not* shifted so that cell_z preserves physical meaning.
    """
    if info == True:  log.info("Projecting to %s, cell size %d km ...", target_crs, cell_size_km)

    transformer = Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
    x_m, y_m = transformer.transform(df["longitude"].values, df["latitude"].values)

    x_km = x_m / 1000.0
    y_km = y_m / 1000.0
    z_km = df["depth_km"].values

    x_shifted = x_km - x_km.min()
    y_shifted = y_km - y_km.min()

    cx = pd.Series(np.floor(x_shifted / cell_size_km).astype(int), index=df.index)
    cy = pd.Series(np.floor(y_shifted / cell_size_km).astype(int), index=df.index)
    cz = pd.Series(np.floor(z_km      / cell_size_km).astype(int), index=df.index)

    return df.assign(
        x_km=pd.Series(x_km, index=df.index),
        y_km=pd.Series(y_km, index=df.index),
        cell_x=cx,
        cell_y=cy,
        cell_z=cz,
        cell_id=cx.astype(str) + "_" + cy.astype(str) + "_" + cz.astype(str),
    )


def build_abe_suzuki_network(
    df: pd.DataFrame,
    cell_size_km: float,
    target_crs: str = "epsg:5070",
    info = True
) -> nx.DiGraph:
    """
    Build the Abe-Suzuki directed network from a chronologically sorted
    earthquake catalog.

    Nodes are 3-D spatial cells; a directed edge u → v means that at least
    one earthquake in cell u was immediately followed by one in cell v.
    Edge attribute ``weight`` counts the number of such transitions.
    Node attributes ``lat`` and ``lon`` store the true geometric centre of
    each cell (inverse-projected from the cell's midpoint in metric space).

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog **sorted by time**, with columns ``latitude``,
        ``longitude``, ``depth_km``.
    cell_size_km : float
        Edge length of each cubic cell in kilometres.
    target_crs : str
        Projected CRS passed through to ``discretize_space_3d``. Use
        ``"epsg:5070"`` for US data (default) or ``"epsg:32632"`` for Italy.

    Returns
    -------
    nx.DiGraph
        Weighted directed graph.

    Notes
    -----
    By construction, every interior node has equal weighted in-degree and
    out-degree. Only the first and last nodes in the time series will differ
    by exactly 1.
    """
    t0 = time.time()

    df_grid = discretize_space_3d(df, cell_size_km, target_crs=target_crs, info=info)
    seq = df_grid["cell_id"].tolist()

    G: nx.DiGraph = nx.DiGraph()
    G.add_nodes_from(set(seq))

    edge_counts: Counter = Counter(zip(seq[:-1], seq[1:]))
    G.add_weighted_edges_from((u, v, w) for (u, v), w in edge_counts.items())

    # True geometric centre of each cell: (cx + 0.5) * cell_size + origin,
    # inverse-projected back to WGS-84. This is exact regardless of how
    # many earthquakes fell in the cell (unlike the previous mean of events).
    inv = Transformer.from_crs(target_crs, "epsg:4326", always_xy=True)
    x_origin = df_grid["x_km"].min()
    y_origin = df_grid["y_km"].min()
    cell_info = (
        df_grid[["cell_id", "cell_x", "cell_y"]]
        .drop_duplicates("cell_id")
        .set_index("cell_id")
    )
    for node in G.nodes():
        if node not in cell_info.index:
            continue
        cx = cell_info.at[node, "cell_x"]
        cy = cell_info.at[node, "cell_y"]
        x_m = ((cx + 0.5) * cell_size_km + x_origin) * 1000.0
        y_m = ((cy + 0.5) * cell_size_km + y_origin) * 1000.0
        lon, lat = inv.transform(x_m, y_m)
        G.nodes[node]["lat"] = float(lat)
        G.nodes[node]["lon"] = float(lon)

    if info == True:  log.info(
        "Network (%d km, %s): %d nodes, %d edges, %d self-loops — %.1fs",
        cell_size_km,
        target_crs,
        G.number_of_nodes(),
        G.number_of_edges(),
        nx.number_of_selfloops(G),
        time.time() - t0,
    )
    return G
