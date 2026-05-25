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


# ============================================================================    


def build_abe_suzuki_network_custom(
    df: pd.DataFrame,
    cell_size_km: float,
    target_crs: str = "epsg:5070",
    alpha: float = 0.7,        # magnitude scaling
    tau: float = 86400.0,      # temporal scale (seconds) ~ 1 day
    r0: float = 10.0,          # spatial scale (km)
    info: bool = True
) -> nx.DiGraph:
    """
    Custom Abe-Suzuki network with weighted edges:

    w_ij ∝ Σ 10^(alpha * M_i) * exp(-Δt / tau) * exp(-Δr / r0)

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by time and include:
        time, latitude, longitude, depth_km, magnitude
    alpha : float
        Controls importance of magnitude
    tau : float
        Temporal decay scale (seconds)
    r0 : float
        Spatial decay scale (km)
    """

    t0 = time.time()

    # Discretize space
    df_grid = discretize_space_3d(df, cell_size_km, target_crs=target_crs, info=info)

    # Extract sequences
    seq = df_grid["cell_id"].values
    times = df_grid["time"].values
    mags = df_grid["magnitude"].values

    # Use projected coordinates (already in km)
    x = df_grid["x_km"].values
    y = df_grid["y_km"].values

    G = nx.DiGraph()
    G.add_nodes_from(set(seq))

    edge_weights = {}

    # Loop over consecutive events
    for i in range(len(df_grid) - 1):
        u = seq[i]
        v = seq[i + 1]

        # Δt in seconds
        dt = (times[i + 1] - times[i]) / np.timedelta64(1, 's')

        # Δr in km (Euclidean in projected space)
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        dr = np.sqrt(dx**2 + dy**2)

        # Weight components
        # mag_weight = 10 ** (alpha * mags[i])                        # source-based version
        mag_weight = 10 ** (alpha * (mags[i] + mags[i+1]) / 2)      # pair-based version (so both source and target earthquakes)
        time_weight = np.exp(-dt / tau)
        space_weight = np.exp(-dr / r0)

        w = mag_weight * time_weight * space_weight

        if (u, v) in edge_weights:
            edge_weights[(u, v)] += w
        else:
            edge_weights[(u, v)] = w

    # Add edges
    G.add_weighted_edges_from((u, v, w) for (u, v), w in edge_weights.items())

    # --- Node positions (same as before) ---
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

    if info:
        log.info(
            "Custom Network (%d km): %d nodes, %d edges, %.1fs",
            cell_size_km,
            G.number_of_nodes(),
            G.number_of_edges(),
            time.time() - t0,
        )

    return G





    # ==============================================================================00000

def haversine_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_abe_suzuki_network_custom_hard(
    df: pd.DataFrame,
    cell_size_km: float,
    spatial_threshold_km: float = 50.0,
    time_threshold_sec: float = 24 * 3600,  # 1 day default
    target_crs: str = "epsg:5070",
    info: bool = True
) -> nx.DiGraph:

    t0 = time.time()

    # ensure chronological order
    df = df.sort_values("time").reset_index(drop=True)

    # spatial discretization
    df_grid = discretize_space_3d(df, cell_size_km, target_crs=target_crs, info=info)

    seq = df_grid["cell_id"].tolist()

    # build graph
    G = nx.DiGraph()
    G.add_nodes_from(set(seq))

    edge_counts = Counter()

    # iterate consecutive events with filtering
    for i in range(len(df_grid) - 1):

        t1 = pd.to_datetime(df_grid.loc[i, "time"])
        t2 = pd.to_datetime(df_grid.loc[i + 1, "time"])

        dt = (t2 - t1).total_seconds()

        if dt > time_threshold_sec:
            continue

        lat1, lon1 = df_grid.loc[i, "latitude"], df_grid.loc[i, "longitude"]
        lat2, lon2 = df_grid.loc[i + 1, "latitude"], df_grid.loc[i + 1, "longitude"]

        dist = haversine_km(lat1, lon1, lat2, lon2)

        if dist > spatial_threshold_km:
            continue

        u = seq[i]
        v = seq[i + 1]

        edge_counts[(u, v)] += 1

    G.add_weighted_edges_from((u, v, w) for (u, v), w in edge_counts.items())

    # node centroids (unchanged from original)
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

    if info:
        print(
            f"Custom Abe-Suzuki (hard thresholds): "
            f"{G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges, "
            f"{nx.number_of_selfloops(G)} self-loops, "
            f"time {time.time() - t0:.2f}s"
        )

    return G