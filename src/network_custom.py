"""
Custom (hybrid) variant of the Abe-Suzuki earthquake network.

:func:`build_abe_suzuki_network_custom_hybrid` modifies the original
construction in two ways requested by the project brief:

1. **Link criterion** – a consecutive pair is linked only if Δt ≤
   ``time_threshold_sec`` **and** great-circle Δr ≤ ``spatial_threshold_km``,
   which removes spurious far/late transitions.
2. **Edge weights** – surviving links are weighted by smooth exponential
   decays in time and space and by the Gutenberg-Richter energy proxy of the
   pair, so links between strong, close, prompt events stand out instead of
   the weight being a pure transition count:

      w_ij ∝ Σ 10^(α·(M_i+M_j)/2) · exp(-Δt/τ) · exp(-Δr/r₀)

References
----------
Abe, S., & Suzuki, N. (2004). Scale-free network of earthquakes.
Europhysics Letters, 65(4), 581-586.

Gutenberg, B., & Richter, C. F. (1944). Frequency of earthquakes in
California. Bulletin of the Seismological Society of America, 34(4), 185-188.
"""

import logging
import time

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer

# Re-export so notebooks can do `from src.network_custom import discretize_space_3d`.
from src.network import discretize_space_3d  # noqa: F401

log = logging.getLogger(__name__)


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """
    Great-circle distance between two points (or two coordinate arrays).

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : array_like or float
        Latitudes and longitudes in decimal degrees.  Broadcasting follows
        NumPy rules, so passing equal-length arrays returns the element-wise
        distance between paired points.

    Returns
    -------
    np.ndarray or float
        Distance(s) in kilometres (Earth radius 6371 km).
    """
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlambda = np.radians(np.asarray(lon2) - np.asarray(lon1))

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_abe_suzuki_network_custom_hybrid(
    df: pd.DataFrame,
    cell_size_km: float,
    spatial_threshold_km: float = 100.0,   # HARD filter (looser than pure hard version)
    time_threshold_sec: float = 48 * 3600, # HARD filter
    target_crs: str = "epsg:5070",
    alpha: float = 0.7,        # magnitude scaling
    tau: float = 86400.0,      # temporal decay (seconds)
    r0: float = 10.0,          # spatial decay (km)
    info: bool = True
) -> nx.DiGraph:

    t0 = time.time()

    df_grid = discretize_space_3d(df, cell_size_km, target_crs=target_crs, info=info)

    seq = df_grid["cell_id"].values
    times = df_grid["time"].values
    mags = df_grid["magnitude"].values

    x = df_grid["x_km"].values
    y = df_grid["y_km"].values

    lat = df_grid["latitude"].values
    lon = df_grid["longitude"].values

    G = nx.DiGraph()
    G.add_nodes_from(set(seq))

    edge_weights = {}

    for i in range(len(df_grid) - 1):

        # --- TIME DIFFERENCE ---
        dt = (times[i + 1] - times[i]) / np.timedelta64(1, 's')

        if dt > time_threshold_sec:
            continue

        # --- SPACE DIFFERENCE (HARD FILTER via haversine) ---
        dist = haversine_km(lat[i], lon[i], lat[i+1], lon[i+1])

        if dist > spatial_threshold_km:
            continue

        # --- EDGE ---
        u = seq[i]
        v = seq[i + 1]

        # Self-loops (consecutive events in the same cell) are kept: they capture
        # same-cell recurrence / aftershock bursts, which are part of the seismic
        # transition dynamics.

        # --- SOFT WEIGHTS (same as soft version) ---
        mag_weight = 10 ** (alpha * (mags[i] + mags[i+1]) / 2)
        time_weight = np.exp(-dt / tau)

        # spatial decay (use projected coords for consistency with soft version)
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        dr = np.sqrt(dx**2 + dy**2)

        space_weight = np.exp(-dr / r0)

        w = mag_weight * time_weight * space_weight

        if (u, v) in edge_weights:
            edge_weights[(u, v)] += w
        else:
            edge_weights[(u, v)] = w

    G.add_weighted_edges_from((u, v, w) for (u, v), w in edge_weights.items())

    # --- Node positions  ---
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

        lon_, lat_ = inv.transform(x_m, y_m)
        G.nodes[node]["lat"] = float(lat_)
        G.nodes[node]["lon"] = float(lon_)

    if info:
        print(
            f"Hybrid Abe-Suzuki: "
            f"{G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges, "
            f"time {time.time() - t0:.2f}s"
        )

    return G