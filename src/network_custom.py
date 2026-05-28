"""
Custom variants of the Abe-Suzuki earthquake network.

Both builders modify the original construction in two ways requested by the
project brief:

1. **Link criterion** — a threshold on the spatial and/or temporal distance
   between an earthquake and the next one (whether a link is created at all).
2. **Edge weights** — magnitude is folded into the weight so that links
   between strong events stand out, instead of the weight being a pure
   transition count.

Two strategies are provided:

* :func:`build_abe_suzuki_network_custom` — *soft* model.  No edge is ever
  deleted; instead each transition is down-weighted by smooth exponential
  decays in time and space, and up-weighted by the Gutenberg-Richter energy
  proxy of the pair.  Avoids the arbitrary discontinuities of a hard cut and
  preserves physically real long-range triggering.

      w_ij ∝ Σ 10^(α·(M_i+M_j)/2) · exp(-Δt/τ) · exp(-Δr/r₀)

* :func:`build_abe_suzuki_network_custom_hard` — *hard* model.  A consecutive
  pair is linked only if Δt ≤ ``time_threshold_sec`` **and** great-circle
  Δr ≤ ``spatial_threshold_km``; surviving links keep the original
  transition-count weight.  Simple and interpretable, at the cost of
  sensitivity to the threshold choice.

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


def _assign_node_coords(
    G: nx.DiGraph,
    df_grid: pd.DataFrame,
    cell_size_km: float,
    target_crs: str,
) -> None:
    """
    Attach ``lat``/``lon`` attributes to every node, in place.

    Each node id is a ``"cx_cy_cz"`` cell key; the cell centre is inverse-
    projected from ``target_crs`` back to EPSG:4326 (lon/lat).  Mirrors the
    centroid logic in :func:`src.network.build_abe_suzuki_network`.
    """
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


def build_abe_suzuki_network_custom(
    df: pd.DataFrame,
    cell_size_km: float,
    target_crs: str = "epsg:5070",
    alpha: float = 0.7,
    tau: float = 86400.0,
    r0: float = 10.0,
    info: bool = True,
) -> nx.DiGraph:
    """
    Soft (decay-weighted, magnitude-aware) Abe-Suzuki network.

    No link is removed.  Each consecutive transition ``i -> i+1`` contributes a
    weight that decays smoothly with the time and space separation of the two
    events and grows with their Gutenberg-Richter energy:

    .. math::

        w_{ij} \\propto \\sum 10^{\\alpha (M_i + M_j)/2}\\;
                          e^{-\\Delta t / \\tau}\\; e^{-\\Delta r / r_0}

    Parallel transitions between the same ordered cell pair are summed.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog sorted by ``time`` with columns ``time``, ``latitude``,
        ``longitude``, ``depth_km``, ``magnitude``.
    cell_size_km : float
        Cubic cell edge length in km.
    target_crs : str
        Projected CRS for the metric grid (``"epsg:32632"`` for Italy).
    alpha : float
        Magnitude scaling exponent; larger values amplify strong-event links.
    tau : float
        Temporal decay scale in seconds (e.g. ``86400`` ≈ 1 day).
    r0 : float
        Spatial decay scale in km.
    info : bool
        If True, log node/edge counts and elapsed time.

    Returns
    -------
    nx.DiGraph
        Directed weighted network; nodes carry ``lat``/``lon`` attributes.
        Self-loops (consecutive events in the same cell) are kept.
    """
    t0 = time.time()
    df_grid = discretize_space_3d(df, cell_size_km, target_crs=target_crs, info=info)

    seq = df_grid["cell_id"].to_numpy()
    times = df_grid["time"].to_numpy(dtype="datetime64[ns]")  # tz-aware → UTC naive
    mags = df_grid["magnitude"].to_numpy(dtype=float)
    x = df_grid["x_km"].to_numpy(dtype=float)
    y = df_grid["y_km"].to_numpy(dtype=float)

    # Vectorised separations between consecutive events.
    dt = (times[1:] - times[:-1]) / np.timedelta64(1, "s")
    dr = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])

    mag_weight = 10.0 ** (alpha * (mags[:-1] + mags[1:]) / 2.0)
    w = mag_weight * np.exp(-dt / tau) * np.exp(-dr / r0)

    edges = (
        pd.DataFrame({"u": seq[:-1], "v": seq[1:], "w": w})
        .groupby(["u", "v"], sort=False)["w"]
        .sum()
        .reset_index()
    )

    G = nx.DiGraph()
    G.add_nodes_from(set(seq))
    G.add_weighted_edges_from(edges.itertuples(index=False, name=None))

    _assign_node_coords(G, df_grid, cell_size_km, target_crs)

    if info:
        log.info(
            "Custom soft network (%d km): %d nodes, %d edges, %.1fs",
            cell_size_km,
            G.number_of_nodes(),
            G.number_of_edges(),
            time.time() - t0,
        )
    return G


def build_abe_suzuki_network_custom_hard(
    df: pd.DataFrame,
    cell_size_km: float,
    spatial_threshold_km: float = 50.0,
    time_threshold_sec: float = 24 * 3600,
    target_crs: str = "epsg:5070",
    info: bool = True,
) -> nx.DiGraph:
    """
    Hard-threshold Abe-Suzuki network.

    A consecutive pair ``i -> i+1`` is linked only when **both** the temporal
    gap and the great-circle distance fall within the thresholds:

        Δt ≤ ``time_threshold_sec``  and  Δr ≤ ``spatial_threshold_km``

    Surviving links keep the original Abe-Suzuki transition-count weight
    (number of times that ordered cell pair occurs as a kept transition).

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with ``time``, ``latitude``, ``longitude``, ``depth_km``.
        Re-sorted by ``time`` internally.
    cell_size_km : float
        Cubic cell edge length in km.
    spatial_threshold_km : float
        Maximum great-circle separation for a link.
    time_threshold_sec : float
        Maximum inter-event time for a link, in seconds.
    target_crs : str
        Projected CRS for cell centroids.
    info : bool
        If True, log node/edge/self-loop counts and elapsed time.

    Returns
    -------
    nx.DiGraph
        Directed weighted network; nodes carry ``lat``/``lon`` attributes.
        Self-loops are kept.

    Notes
    -----
    A hard cut introduces an arbitrary discontinuity and can sever physically
    real long-range triggering; results are sensitive to the threshold values.
    See :func:`build_abe_suzuki_network_custom` for a smooth alternative.
    """
    t0 = time.time()
    df = df.sort_values("time").reset_index(drop=True)
    df_grid = discretize_space_3d(df, cell_size_km, target_crs=target_crs, info=info)

    seq = df_grid["cell_id"].to_numpy()
    times = df_grid["time"].to_numpy(dtype="datetime64[ns]")  # tz-aware → UTC naive
    lat = df_grid["latitude"].to_numpy(dtype=float)
    lon = df_grid["longitude"].to_numpy(dtype=float)

    dt = (times[1:] - times[:-1]) / np.timedelta64(1, "s")
    dist = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])

    keep = (dt <= time_threshold_sec) & (dist <= spatial_threshold_km)

    edges = (
        pd.DataFrame({"u": seq[:-1][keep], "v": seq[1:][keep]})
        .groupby(["u", "v"], sort=False)
        .size()
        .reset_index(name="w")
    )

    G = nx.DiGraph()
    G.add_nodes_from(set(seq))
    G.add_weighted_edges_from(edges.itertuples(index=False, name=None))

    _assign_node_coords(G, df_grid, cell_size_km, target_crs)

    if info:
        log.info(
            "Custom hard network (%d km, Δr≤%g km, Δt≤%g s): "
            "%d nodes, %d edges, %d self-loops, %.1fs",
            cell_size_km,
            spatial_threshold_km,
            time_threshold_sec,
            G.number_of_nodes(),
            G.number_of_edges(),
            nx.number_of_selfloops(G),
            time.time() - t0,
        )
    return G
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

    # Discretize space
    df_grid = discretize_space_3d(df, cell_size_km, target_crs=target_crs, info=info)

    # Extract sequences
    seq = df_grid["cell_id"].values
    times = df_grid["time"].values
    mags = df_grid["magnitude"].values

    # Coordinates (projected, km)
    x = df_grid["x_km"].values
    y = df_grid["y_km"].values

    # Lat/lon (for hard filtering)
    lat = df_grid["latitude"].values
    lon = df_grid["longitude"].values

    # Graph
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

        # Skip self-loops (consecutive events in same cell are burst artifacts,
        # not inter-cell transitions; they otherwise dominate the strength
        # distribution with astronomical accumulated weights).
        if u == v:
            continue

        # --- SOFT WEIGHTS (same as soft version) ---

        # magnitude (pair-based)
        mag_weight = 10 ** (alpha * (mags[i] + mags[i+1]) / 2)

        # temporal decay
        time_weight = np.exp(-dt / tau)

        # spatial decay (use projected coords for consistency with soft version)
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        dr = np.sqrt(dx**2 + dy**2)

        space_weight = np.exp(-dr / r0)

        w = mag_weight * time_weight * space_weight

        # accumulate
        if (u, v) in edge_weights:
            edge_weights[(u, v)] += w
        else:
            edge_weights[(u, v)] = w

    # Add edges
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