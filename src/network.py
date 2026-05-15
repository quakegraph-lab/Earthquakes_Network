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
from itertools import permutations

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer

log = logging.getLogger(__name__)


def discretize_space_3d(
    df: pd.DataFrame,
    cell_size_km: float,
    target_crs: str = "epsg:5070",
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
    log.info("Projecting to %s, cell size %d km ...", target_crs, cell_size_km)

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

    df_grid = discretize_space_3d(df, cell_size_km, target_crs=target_crs)
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

    log.info(
        "Network (%d km, %s): %d nodes, %d edges, %d self-loops — %.1fs",
        cell_size_km,
        target_crs,
        G.number_of_nodes(),
        G.number_of_edges(),
        nx.number_of_selfloops(G),
        time.time() - t0,
    )
    return G


def build_baiesi_paczuski_network(
    df: pd.DataFrame,
    b: float = 1.0,
    alpha: float = 1.0,
    d: float = 1.6,
    t_max_days: float = 730.0,
    return_nn_distances: bool = False,
) -> "nx.DiGraph | tuple[nx.DiGraph, np.ndarray]":
    """
    Build the Baiesi-Paczuski directed earthquake network.

    Each earthquake j is linked to a single parent i — the prior event that
    minimises the metric

        n_ij = t_ij^α × r_ij^d × 10^(−b × m_i)

    where t_ij is the time difference (seconds), r_ij is the haversine
    epicentral distance (km), and m_i is the parent magnitude.  The result
    is a directed forest: every non-root node has exactly one in-edge.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog **sorted by time**, with columns ``time``,
        ``latitude``, ``longitude``, ``depth_km``, ``magnitude``.
    b : float
        Gutenberg-Richter b-value. Default 1.0.
    alpha : float
        Omori time exponent. Default 1.0.
    d : float
        Fractal dimension of epicentre distribution. Default 1.6.
    t_max_days : float
        Look-back window in days. Events outside this window are not
        considered as candidate parents. Default 730 (2 years).
    return_nn_distances : bool
        If True, also return a length-N array of log10(nearest-neighbour
        distance) for each event (NaN where no candidate existed).  Used by
        :func:`build_zaliapin_ben_zion_network` to avoid recomputing the
        metric.  Default False.

    Returns
    -------
    nx.DiGraph
        Directed forest. Nodes are integer indices (0 … N-1) matching the
        DataFrame row order after time-sorting.  Node attributes: ``lat``,
        ``lon``, ``depth_km``, ``magnitude``, ``mean_magnitude`` (alias),
        ``mean_depth`` (alias).  Edge attributes: ``weight`` = n_ij (lower =
        stronger link), ``log_n`` = log10(n_ij).
    np.ndarray, optional
        Only returned when ``return_nn_distances=True``.  Length-N array of
        log10(n_{i*j}) — the nearest-neighbour distance in the BP metric
        space for each event j.

    References
    ----------
    Baiesi, M., & Paczuski, M. (2004). Scale-free networks of earthquakes
    and aftershocks. Physical Review E, 69(6), 066106.
    """
    t0 = time.time()
    EPS = 1e-10
    t_max_s = t_max_days * 86400.0
    R_EARTH = 6371.0  # km

    df_s = df.sort_values("time").reset_index(drop=True)
    N = len(df_s)
    log.info("Building BP network: %d events, t_max=%.0f days ...", N, t_max_days)

    # Unix seconds — total_seconds() works for any tz-aware precision (us or ns)
    _epoch  = pd.Timestamp("1970-01-01", tz="UTC")
    times_s = (pd.to_datetime(df_s["time"], utc=True) - _epoch).dt.total_seconds().values
    lats    = df_s["latitude"].values
    lons    = df_s["longitude"].values
    mags    = df_s["magnitude"].values
    depths  = df_s["depth_km"].values

    lats_r = np.radians(lats)
    lons_r = np.radians(lons)

    G: nx.DiGraph = nx.DiGraph()
    for i in range(N):
        G.add_node(
            i,
            lat=float(lats[i]),
            lon=float(lons[i]),
            depth_km=float(depths[i]),
            magnitude=float(mags[i]),
            mean_magnitude=float(mags[i]),  # alias for compute_assortativity
            mean_depth=float(depths[i]),    # alias for compute_assortativity
        )

    edges: list[tuple[int, int, dict]] = []
    log_interval = max(N // 20, 1)
    log_eta_nn = np.full(N, np.nan)

    for j in range(1, N):
        if j % log_interval == 0:
            log.info("  %.0f%%  (%d/%d events linked)", 100.0 * j / N, j, N)

        t_j  = times_s[j]
        left = int(np.searchsorted(times_s[:j], t_j - t_max_s, side="left"))
        if left >= j:
            continue  # no candidates → root

        cands = np.arange(left, j)

        dt   = t_j - times_s[cands]                   # seconds, always > 0

        dlat = lats_r[cands] - lats_r[j]
        dlon = lons_r[cands] - lons_r[j]
        a_h  = (np.sin(dlat / 2) ** 2
                + np.cos(lats_r[j]) * np.cos(lats_r[cands]) * np.sin(dlon / 2) ** 2)
        r_km = R_EARTH * 2.0 * np.arcsin(np.sqrt(np.clip(a_h, 0.0, 1.0)))

        n_ij   = (dt + EPS) ** alpha * (r_km + EPS) ** d * 10.0 ** (-b * mags[cands])
        best   = int(np.argmin(n_ij))
        parent = int(cands[best])
        w      = float(n_ij[best])
        log_eta_nn[j] = float(np.log10(w + EPS))
        edges.append((parent, j, {"weight": w, "log_n": log_eta_nn[j]}))

    G.add_edges_from(edges)

    n_roots = sum(1 for _, deg in G.in_degree() if deg == 0)
    n_trees = nx.number_weakly_connected_components(G)
    log.info(
        "BP network: %d nodes, %d edges, %d roots, %d trees — %.1fs",
        G.number_of_nodes(), G.number_of_edges(), n_roots, n_trees, time.time() - t0,
    )
    if return_nn_distances:
        return G, log_eta_nn
    return G


def build_telesca_lovallo_network(
    df: pd.DataFrame,
    max_look_ahead: int = 10_000,
) -> nx.Graph:
    """
    Build the Telesca-Lovallo Natural Visibility Graph (NVG) from a seismic catalog.

    Two events i and j (i occurring before j) are connected iff the straight
    line joining their (time, magnitude) coordinates lies strictly above all
    intermediate data points, i.e. for every k with i < k < j:

        m_k < m_i + (m_j − m_i) · (t_k − t_i) / (t_j − t_i)

    Equivalently, j is visible from i when

        slope(i, j) = (m_j − m_i) / (t_j − t_i) >  max_{i < k < j} slope(i, k)

    This angular-sweep reformulation allows an O(n) right-side scan per node
    with a running maximum — no nested loop is needed.

    The result is an **undirected, always-connected** graph: every adjacent pair
    (i, i+1) is trivially visible (no intermediate events), so the path
    i — i+1 — … — N-1 is always present.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog **sorted by time**, with columns ``time``,
        ``latitude``, ``longitude``, ``depth_km``, ``magnitude``.
    max_look_ahead : int
        Safety cap on how far ahead each event scans for visible neighbours.
        The natural slope-based termination makes this rarely binding for
        typical seismicity. Default 10 000.

    Returns
    -------
    nx.Graph
        Undirected visibility graph. Nodes are integer indices (0 … N-1)
        matching the DataFrame row order after time-sorting.  Node attributes:
        ``lat``, ``lon``, ``depth_km``, ``magnitude``, ``mean_magnitude``
        (alias), ``mean_depth`` (alias).  All edges have ``weight=1``.

    References
    ----------
    Luque, B., Lacasa, L., Ballesteros, F., & Luque, J. (2009). Horizontal
    visibility graphs: Exact results for random time series.
    Physical Review E, 80(4), 046103.

    Telesca, L., & Lovallo, M. (2012). Analysis of seismic sequences by using
    the method of visibility graphs. EPL, 97(5), 50002.
    """
    t0  = time.time()
    EPS = 1e-10

    df_s = df.sort_values("time").reset_index(drop=True)
    N    = len(df_s)
    log.info("Building TL visibility graph: %d events ...", N)

    _epoch  = pd.Timestamp("1970-01-01", tz="UTC")
    times_s = (pd.to_datetime(df_s["time"], utc=True) - _epoch).dt.total_seconds().values
    mags    = df_s["magnitude"].values
    lats    = df_s["latitude"].values
    lons    = df_s["longitude"].values
    depths  = df_s["depth_km"].values

    G: nx.Graph = nx.Graph()
    for i in range(N):
        G.add_node(
            i,
            lat=float(lats[i]),
            lon=float(lons[i]),
            depth_km=float(depths[i]),
            magnitude=float(mags[i]),
            mean_magnitude=float(mags[i]),
            mean_depth=float(depths[i]),
        )

    log_interval = max(N // 20, 1)
    edges: list[tuple[int, int]] = []

    for i in range(N - 1):
        if i % log_interval == 0:
            log.info("  %.0f%%  (%d/%d events scanned)", 100.0 * i / N, i, N)

        limit  = min(N, i + max_look_ahead)
        slopes = (mags[i + 1:limit] - mags[i]) / (times_s[i + 1:limit] - times_s[i] + EPS)

        run_max = np.empty(len(slopes))
        run_max[0] = -np.inf
        if len(slopes) > 1:
            run_max[1:] = np.maximum.accumulate(slopes[:-1])

        for k in np.where(slopes > run_max)[0]:
            edges.append((i, i + 1 + k))

    G.add_edges_from(edges, weight=1)

    avg_deg = 2 * G.number_of_edges() / max(G.number_of_nodes(), 1)
    log.info(
        "TL network: %d nodes, %d edges, avg degree %.2f — %.1fs",
        G.number_of_nodes(), G.number_of_edges(), avg_deg, time.time() - t0,
    )
    return G


def build_zaliapin_ben_zion_network(
    G_bp: nx.DiGraph,
    log_eta_nn: np.ndarray,
    eta_0: float | None = None,
) -> tuple[nx.DiGraph, float]:
    """
    Build the Zaliapin-Ben-Zion (ZBZ) network from pre-computed BP data.

    Takes the BP graph and its nearest-neighbour distances (returned by
    ``build_baiesi_paczuski_network(..., return_nn_distances=True)``) and
    applies a threshold eta_0 in log10 space to separate aftershock links
    (log10(n_ij) < eta_0) from background events (no parent assigned).

    The threshold is detected automatically via a 2-component Gaussian
    Mixture Model fitted to the bimodal distribution of log10(n_ij*): the
    left mode contains aftershock pairs (small t, small r, large m_parent),
    the right mode contains background pairs.  eta_0 is placed at the trough
    between the two modes.

    Parameters
    ----------
    G_bp : nx.DiGraph
        Output of ``build_baiesi_paczuski_network``.
    log_eta_nn : np.ndarray
        Length-N array of log10(nearest-neighbour distance) per event,
        as returned by ``build_baiesi_paczuski_network(...,
        return_nn_distances=True)``.  NaN for events with no candidate.
    eta_0 : float or None
        Manual log10 threshold.  If None, auto-detected via GMM trough.

    Returns
    -------
    G_zbz : nx.DiGraph
        Sparser directed forest.  Background events are roots (in-degree 0);
        aftershock events keep exactly one parent edge from G_bp.
        Node attributes copied from G_bp.  Edge attributes: ``weight``
        (n_ij), ``log_n`` (log10 n_ij).
    eta_0 : float
        The threshold used (auto or manual).

    References
    ----------
    Zaliapin, I., Gabrielov, A., Keilis-Borok, V., & Wong, H. (2008).
    Clustering analysis of seismicity and aftershock identification.
    Physical Review Letters, 101(1), 018501.
    """
    from sklearn.mixture import GaussianMixture

    t0 = time.time()
    valid_mask = ~np.isnan(log_eta_nn)
    log_eta_valid = log_eta_nn[valid_mask]

    if eta_0 is None:
        gmm = GaussianMixture(n_components=2, random_state=42, max_iter=300)
        gmm.fit(log_eta_valid.reshape(-1, 1))
        mu1, mu2 = sorted(gmm.means_.flatten())
        # Find trough (minimum density) between the two modes
        x_grid = np.linspace(mu1, mu2, 500)
        log_probs = gmm.score_samples(x_grid.reshape(-1, 1))
        eta_0 = float(x_grid[np.argmin(log_probs)])
        log.info("ZBZ GMM: mu1=%.2f  mu2=%.2f  trough eta_0=%.2f", mu1, mu2, eta_0)

    G_zbz: nx.DiGraph = nx.DiGraph()
    G_zbz.add_nodes_from(G_bp.nodes(data=True))

    n_kept = 0
    for u, v, data in G_bp.edges(data=True):
        if log_eta_nn[v] < eta_0:
            G_zbz.add_edge(u, v, **data)
            n_kept += 1

    n_bg    = sum(1 for _, d in G_zbz.in_degree() if d == 0)
    n_trees = nx.number_weakly_connected_components(G_zbz)
    log.info(
        "ZBZ (eta_0=%.2f): %d edges kept, %d background roots, %d trees — %.1fs",
        eta_0, n_kept, n_bg, n_trees, time.time() - t0,
    )
    return G_zbz, eta_0


def build_hvg_network(
    df: pd.DataFrame,
    max_look_ahead: int = 10_000,
) -> nx.Graph:
    """
    Build the Horizontal Visibility Graph (HVG) from the magnitude time series.

    Two events i and j (i < j) are connected iff every intermediate event k
    (i < k < j) satisfies m_k < min(m_i, m_j).  Equivalently, the horizontal
    bar at height min(m_i, m_j) is not obstructed by any intermediate event.

    Compared to the NVG (Telesca-Lovallo), the HVG uses a stricter condition
    that produces fewer long-range edges.  It admits an O(n) construction with
    true early termination: once the running maximum of intermediate magnitudes
    exceeds m_i, no further j can be visible from i.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog **sorted by time**, with columns ``time``,
        ``latitude``, ``longitude``, ``depth_km``, ``magnitude``.
    max_look_ahead : int
        Safety cap on the forward scan per event.  Natural termination
        typically fires long before this limit.  Default 10 000.

    Returns
    -------
    nx.Graph
        Undirected HVG.  Node attributes identical to TL network
        (``lat``, ``lon``, ``depth_km``, ``magnitude``, ``mean_magnitude``,
        ``mean_depth``).  All edges have ``weight=1``.

    References
    ----------
    Luque, B., Lacasa, L., Ballesteros, F., & Luque, J. (2009). Horizontal
    visibility graphs: Exact results for random time series.
    Physical Review E, 80(4), 046103.
    """
    t0 = time.time()

    df_s = df.sort_values("time").reset_index(drop=True)
    N    = len(df_s)
    log.info("Building HVG: %d events ...", N)

    mags   = df_s["magnitude"].values
    lats   = df_s["latitude"].values
    lons   = df_s["longitude"].values
    depths = df_s["depth_km"].values

    G: nx.Graph = nx.Graph()
    for i in range(N):
        G.add_node(
            i,
            lat=float(lats[i]),
            lon=float(lons[i]),
            depth_km=float(depths[i]),
            magnitude=float(mags[i]),
            mean_magnitude=float(mags[i]),
            mean_depth=float(depths[i]),
        )

    log_interval = max(N // 20, 1)
    edges: list[tuple[int, int]] = []

    for i in range(N - 1):
        if i % log_interval == 0:
            log.info("  %.0f%%  (%d/%d)", 100.0 * i / N, i, N)

        edges.append((i, i + 1))           # adjacent pair always visible
        limit       = min(N, i + max_look_ahead)
        running_max = mags[i + 1]          # max of intermediates i+1..j-1

        for j in range(i + 2, limit):
            if running_max >= mags[i]:     # no further j visible from i
                break
            if running_max < mags[j]:      # running_max < min(m_i, m_j)
                edges.append((i, j))
            running_max = max(running_max, mags[j])

    G.add_edges_from(edges, weight=1)

    avg_deg = 2 * G.number_of_edges() / max(N, 1)
    log.info(
        "HVG: %d nodes, %d edges, avg degree %.2f — %.1fs",
        N, G.number_of_edges(), avg_deg, time.time() - t0,
    )
    return G


def build_recurrence_network(
    df: pd.DataFrame,
    epsilon: float | None = None,
    target_degree: float = 20.0,
    feature_weights: dict | None = None,
    target_crs: str = "epsg:32632",
) -> nx.Graph:
    """
    Build a Recurrence Network (RN) from the earthquake catalog.

    Two events i and j are connected iff their Euclidean distance in a
    normalised (5-dimensional) feature space is below threshold ε:

        G = {(i,j) : ||x_i - x_j||_2 < ε}

    Feature vector: x_k = (t_k*, x_k*, y_k*, z_k*, m_k*) where each
    dimension is independently normalised to [0, 1] using the catalog
    min/max.  Optional per-feature weights scale the normalised features
    before distance computation.

    If ``epsilon`` is None (default), ε is auto-selected via binary search
    on ``scipy.spatial.cKDTree.count_neighbors`` to achieve
    ``target_degree`` average connections.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog **sorted by time**, with columns ``time``,
        ``latitude``, ``longitude``, ``depth_km``, ``magnitude``.
    epsilon : float or None
        Recurrence threshold in normalised feature space.  If None,
        auto-selected to match ``target_degree``.
    target_degree : float
        Target average degree ⟨k⟩ for auto ε selection.  Default 20.
    feature_weights : dict or None
        Optional multipliers for each feature dimension.  Keys:
        ``"time"``, ``"x"``, ``"y"``, ``"depth"``, ``"magnitude"``.
        Missing keys default to 1.0.  Higher weight → feature matters more
        in the distance calculation.
    target_crs : str
        Projected CRS for km conversion. Default ``"epsg:32632"`` (UTM 32N,
        Italy). Use ``"epsg:5070"`` for US, ``"epsg:32654"`` for Japan.

    Returns
    -------
    nx.Graph
        Undirected recurrence network.  Nodes are integer indices matching
        DataFrame row order.  Node attributes: ``lat``, ``lon``,
        ``depth_km``, ``magnitude``, ``mean_magnitude``, ``mean_depth``.
        All edges have ``weight=1``.

    References
    ----------
    Donner, R. V., Zou, Y., Donges, J. F., Marwan, N., & Kurths, J. (2010).
    Recurrence networks — a novel paradigm for nonlinear time series analysis.
    New Journal of Physics, 12, 033025.
    """
    from scipy.spatial import cKDTree

    t0  = time.time()
    EPS = 1e-10

    df_s = df.sort_values("time").reset_index(drop=True)
    N    = len(df_s)
    log.info("Building RN: %d events, target ⟨k⟩=%.1f ...", N, target_degree)

    # Time in days from catalog start
    _epoch    = pd.Timestamp("1970-01-01", tz="UTC")
    times_s   = (pd.to_datetime(df_s["time"], utc=True) - _epoch).dt.total_seconds().values
    times_day = (times_s - times_s.min()) / 86400.0

    # Project to km
    transformer = Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
    x_m, y_m   = transformer.transform(df_s["longitude"].values, df_s["latitude"].values)
    x_km = x_m / 1000.0
    y_km = y_m / 1000.0
    z_km = df_s["depth_km"].values
    mags = df_s["magnitude"].values
    lats = df_s["latitude"].values
    lons = df_s["longitude"].values

    def _norm(arr: np.ndarray) -> np.ndarray:
        r = arr.max() - arr.min()
        return (arr - arr.min()) / (r + EPS)

    X = np.column_stack([
        _norm(times_day),
        _norm(x_km),
        _norm(y_km),
        _norm(z_km),
        _norm(mags),
    ])

    # Apply optional feature weights
    if feature_weights is not None:
        _keys = ["time", "x", "y", "depth", "magnitude"]
        w = np.array([feature_weights.get(k, 1.0) for k in _keys], dtype=float)
        X *= w[np.newaxis, :]

    tree = cKDTree(X)

    # Auto-select epsilon: binary search using count_neighbors (avoids storing pairs)
    if epsilon is None:
        lo, hi = 0.0, 0.1
        # Expand hi until we can exceed target_degree
        for _ in range(25):
            cnt   = int(tree.count_neighbors(tree, hi))
            avg_k = (cnt - N) / N   # subtract N self-pairs; each unordered pair counted twice
            if avg_k >= target_degree:
                break
            hi *= 2.0
        # Refine with 35 bisection steps (< 1e-10 relative error)
        for _ in range(35):
            mid   = (lo + hi) / 2.0
            cnt   = int(tree.count_neighbors(tree, mid))
            avg_k = (cnt - N) / N
            if avg_k < target_degree:
                lo = mid
            else:
                hi = mid
        epsilon = (lo + hi) / 2.0
        log.info("RN auto-ε=%.6f for target ⟨k⟩=%.1f", epsilon, target_degree)

    # Build edge set (single call)
    pairs = tree.query_pairs(epsilon)
    log.info("RN pairs found: %d", len(pairs))

    G: nx.Graph = nx.Graph()
    for i in range(N):
        G.add_node(
            i,
            lat=float(lats[i]),
            lon=float(lons[i]),
            depth_km=float(z_km[i]),
            magnitude=float(mags[i]),
            mean_magnitude=float(mags[i]),
            mean_depth=float(z_km[i]),
        )
    G.add_edges_from(pairs, weight=1)

    avg_k = 2 * G.number_of_edges() / max(N, 1)
    log.info(
        "RN (ε=%.6f): %d nodes, %d edges, ⟨k⟩=%.2f — %.1fs",
        epsilon, N, G.number_of_edges(), avg_k, time.time() - t0,
    )
    return G


def build_ordinal_transition_network(
    df: pd.DataFrame,
    word_size: int = 4,
    lag: int = 1,
    magnitude_col: str = "magnitude",
) -> tuple["nx.DiGraph", np.ndarray]:
    """
    Build the Ordinal Transition Network (OTN) from the magnitude time series.

    Each consecutive window of ``word_size`` magnitudes is mapped to its
    ordinal rank permutation π ∈ S_{word_size}.  Consecutive patterns
    (offset by ``lag``) are connected by a directed edge weighted by the
    transition count.

    The result is a compact weighted digraph with at most word_size! nodes
    (e.g. 24 for word_size=4), suitable for entropy and flow analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog **sorted by time**, with column ``magnitude_col``.
    word_size : int
        Number of values per ordinal window (embedding dimension d).
        Default 4 → at most 4! = 24 patterns.
    lag : int
        Step between consecutive pattern windows (stride).  ``lag=1``
        produces maximally overlapping windows (standard for permutation
        entropy).  ``lag=word_size`` gives non-overlapping blocks.
    magnitude_col : str
        Column name for the magnitude series.  Default ``"magnitude"``.

    Returns
    -------
    G : nx.DiGraph
        Weighted directed transition graph.  Nodes are integer pattern IDs
        with attributes ``pattern_tuple`` (rank permutation as tuple),
        ``frequency`` (occurrence count), ``label`` (short string, e.g.
        ``"0213"``).  Edges carry ``weight`` = transition count.
    patterns_arr : np.ndarray
        Integer array of length ⌊(N - word_size) / lag⌋ + 1 giving the
        pattern ID at each window position (aligned to catalog order).

    References
    ----------
    Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity
    measure for time series. Physical Review Letters, 88(17), 174102.

    Pessa, A. A. B., & Ribeiro, H. V. (2019). Characterizing stochastic
    time series with ordinal networks. Physical Review E, 100(4), 042304.
    """
    t0 = time.time()

    df_s = df.sort_values("time").reset_index(drop=True)
    mags = df_s[magnitude_col].values.astype(float)
    N    = len(mags)
    log.info("Building OTN: %d events, word_size=%d, lag=%d ...", N, word_size, lag)

    # Canonical pattern ordering: all permutations of (0..word_size-1), lexicographic
    all_patterns = list(permutations(range(word_size)))
    pattern_to_id: dict[tuple, int] = {p: i for i, p in enumerate(all_patterns)}
    n_canonical = len(all_patterns)  # word_size!

    # Compute windows using advanced indexing (memory-efficient)
    start_idx  = np.arange(0, N - word_size + 1, lag)
    offset_idx = start_idx[:, None] + np.arange(word_size)   # shape (n_windows, word_size)
    windows    = mags[offset_idx]                              # shape (n_windows, word_size)

    # Ordinal patterns via row-wise argsort (stable to handle ties consistently)
    rank_matrix   = np.argsort(windows, axis=1, kind="stable")
    patterns_arr  = np.array([pattern_to_id[tuple(row)] for row in rank_matrix], dtype=np.int32)

    # Count frequencies and transitions
    freq    = np.bincount(patterns_arr, minlength=n_canonical)
    n_wins  = len(patterns_arr)

    # Build DiGraph
    G: nx.DiGraph = nx.DiGraph()
    for pid, ptup in enumerate(all_patterns):
        G.add_node(
            pid,
            pattern_tuple=ptup,
            frequency=int(freq[pid]),
            label="".join(str(x) for x in ptup),
        )

    # Count transitions between consecutive patterns
    if n_wins > 1:
        src = patterns_arr[:-1]
        dst = patterns_arr[1:]
        for s, d in zip(src, dst):
            if G.has_edge(int(s), int(d)):
                G[int(s)][int(d)]["weight"] += 1
            else:
                G.add_edge(int(s), int(d), weight=1)

    n_observed  = int(np.sum(freq > 0))
    n_forbidden = n_canonical - n_observed
    log.info(
        "OTN: %d/%d patterns observed (%d forbidden), %d transitions — %.1fs",
        n_observed, n_canonical, n_forbidden, n_wins - 1, time.time() - t0,
    )
    return G, patterns_arr


def build_etas_network(
    df: pd.DataFrame,
    K: float = 0.013,
    alpha_etas: float = 1.23,
    c_days: float = 0.05,
    p: float = 1.07,
    q: float = 1.5,
    d_km: float = 1.5,
    t_max_days: float = 730.0,
    mu_threshold: float | None = None,
    target_crs: str = "epsg:32632",
) -> nx.DiGraph:
    """
    Build an ETAS-derived directed network from the earthquake catalog.

    The Epidemic-Type Aftershock Sequence (ETAS) model assigns a
    triggering intensity from past event i to future event j:

        κ_ij = K · exp(α(m_i − m_min)) · (t_ij + c)^(−p) · (r_ij² + d²)^(−q)

    where t_ij is the time difference in days, r_ij is the epicentral
    distance in km, and m_min is the catalog minimum magnitude.

    Parent assignment (stochastic declustering):
    - For each j compute κ_ij for all i < j within t_max_days.
    - If max_i κ_ij > μ_threshold: parent = argmax; edge parent→j is added.
    - Otherwise: j is a background root (no parent).

    When ``mu_threshold`` is None the threshold is auto-estimated as the
    35th percentile of max(κ_ij) across all events — targeting ~35%
    background events (typical for tectonically active Italy).

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog **sorted by time**, with columns ``time``,
        ``latitude``, ``longitude``, ``depth_km``, ``magnitude``.
    K : float
        ETAS productivity constant.  Default 0.013 (Console et al. 2003).
    alpha_etas : float
        Magnitude scaling exponent.  Default 1.23.
    c_days : float
        Omori time offset in days.  Default 0.05.
    p : float
        Omori decay exponent.  Default 1.07.
    q : float
        Spatial decay exponent (power of r²+d²).  Default 1.5.
    d_km : float
        Spatial scale factor in km.  Default 1.5.
    t_max_days : float
        Look-back window in days.  Default 730 (2 years).
    mu_threshold : float or None
        Background threshold on κ units.  If None, auto-estimated.
    target_crs : str
        Projected CRS for km distance computation.
        Default ``"epsg:32632"`` (UTM 32N, Italy).

    Returns
    -------
    nx.DiGraph
        Directed forest.  Nodes are integer indices (0 … N-1).  Node
        attributes: ``lat``, ``lon``, ``depth_km``, ``magnitude``,
        ``mean_magnitude``, ``mean_depth``, ``is_background`` (bool).
        Edge attribute: ``weight`` = κ_ij of the winning parent link.

    References
    ----------
    Ogata, Y. (1988). Statistical models for earthquake occurrences and
    residual analysis for point processes. Journal of the American
    Statistical Association, 83(401), 9–27.

    Console, R., & Murru, M. (2001). A simple and testable model for
    earthquake clustering. Journal of Geophysical Research, 106(B5),
    8699–8711.

    Zhuang, J., Ogata, Y., & Vere-Jones, D. (2002). Stochastic declustering
    of space-time earthquake occurrences. Journal of the American Statistical
    Association, 97(458), 369–380.
    """
    t0      = time.time()
    EPS     = 1e-10
    t_max_s = t_max_days * 86400.0
    R_EARTH = 6371.0  # km

    df_s = df.sort_values("time").reset_index(drop=True)
    N    = len(df_s)
    log.info("Building ETAS network: %d events, t_max=%.0f days ...", N, t_max_days)

    _epoch  = pd.Timestamp("1970-01-01", tz="UTC")
    times_s = (pd.to_datetime(df_s["time"], utc=True) - _epoch).dt.total_seconds().values
    lats    = df_s["latitude"].values
    lons    = df_s["longitude"].values
    mags    = df_s["magnitude"].values
    depths  = df_s["depth_km"].values
    m_min   = float(mags.min())

    lats_r = np.radians(lats)
    lons_r = np.radians(lons)

    # Pre-compute projected km coordinates for Euclidean distance
    # (accurate enough for Italy's scale; avoids arcsin in the inner loop)
    transformer = Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
    x_m, y_m   = transformer.transform(lons, lats)
    x_km = x_m / 1000.0
    y_km = y_m / 1000.0

    G: nx.DiGraph = nx.DiGraph()
    for i in range(N):
        G.add_node(
            i,
            lat=float(lats[i]),
            lon=float(lons[i]),
            depth_km=float(depths[i]),
            magnitude=float(mags[i]),
            mean_magnitude=float(mags[i]),
            mean_depth=float(depths[i]),
            is_background=True,   # default; updated below
        )

    # First pass: compute max κ and best parent for every event
    max_kappa   = np.zeros(N, dtype=np.float64)
    best_parent = np.full(N, -1, dtype=np.int64)

    log_interval = max(N // 20, 1)
    d_km2 = d_km ** 2

    for j in range(1, N):
        if j % log_interval == 0:
            log.info("  %.0f%%  (%d/%d events processed)", 100.0 * j / N, j, N)

        t_j  = times_s[j]
        left = int(np.searchsorted(times_s[:j], t_j - t_max_s, side="left"))
        if left >= j:
            continue  # no candidates → root

        cands   = np.arange(left, j)
        dt_days = (t_j - times_s[cands]) / 86400.0   # days, always > 0

        dx_km = x_km[cands] - x_km[j]
        dy_km = y_km[cands] - y_km[j]
        r2    = dx_km ** 2 + dy_km ** 2               # km²

        kappa = (
            K
            * np.exp(alpha_etas * (mags[cands] - m_min))
            / (dt_days + c_days) ** p
            / (r2 + d_km2) ** q
        )

        best              = int(np.argmax(kappa))
        max_kappa[j]      = float(kappa[best])
        best_parent[j]    = int(cands[best])

    # Auto-threshold at 35th percentile of non-zero max_kappa values
    if mu_threshold is None:
        nonzero = max_kappa[max_kappa > 0]
        if len(nonzero) == 0:
            mu_threshold = 0.0
        else:
            mu_threshold = float(np.percentile(nonzero, 35.0))
        log.info("ETAS auto mu_threshold=%.4e (35th pctile of max κ)", mu_threshold)

    # Second pass: build edges where max_kappa > threshold
    n_linked = 0
    for j in range(1, N):
        if max_kappa[j] > mu_threshold and best_parent[j] >= 0:
            parent = int(best_parent[j])
            G.add_edge(parent, j, weight=float(max_kappa[j]))
            G.nodes[j]["is_background"] = False
            n_linked += 1

    n_roots = sum(1 for _, d in G.in_degree() if d == 0)
    n_trees = nx.number_weakly_connected_components(G)
    log.info(
        "ETAS (μ=%.3e): %d edges, %d roots (%.1f%% background), %d trees — %.1fs",
        mu_threshold, n_linked, n_roots, 100.0 * n_roots / N, n_trees, time.time() - t0,
    )
    return G
