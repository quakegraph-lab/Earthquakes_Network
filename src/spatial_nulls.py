"""
Spatial interaction null model for the Abe-Suzuki earthquake network.

Tests whether edge weights can be explained by a gravity model:

    λ_ij = C · A_i^α · A_j^β · d_ij^{-η}

where A_i = out-strength of source cell, A_j = in-strength of target cell,
d_ij = great-circle distance between cell centroids (km).  Parameters
(α, β, η, C) are estimated by log-linear OLS on the observed edge set.

The *structural excess* w_ij / λ_ij flags transitions stronger than
geography and local activity can explain – candidate long-range
stress-transfer corridors.

References
----------
Wilson, A. G. (1971). A family of spatial interaction models and their
  derivations. Environment and Planning A, 3(1), 1–32.
Krings, G. et al. (2009). Urban gravity: a model for inter-city
  telecommunication flows. Journal of Statistical Mechanics, L07003.
"""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)


# ── Distance utility ──────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6_371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Model fitting ─────────────────────────────────────────────────────────────

def fit_gravity_model(
    G: nx.DiGraph,
    min_distance_km: float = 1.0,
) -> tuple[dict, pd.DataFrame]:
    """
    Fit a gravity model to all directed edges of an Abe-Suzuki network.

    Log-linear OLS is used on edges with geographic distance ≥ ``min_distance_km``
    (vertical cell-to-cell transitions with identical lat/lon are excluded because
    they carry no geographic information for a 2-D gravity model).

    Parameters
    ----------
    G : nx.DiGraph
        Directed earthquake network; nodes must carry ``lat`` and ``lon``
        attributes (set by :func:`src.network.build_abe_suzuki_network`).
    min_distance_km : float
        Edges between cells closer than this are excluded from the fit.

    Returns
    -------
    params : dict
        Fitted parameters: ``alpha``, ``beta``, ``eta``, ``intercept``, ``r2``,
        ``n_edges`` (used in the fit), ``n_vertical`` (excluded).
    df_edges : pd.DataFrame
        One row per fitted edge with columns: ``u``, ``v``, ``w``,
        ``d_km``, ``A_u``, ``A_v``, ``lambda_``, ``log_excess``, ``excess``,
        ``lat_u``, ``lon_u``, ``lat_v``, ``lon_v``.
    """
    out_deg = dict(G.out_degree(weight="weight"))
    in_deg  = dict(G.in_degree(weight="weight"))
    records = []
    n_vertical = 0

    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        nu, nv = G.nodes[u], G.nodes[v]
        if "lat" not in nu or "lat" not in nv:
            continue
        d = _haversine_km(nu["lat"], nu["lon"], nv["lat"], nv["lon"])
        if d < min_distance_km:
            n_vertical += 1
            continue
        w = float(data.get("weight", 1.0))
        A_u = float(out_deg[u])
        A_v = float(in_deg[v])
        if A_u <= 0 or A_v <= 0 or w <= 0:
            continue
        records.append({
            "u": u, "v": v,
            "w": w, "d_km": d,
            "A_u": A_u, "A_v": A_v,
            "lat_u": nu["lat"], "lon_u": nu["lon"],
            "lat_v": nv["lat"], "lon_v": nv["lon"],
        })

    df = pd.DataFrame(records)
    log.info("Gravity fit: %d edges  (%d vertical excluded)", len(df), n_vertical)

    # Log-linear OLS: log(w) = c + α·log(A_u) + β·log(A_v) − η·log(d)
    log_w = np.log(df["w"].values)
    X = np.column_stack([
        np.ones(len(df)),
        np.log(df["A_u"].values),
        np.log(df["A_v"].values),
        np.log(df["d_km"].values),
    ])
    coeffs, *_ = np.linalg.lstsq(X, log_w, rcond=None)
    c, alpha, beta, eta_coeff = coeffs
    # eta_coeff is the coefficient on log(d); typically negative (decay)

    log_lambda = X @ coeffs
    ss_res = np.sum((log_w - log_lambda) ** 2)
    ss_tot = np.sum((log_w - log_w.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    df = df.assign(
        log_lambda=log_lambda,
        lambda_=np.exp(log_lambda),
        log_excess=log_w - log_lambda,
        excess=np.exp(log_w - log_lambda),
    )

    params = {
        "alpha":      round(float(alpha), 4),
        "beta":       round(float(beta),  4),
        "eta":        round(float(-eta_coeff), 4),  # positive decay exponent
        "intercept":  round(float(c), 4),
        "r2":         round(float(r2), 4),
        "n_edges":    len(df),
        "n_vertical": n_vertical,
    }
    log.info("Gravity params: α=%.3f  β=%.3f  η=%.3f  R²=%.3f",
             params["alpha"], params["beta"], params["eta"], params["r2"])
    return params, df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_gravity_fit(
    df_edges: pd.DataFrame,
    params: dict,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Scatter plot of observed vs gravity-model-predicted edge weights (log scale).

    Points are coloured by log-excess: red = stronger than expected (non-trivial),
    blue = weaker than expected (suppressed by distance / activity).

    Parameters
    ----------
    df_edges : pd.DataFrame
        Output ``df_edges`` from :func:`fit_gravity_model`.
    params : dict
        Output ``params`` from :func:`fit_gravity_model`.
    title : str
        Figure title suffix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    excess = df_edges["log_excess"].values
    vmax = np.percentile(np.abs(excess), 95)
    sc = ax.scatter(
        df_edges["lambda_"], df_edges["w"],
        c=excess, cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        alpha=0.25, s=6, linewidths=0,
    )
    lim_min = min(df_edges["lambda_"].min(), df_edges["w"].min())
    lim_max = max(df_edges["lambda_"].max(), df_edges["w"].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k-", lw=2.5, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    # Clip axes to actual scatter data range – don't let the y=x line stretch x beyond lambda_max
    x_lo = df_edges["lambda_"].min() * 0.8
    x_hi = df_edges["lambda_"].max() * 1.25
    y_lo = df_edges["w"].min() * 0.8
    y_hi = df_edges["w"].max() * 1.25
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    # Annotate the y=x line which is clipped to the bottom-left corner
    x_mid = np.sqrt(x_lo * x_hi)          # geometric mid of x range
    ax.annotate(
        "y = x  (perfect fit)",
        xy=(x_mid, x_mid),                # tip: on the line
        xytext=(x_mid * 1.4, x_mid * 8),  # text: above and slightly right
        fontsize=8, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )
    ax.set_xlabel("Gravity model prediction λ (transitions)", fontsize=11)
    ax.set_ylabel("Observed weight w (transitions)", fontsize=11)
    ax.set_title(
        f"Spatial Null Model – Observed vs Predicted – {title}\n"
        f"α={params['alpha']}  β={params['beta']}  η={params['eta']}  R²={params['r2']}",
        fontsize=12,
    )
    plt.colorbar(sc, ax=ax, label="log-excess log(w/λ)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        savefig(f"spatial_null_fit_{_slug(title)}")
    plt.show()


def plot_distance_decay(
    df_edges: pd.DataFrame,
    params: dict,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Log-log scatter of edge weight vs distance with fitted decay line.

    Shows the marginal distance effect after collapsing activity (A_u, A_v).
    Slope ≈ −η on this plot confirms the gravity decay exponent.

    Parameters
    ----------
    df_edges : pd.DataFrame
        Output ``df_edges`` from :func:`fit_gravity_model`.
    params : dict
        Output ``params`` from :func:`fit_gravity_model`.
    title : str
        Figure title suffix.
    """
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(8, 5))

    log_x = np.log10(df_edges["d_km"].clip(lower=1e-6))
    log_y = np.log10(df_edges["w"].clip(lower=1e-6))
    xy = np.vstack([log_x, log_y])
    density = gaussian_kde(xy)(xy)
    order = density.argsort()
    sc = ax.scatter(
        df_edges["d_km"].values[order], df_edges["w"].values[order],
        c=density[order], cmap="plasma", alpha=0.4, s=5, linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label="Point density")

    d_range = np.logspace(
        np.log10(df_edges["d_km"].min()),
        np.log10(df_edges["d_km"].max()),
        200,
    )
    # Evaluate the model holding A_u=A_v at their geometric means
    mean_logA_u = np.log(df_edges["A_u"]).mean()
    mean_logA_v = np.log(df_edges["A_v"]).mean()
    log_fit = (
        params["intercept"]
        + params["alpha"] * mean_logA_u
        + params["beta"]  * mean_logA_v
        - params["eta"]   * np.log(d_range)
    )
    ax.plot(d_range, np.exp(log_fit), "r-", lw=2,
            label=f"Gravity fit  η = {params['eta']:.3f}")

    # Binned mean + 90th-percentile – 20 log-spaced distance bins
    bin_edges = np.logspace(
        np.log10(df_edges["d_km"].min()),
        np.log10(df_edges["d_km"].max()),
        21,
    )
    bin_idx = np.digitize(df_edges["d_km"].values, bin_edges) - 1
    bin_mids, bin_means, bin_p90 = [], [], []
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        if mask.sum() >= 5:
            w_bin = df_edges["w"].values[mask]
            bin_mids.append(np.sqrt(bin_edges[b] * bin_edges[b + 1]))
            bin_means.append(np.mean(w_bin))
            bin_p90.append(np.percentile(w_bin, 90))
    if bin_mids:
        ax.plot(bin_mids, bin_means, "o--", color="steelblue", lw=1.5,
                ms=5, label="Binned mean w")
        ax.plot(bin_mids, bin_p90, "s--", color="darkorange", lw=1.5,
                ms=5, label="Binned 90th pct w")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Cell-to-cell distance (km)", fontsize=11)
    ax.set_ylabel("Edge weight (transitions)", fontsize=11)
    ax.set_title(f"Distance Decay of Seismic Transitions – {title}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        savefig(f"spatial_null_distance_decay_{_slug(title)}")
    plt.show()


def plot_excess_map(
    df_edges: pd.DataFrame,
    title: str = "",
    n_top: int = 200,
    scope: str = "world",       # kept for API compatibility; no longer used
    center_lat: float = 41.9,
    center_lon: float = 12.5,
    zoom: float = 0,
    bounds: dict | None = None,
    height: int = 600,
    width: int = 1100,
    save: bool = True,
) -> None:
    """
    Interactive map of the top-``n_top`` excess edges (strongest relative to gravity).

    Each edge is drawn as a line coloured by log-excess magnitude.
    These are the seismic transitions that the gravity model cannot explain –
    candidate long-range stress-transfer corridors or induced-seismicity links.

    Parameters
    ----------
    df_edges : pd.DataFrame
        Output ``df_edges`` from :func:`fit_gravity_model` (must contain
        ``lat_u``, ``lon_u``, ``lat_v``, ``lon_v``, ``log_excess``, ``w``).
    title : str
        Figure title suffix.
    n_top : int
        Number of highest-excess edges to display.
    center_lat, center_lon, zoom : float
        Map view parameters (tile-map convention).
    bounds : dict or None
        Optional ``dict(west=, east=, south=, north=)`` to constrain the
        visible viewport (passed directly to Plotly's ``map.bounds``).
    """
    df = df_edges.nlargest(n_top, "log_excess").reset_index(drop=True)

    exc = df["log_excess"].values
    exc_norm = (exc - exc.min()) / max(exc.max() - exc.min(), 1e-9)

    # Build line segments with None-separator trick
    lats, lons = [], []
    for _, row in df.iterrows():
        lats.extend([row["lat_u"], row["lat_v"], None])
        lons.extend([row["lon_u"], row["lon_v"], None])

    node_lat = np.concatenate([[r["lat_u"], r["lat_v"]] for _, r in df.iterrows()])
    node_lon = np.concatenate([[r["lon_u"], r["lon_v"]] for _, r in df.iterrows()])
    node_exc = np.concatenate([[e, e] for e in exc_norm])

    # Sort ascending so high-excess points render on top
    sort_idx = np.argsort(node_exc)
    node_lat = node_lat[sort_idx]
    node_lon = node_lon[sort_idx]
    node_exc = node_exc[sort_idx]

    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=lats, lon=lons,
        mode="lines",
        line=dict(width=1.0, color="rgba(200,50,50,0.4)"),
        name="Excess edge",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scattermap(
        lat=node_lat, lon=node_lon,
        mode="markers",
        marker=dict(
            size=6,
            color=node_exc,
            colorscale="Plasma",
            colorbar=dict(title="Norm. excess"),
            opacity=0.8,
        ),
        name="Cell centroid",
        hoverinfo="skip",
    ))

    map_cfg: dict = {
        "style": "carto-positron",
        "center": {"lat": center_lat, "lon": center_lon},
        "zoom": zoom,
    }
    if bounds is not None:
        map_cfg["bounds"] = bounds

    fig.update_layout(
        title=(f"Spatial Null Model – Top {n_top} Excess Edges (beyond gravity): {title}\n"
               "Red lines = transitions stronger than proximity + activity predicts"),
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        width=width, height=height,
        showlegend=False,
        map=map_cfg,
    )
    if save:
        save_plotly(fig, f"spatial_null_excess_map_{_slug(title)}")
    fig.show()
