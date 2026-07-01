"""
Empirical seismological law fitting: Gutenberg-Richter and Omori-Utsu.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

from src.plotutils import savefig, save_plotly, _slug


def fit_gr_law(
    df: pd.DataFrame,
    max_mag: float,
    M_c: float = 1.5,
    plot: bool = False,
    save: bool = True,
) -> dict:
    """
    Fit the Gutenberg-Richter law: log₁₀ N(≥M) = a − bM.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with a ``magnitude`` column.
    max_mag : float
        Upper magnitude limit for fitting.
    M_c : float
        Completeness magnitude (lower bound).
    plot : bool
        If True, display the fit plot.

    Returns
    -------
    dict
        Keys: ``max_mag``, ``a_value``, ``b_value``, ``r_squared``,
        ``std_err``.

    Notes
    -----
    b ≈ 1 is typical. Higher b means more small events relative to large ones.
    The b-value is a proxy for tectonic stress: lower b → higher differential
    stress (e.g. subduction zones).

    References
    ----------
    Gutenberg, B., & Richter, C. F. (1944). Frequency of earthquakes in
    California. Bulletin of the Seismological Society of America, 34(4),
    185-188.
    """
    mag_counts   = df["magnitude"].value_counts().sort_index()
    cumulative_N = mag_counts[::-1].cumsum()[::-1]

    fit_mask = (cumulative_N.index >= M_c) & (cumulative_N.index <= max_mag)
    mags_fit = cumulative_N.index[fit_mask]
    logN_fit = np.log10(cumulative_N.values[fit_mask])

    slope, intercept, r, _, std_err = linregress(mags_fit, logN_fit)
    b_value = -slope

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(cumulative_N.index, np.log10(cumulative_N.values),
                   alpha=0.7, edgecolors="k", label="Observed")
        ax.plot(mags_fit, intercept - b_value * mags_fit, linewidth=2, color='red', linestyle='--', alpha=0.6,
                label=f"Fit M≤{max_mag}  b={b_value:.2f}  a={intercept:.2f}")
        ax.set_title(f"Gutenberg-Richter (M ≤ {max_mag})")
        ax.set_xlabel("Magnitude M")
        ax.set_ylabel("$\\log_{10} N(\\geq M)$")
        ax.legend()
        plt.tight_layout()
        if save:
            savefig(f"gutenberg_richter_{_slug(str(max_mag))}")
        plt.show()

    return {
        "max_mag":   max_mag,
        "a_value":   intercept,
        "b_value":   b_value,
        "r_squared": r**2,
        "std_err":   std_err,
    }


def omori_law(t: np.ndarray, K: float, c: float, p: float) -> np.ndarray:
    """
    Modified Omori-Utsu law: n(t) = K / (c + t)^p.

    Parameters
    ----------
    t : array-like
        Time since mainshock in days.
    K : float
        Productivity (amplitude).
    c : float
        Time offset; regularises the singularity at t=0.
    p : float
        Decay exponent, typically 0.7–1.5.

    Returns
    -------
    np.ndarray
        Aftershock rate n(t).

    References
    ----------
    Utsu, T. (1961). A statistical study on the occurrence of aftershocks.
    Geophysical Magazine, 30, 521-605.
    """
    return K / (c + t) ** p


def fit_omori(
    df: pd.DataFrame,
    mainshock_time: pd.Timestamp,
    days: int,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    event_name: str = "Event",
    mag_cut: float = 1.5,
    save: bool = True,
) -> dict:
    """
    Fit the Omori-Utsu law to an aftershock sequence.

    Filters the catalog spatiotemporally around the mainshock, bins daily
    aftershock counts, fits both a log-log linear approximation and the full
    nonlinear Omori law, and displays a comparison plot.

    Parameters
    ----------
    df : pd.DataFrame
        Full catalog with columns ``time``, ``latitude``, ``longitude``,
        ``magnitude``.
    mainshock_time : pd.Timestamp
        Origin time of the mainshock (timezone-aware UTC).
    days : int
        Number of days to analyse after the mainshock.
    lat_range : tuple of float
        (lat_min, lat_max) bounding box.
    lon_range : tuple of float
        (lon_min, lon_max) bounding box.
    event_name : str
        Label used in plot titles and printed output.
    mag_cut : float
        Minimum magnitude filter.

    Returns
    -------
    dict
        Keys: ``p_approx``, ``K``, ``c``, ``p_omori``.

    Notes
    -----
    * p ≈ 1 → typical decay; p < 1 → slow decay; p > 1 → fast decay.
    * Large c values indicate early-time catalog incompleteness right after
      the mainshock.
    """
    end_time = mainshock_time + pd.Timedelta(days=days)
    seq = df[
        (df["time"] >= mainshock_time) & (df["time"] <= end_time)
        & (df["latitude"]  >= lat_range[0]) & (df["latitude"]  <= lat_range[1])
        & (df["longitude"] >= lon_range[0]) & (df["longitude"] <= lon_range[1])
        & (df["magnitude"] >= mag_cut)
    ].copy()

    seq["days_since"] = (
        (seq["time"] - mainshock_time).dt.total_seconds() / 86400.0
    )

    bins = np.arange(0, days + 1, 1)
    counts, edges = np.histogram(seq["days_since"], bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    mask = counts > 0
    t_v  = centers[mask]
    n_v  = counts[mask].astype(float)

    slope, intercept, *_ = linregress(np.log10(t_v), np.log10(n_v))
    p_approx = -slope
    n_approx = 10 ** (intercept + slope * np.log10(t_v))

    p0 = [float(n_v.max()), 0.1, 1.0]
    params, _ = curve_fit(omori_law, t_v, n_v, p0=p0, maxfev=10_000)
    K_fit, c_fit, p_fit = params
    t_smooth = np.linspace(t_v.min(), t_v.max(), 200)
    n_smooth = omori_law(t_smooth, K_fit, c_fit, p_fit)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(t_v, n_v, alpha=0.7, edgecolors="k", s=50,
               label="Observed daily aftershocks")
    ax.plot(t_v, n_approx, "--", linewidth=2, color="orange",
            label=rf"Approx. power-law ($p \approx {p_approx:.2f}$)")
    ax.plot(t_smooth, n_smooth, "--", linewidth=2, color="green",
            label=rf"Omori fit ($p={p_fit:.2f}$, $c={c_fit:.2f}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Omori Law — {event_name}", fontsize=16)
    ax.set_xlabel("Time since mainshock (days)", fontsize=14)
    ax.set_ylabel("Aftershock rate $n(t)$", fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    if save:
        savefig(f"omori_{_slug(event_name)}")
    plt.show()

    print(f"[{event_name}] p_approx={p_approx:.3f} | "
          f"K={K_fit:.1f}  c={c_fit:.2f}  p={p_fit:.3f}")

    return {"p_approx": p_approx, "K": K_fit, "c": c_fit, "p_omori": p_fit}
