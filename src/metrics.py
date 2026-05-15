"""
Statistical and graph-theoretic metrics for the Abe-Suzuki earthquake network.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def test_power_law(degrees: list[float], k_min: float) -> dict:
    """
    Statistically test whether the degree distribution follows a power law
    using the Clauset-Shalizi-Newman (2009) method.

    Fits both a power law and an exponential distribution to the tail
    (values ≥ k_min) and computes a log-likelihood ratio test.

    Parameters
    ----------
    degrees : list of float
        Node degree (or strength) values.
    k_min : float
        Lower cutoff for the tail (same value used in ``estimate_gamma_mle``).

    Returns
    -------
    dict
        Keys:

        * ``gamma``   — MLE power-law exponent from the powerlaw library.
        * ``sigma``   — Standard error on gamma.
        * ``k_min``   — Effective xmin used (may differ from input if
          powerlaw auto-selects it; here we fix it to ``k_min``).
        * ``R``       — Log-likelihood ratio: positive → power law fits
          better than exponential; negative → exponential wins.
        * ``p_value`` — Two-sided p-value for the likelihood ratio test.
          p < 0.05 means the direction of R is statistically significant.
        * ``verdict`` — ``"power law"`` if R > 0 and p < 0.05,
          ``"not significant"`` otherwise.

    Raises
    ------
    ImportError
        If the ``powerlaw`` package is not installed.

    Notes
    -----
    ``R > 0`` and ``p < 0.05`` together constitute strong evidence for
    power-law behaviour over an exponential alternative (Le3 / Clauset 2009).
    The exponential is the Poisson-like null; rejecting it supports
    scale-free structure.

    References
    ----------
    Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law
    distributions in empirical data. SIAM Review, 51(4), 661-703.
    """
    try:
        import powerlaw  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pip install powerlaw  (required for test_power_law)"
        ) from exc

    tail = [d for d in degrees if d >= k_min]
    if len(tail) < 10:
        return {
            "gamma": float("nan"), "sigma": float("nan"),
            "k_min": k_min, "R": float("nan"),
            "p_value": float("nan"), "verdict": "insufficient data",
        }

    fit = powerlaw.Fit(tail, xmin=k_min, discrete=True, verbose=False)
    R, p = fit.distribution_compare("power_law", "exponential")

    return {
        "gamma":   round(float(fit.power_law.alpha), 3),
        "sigma":   round(float(fit.power_law.sigma), 3),
        "k_min":   fit.xmin,
        "R":       round(float(R), 3),
        "p_value": round(float(p), 4),
        "verdict": "power law" if (R > 0 and p < 0.05) else "not significant",
    }


def estimate_gamma_mle(degrees: list[float], k_min: float) -> float:
    """
    Estimate the power-law exponent γ via Maximum Likelihood Estimation.

    Parameters
    ----------
    degrees : list of float
        Node degree (or strength) values.
    k_min : float
        Lower cutoff; only values ≥ k_min enter the estimator.

    Returns
    -------
    float
        MLE estimate of γ, or ``nan`` if fewer than 2 values exceed k_min.

    Notes
    -----
    Closed-form MLE (Clauset et al. 2009):

        γ = 1 + n · [Σ ln(k_i / k_min)]^{-1}

    References
    ----------
    Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law
    distributions in empirical data. SIAM Review, 51(4), 661-703.
    """
    arr = np.asarray(degrees, dtype=float)
    tail = arr[arr >= k_min]
    if len(tail) < 2:
        return float("nan")
    # Discrete MLE via powerlaw library (Clauset et al. 2009, discrete case).
    # Falls back to the continuous closed-form 1 + n/Σln(kᵢ/kmin) if powerlaw
    # is unavailable — the continuous formula underestimates γ slightly for
    # integer-valued degrees but remains a valid approximation.
    try:
        import powerlaw as _pw
        fit = _pw.Fit(tail, xmin=k_min, discrete=True, verbose=False)
        return float(fit.alpha)
    except Exception:
        n = len(tail)
        return 1.0 + n / np.sum(np.log(tail / k_min))


def measure_preferential_attachment(
    df: pd.DataFrame,
    cell_size_km: float = 10.0,
    target_crs: str = "epsg:5070",
    k_min: int = 1,
    n_bins: int = 20,
    split: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Measure the empirical preferential attachment kernel π(k).

    Uses the **two-snapshot estimator** (Barabási *et al.* 2002):

    1. Split the sorted event sequence at fraction ``split`` (default 50 %).
    2. Build the network on the first half → degree :math:`k_i` for each cell.
    3. Count how many times each cell is the **target** (next earthquake) in
       the second half → :math:`\\Delta k_i`.
    4. Bin nodes by :math:`k_i` on a log scale and compute
       :math:`\\pi(k) = \\langle \\Delta k_i \\rangle_{k_i \\approx k}`.

    This avoids the consecutive-step saturation artefact of step-by-step
    estimators: the first-half degree is a stable snapshot, and the second-half
    target counts are a clean observation of which degree classes attract new
    edges.

    :math:`\\alpha \\approx 1` confirms linear preferential attachment (BA);
    :math:`\\alpha < 1` sub-linear; :math:`\\alpha > 1` super-linear.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog (columns: ``time``, ``latitude``, ``longitude``,
        ``depth_km``).
    cell_size_km : float
        Grid resolution matching the network.
    target_crs : str
        Projection CRS.
    k_min : int
        Minimum first-half degree to include.
    n_bins : int
        Number of logarithmic degree bins.
    split : float
        Fraction of events in the first (reference) half (default 0.5).

    Returns
    -------
    ks : np.ndarray
        Log-bin centres (degree).
    pi_k : np.ndarray
        Mean Δk per bin (proportional to π(k)).
    alpha : float
        Power-law exponent from OLS log-log fit.

    References
    ----------
    Barabási A.-L. *et al.* (2002). Evolution of the social network of
    scientific collaborations. *Physica A*, 311, 590–614.

    Jeong H., Néda Z. & Barabási A.-L. (2003). Measuring preferential
    attachment in evolving networks. *Europhysics Letters* 61, 567–572.
    """
    from collections import Counter
    from src.network import discretize_space_3d, build_abe_suzuki_network  # noqa: PLC0415

    df_s = df.sort_values("time").reset_index(drop=True)
    cut = int(len(df_s) * split)
    df_first  = df_s.iloc[:cut].reset_index(drop=True)
    df_second = df_s.iloc[cut:].reset_index(drop=True)

    # First-half network → degree k_i for each cell
    G_first = build_abe_suzuki_network(
        df_first, cell_size_km=cell_size_km, target_crs=target_crs,
    )
    k_first: dict[str, int] = dict(G_first.degree())

    # Second-half target counts Δk_i (times each cell is the next earthquake)
    df_grid2 = discretize_space_3d(df_second, cell_size_km=cell_size_km,
                                   target_crs=target_crs)
    seq2 = df_grid2["cell_id"].tolist()
    target_counts: Counter[str] = Counter(seq2[1:])  # every position except first is a target

    # Pair (k_first[cell], Δk[cell]) for cells in the first-half network
    pairs = [
        (k_first[n], target_counts.get(n, 0))
        for n in G_first.nodes()
        if k_first.get(n, 0) >= k_min
    ]
    if not pairs:
        return np.array([]), np.array([]), float("nan")

    ks_node = np.array([p[0] for p in pairs], dtype=float)
    dk_node = np.array([p[1] for p in pairs], dtype=float)

    # Log-bin: mean Δk per degree bin
    bin_edges = np.logspace(np.log10(ks_node.min()), np.log10(ks_node.max()), n_bins + 1)
    k_binned, pi_binned = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask_bin = (ks_node >= lo) & (ks_node < hi)
        if mask_bin.sum() >= 2:
            k_binned.append(np.sqrt(lo * hi))
            pi_binned.append(dk_node[mask_bin].mean())
    ks   = np.array(k_binned)
    pi_k = np.array(pi_binned)

    valid = pi_k > 0
    ks, pi_k = ks[valid], pi_k[valid]

    if len(ks) >= 3:
        log_c: float
        alpha, log_c = np.polyfit(np.log10(ks), np.log10(pi_k), 1)
    else:
        alpha = float("nan")

    log.info(
        "Preferential attachment (two-snapshot): α = %.3f (%d bins, k_min=%d)",
        alpha, len(ks), k_min,
    )
    return ks, pi_k, float(alpha)


def plot_preferential_attachment(
    ks: np.ndarray,
    pi_k: np.ndarray,
    alpha: float,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Plot the empirical preferential attachment kernel π(k) on log-log axes.

    A slope of α ≈ 1 on this plot confirms linear preferential attachment
    (Barabási-Albert). Deviation from linearity indicates a more complex
    growth mechanism.

    Parameters
    ----------
    ks : np.ndarray
        Degree bins from :func:`measure_preferential_attachment`.
    pi_k : np.ndarray
        Empirical π(k) values.
    alpha : float
        Fitted power-law exponent.
    title : str
        Figure title suffix.
    save : bool
        Whether to save the figure to disk.
    """
    from src.plotutils import savefig, _slug  # noqa: PLC0415

    mask = (ks >= 1) & (pi_k > 0)
    ks_fit, pi_fit_data = ks[mask], pi_k[mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ks_fit, pi_fit_data, s=45, alpha=0.85, color="steelblue",
               edgecolors="k", linewidths=0.4, zorder=3,
               label=r"Empirical $\pi(k)$ (log-binned)")

    if not np.isnan(alpha) and len(ks_fit) >= 3:
        log_c = float(np.mean(np.log10(pi_fit_data) - alpha * np.log10(ks_fit)))
        k_line = np.logspace(np.log10(ks_fit.min()), np.log10(ks_fit.max()), 200)
        ax.plot(k_line, 10 ** log_c * k_line ** alpha, "r--", linewidth=1.8,
                label=rf"Fit $\pi(k)\propto k^{{{alpha:.2f}}}$")
        # Linear PA reference (α = 1)
        ax.plot(k_line, 10 ** log_c * k_line ** 1.0, "g:", linewidth=1.4,
                alpha=0.7, label=r"Linear PA reference ($\alpha=1$)")

    verdict = (
        "consistent with linear PA (BA)"  if 0.85 <= alpha <= 1.15 else
        "sub-linear attachment"           if alpha < 0.85 else
        "super-linear attachment"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree $k$", fontsize=12)
    ax.set_ylabel(r"$\pi(k)$", fontsize=12)
    ax.set_title(
        rf"Preferential attachment kernel — {title}" + "\n"
        rf"$\alpha = {alpha:.3f}$ → {verdict}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save:
        savefig(f"preferential_attachment_{_slug(title)}")
    plt.show()


def measure_pa_forest(
    G: nx.DiGraph,
    df: pd.DataFrame,
    k_min: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Measure the preferential attachment kernel for a directed causal forest.

    In BP / ZBZ / ETAS networks each non-root event *j* has exactly one parent
    *i* (predecessor in *G*).  Replaying the forest chronologically, when child
    *j* attaches to parent *i*, the current out-degree of *i* (number of
    children it already has) is recorded.  The kernel

    .. math::

        \\pi(k_{\\text{out}}) = \\frac{\\sum_{i:\\,k_i^{\\text{out}}(t)=k}
        \\Delta k_i}{\\#\\{i:\\,k_i^{\\text{out}}(t)=k\\}}

    measures whether productive parents (high out-degree) attract more
    children — the direct seismological analogue of linear preferential
    attachment.

    Parameters
    ----------
    G : nx.DiGraph
        Directed causal forest. Node IDs are integers 0…N-1 matching the
        time-sorted DataFrame row order. Edges run parent→child.
    df : pd.DataFrame
        Earthquake catalog with a ``time`` column. Row *i* corresponds to
        node *i* in *G*.
    k_min : int
        Minimum out-degree to include in the fit.

    Returns
    -------
    ks : np.ndarray
        Out-degree values for which π(k) was estimated.
    pi_k : np.ndarray
        Empirical π(k) values.
    alpha : float
        Power-law exponent from log-log fit (nan if fit fails).

    References
    ----------
    Jeong H., Néda Z. & Barabási A.-L. (2003). Measuring preferential
    attachment in evolving networks. *Europhysics Letters* 61, 567–572.
    """
    from collections import defaultdict  # noqa: PLC0415

    # Edges are parent→child; predecessors(j) gives parent of child j
    parent_of: dict[int, int] = {}
    for j in G.nodes():
        preds = list(G.predecessors(j))
        if preds:
            parent_of[j] = preds[0]

    # Children sorted by event index = chronological order (df already time-sorted)
    children_sorted = sorted(parent_of.keys())

    out_deg: dict[int, int]    = defaultdict(int)
    delta_k: dict[int, float]  = defaultdict(float)
    count_k: dict[int, int]    = defaultdict(int)

    for child in children_sorted:
        parent = parent_of[child]
        k_p = out_deg[parent]
        delta_k[k_p] += 1
        count_k[k_p] += 1
        out_deg[parent] += 1

    ks_all = sorted(delta_k.keys())
    pi_all = np.array([delta_k[k] / count_k[k] for k in ks_all], dtype=float)
    ks_all = np.array(ks_all, dtype=float)

    mask = ks_all >= k_min
    ks   = ks_all[mask]
    pi_k = pi_all[mask]

    if mask.sum() >= 3:
        alpha = float(np.polyfit(np.log10(ks), np.log10(pi_k), 1)[0])
    else:
        alpha = float("nan")

    log.info(
        "PA forest: α = %.3f (%d degree bins, k_min=%d)",
        alpha, len(ks), k_min,
    )
    return ks, pi_k, float(alpha)


def measure_pa_growing_graph(
    G: nx.Graph,
    df: pd.DataFrame,
    k_min: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Measure the preferential attachment kernel for an undirected growing graph.

    In TL / HVG visibility graphs each new event *j* (node index *j*) connects
    to a subset of earlier events *i* < *j*.  When event *j* arrives, the
    degree of each earlier neighbour *i* is recorded before *j*'s edges are
    added (batch arrival).  The kernel π(k) is estimated identically to
    :func:`measure_pa_forest` but for undirected degree.

    Parameters
    ----------
    G : nx.Graph
        Undirected visibility graph. Node IDs are integers 0…N-1 matching
        the time-sorted DataFrame row order.
    df : pd.DataFrame
        Earthquake catalog with a ``time`` column.
    k_min : int
        Minimum degree to include in the fit.

    Returns
    -------
    ks : np.ndarray
        Degree values for which π(k) was estimated.
    pi_k : np.ndarray
        Empirical π(k) values.
    alpha : float
        Power-law exponent from log-log fit (nan if fit fails).

    References
    ----------
    Jeong H., Néda Z. & Barabási A.-L. (2003). Measuring preferential
    attachment in evolving networks. *Europhysics Letters* 61, 567–572.
    """
    from collections import defaultdict  # noqa: PLC0415

    deg: dict[int, int]        = defaultdict(int)
    delta_k: dict[int, float]  = defaultdict(float)
    count_k: dict[int, int]    = defaultdict(int)

    N = G.number_of_nodes()
    for j in range(N):
        earlier = [i for i in G.neighbors(j) if i < j]
        if not earlier:
            continue
        # Record degree of each earlier neighbour BEFORE j's edges are added
        for i in earlier:
            k_i = deg[i]
            delta_k[k_i] += 1
            count_k[k_i] += 1
        # Add edges (update degrees)
        for i in earlier:
            deg[i] += 1
        deg[j] += len(earlier)

    ks_all = sorted(delta_k.keys())
    pi_all = np.array([delta_k[k] / count_k[k] for k in ks_all], dtype=float)
    ks_all = np.array(ks_all, dtype=float)

    mask = ks_all >= k_min
    ks   = ks_all[mask]
    pi_k = pi_all[mask]

    if mask.sum() >= 3:
        alpha = float(np.polyfit(np.log10(ks), np.log10(pi_k), 1)[0])
    else:
        alpha = float("nan")

    log.info(
        "PA growing graph: α = %.3f (%d degree bins, k_min=%d)",
        alpha, len(ks), k_min,
    )
    return ks, pi_k, float(alpha)


def verify_balanced_degrees(G: nx.DiGraph) -> bool | list:
    """
    Check that weighted in-degree equals weighted out-degree for every node.

    Parameters
    ----------
    G : nx.DiGraph
        The Abe-Suzuki network.

    Returns
    -------
    bool or list
        ``True`` if balanced; otherwise a list of unbalanced node IDs.

    Notes
    -----
    In the Abe-Suzuki construction every interior earthquake is both a target
    and a source, so in-strength == out-strength for all nodes except the
    first (out only) and last (in only) events in the time series.
    """
    unbalanced = [
        n for n in G.nodes()
        if G.in_degree(n, weight="weight") != G.out_degree(n, weight="weight")
    ]
    if not unbalanced:
        log.info("Network balanced: in-strength == out-strength for all nodes.")
        return True
    log.warning("Found %d unbalanced nodes.", len(unbalanced))
    return unbalanced
