"""
Statistical and graph-theoretic metrics for the Abe-Suzuki earthquake network.
"""

import logging

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

        * ``gamma``   – MLE power-law exponent from the powerlaw library.
        * ``sigma``   – Standard error on gamma.
        * ``k_min``   – Effective xmin used (may differ from input if
          powerlaw auto-selects it; here we fix it to ``k_min``).
        * ``R``       – Log-likelihood ratio: positive → power law fits
          better than exponential; negative → exponential wins.
        * ``p_value`` – Two-sided p-value for the likelihood ratio test.
          p < 0.05 means the direction of R is statistically significant.
        * ``verdict`` – ``"power law"`` if R > 0 and p < 0.05,
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
    # is unavailable – the continuous formula underestimates γ slightly for
    # integer-valued degrees but remains a valid approximation.
    try:
        import powerlaw as _pw
        fit = _pw.Fit(tail, xmin=k_min, discrete=True, verbose=False)
        return float(fit.alpha)
    except Exception:
        n = len(tail)
        return 1.0 + n / np.sum(np.log(tail / k_min))


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
        if not np.isclose(
            G.in_degree(n, weight="weight"),
            G.out_degree(n, weight="weight"),
        )
    ]
    if not unbalanced:
        log.info("Network balanced: in-strength == out-strength for all nodes.")
        return True
    log.warning("Found %d unbalanced nodes.", len(unbalanced))
    return unbalanced


def fit_powerlaw_hybrid(
    strengths: list[float],
    k_min: float | None = None,
) -> dict:
    """
    Power-law (Pareto) MLE fit of node strengths (Clauset, Shalizi &
    Newman 2009).

    Parameters
    ----------
    strengths : list of float
        Node strength (weighted degree) values; non-positive entries are
        dropped (the log-likelihood is undefined there).
    k_min : float or None
        Lower cutoff for the fit. If ``None`` (default), the powerlaw library
        auto-selects ``xmin`` by KS minimisation (Clauset et al. 2009);
        otherwise the supplied value is used.

    Returns
    -------
    dict with keys:
        ``gamma`` : float – power-law exponent (``fit.power_law.alpha``)
        ``xmin``  : float – lower cutoff of the fitted tail
    """
    arr = np.asarray(strengths, dtype=float)
    arr = arr[arr > 0]
    nan_result = {"gamma": float("nan"), "xmin": float("nan")}
    if len(arr) < 2:
        return nan_result

    try:
        import powerlaw as _pw
    except ImportError:
        return nan_result

    fit = _pw.Fit(arr, discrete=False) if k_min is None else _pw.Fit(arr, xmin=k_min, discrete=False)
    return {
        "gamma": float(fit.power_law.alpha),
        "xmin":  float(fit.power_law.xmin),
    }
