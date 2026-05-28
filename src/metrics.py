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
def fit_strength_distribution_hybrid(
    strengths: list[float],
    k_min: float | None = None,
) -> dict:
    """
    Fit both power-law and lognormal to node strengths and compare via the
    Vuong likelihood-ratio test (Clauset, Shalizi & Newman 2009, §6.3).

    The hybrid Abe-Suzuki weights are a product of three exponentially
    decaying factors (magnitude, time, space), so by the multiplicative CLT
    the strength distribution is approximately lognormal. A power-law fit
    will succeed numerically on almost any heavy-tailed data, so we must
    *compare* it against the lognormal null before claiming scale-freeness.

    Parameters
    ----------
    strengths : list of float
        Node strength (weighted degree) values.
    k_min : float or None
        Lower cutoff for fitting. If ``None`` (default), powerlaw's KS
        minimisation auto-selects ``xmin`` (Clauset et al. 2009). Both
        candidate distributions are fit on the same tail.

    Returns
    -------
    dict with keys:
        ``power_law``  : {``gamma``, ``xmin``}     — Pareto MLE
        ``lognormal``  : {``mu``, ``sigma``, ``xmin``} — log-mean, log-std
        ``comparison`` : {``R``, ``p``, ``preferred``}
            ``R``         — normalised log-likelihood ratio (power_law / lognormal)
            ``p``         — significance of R under Vuong's test
            ``preferred`` — ``"power_law"`` if R>0 and p<0.1, ``"lognormal"``
                            if R<0 and p<0.1, else ``"inconclusive"``
    """
    arr = np.asarray(strengths, dtype=float)
    arr = arr[arr > 0]  # drop non-positive strengths (log undefined)
    nan_result = {
        "power_law": {"gamma": float("nan"), "xmin": float("nan")},
        "lognormal": {"mu": float("nan"), "sigma": float("nan"), "xmin": float("nan")},
        "comparison": {"R": float("nan"), "p": float("nan"), "preferred": "insufficient_data"},
    }
    if len(arr) < 2:
        return nan_result

    try:
        import powerlaw as _pw
    except ImportError:
        return nan_result

    if k_min is None:
        fit = _pw.Fit(arr, discrete=False)
    else:
        fit = _pw.Fit(arr, xmin=k_min, discrete=False)

    R, p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    if p < 0.1:
        preferred = "power_law" if R > 0 else "lognormal"
    else:
        preferred = "inconclusive"

    return {
        "power_law": {
            "gamma": float(fit.power_law.alpha),
            "xmin":  float(fit.power_law.xmin),
        },
        "lognormal": {
            "mu":    float(fit.lognormal.mu),
            "sigma": float(fit.lognormal.sigma),
            "xmin":  float(fit.lognormal.xmin),
        },
        "comparison": {
            "R": float(R),
            "p": float(p),
            "preferred": preferred,
        },
    }


def estimate_gamma_mle_hybrid(
    strengths: list[float],
    k_min: float | None = None,
) -> tuple[float, float]:
    """
    Power-law-only convenience wrapper around
    :func:`fit_strength_distribution_hybrid`.

    Returns ``(gamma, xmin)``. Prefer the richer function for new code —
    a lognormal vs power-law comparison is necessary to claim scale-freeness
    on the hybrid network's lognormal-shaped weight distribution.
    """
    result = fit_strength_distribution_hybrid(strengths, k_min=k_min)
    return result["power_law"]["gamma"], result["power_law"]["xmin"]


def fit_lognormal_full_hybrid(strengths: list[float]) -> tuple[float, float]:
    """
    Fit a lognormal distribution to the *full* strength distribution
    (no tail truncation) via the closed-form MLE:

    .. math::

        \\hat{\\mu} = \\frac{1}{N}\\sum_i \\ln s_i, \\qquad
        \\hat{\\sigma} = \\sqrt{\\frac{1}{N}\\sum_i (\\ln s_i - \\hat{\\mu})^2}.

    Use this when you want interpretable parameters for a histogram overlay
    — :func:`fit_strength_distribution_hybrid` fits lognormal only on the
    tail above ``xmin`` (so its body is forced absurdly far left to match
    the right tail; the μ it reports is not a description of the actual
    distribution).

    Parameters
    ----------
    strengths : list of float
        Node strength values; non-positive entries are dropped.

    Returns
    -------
    (mu, sigma) : tuple of float
        Maximum-likelihood lognormal parameters in *natural log* space.
        Median strength is ``exp(mu)``; geometric mean is ``exp(mu)``;
        the distribution is symmetric in ``log(s)`` around ``mu``.
    """
    arr = np.asarray(strengths, dtype=float)
    arr = arr[arr > 0]
    if len(arr) < 2:
        return float("nan"), float("nan")
    log_arr = np.log(arr)
    mu = float(np.mean(log_arr))
    sigma = float(np.std(log_arr, ddof=0))   # MLE uses ddof=0
    return mu, sigma