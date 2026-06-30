"""
Statistical and graph-theoretic metrics for the Abe-Suzuki earthquake network.
"""

import logging

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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

