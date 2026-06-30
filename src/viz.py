"""
Plotting utilities for degree-distribution and network analysis figures.

All functions display a matplotlib figure and print a summary line.
Call ``sns.set_theme(style="whitegrid")`` in the notebook before use.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.plotutils import savefig, save_plotly, _slug


def analyze_degree_distribution(
    G: nx.Graph,
    title: str,
    fit: bool = False,
    save: bool = True,
) -> None:
    """
    Log-log scatter of total degree distribution

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    title : str
        Figure title suffix.
    fit : bool
    """

    # degree distribution on the undirected projection
    G_und = G.to_undirected()

    degrees = [d for _, d in G_und.degree() if d > 0]

    counts = pd.Series(degrees).value_counts().sort_index()
    k = counts.index.values
    P_k = counts.values / len(degrees)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(k, P_k, color="royalblue", alpha=0.7,
               edgecolors="k", label="Observed")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Degree Distribution: {title}" if title else "Degree Distribution")
    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("Probability $P(k)$")

    ax.grid(True, which="both", ls="--", alpha=0.3)

    if fit and len(k[k >= 2]) > 2:
        mask = k >= 2
        slope, intercept, r, *_ = linregress(
            np.log10(k[mask]), np.log10(P_k[mask])
        )
        gamma = -slope

        ax.plot(
            k[mask],
            10**intercept * k[mask]**(-gamma),
            "r--",
            linewidth=2,
            label=rf"Fit ($\gamma \approx {gamma:.2f}$)"
        )

        print(f"[{title}] γ={gamma:.3f}  R²={r**2:.3f}")

    ax.legend()
    plt.tight_layout()

    if save:
        savefig(f"degree_distribution_{_slug(title)}")

    plt.show()

def analyze_degree_distribution_log_binning(
    G: nx.Graph,
    title: str,
    k_min_fit: float = 10,
    n_bins: int = 25,
    gamma_mle: float | None = None,
    save: bool = True,
) -> None:
    """
    Degree distribution (UNDIRECTED) with logarithmic binning and optional MLE power-law fit.

    Log-binning normalises by bin width, giving probability *density* P(k).
    When ``gamma_mle`` is supplied the amplitude is fitted by least-squares
    on the log-binned tail ($k \\geq$ ``k_min_fit``) while the slope is fixed
    to the MLE exponent, so the line reflects the MLE rather than an OLS fit.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    title : str
        Figure title suffix.
    k_min_fit : float
        Minimum degree used for the fit amplitude and the fit overlay.
    n_bins : int
        Number of logarithmically spaced bins.
    gamma_mle : float or None
        MLE power-law exponent.  If given, a fit line is drawn on the tail.
    """

    # undirected degree
    G_und = G.to_undirected()
    degrees = [d for _, d in G_und.degree() if d > 0]

    if not degrees:
        print("Graph has no edges.")
        return

    # logarithmic binning
    bins = np.logspace(
        np.log10(min(degrees)),
        np.log10(max(degrees)),
        n_bins + 1
    )

    counts, edges = np.histogram(degrees, bins=bins)

    centers = (edges[:-1] + edges[1:]) / 2
    widths  = np.diff(edges)

    # PDF normalization
    P_k = counts / (len(degrees) * widths)

    # remove empty bins
    valid = (P_k > 0) & (centers > 0)
    k_v = centers[valid]
    P_v = P_k[valid]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        k_v, P_v,
        alpha=0.85,
        edgecolors="k",
        s=60,
        label="Log-binned"
    )

    # MLE power-law overlay
    if gamma_mle is not None:
        fit_mask = k_v >= k_min_fit

        if fit_mask.sum() > 1:
            log_k = np.log10(k_v[fit_mask])
            log_P = np.log10(P_v[fit_mask])

            # amplitude fit (slope fixed)
            intercept = np.mean(log_P + gamma_mle * log_k)

            fit_k = np.logspace(
                np.log10(k_v[fit_mask][0]),
                np.log10(k_v[-1]),
                200
            )
            fit_P = 10**intercept * fit_k**(-gamma_mle)

            ax.plot(
                fit_k, fit_P,
                linestyle="--",
                linewidth=2.5,
                label=rf"MLE fit ($k \geq {k_min_fit}$): $\gamma={gamma_mle:.2f}$"
            )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(f"Degree Distribution (Log-Binning): {title}" if title else "Degree Distribution (Log-Binning)")
    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("Probability Density $P(k)$")

    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save:
        savefig(f"degree_distribution_log_binning_{_slug(title)}")

    plt.show()

    print(f"[{title}] log-binned degree distribution plotted.")

