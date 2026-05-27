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


def plot_degree_distribution_linear(
    G: nx.Graph,
    title: str,
    max_degree: int = 50,
    save: bool = True,
    weighted: bool = True,
) -> None:
    """
    Bar chart of in- and out-degree distributions (log y-axis, linear x-axis).

    Parameters
    ----------
    G : nx.Graph
        Directed NetworkX graph with ``weight`` edge attribute.
    title : str
        Figure title suffix.
    max_degree : int
        Degree values are capped at this value before plotting.
    weighted : bool
        If True, use weighted degree (``weight="weight"``). Set False for
        graphs where edge weights are not transition counts (e.g. BP, TL).
    """
    def _prob_dist(degrees: list, max_k: int):
        arr = np.clip(np.array([d for d in degrees if d > 0]), 0, max_k)
        counts = pd.Series(arr).value_counts().sort_index()
        return counts.index.values, counts.values / len(arr)

    w = "weight" if weighted else None
    fig, ax = plt.subplots(figsize=(10, 6))

    if G.is_directed():
        k_in,  P_in  = _prob_dist([d for _, d in G.in_degree(weight=w)],  max_degree)
        k_out, P_out = _prob_dist([d for _, d in G.out_degree(weight=w)], max_degree)
        ax.bar(k_in,        P_in,  color="dodgerblue", alpha=0.5, label="In-Degree",  width=0.8)
        ax.bar(k_out + 0.2, P_out, color="salmon",     alpha=0.5, label="Out-Degree", width=0.8)
        ax.set_title(f"In vs Out Degree Distribution (capped at {max_degree}): {title}", fontsize=15)
        ax.legend(fontsize=11)
    else:
        k, P = _prob_dist([d for _, d in G.degree(weight=w)], max_degree)
        ax.bar(k, P, color="dodgerblue", alpha=0.7, width=0.8)
        ax.set_title(f"Degree Distribution (capped at {max_degree}): {title}", fontsize=15)

    ax.set_yscale("log")
    ax.set_xlabel("Degree $k$", fontsize=13)
    ax.set_ylabel("Probability $P(k)$", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    if save:
        savefig(f"degree_distribution_linear_{_slug(title)}")
    plt.show()


def analyze_in_out_degree_distribution(
    G: nx.Graph,
    title: str,
    fit: bool = False,
    save: bool = True,
) -> None:
    """
    Log-log scatter of weighted in- and out-degree distributions with
    optional OLS power-law fit lines.

    Parameters
    ----------
    G : nx.Graph
        Directed NetworkX graph.
    title : str
        Figure title suffix.
    fit : bool
        If True, overlay OLS power-law regression lines (k ≥ 2).
    """
    def _get_data(degrees: list, perform_fit: bool):
        arr = [d for d in degrees if d > 0]
        counts = pd.Series(arr).value_counts().sort_index()
        k, P_k = counts.index.values, counts.values / len(arr)
        if perform_fit and k[k >= 2].size > 2:
            mask = k >= 2
            slope, intercept, *_ = linregress(np.log10(k[mask]), np.log10(P_k[mask]))
            gamma = -slope
            return k, P_k, k[mask], 10**intercept * k[mask]**(-gamma), gamma
        return k, P_k, None, None, None

    k_in,  P_in,  kf_in,  lf_in,  g_in  = _get_data(
        [d for _, d in G.in_degree(weight="weight")],  fit)
    k_out, P_out, kf_out, lf_out, g_out = _get_data(
        [d for _, d in G.out_degree(weight="weight")], fit)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(k_in,  P_in,  color="dodgerblue", alpha=0.6, label="In-Degree")
    ax.scatter(k_out, P_out, color="salmon",      alpha=0.6, label="Out-Degree")
    if fit and kf_in is not None:
        ax.plot(kf_in,  lf_in,  "b--", label=rf"In fit ($\gamma={g_in:.2f}$)")
        ax.plot(kf_out, lf_out, "r--", label=rf"Out fit ($\gamma={g_out:.2f}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"In vs Out Degree Distribution: {title}", fontsize=15)
    ax.set_xlabel("Degree $k$ (log)", fontsize=13)
    ax.set_ylabel("Probability $P(k)$ (log)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    if save:
        savefig(f"in_out_degree_distribution_{_slug(title)}")
    plt.show()

    if fit and g_in is not None:
        print(f"[{title}] In γ={g_in:.3f}, Out γ={g_out:.3f}")
    else:
        print(f"[{title}] Degree distribution plotted (no fit).")


def analyze_degree_distribution(
    G: nx.Graph,
    title: str,
    fit: bool = False,
    save: bool = True,
) -> None:
    """
    Log-log scatter of total degree (strength) distribution with optional
    OLS power-law fit (k ≥ 2).

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    title : str
        Figure title suffix.
    fit : bool
        If True, overlay OLS power-law regression line.
    """
    degrees = [d for _, d in G.degree(weight="weight") if d > 0]
    counts = pd.Series(degrees).value_counts().sort_index()
    k, P_k = counts.index.values, counts.values / len(degrees)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(k, P_k, color="royalblue", alpha=0.7, edgecolors="k", label="Observed")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Degree Distribution: {title}", fontsize=15)
    ax.set_xlabel("Degree $k$", fontsize=13)
    ax.set_ylabel("Probability $P(k)$", fontsize=13)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    if fit:
        mask = k >= 2
        slope, intercept, r, *_ = linregress(np.log10(k[mask]), np.log10(P_k[mask]))
        gamma = -slope
        ax.plot(k[mask], 10**intercept * k[mask]**(-gamma), "r--", linewidth=2,
                label=rf"Power-law fit ($\gamma \approx {gamma:.2f}$)")
        print(f"[{title}] γ={gamma:.3f}  R²={r**2:.3f}")
    else:
        print(f"[{title}]")

    ax.legend(fontsize=12)
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
    Degree distribution with logarithmic binning and optional MLE power-law fit.

    Log-binning normalises by bin width, giving probability *density* P(k).
    When ``gamma_mle`` is supplied the amplitude is fitted by least-squares
    on the log-binned tail ($k \geq$ ``k_min_fit``) while the slope is fixed
    to the MLE exponent — giving a visually honest fit without OLS bias.

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
    degrees = [d for _, d in G.degree(weight="weight") if d > 0]
    if not degrees:
        print("Graph has no edges.")
        return

    bins = np.logspace(np.log10(min(degrees)), np.log10(max(degrees)), n_bins)
    counts, edges = np.histogram(degrees, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths  = np.diff(edges)
    P_k     = counts / (len(degrees) * widths)

    valid    = P_k > 0
    k_v, P_v = centers[valid], P_k[valid]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(k_v, P_v, color="darkviolet", alpha=0.8, edgecolors="k", s=60,
               label="Log-binned")

    if gamma_mle is not None:
        fit_mask = k_v >= k_min_fit
        if fit_mask.sum() > 1:
            log_k = np.log10(k_v[fit_mask])
            log_P = np.log10(P_v[fit_mask])
            intercept = np.mean(log_P + gamma_mle * log_k)
            fit_k  = np.logspace(np.log10(k_v[fit_mask][0]),
                                  np.log10(k_v[-1]), 200)
            fit_P  = 10**intercept * fit_k**(-gamma_mle)
            ax.plot(fit_k, fit_P, color="crimson", linewidth=2.5, linestyle="--",
                    label=rf"MLE fit ($k \geq {k_min_fit:.0f}$): $\hat{{\gamma}} = {gamma_mle:.2f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Degree Distribution (Log-Binning): {title}", fontsize=15)
    ax.set_xlabel("Degree $k$", fontsize=13)
    ax.set_ylabel("Probability Density $P(k)$", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    if save:
        savefig(f"degree_distribution_log_binning_{_slug(title)}")
    plt.show()

    print(f"[{title}] log-binned distribution plotted.")


def plot_ccdf_with_fit(
    G: nx.Graph,
    title: str,
    k_min_fit: float = 10,
    gamma_mle: float | None = None,
    save: bool = True,
) -> None:
    """
    CCDF of node degree with optional MLE power-law fit.

    For $P(k) \propto k^{-\gamma}$ the CCDF scales as $k^{-(\gamma-1)}$.
    When ``gamma_mle`` is supplied the amplitude is fitted on the tail while
    the slope is fixed — giving a clean MLE-anchored overlay.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    title : str
        Figure title suffix.
    k_min_fit : float
        Minimum degree for the fit overlay.
    gamma_mle : float or None
        MLE power-law exponent.  If given, a fit line is drawn on the tail.
    """
    degrees = np.array([d for _, d in G.degree(weight="weight") if d > 0])
    if len(degrees) == 0:
        print("Empty graph.")
        return

    k_vals = np.sort(np.unique(degrees))
    ccdf   = np.array([np.mean(degrees >= k) for k in k_vals])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(k_vals, ccdf, s=40, alpha=0.8, label="CCDF")

    if gamma_mle is not None:
        mask = k_vals >= k_min_fit
        if mask.sum() > 1:
            ccdf_exp = gamma_mle - 1
            log_k    = np.log10(k_vals[mask])
            log_c    = np.log10(ccdf[mask])
            intercept = np.mean(log_c + ccdf_exp * log_k)
            fit_k = np.logspace(np.log10(k_vals[mask][0]),
                                 np.log10(k_vals[-1]), 200)
            fit_c = 10**intercept * fit_k**(-ccdf_exp)
            ax.plot(fit_k, fit_c, color="crimson", linewidth=2.5, linestyle="--",
                    label=rf"MLE fit: $\hat{{\gamma}} = {gamma_mle:.2f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree $k$", fontsize=12)
    ax.set_ylabel("$P(K \\geq k)$", fontsize=12)
    ax.set_title(f"CCDF Degree Distribution: {title}", fontsize=14)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    if save:
        savefig(f"ccdf_{_slug(title)}")
    plt.show()

    print(f"[{title}] CCDF plotted.")







# =================================================================================
# =================================================================================
# ======================== HYBRID PART ============================================
# =================================================================================
# =================================================================================



def analyze_in_out_degree_distribution_hybrid(
    G: nx.DiGraph,
    title: str,
    bins: int = 30,
    fit: bool = False,
    save: bool = True,
) -> None:

    def _binned_distribution(values, bins):
        values = np.array([v for v in values if v > 0])

        hist, bin_edges = np.histogram(values, bins=bins, density=True)

        # avoid zero bins for log-log plot
        mask = hist > 0
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return centers[mask], hist[mask]

    # weighted in/out strengths
    in_strength  = [d for _, d in G.in_degree(weight="weight")]
    out_strength = [d for _, d in G.out_degree(weight="weight")]

    k_in, P_in   = _binned_distribution(in_strength, bins)
    k_out, P_out = _binned_distribution(out_strength, bins)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(k_in, P_in, color="dodgerblue", alpha=0.6, label="In-strength")
    ax.scatter(k_out, P_out, color="salmon", alpha=0.6, label="Out-strength")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(f"In vs Out Strength Distribution (Hybrid): {title}")
    ax.set_xlabel("Strength (log)")
    ax.set_ylabel("P(s)")

    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save:
        savefig(f"in_out_strength_distribution_hybrid_{_slug(title)}")
    plt.show()






def analyze_degree_distribution_hybrid(
    G: nx.DiGraph,
    title: str,
    bins: int = 30,
    fit: bool = False,
    save: bool = True,
) -> None:

    def _binned_distribution(values, bins):
        values = np.array([v for v in values if v > 0])

        hist, bin_edges = np.histogram(values, bins=bins, density=True)

        mask = hist > 0
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return centers[mask], hist[mask]

    strengths = [d for _, d in G.degree(weight="weight")]

    k, P_k = _binned_distribution(strengths, bins)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(k, P_k, color="royalblue", alpha=0.7, edgecolors="k", label="Observed")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(f"Strength Distribution (Hybrid): {title}")
    ax.set_xlabel("Strength s")
    ax.set_ylabel("P(s)")
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # optional power-law fit (now meaningful again)
    if fit and len(k) > 5:
        mask = k >= np.percentile(k, 20)  # safer cutoff than k>=2
        slope, intercept, r, *_ = linregress(np.log10(k[mask]), np.log10(P_k[mask]))
        gamma = -slope

        ax.plot(k[mask],
                10**intercept * k[mask]**(-gamma),
                "r--",
                label=rf"Fit ($\gamma \approx {gamma:.2f}$)")

        print(f"[{title}] γ={gamma:.3f}  R²={r**2:.3f}")

    ax.legend()
    plt.tight_layout()

    if save:
        savefig(f"strength_distribution_hybrid_{_slug(title)}")

    plt.show()






def analyze_strength_distribution_log_binning_hybrid(
    G: nx.DiGraph,
    title: str,
    k_min_fit: float = 1e-13,
    n_bins: int = 25,
    gamma_mle: float | None = None,
    save: bool = True,
) -> None:
    """
    Log-binned distribution of node strengths (hybrid Abe–Suzuki network).
    """

    strengths = [s for _, s in G.degree(weight="weight") if s > 0]

    if not strengths:
        print("Graph has no edges.")
        return

    bins = np.logspace(np.log10(min(strengths)), np.log10(max(strengths)), n_bins)
    counts, edges = np.histogram(strengths, bins=bins)

    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    P_k = counts / (len(strengths) * widths)

    valid = P_k > 0
    k_v, P_v = centers[valid], P_k[valid]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(k_v, P_v, color="darkviolet", alpha=0.8,
               edgecolors="k", s=60, label="Log-binned")

    if gamma_mle is not None:
        fit_mask = k_v >= k_min_fit

        if fit_mask.sum() > 1:
            log_k = np.log10(k_v[fit_mask])
            log_P = np.log10(P_v[fit_mask])

            intercept = np.mean(log_P + gamma_mle * log_k)

            fit_k = np.logspace(np.log10(k_v[fit_mask][0]),
                                np.log10(k_v[-1]), 200)

            fit_P = 10**intercept * fit_k**(-gamma_mle)

            ax.plot(
                fit_k, fit_P,
                color="crimson",
                linewidth=2.5,
                linestyle="--",
                label=rf"MLE fit ($k \geq {k_min_fit:.0e}$): $\hat{{\gamma}} = {gamma_mle:.2f}$"
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Strength Distribution (Log-Binning): {title}", fontsize=15)
    ax.set_xlabel("Strength $s$", fontsize=13)
    ax.set_ylabel("Probability Density $P(s)$", fontsize=13)

    ax.legend(fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()

    if save:
        savefig(f"strength_distribution_log_binning_{_slug(title)}")

    plt.show()

    print(f"[{title}] log-binned strength distribution plotted.")






def plot_ccdf_with_fit_hybrid(
    G: nx.Graph,
    title: str,
    k_min_fit: float = 1e-13,
    gamma_mle: float | None = None,
    save: bool = True,
) -> None:
    """
    CCDF of node strength (hybrid Abe–Suzuki network) with optional MLE fit.

    For a power-law:
        P(s) ~ s^{-γ}
    the CCDF scales as:
        P(S ≥ s) ~ s^{-(γ - 1)}
    """

    strengths = np.array([s for _, s in G.degree(weight="weight") if s > 0])

    if len(strengths) == 0:
        print("Empty graph.")
        return

    # CCDF (same definition, but on strengths)
    s_vals = np.sort(np.unique(strengths))
    ccdf = np.array([np.mean(strengths >= s) for s in s_vals])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(s_vals, ccdf, s=40, alpha=0.8, label="CCDF (strength)")

    if gamma_mle is not None:
        mask = s_vals >= k_min_fit

        if mask.sum() > 1:
            # CCDF exponent relation
            ccdf_exp = gamma_mle - 1

            log_s = np.log10(s_vals[mask])
            log_c = np.log10(ccdf[mask])

            intercept = np.mean(log_c + ccdf_exp * log_s)

            fit_s = np.logspace(
                np.log10(s_vals[mask][0]),
                np.log10(s_vals[-1]),
                200
            )

            fit_c = 10**intercept * fit_s**(-ccdf_exp)

            ax.plot(
                fit_s,
                fit_c,
                color="crimson",
                linewidth=2.5,
                linestyle="--",
                label=rf"MLE fit: $\hat{{\gamma}} = {gamma_mle:.2f}$"
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Strength $s$", fontsize=12)
    ax.set_ylabel("$P(S \\geq s)$", fontsize=12)
    ax.set_title(f"CCDF Strength Distribution: {title}", fontsize=14)

    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()

    if save:
        savefig(f"ccdf_strength_{_slug(title)}")

    plt.show()

    print(f"[{title}] CCDF plotted.")