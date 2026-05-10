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
    """
    def _prob_dist(degrees: list, max_k: int):
        arr = np.clip(np.array([d for d in degrees if d > 0]), 0, max_k)
        counts = pd.Series(arr).value_counts().sort_index()
        return counts.index.values, counts.values / len(arr)

    k_in,  P_in  = _prob_dist([d for _, d in G.in_degree(weight="weight")],  max_degree)
    k_out, P_out = _prob_dist([d for _, d in G.out_degree(weight="weight")], max_degree)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(k_in,        P_in,  color="dodgerblue", alpha=0.5, label="In-Degree",  width=0.8)
    ax.bar(k_out + 0.2, P_out, color="salmon",     alpha=0.5, label="Out-Degree", width=0.8)
    ax.set_yscale("log")
    ax.set_title(f"In vs Out Degree Distribution (capped at {max_degree}): {title}", fontsize=15)
    ax.set_xlabel("Degree $k$", fontsize=13)
    ax.set_ylabel("Probability $P(k)$", fontsize=13)
    ax.legend(fontsize=11)
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
    save: bool = True,
) -> None:
    """
    Degree distribution with logarithmic binning and an OLS power-law fit
    on the tail.

    Log-binning normalises by bin width, giving probability *density* P(k).
    This is the standard approach for heavy-tailed distributions where
    linear binning leaves the tail noisy.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    title : str
        Figure title suffix.
    k_min_fit : float
        Minimum degree for the tail fit.
    n_bins : int
        Number of logarithmically spaced bins.
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

    valid   = P_k > 0
    k_v, P_v = centers[valid], P_k[valid]

    gamma, r_val = 0.0, 0.0
    fit_line = np.array([])
    fit_mask = k_v >= k_min_fit
    if fit_mask.sum() > 2:
        slope, intercept, r_val, *_ = linregress(
            np.log10(k_v[fit_mask]), np.log10(P_v[fit_mask]))
        gamma    = -slope
        fit_line = 10**intercept * k_v[fit_mask]**(-gamma)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(k_v, P_v, color="darkviolet", alpha=0.8, edgecolors="k", s=60,
               label="Log-binned")
    if fit_line.size:
        ax.plot(k_v[fit_mask], fit_line, "r--", linewidth=2.5,
                label=rf"Fit ($k \geq {k_min_fit}$): $\gamma \approx {gamma:.2f}$")
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

    print(f"[{title}] γ={gamma:.3f}  R²={r_val**2:.3f}")


def plot_ccdf_with_fit(
    G: nx.Graph,
    title: str,
    k_min_fit: float = 10,
    save: bool = True,
) -> None:
    """
    CCDF of node degree with an OLS power-law fit on the tail.

    For a power law P(k) ∝ k^{-γ}, the CCDF scales as k^{-(γ-1)},
    so γ = 1 − slope.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    title : str
        Figure title suffix.
    k_min_fit : float
        Minimum degree for the tail fit.
    """
    degrees = np.array([d for _, d in G.degree(weight="weight") if d > 0])
    if len(degrees) == 0:
        print("Empty graph.")
        return

    k_vals = np.sort(np.unique(degrees))
    ccdf   = np.array([np.mean(degrees >= k) for k in k_vals])

    gamma, r, fit_line = float("nan"), 0.0, None
    mask = k_vals >= k_min_fit
    if mask.sum() > 2:
        slope, intercept, r, *_ = linregress(
            np.log10(k_vals[mask]), np.log10(ccdf[mask]))
        gamma    = 1 - slope
        fit_line = 10**intercept * k_vals[mask]**slope

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(k_vals, ccdf, s=40, alpha=0.8, label="CCDF")
    if fit_line is not None:
        ax.plot(k_vals[mask], fit_line, "r--",
                label=rf"Fit: $\gamma \approx {gamma:.2f}$")
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

    print(f"[{title}] γ≈{gamma:.3f}  R²≈{r**2:.3f}")
