"""
Assortativity analysis for the Abe-Suzuki earthquake network.

Computes structural (degree-degree) and physical attribute (depth, magnitude)
assortativity, then visualizes the mixing patterns via binned and scalar plots.
Includes a suite of custom structural constraint diagnostics:

* ``preprocess_to_binned_df``      – Reduces a directed network to a simple graph
                                    and applies logarithmic binning to smooth noise.
* ``fit_intrinsic_slope``          – Computes the true mixing exponent (μ) by fitting
                                    binned data strictly below the structural cutoff.
* ``analyze_degree_correlations``  – Complete degree-mixing pipeline mapping raw data,
                                    log-binned points, and the intrinsic fit line.
* ``run_binned_randomization_test``– Generates an edge-swapped null model baseline to
                                    isolate genuine physics from finite-size constraints.

Seismological interpretation
----------------------------
* Intrinsic Degree Mixing (μ ≈ 0): The bulk network behaving identically to the 
  randomized baseline indicates an intrinsically neutral system. Outside of forced 
  finite-size artifacts, different seismic cells trigger consecutive earthquakes 
  independently of their overall activity level.

* Structural Disassortativity (k > k_str): The sharp drop in neighbor degrees 
  observed past the structural cutoff is a purely mathematical consequence of 
  heavy-tailed degree sequences (γ < 2). Massive seismic hubs physically run out 
  of other hubs to connect to, forcing them to link to smaller peripheral cells.

* Depth Assortativity (r > 0): A positive attribute correlation indicates that 
  events preferentially trigger sequence steps within the same crustal horizon, 
  highlighting vertical stratification of regional seismogenic zones.

* Magnitude Assortativity (r < 0): A negative attribute correlation represents a 
  clear signature of mainshock-aftershock relaxation, where a singular high-energy 
  spatial cell dissipates stress into a cascade of lower-magnitude surrounding grid units.
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)


def analyze_degree_correlations(
    G: nx.DiGraph, 
    gamma: float, 
    num_bins: int = 15, 
    save: bool = True ):
    """
    Complete pipeline for analyzing network assortativity and structural cutoffs.
    
    1. Preprocesses G into a simple graph.
    2. Computes raw k_nn(k).
    3. Computes exact structural and natural cutoffs.
    4. Performs logarithmic binning on the degree sequence.
    5. Fits a power-law exponent (mu) ONLY to binned data below the structural cutoff.
    6. Plots results on a clean log-log scale.
    """
    # ----------------------------------------------------
    # 1. GRAPH PREPROCESSING & RAW K_NN COMPUTATION
    # ----------------------------------------------------
    # Force conversion to a simple, undirected graph (removes weights, directions, multi-edges)
    G_simple = nx.Graph(G)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
    
    N = G_simple.number_of_nodes()
    L = G_simple.number_of_edges()
    
    # Compute raw k_nn(k)
    knn_dict = nx.average_degree_connectivity(G_simple)
    degrees = [d for n, d in G_simple.degree()]
    counts = pd.Series(degrees).value_counts().to_dict()

    df_raw = pd.DataFrame([
        {"k": k, "knn": knn, "count": counts[k]} 
        for k, knn in knn_dict.items()
    ]).sort_values("k").reset_index(drop=True)
    
    # Filter out any isolated nodes or non-positive entries
    df_raw = df_raw[(df_raw["k"] > 0) & (df_raw["knn"] > 0)]

    # ----------------------------------------------------
    # 2. COMPUTE FINITE-SIZE CUTOFFS
    # ----------------------------------------------------
    k_str = np.sqrt(2 * L)
    k_nat = N ** (1.0 / (gamma - 1.0)) if gamma > 1.0 else np.nan

    # ----------------------------------------------------
    # 3. LOGARITHMIC BINNING
    # ----------------------------------------------------
    # Create log-spaced bin edges from min degree to max degree
    k_min, k_max = df_raw["k"].min(), df_raw["k"].max()
    bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), num=num_bins + 1)
    
    # Adjust outermost boundaries slightly to avoid floating-point edge exclusions
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    
    df_raw["bin"] = pd.cut(df_raw["k"], bins=bin_edges)
    
    # Aggregate within bins using weighted averages based on node counts
    binned_rows = []
    for _, group in df_raw.groupby("bin", observed=False):
        if len(group) == 0:
            continue
        total_nodes = group["count"].sum()
        # True ensemble average within the log-interval
        w_k = np.average(group["k"], weights=group["count"])
        w_knn = np.average(group["knn"], weights=group["count"])
        
        binned_rows.append({"k": w_k, "knn": w_knn, "count": total_nodes})
        
    df_binned = pd.DataFrame(binned_rows)

    # ----------------------------------------------------
    # 4. BOUNDED POWER-LAW FITTING (BEFORE STRUCTURAL CUTOFF)
    # ----------------------------------------------------
    # Select log-binned data points that lie strictly below the structural cutoff
    df_fit = df_binned[df_binned["k"] < k_str]
    # Alternative: drops the first bin, i.e. any points where average k <= 1.5:
    #df_fit = df_binned[(df_binned["k"] > 1.5) & (df_binned["k"] < k_str)]
    
    if len(df_fit) >= 2:
        log_k = np.log(df_fit["k"].values)
        log_knn = np.log(df_fit["knn"].values)
        mu, b = np.polyfit(log_k, log_knn, 1)
        has_fit = True
    else:
        mu, b = np.nan, np.nan
        has_fit = False
        print("Warning: Not enough log-binned points below k_str to compute a linear fit.")

    # ----------------------------------------------------
    # 5. MATPLOTLIB PLOTTING
    # ----------------------------------------------------
    plt.figure(figsize=(15, 6))
    
    # All raw data in gray
    plt.scatter(df_raw["k"], df_raw["knn"], color="gray", alpha=0.3, s=15, label="Raw Data (All Nodes)")
    
    # Log-binned data in purple
    plt.scatter(df_binned["k"], df_binned["knn"], color="purple", alpha=0.9, s=40, edgecolor="black", zorder=4, label="Log-Binned Data")
    
    # Fitted line in dashed green (plotted up to the structural cutoff boundary)
    if has_fit:
        k_line = np.linspace(df_fit["k"].min(), k_str, 200)
        knn_fit = np.exp(b) * k_line**mu
        plt.plot(k_line, knn_fit, "mediumpurple", ls="--", linewidth=2, zorder=5, label=f"Intrinsic Fit ($\\mu$ = {mu:.3f})")
        
    # Vertical line for the structural cutoff in dashed red
    plt.axvline(k_str, color="red", linestyle="--", linewidth=1.5, label=f"Structural Cutoff ($k_{{str}}$ = {k_str:.1f})")

    # Layout adjustments
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree $k$", fontsize=15)
    plt.ylabel("$k_{nn}(k)$", fontsize=15)
    plt.title("Assortativity Fit", fontsize=15, pad=10)
    plt.legend(fontsize=9, loc="best")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    if save:
        savefig(f"knn_assortativity_fit")
    plt.show()

    # ----------------------------------------------------
    # 6. OUTPUT VALUES AND INTERPRETATION
    # ----------------------------------------------------
    print("==============================================")
    print("        NETWORK SCALING ANALYSIS METRICS       ")
    print("==============================================")
    print(f"Network Size (N)     : {N}")
    print(f"Network Edges (L)    : {L}")
    print(f"Power-Law Exponent(γ): {gamma:.2f}")
    print(f"Structural Cutoff(k_s): {k_str:.2f}")
    print(f"Natural Cutoff (k_nat): {k_nat:.2f}")
    print("----------------------------------------------")
    if has_fit:
        print(f"Fitted Slope (μ)     : {mu:.4f}")
        print(f"Fitted Intercept (b) : {b:.4f}")
        
        if mu > 0.05:
            interp = "Assortative (Hubs structurally free to connect to other hubs)"
        elif mu < -0.05:
            interp = "Disassortative (Nodes show intrinsic preference for smaller neighbors)"
        else:
            interp = "Neutral (No physical correlation; topology behaves randomly)"
        print(f"Intrinsic Behavior   : {interp}")
    print("==============================================")
    
    return df_raw, df_binned, {"mu": mu, "intercept": b, "k_str": k_str, "k_nat": k_nat}









def preprocess_to_binned_df(
    G_simple: nx.Graph, 
    num_bins: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes a simple undirected graph, computes raw k_nn, 
    and applies logarithmic binning. Returns (df_raw, df_binned).
    """
    knn_dict = nx.average_degree_connectivity(G_simple)
    degrees = [d for n, d in G_simple.degree()]
    counts = pd.Series(degrees).value_counts().to_dict()

    df_raw = pd.DataFrame([
        {"k": k, "knn": knn, "count": counts[k]} 
        for k, knn in knn_dict.items()
    ]).sort_values("k").reset_index(drop=True)
    
    df_raw = df_raw[(df_raw["k"] > 0) & (df_raw["knn"] > 0)]

    # Logarithmic binning
    k_min, k_max = df_raw["k"].min(), df_raw["k"].max()
    bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), num=num_bins + 1)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    
    df_raw["bin"] = pd.cut(df_raw["k"], bins=bin_edges)
    
    binned_rows = []
    for _, group in df_raw.groupby("bin", observed=False):
        if len(group) == 0:
            continue
        total_nodes = group["count"].sum()
        w_k = np.average(group["k"], weights=group["count"])
        w_knn = np.average(group["knn"], weights=group["count"])
        binned_rows.append({"k": w_k, "knn": w_knn, "count": total_nodes})
        
    return df_raw, pd.DataFrame(binned_rows)







def fit_intrinsic_slope(df_binned: pd.DataFrame, k_str: float) -> tuple[float, float]:
    """Fits log-binned before the structural cutoff."""
    df_fit = df_binned[ df_binned["k"] < k_str ]
    if len(df_fit) >= 2:
        log_k = np.log(df_fit["k"].values)
        log_knn = np.log(df_fit["knn"].values)
        mu, b = np.polyfit(log_k, log_knn, 1)
        return mu, b
    return np.nan, np.nan




def run_binned_randomization_test(
    G: nx.DiGraph, 
    gamma: float, 
    num_bins: int = 15, 
    n_swaps_per_edge: int = 10,
    save: bool = True
):
    """
    1. Preprocesses G to simple graph.
    2. Runs degree-preserving randomization (rewiring).
    3. Computes log-binned curves for both original and rewired.
    4. Plots them together (Original in Purple/Green, Rewired in Orange/Red).
    5. Fits slopes below k_str and outputs all parameter evaluations.
    """
    # ---- 1. SETUP ORIGINAL NETWORK ----
    G_orig = nx.Graph(G)
    G_orig.remove_edges_from(nx.selfloop_edges(G_orig))
    
    N = G_orig.number_of_nodes()
    L = G_orig.number_of_edges()
    k_str = np.sqrt(2 * L)
    k_nat = N ** (1.0 / (gamma - 1.0)) if gamma > 1.0 else np.nan
    
    df_raw_orig, df_bin_orig = preprocess_to_binned_df(G_orig, num_bins)
    mu_orig, b_orig = fit_intrinsic_slope(df_bin_orig, k_str)

    # ---- 2. RANDOMIZE NETWORK (DEGREE-PRESERVING) ----
    print("Scrambling network topology (preserving exact degrees)...")
    G_rand = G_orig.copy()
    nx.double_edge_swap(G_rand, nswap=L * n_swaps_per_edge, max_tries=L * n_swaps_per_edge * 10)
    
    df_raw_rand, df_bin_rand = preprocess_to_binned_df(G_rand, num_bins)
    mu_rand, b_rand = fit_intrinsic_slope(df_bin_rand, k_str)

    # ---- 3. MATPLOTLIB COMPARISON PLOT ----
    plt.figure(figsize=(15, 6))
    
    # Plot Original Network Layer
    plt.scatter(df_raw_orig["k"], df_raw_orig["knn"], color="gray", alpha=0.2, s=12, label="Raw Data (Original)")
    plt.scatter(df_bin_orig["k"], df_bin_orig["knn"], color="purple", alpha=0.9, s=40, edgecolor="black", zorder=4, label="Original Binned")
    if not np.isnan(mu_orig):
        k_line_orig = np.linspace(df_bin_orig[ df_bin_orig["k"] < k_str]["k"].min(), k_str, 200)
        plt.plot(k_line_orig, np.exp(b_orig) * k_line_orig**mu_orig, color="mediumpurple", ls="--", linewidth=2, zorder=5, label=f"Original Fit ($\\mu$ = {mu_orig:.3f})")

    # Plot Randomized Network Layer
    plt.scatter(df_bin_rand["k"], df_bin_rand["knn"], color="darkorange", alpha=0.9, s=40, marker="D", edgecolor="black", zorder=4, label="Rewired Binned")
    if not np.isnan(mu_rand):
        # We perform the fit completely without plotting the actual line if strictly requested, 
        # but plotting it helps visual diagnosis. If you prefer to hide the orange dashed line, comment out the next two lines:
        k_line_rand = np.linspace(df_bin_rand[df_bin_rand["k"] < k_str]["k"].min(), k_str, 200)
        plt.plot(k_line_rand, np.exp(b_rand) * k_line_rand**mu_rand, color="orange", linestyle=":", linewidth=2, zorder=5, label=f"Rewired Fit ($\\mu$ = {mu_rand:.3f})")

    # Structural Cutoff Line
    plt.axvline(k_str, color="red", linestyle="-.", linewidth=1.5, label=f"Structural Cutoff ($k_{{str}}$ = {k_str:.1f})")

    # Layout styling
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree $k$", fontsize=15)
    plt.ylabel("$k_{nn}(k)$", fontsize=15)
    plt.title("Assortativity Fit with Randomization Test", fontsize=15, pad=10)
    plt.legend(fontsize=9, loc="best")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    if save:
        savefig(f"knn_assortativity_fit_with_rewiring")
    plt.show()

    # ---- 4. CONSOLE OUTPUT METRICS ----
    print("==============================================")
    print("        RANDOMIZATION METRICS REPORT          ")
    print("==============================================")
    print(f"Structural Cutoff (k_str) : {k_str:.2f}")
    print(f"Natural Cutoff (k_nat)    : {k_nat:.2f}")
    print("----------------------------------------------")
    print(f"ORIGINAL Network Slope (μ): {mu_orig:.4f} (b={b_orig:.2f})")
    print(f"REWIRED Network Slope (μ) : {mu_rand:.4f} (b={b_rand:.2f})")
    print("----------------------------------------------")
    
    diff = abs(mu_orig - mu_rand)
    if diff < 0.03:
        print("DIAGNOSIS: The trends are practically identical.")
        print("Your network's mixing properties below the cutoff behave exactly like a random graph.")
    else:
        print("DIAGNOSIS: The trends differ significantly.")
        print("The physical spacing of your earthquake sequences holds non-structural ordering patterns!")
    print("==============================================")