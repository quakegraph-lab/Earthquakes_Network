import logging
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)


def analyze_degree_correlations_hybrid(
    G: nx.Graph,
    gamma: float,
    num_bins: int = 15,
    weighted: bool = False,
    save: bool = True
):
    """
    Assortativity analysis for hybrid seismic network.

    If weighted=True, uses edge weights (interaction strength).
    Otherwise uses purely topological structure.
    """

    # ─────────────────────────────────────────────
    # 1. PREPROCESS GRAPH
    # ─────────────────────────────────────────────
    G_simple = nx.Graph(G)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))

    N = G_simple.number_of_nodes()
    L = G_simple.number_of_edges()

    # ─────────────────────────────────────────────
    # 2. DEGREE OR STRENGTH
    # ─────────────────────────────────────────────
    if weighted:
        degrees = dict(G_simple.degree(weight="weight"))
        deg_label = "Strength s"
    else:
        degrees = dict(G_simple.degree())
        deg_label = "Degree k"

    # k_nn(k)
    knn_dict = nx.average_degree_connectivity(G_simple, weight="weight" if weighted else None)

    df_raw = pd.DataFrame([
        {"k": k, "knn": v}
        for k, v in knn_dict.items()
    ]).sort_values("k")

    df_raw = df_raw[(df_raw["k"] > 0) & (df_raw["knn"] > 0)]

    # ─────────────────────────────────────────────
    # 3. CUTOFFS
    # ─────────────────────────────────────────────
    k_str = np.sqrt(2 * L)
    k_nat = N ** (1.0 / (gamma - 1.0)) if gamma > 1.0 else np.nan

    # ─────────────────────────────────────────────
    # 4. LOG BINNING
    # ─────────────────────────────────────────────
    k_min, k_max = df_raw["k"].min(), df_raw["k"].max()
    bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), num_bins + 1)

    df_raw["bin"] = pd.cut(df_raw["k"], bins=bin_edges, include_lowest=True)

    binned = []
    for _, g in df_raw.groupby("bin"):
        if len(g) == 0:
            continue
        binned.append({
            "k": np.average(g["k"]),
            "knn": np.average(g["knn"])
        })

    df_binned = pd.DataFrame(binned)

    # ─────────────────────────────────────────────
    # 5. FIT (ONLY BELOW STRUCTURAL CUTOFF)
    # ─────────────────────────────────────────────
    df_fit = df_binned[df_binned["k"] < k_str]

    if len(df_fit) >= 2:
        mu, b = np.polyfit(np.log(df_fit["k"]), np.log(df_fit["knn"]), 1)
        has_fit = True
    else:
        mu, b = np.nan, np.nan
        has_fit = False

    # ─────────────────────────────────────────────
    # 6. PLOT
    # ─────────────────────────────────────────────
    plt.figure(figsize=(15, 6))

    plt.scatter(df_raw["k"], df_raw["knn"],
                color="gray", alpha=0.3, s=15, label="Raw")

    plt.scatter(df_binned["k"], df_binned["knn"],
                color="purple", s=40, edgecolor="black", label="Binned")

    if has_fit:
        k_line = np.linspace(df_fit["k"].min(), k_str, 200)
        plt.plot(k_line, np.exp(b) * k_line**mu,
                 "--", color="green", label=f"Fit μ={mu:.3f}")

    plt.axvline(k_str, color="red", linestyle="--",
                label=f"k_str={k_str:.1f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(deg_label)
    plt.ylabel("k_nn(k)")
    plt.title("Assortativity Analysis (Hybrid Network)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)

    if save:
        savefig("knn_assortativity_hybrid")

    plt.show()

    # ─────────────────────────────────────────────
    # 7. OUTPUT
    # ─────────────────────────────────────────────
    print("N:", N, "L:", L)
    print("γ:", gamma)
    print("k_str:", k_str)
    print("k_nat:", k_nat)

    if has_fit:
        print("μ:", mu)

    return df_raw, df_binned, {"mu": mu, "k_str": k_str, "k_nat": k_nat}





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





def analyze_assortativity_with_randomization_hybrid(
    G: nx.DiGraph,
    gamma: float,
    num_bins: int = 15,
    n_swaps_per_edge: int = 10,
    use_weighted: bool = False,
    save: bool = True
):
    """
    Degree-preserving randomization test for hybrid seismic network.

    IMPORTANT:
    - rewiring preserves ONLY topology (not weights)
    - weights are ignored in null model
    - interpretation is purely structural
    """

    # ─────────────────────────────────────────────
    # 1. ORIGINAL NETWORK
    # ─────────────────────────────────────────────
    G_orig = nx.Graph(G)
    G_orig.remove_edges_from(nx.selfloop_edges(G_orig))

    N = G_orig.number_of_nodes()
    L = G_orig.number_of_edges()

    k_str = np.sqrt(2 * L)
    k_nat = N ** (1.0 / (gamma - 1.0)) if gamma > 1.0 else np.nan

    df_raw_orig, df_bin_orig = preprocess_to_binned_df(G_orig, num_bins)
    mu_orig, b_orig = fit_intrinsic_slope(df_bin_orig, k_str)

    # ─────────────────────────────────────────────
    # 2. RANDOMIZATION (TOPOLOGY ONLY)
    # ─────────────────────────────────────────────
    print("Rewiring network (degree-preserving null model)...")

    G_rand = G_orig.copy()

    nx.double_edge_swap(
        G_rand,
        nswap=L * n_swaps_per_edge,
        max_tries=L * n_swaps_per_edge * 10
    )

    df_raw_rand, df_bin_rand = preprocess_to_binned_df(G_rand, num_bins)
    mu_rand, b_rand = fit_intrinsic_slope(df_bin_rand, k_str)

    # ─────────────────────────────────────────────
    # 3. PLOT
    # ─────────────────────────────────────────────
    plt.figure(figsize=(15, 6))

    # original
    plt.scatter(df_bin_orig["k"], df_bin_orig["knn"],
                color="purple", label="Original")

    # randomized
    plt.scatter(df_bin_rand["k"], df_bin_rand["knn"],
                color="orange", label="Rewired")

    # fits
    if not np.isnan(mu_orig):
        k_line = np.linspace(df_bin_orig["k"].min(), k_str, 200)
        plt.plot(k_line, np.exp(b_orig) * k_line**mu_orig,
                 "--", color="purple")

    if not np.isnan(mu_rand):
        k_line = np.linspace(df_bin_rand["k"].min(), k_str, 200)
        plt.plot(k_line, np.exp(b_rand) * k_line**mu_rand,
                 "--", color="orange")

    plt.axvline(k_str, color="red", linestyle="--")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree k")
    plt.ylabel("k_nn(k)")
    plt.title("Assortativity: Original vs Degree-Preserving Null Model")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)

    if save:
        savefig("knn_randomization_hybrid")

    plt.show()

    # ─────────────────────────────────────────────
    # 4. OUTPUT
    # ─────────────────────────────────────────────
    print("===================================")
    print(f"k_str = {k_str:.3f}")
    print(f"μ_orig = {mu_orig:.4f}")
    print(f"μ_rand = {mu_rand:.4f}")
    print(f"Δμ = {abs(mu_orig - mu_rand):.4f}")

    if abs(mu_orig - mu_rand) < 0.03:
        print("→ structurally random mixing")
    else:
        print("→ non-trivial structural correlations")

    print("===================================")

    return df_raw_orig, df_bin_orig, df_raw_rand, df_bin_rand






