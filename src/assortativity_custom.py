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
    save: bool = True
):
    """
    Topological assortativity analysis (degree-based) for hybrid network.

    IMPORTANT:
    - Uses pure degree (NOT weights)
    - Suitable for comparison with ER / configuration model
    """

    # ─────────────────────────────────────────────
    # 1. PREPROCESS GRAPH
    # ─────────────────────────────────────────────
    G_simple = nx.Graph(G)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))

    N = G_simple.number_of_nodes()
    L = G_simple.number_of_edges()

    # ─────────────────────────────────────────────
    # 2. k_nn(k)
    # ─────────────────────────────────────────────
    knn_dict = nx.average_degree_connectivity(G_simple)

    # degree counts (important for weighting!)
    degrees = [d for _, d in G_simple.degree()]
    counts = pd.Series(degrees).value_counts().to_dict()

    df_raw = pd.DataFrame([
        {"k": k, "knn": knn_dict[k], "count": counts.get(k, 1)}
        for k in knn_dict
    ]).sort_values("k").reset_index(drop=True)

    df_raw = df_raw[(df_raw["k"] > 0) & (df_raw["knn"] > 0)]

    # ─────────────────────────────────────────────
    # 3. CUTOFFS
    # ─────────────────────────────────────────────
    k_str = np.sqrt(2 * L)
    k_nat = N ** (1.0 / (gamma - 1.0)) if gamma > 1.0 else np.nan

    # ─────────────────────────────────────────────
    # 4. LOG BINNING (weighted!)
    # ─────────────────────────────────────────────
    k_min, k_max = df_raw["k"].min(), df_raw["k"].max()

    bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), num_bins + 1)
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

        binned_rows.append({
            "k": w_k,
            "knn": w_knn,
            "count": total_nodes
        })

    df_binned = pd.DataFrame(binned_rows)

    # ─────────────────────────────────────────────
    # 5. FIT (below structural cutoff)
    # ─────────────────────────────────────────────
    df_fit = df_binned[df_binned["k"] < k_str]

    if len(df_fit) >= 2:
        log_k = np.log(df_fit["k"].values)
        log_knn = np.log(df_fit["knn"].values)

        mu, b = np.polyfit(log_k, log_knn, 1)
        has_fit = True
    else:
        mu, b = np.nan, np.nan
        has_fit = False
        print("Warning: Not enough points for fit.")

    # ─────────────────────────────────────────────
    # 6. PLOT
    # ─────────────────────────────────────────────
    plt.figure(figsize=(15, 6))

    # raw
    plt.scatter(
        df_raw["k"], df_raw["knn"],
        color="gray", alpha=0.3, s=15,
        label="Raw data"
    )

    # binned
    plt.scatter(
        df_binned["k"], df_binned["knn"],
        color="purple", s=40, edgecolor="black",
        label="Log-binned"
    )

    # fit
    if has_fit:
        k_line = np.linspace(df_fit["k"].min(), k_str, 200)
        knn_fit = np.exp(b) * k_line**mu

        plt.plot(
            k_line, knn_fit,
            "--", color="mediumpurple", linewidth=2,
            label=f"Fit μ = {mu:.3f}"
        )

    # cutoff
    plt.axvline(
        k_str, color="red", linestyle="--",
        label=f"k_str = {k_str:.1f}"
    )

    # layout
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree k")
    plt.ylabel(r"$k_{nn}(k)$")
    plt.title("Assortativity Analysis (Degree-based, Hybrid Network)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)

    if save:
        savefig("knn_assortativity_hybrid_degree")

    plt.show()

    # ─────────────────────────────────────────────
    # 7. OUTPUT
    # ─────────────────────────────────────────────
    print("====================================")
    print("N:", N, "L:", L)
    print("γ:", gamma)
    print("k_str:", k_str)
    print("k_nat:", k_nat)

    if has_fit:
        print("μ:", mu)

        if mu > 0.05:
            print("→ Assortative")
        elif mu < -0.05:
            print("→ Disassortative")
        else:
            print("→ Neutral")

    print("====================================")

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
    G: nx.Graph,
    gamma: float,
    num_bins: int = 15,
    n_swaps_per_edge: int = 10,
    save: bool = True
):
    """
    Assortativity analysis with degree-preserving randomization.

    Uses ONLY degree (topological assortativity).
    Compares original vs rewired network.
    """

    # ─────────────────────────────────────────────
    # 1. PREPROCESS GRAPH
    # ─────────────────────────────────────────────
    G_simple = nx.Graph(G)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))

    N = G_simple.number_of_nodes()
    L = G_simple.number_of_edges()

    # ─────────────────────────────────────────────
    # 2. k_nn(k) ORIGINAL
    # ─────────────────────────────────────────────
    knn_dict = nx.average_degree_connectivity(G_simple)

    degrees = [d for _, d in G_simple.degree()]
    counts = pd.Series(degrees).value_counts().to_dict()

    df_raw = pd.DataFrame([
        {"k": k, "knn": knn_dict[k], "count": counts[k]}
        for k in knn_dict
    ]).sort_values("k")

    df_raw = df_raw[(df_raw["k"] > 0) & (df_raw["knn"] > 0)]

    # ─────────────────────────────────────────────
    # 3. CUTOFFS
    # ─────────────────────────────────────────────
    k_str = np.sqrt(2 * L)
    k_nat = N ** (1.0 / (gamma - 1.0)) if gamma > 1 else np.nan

    # ─────────────────────────────────────────────
    # 4. LOG BINNING (ORIGINAL)
    # ─────────────────────────────────────────────
    def log_bin(df):
        k_min, k_max = df["k"].min(), df["k"].max()
        bins = np.logspace(np.log10(k_min), np.log10(k_max), num_bins + 1)
        bins[0] -= 1e-5
        bins[-1] += 1e-5

        df["bin"] = pd.cut(df["k"], bins=bins)

        rows = []
        for _, g in df.groupby("bin", observed=False):
            if len(g) == 0:
                continue
            rows.append({
                "k": np.average(g["k"], weights=g["count"]),
                "knn": np.average(g["knn"], weights=g["count"]),
                "count": g["count"].sum()
            })
        return pd.DataFrame(rows)

    df_binned = log_bin(df_raw.copy())

    # ─────────────────────────────────────────────
    # 5. FIT ORIGINAL
    # ─────────────────────────────────────────────
    def fit(df):
        df_fit = df[df["k"] < k_str]
        if len(df_fit) >= 2:
            mu, b = np.polyfit(np.log(df_fit["k"]), np.log(df_fit["knn"]), 1)
            return mu, b
        return np.nan, np.nan

    mu_orig, b_orig = fit(df_binned)

    # ─────────────────────────────────────────────
    # 6. RANDOMIZATION (DEGREE-PRESERVING)
    # ─────────────────────────────────────────────
    print("Rewiring network (degree-preserving)...")

    G_rand = G_simple.copy()
    nx.double_edge_swap(
        G_rand,
        nswap=L * n_swaps_per_edge,
        max_tries=L * n_swaps_per_edge * 10
    )

    knn_rand = nx.average_degree_connectivity(G_rand)

    degrees_rand = [d for _, d in G_rand.degree()]
    counts_rand = pd.Series(degrees_rand).value_counts().to_dict()

    df_raw_rand = pd.DataFrame([
        {"k": k, "knn": knn_rand[k], "count": counts_rand[k]}
        for k in knn_rand
    ]).sort_values("k")

    df_raw_rand = df_raw_rand[(df_raw_rand["k"] > 0) & (df_raw_rand["knn"] > 0)]

    df_binned_rand = log_bin(df_raw_rand.copy())

    mu_rand, b_rand = fit(df_binned_rand)

    # ─────────────────────────────────────────────
    # 7. PLOT
    # ─────────────────────────────────────────────
    plt.figure(figsize=(15, 6))

    # raw original
    plt.scatter(df_raw["k"], df_raw["knn"],
                color="gray", alpha=0.2, s=12,
                label="Raw (original)")

    # binned original
    plt.scatter(df_binned["k"], df_binned["knn"],
                color="purple", s=40, edgecolor="black",
                label="Binned (original)")

    # binned random
    plt.scatter(df_binned_rand["k"], df_binned_rand["knn"],
                color="orange", marker="D", s=40, edgecolor="black",
                label="Binned (rewired)")

    # fit original
    if not np.isnan(mu_orig):
        k_line = np.linspace(df_binned["k"].min(), k_str, 200)
        plt.plot(k_line, np.exp(b_orig) * k_line**mu_orig,
                 "--", color="mediumpurple",
                 label=f"Original μ={mu_orig:.3f}")

    # fit random
    if not np.isnan(mu_rand):
        k_line = np.linspace(df_binned_rand["k"].min(), k_str, 200)
        plt.plot(k_line, np.exp(b_rand) * k_line**mu_rand,
                 ":", color="orange",
                 label=f"Rewired μ={mu_rand:.3f}")

    # cutoff
    plt.axvline(k_str, color="red", linestyle="--",
                label=f"k_str={k_str:.1f}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree k")
    plt.ylabel("k_nn(k)")
    plt.title("Assortativity with Randomization (Hybrid Network)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)

    if save:
        savefig("knn_assortativity_randomized_hybrid")

    plt.show()

    # ─────────────────────────────────────────────
    # 8. INTERPRETATION
    # ─────────────────────────────────────────────
    diff = abs(mu_orig - mu_rand)

    print("\n========== RESULTS ==========")
    print(f"μ original : {mu_orig:.4f}")
    print(f"μ random   : {mu_rand:.4f}")
    print(f"|Δμ|       : {diff:.4f}")
    print("----------------------------")

    if diff < 0.03:
        print("→ Neutral mixing (structure-driven only)")
    else:
        print("→ Genuine correlations present")

    return {
        "mu_orig": mu_orig,
        "mu_rand": mu_rand,
        "k_str": k_str,
        "k_nat": k_nat
    }






