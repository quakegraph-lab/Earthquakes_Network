"""
Link prediction as seismic forecasting for the Abe-Suzuki earthquake network.

The experiment:
  1. Build a "past" network from events up to ``train_end_year``.
  2. Build a "full" network from all events.
  3. Test edges = edges present in the full network but absent in the past
     network AND connecting nodes that already existed in the past network.
     These are connections that a predictor, working only from historical
     data, must anticipate.
  4. Negative samples = equal number of random non-edges between past nodes.
  5. Six predictors are scored and evaluated by AUC-ROC.

Predictors
----------
Common Neighbors  : |N(u) ∩ N(v)|
Adamic-Adar       : Σ 1/log₂(deg(w)) for w in N(u)∩N(v)
Resource Alloc.   : Σ 1/deg(w)
Jaccard           : |N(u)∩N(v)| / |N(u)∪N(v)|
Katz (β=0.005)    : Σ βˡ |paths of length l from u to v| (truncated l≤4)
Pers. PageRank    : PPR score of v when personalizing on u (symmetric mean)

References
----------
Liben-Nowell, D., & Kleinberg, J. (2007). The link-prediction problem for
  social networks. JASIST, 58(7), 1019–1031.
"""

import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics import auc, roc_curve

from src.plotutils import savefig, save_plotly, _slug

log = logging.getLogger(__name__)


# ── Edge-set utilities ────────────────────────────────────────────────────────

def split_edges_temporal(
    G_past: nx.Graph,
    G_full: nx.Graph,
    n_negatives: int | None = None,
    seed: int = 42,
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """
    Derive test-positive and test-negative edge sets from a temporal split.

    Parameters
    ----------
    G_past : nx.Graph
        Network built on the training period (undirected, no self-loops).
    G_full : nx.Graph
        Network built on the full period (undirected, no self-loops).
    n_negatives : int, optional
        Number of negative (non-edge) pairs to sample. Defaults to the
        number of positive test edges.
    seed : int
        RNG seed for negative sampling.

    Returns
    -------
    past_nodes : list[tuple]
        Nodes present in G_past.
    test_pos : list[tuple]
        Edges in G_full but not in G_past, restricted to G_past's node set.
    test_neg : list[tuple]
        Randomly sampled non-edges from G_past's node set.
    """
    past_nodes = set(G_past.nodes())
    past_edges = set(G_past.edges())
    past_edges_sym = past_edges | {(v, u) for u, v in past_edges}

    # Positive: new edges in G_full between nodes already in G_past
    test_pos = [
        (u, v) for u, v in G_full.edges()
        if u in past_nodes and v in past_nodes
        and (u, v) not in past_edges_sym
    ]

    if not test_pos:
        log.warning("No new edges found between past nodes – check year split.")
        return list(past_nodes), [], []

    # Negative: random non-edges (use a set for O(1) membership tests)
    rng = np.random.default_rng(seed)
    n_neg = n_negatives if n_negatives is not None else len(test_pos)
    nodes = list(past_nodes)
    neg_set: set[tuple] = set()
    attempts = 0
    while len(neg_set) < n_neg and attempts < n_neg * 50:
        u, v = rng.choice(nodes, 2, replace=False)
        if u != v and (u, v) not in past_edges_sym and (u, v) not in neg_set:
            neg_set.add((u, v))
        attempts += 1

    test_neg = list(neg_set)
    if len(test_neg) < n_neg:
        log.warning(
            "Negative sampling: requested %d, got %d "
            "(graph too dense or attempt budget exhausted)",
            n_neg, len(test_neg),
        )
    log.info("Link prediction split: %d pos, %d neg test edges",
             len(test_pos), len(test_neg))
    return nodes, test_pos, test_neg


# ── Predictors ────────────────────────────────────────────────────────────────

def _score_nx_predictor(G: nx.Graph, pairs: list[tuple], method: str) -> np.ndarray:
    """Wrapper for NetworkX built-in link predictors."""
    func = {
        "cn":  nx.common_neighbor_centrality,
        "aa":  nx.adamic_adar_index,
        "ra":  nx.resource_allocation_index,
        "jac": nx.jaccard_coefficient,
    }[method]
    ebunch = [(u, v) for u, v in pairs if u in G and v in G]
    scores_map = {(u, v): s for u, v, s in func(G, ebunch)}
    return np.array([scores_map.get((u, v), scores_map.get((v, u), 0.0))
                     for u, v in pairs])


def _score_katz(
    G: nx.Graph,
    pairs: list[tuple],
    beta: float = 0.005,
    max_len: int = 4,
) -> np.ndarray:
    """
    Katz similarity: sum over paths of length 1..max_len weighted by beta^l.
    Uses scipy sparse operations – O(nnz) per power, not O(N²).
    Beta is clamped to < 1/ρ(A) so the series converges.
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr")
    w_max = A.data.max() if A.nnz > 0 else 1.0
    if w_max > 0:
        A = A / w_max

    # Clamp beta to < 1/spectral_radius so the Katz series converges
    try:
        rho = float(eigsh(A, k=1, which="LM", return_eigenvectors=False)[0])
        if rho > 0 and beta >= 1.0 / rho:
            beta = 0.9 / rho
            log.warning("Katz: beta clamped to %.5f (spectral radius ρ=%.3f)", beta, rho)
    except Exception as exc:
        log.debug("Katz: eigsh failed (%s), using beta=%g as-is", exc, beta)

    us = np.array([idx[u] for u, v in pairs if u in idx and v in idx], dtype=np.int32)
    vs = np.array([idx[v] for u, v in pairs if u in idx and v in idx], dtype=np.int32)
    valid = np.array([u in idx and v in idx for u, v in pairs])

    scores = np.zeros(len(us))
    Ak = sp.eye(A.shape[0], format="csr")
    for l in range(1, max_len + 1):
        Ak = Ak @ A
        # np.asarray handles both csr_array (1-D) and csr_matrix ((1,n) matrix)
        scores += (beta ** l) * np.asarray(Ak[us, vs]).ravel()

    result = np.zeros(len(pairs))
    result[valid] = scores
    return result


def _score_ppr(
    G: nx.Graph,
    pairs: list[tuple],
    alpha: float = 0.85,
) -> np.ndarray:
    """
    Personalised PageRank similarity: symmetric mean PPR(u→v) + PPR(v→u).

    Builds the sparse row-stochastic transition matrix W once, then runs
    power iteration per unique source node.  Each source's N-dim PPR vector
    is computed, used to update pair scores, then discarded – peak memory is
    O(N) rather than O(N × |sources|).
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight", format="csr")
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    # D_inv @ A = row-stochastic W; .T gives column-stochastic W_T
    # (use matrix multiply, not element-wise .multiply, which would zero off-diagonal)
    D_inv = sp.diags(1.0 / row_sums)
    W_T = (D_inv @ A).T.tocsr()

    # Index pairs by source so each PPR vector is used immediately and freed
    by_source: dict = defaultdict(list)
    for i, (u, v) in enumerate(pairs):
        if u in idx:
            by_source[u].append((i, v))
        if v in idx:
            by_source[v].append((i, u))

    half_scores = np.zeros(len(pairs))
    e_s = np.zeros(N)
    r = np.empty(N)

    for s, pair_list in by_source.items():
        e_s[:] = 0.0
        e_s[idx[s]] = alpha
        r[:] = 1.0 / N
        for _ in range(100):
            r_new = (1 - alpha) * (W_T @ r) + e_s
            if np.linalg.norm(r_new - r, 1) < 1e-4:
                break
            r = r_new
        else:
            r_new = r  # use last iterate if not converged
        for i, target in pair_list:
            if target in idx:
                half_scores[i] += 0.5 * r_new[idx[target]]

    return half_scores


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_predictors(
    G_past: nx.Graph,
    test_pos: list[tuple],
    test_neg: list[tuple],
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """
    Score all predictors on the test set and return AUC-ROC for each.

    Parameters
    ----------
    G_past : nx.Graph
        Training network (undirected, no self-loops).
    test_pos : list[tuple]
        Positive test edges (new connections in held-out period).
    test_neg : list[tuple]
        Negative test edges (random non-edges).
    methods : list[str], optional
        Subset of ``["CN", "AA", "RA", "Jaccard", "Katz", "PPR"]``.
        Defaults to all six.

    Returns
    -------
    pd.DataFrame
        Columns: ``method``, ``AUC``, ``n_pos``, ``n_neg``.
    """
    if methods is None:
        methods = ["CN", "AA", "RA", "Jaccard", "Katz", "PPR"]

    all_pairs = test_pos + test_neg
    labels = np.array([1] * len(test_pos) + [0] * len(test_neg))

    # Restrict G_past to largest connected component for PPR convergence
    gcc_nodes = max(nx.connected_components(G_past), key=len)
    G_gcc = G_past.subgraph(gcc_nodes).copy()

    results = []
    _map = {"CN": "cn", "AA": "aa", "RA": "ra", "Jaccard": "jac"}

    for method in methods:
        log.info("  Scoring %s...", method)
        print(f"    {method}...", end=" ", flush=True)
        try:
            if method in _map:
                scores = _score_nx_predictor(G_gcc, all_pairs, _map[method])
            elif method == "Katz":
                scores = _score_katz(G_gcc, all_pairs)
            elif method == "PPR":
                scores = _score_ppr(G_gcc, all_pairs)
            else:
                log.warning("Unknown method: %s", method)
                continue

            fpr, tpr, _ = roc_curve(labels, scores)
            auc_score = auc(fpr, tpr)
            print(f"AUC={auc_score:.3f}")
            results.append({"method": method, "AUC": round(auc_score, 4),
                            "n_pos": len(test_pos), "n_neg": len(test_neg)})
        except Exception as exc:
            print(f"FAILED ({exc})")
            log.warning("%s failed: %s", method, exc)
            results.append({"method": method, "AUC": float("nan"),
                            "n_pos": len(test_pos), "n_neg": len(test_neg)})

    return pd.DataFrame(results)


def plot_auc_comparison(
    df_results: pd.DataFrame,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Horizontal bar chart of AUC scores per predictor.

    Bars are coloured by performance: green ≥ 0.7, orange 0.6–0.7, red < 0.6.
    A vertical line at AUC = 0.5 marks random-chance baseline.

    Parameters
    ----------
    df_results : pd.DataFrame
        Output of :func:`evaluate_predictors`.
    title : str
        Figure title suffix.
    """
    df = df_results.dropna(subset=["AUC"]).sort_values("AUC", ascending=True)

    colors = [
        "#2a9d8f" if a >= 0.70 else "#f4a261" if a >= 0.60 else "#e63946"
        for a in df["AUC"]
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(df["method"], df["AUC"], color=colors,
                   edgecolor="k", linewidth=0.6)
    for bar, val in zip(bars, df["AUC"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.axvline(0.5, color="gray", ls="--", lw=1.2, label="Random baseline (0.5)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC-ROC", fontsize=12)
    ax.set_title(f"Link Prediction AUC – {title}", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save:
        savefig(f"link_prediction_auc_{_slug(title)}")
    plt.show()
