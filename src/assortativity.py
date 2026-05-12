"""
Assortativity analysis for the Abe-Suzuki earthquake network.

Computes structural (degree-degree) and attribute (depth, magnitude)
assortativity, then visualises the mixing patterns via scatter plots.
Includes three extended diagnostics:

* ``plot_knn``           — average nearest-neighbour degree k_nn(k) vs k,
                           log-log fit giving the mixing exponent μ
* ``plot_directed_mixing`` — four in/out degree mixing panels (Foster 2010)
* ``plot_rich_club``     — normalised rich-club coefficient φ_norm(k)
                           vs configuration-model null (Colizza 2006)

Seismological interpretation
----------------------------
* Degree assortativity r < 0 (disassortative) is the hallmark of scale-free
  networks: hubs connect to low-degree peripheral cells rather than to other
  hubs. This is directly related to the star-like topology of aftershock trees.

* Depth assortativity r > 0 (assortative) would indicate that deep events
  preferentially trigger other deep events — a signature of distinct
  seismogenic depth horizons (e.g. crustal vs subduction-zone seismicity).

* Magnitude assortativity r > 0 would indicate that high-magnitude regions
  cluster together in the temporal sequence — possible evidence of
  mainshock–aftershock structuring.

References
----------
Newman, M. E. J. (2002). Assortative mixing in networks.
  Physical Review Letters, 89(20), 208701.

Pastor-Satorras R., Vázquez A. & Vespignani A. (2001). Dynamical and
  correlation properties of the Internet. PRL 87, 258701.

Foster J.G., Foster D.V., Grassberger P. & Paczuski M. (2010).
  Edge direction and the structure of networks. PNAS 107, 10815–10820.

Colizza V., Flammini A., Serrano M.A. & Vespignani A. (2006).
  Detecting rich-club ordering in complex networks. Nature Physics 2, 110–115.
"""

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


def attach_catalog_attrs(
    G: nx.DiGraph,
    df_net: pd.DataFrame,
    cell_size_km: float = 10.0,
    target_crs: str = "epsg:5070",
) -> None:
    """
    Attach mean magnitude and mean depth to each node of G.

    Re-discretises the catalog (fast: no edge construction) to obtain
    per-cell statistics, then sets ``mean_magnitude`` and ``mean_depth``
    as node attributes. Mutates G in place.

    Parameters
    ----------
    G : nx.DiGraph
        Earthquake network; nodes are cell_id strings.
    df_net : pd.DataFrame
        Full earthquake catalog with columns ``magnitude`` and ``depth_km``.
    cell_size_km : float
        Must match the resolution used to build G.
    target_crs : str
        Projection CRS (``"epsg:5070"`` for US, ``"epsg:32632"`` for Italy).
    """
    from src.network import discretize_space_3d  # noqa: PLC0415

    log.info("Re-discretising catalog for node attribute attachment...")
    df_grid = discretize_space_3d(df_net, cell_size_km=cell_size_km,
                                  target_crs=target_crs)

    agg_cols = {}
    if "magnitude" in df_grid.columns:
        agg_cols["mean_magnitude"] = ("magnitude", "mean")
    if "depth_km" in df_grid.columns:
        agg_cols["mean_depth"] = ("depth_km", "mean")

    if not agg_cols:
        log.warning("No magnitude or depth_km column found in catalog.")
        return

    cell_stats = df_grid.groupby("cell_id").agg(**agg_cols)

    n_attached = 0
    for node in G.nodes():
        if node in cell_stats.index:
            for col in cell_stats.columns:
                G.nodes[node][col] = float(cell_stats.at[node, col])
            n_attached += 1

    log.info("Attached attributes to %d / %d nodes.", n_attached, G.number_of_nodes())


def compute_assortativity(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute degree-degree and attribute assortativity coefficients.

    Uses the *undirected* version of G (removing self-loops) for all
    measures, so that Newman's standard formula applies.

    In addition to the scalar Newman *r*, the table includes the
    **degree-mixing exponent** :math:`\\mu` extracted from the log-log slope of
    :math:`\\bar{k}_{\\mathrm{nn}}(k)` vs :math:`k`
    (Pastor-Satorras *et al.* 2001).  Negative :math:`\\mu` confirms
    disassortativity across the full degree spectrum; the magnitude of
    :math:`\\mu` describes how steeply high-degree nodes avoid other hubs.

    Parameters
    ----------
    G : nx.DiGraph
        Network; may have ``mean_magnitude`` and ``mean_depth`` node attrs
        (set by :func:`attach_catalog_attrs`).

    Returns
    -------
    pd.DataFrame
        Index = attribute name, columns = ``r`` (coefficient) and
        ``interpretation``.

    References
    ----------
    Newman M.E.J. (2002). Assortative mixing in networks.
      *Physical Review Letters* 89, 208701.

    Pastor-Satorras R., Vázquez A. & Vespignani A. (2001). Dynamical and
      correlation properties of the Internet. *PRL* 87, 258701.
    """
    from collections import defaultdict

    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    rows: list[dict] = []

    # ── 1. Structural degree-degree assortativity ────────────────────────────
    r_deg = nx.degree_assortativity_coefficient(G_und)
    rows.append({
        "attribute":     "degree (Newman r)",
        "r":             round(r_deg, 4),
        "interpretation": (
            "disassortative (hubs→periphery, typical scale-free)"
            if r_deg < -0.05 else
            "assortative (hubs→hubs)" if r_deg > 0.05
            else "neutral"
        ),
    })

    # ── 2. Degree-mixing exponent μ from k_nn(k) log-log slope ──────────────
    try:
        knn = nx.average_neighbor_degree(G_und, weight=None)
        degrees = dict(G_und.degree())
        bin_knn: dict[int, list[float]] = defaultdict(list)
        for n, knn_val in knn.items():
            bin_knn[degrees[n]].append(knn_val)
        ks = np.array(sorted(bin_knn.keys()), dtype=float)
        knn_means = np.array([np.mean(bin_knn[int(k)]) for k in ks])
        mask = ks >= 2
        if mask.sum() >= 3:
            mu, _ = np.polyfit(np.log10(ks[mask]), np.log10(knn_means[mask]), 1)
        else:
            mu = float("nan")
    except Exception as exc:
        log.warning("k_nn exponent μ failed: %s", exc)
        mu = float("nan")

    rows.append({
        "attribute":     "degree (k_nn exponent μ)",
        "r":             round(mu, 4),
        "interpretation": (
            f"disassortative (μ={mu:.3f} < 0, hubs avoid hubs across full k range)"
            if not np.isnan(mu) and mu < 0 else
            f"assortative (μ={mu:.3f} > 0)" if not np.isnan(mu) and mu > 0
            else "neutral / undetermined"
        ),
    })

    # ── 3. Attribute assortativity ───────────────────────────────────────────
    for attr, phys_label in [
        ("mean_depth",     "depth (km)"),
        ("mean_magnitude", "magnitude"),
    ]:
        nodes_ok = [n for n in G_und.nodes() if attr in G_und.nodes[n]]
        if len(nodes_ok) < 10:
            log.warning("Skipping %s: too few nodes have the attribute.", attr)
            continue
        H = G_und.subgraph(nodes_ok)
        if H.number_of_edges() < 5:
            log.warning("Skipping %s: too few edges after filtering.", attr)
            continue
        try:
            r_attr = nx.numeric_assortativity_coefficient(H, attr)
        except Exception as exc:
            log.warning("Assortativity for %s failed: %s", attr, exc)
            r_attr = float("nan")
        rows.append({
            "attribute":     phys_label,
            "r":             round(r_attr, 4),
            "interpretation": (
                f"assortative: {phys_label} clusters in time"
                if r_attr > 0.05 else
                f"disassortative: {phys_label} alternates in time"
                if r_attr < -0.05 else
                "neutral"
            ),
        })

    # ── 4. Depth E-I index (homophily by seismogenic layer) ──────────────────
    nodes_with_depth = {
        n: G_und.nodes[n]["mean_depth"]
        for n in G_und.nodes()
        if "mean_depth" in G_und.nodes[n]
    }
    if len(nodes_with_depth) >= 10:
        def _depth_layer(d: float) -> str:
            if d <= 15.0:
                return "shallow"
            if d <= 35.0:
                return "intermediate"
            return "deep"

        e_count = 0
        i_count = 0
        for u, v in G_und.edges():
            du = nodes_with_depth.get(u)
            dv = nodes_with_depth.get(v)
            if du is None or dv is None:
                continue
            if _depth_layer(du) == _depth_layer(dv):
                i_count += 1
            else:
                e_count += 1

        total_ei = e_count + i_count
        ei = (e_count - i_count) / total_ei if total_ei > 0 else float("nan")
        rows.append({
            "attribute": "depth E-I index",
            "r": round(ei, 4),
            "interpretation": (
                "heterophilic (cross-layer triggering dominant, E-I > 0)"
                if not np.isnan(ei) and ei > 0.05 else
                "homophilic (within-layer triggering dominant, E-I < -0.05)"
                if not np.isnan(ei) and ei < -0.05 else
                "neutral layer mixing"
            ),
        })
    else:
        log.warning("Skipping depth E-I index: too few nodes with mean_depth attribute.")

    return pd.DataFrame(rows).set_index("attribute")


def plot_knn(
    G: nx.DiGraph,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Average nearest-neighbour degree k_nn(k) as a function of degree k.

    For a network with degree-degree correlations, the average degree of
    the neighbours of a node with degree :math:`k` is

    .. math::

        \\bar{k}_{\\mathrm{nn}}(k) = \\frac{1}{N_k}\\sum_{i:\\,k_i = k}
            \\frac{1}{k_i}\\sum_{j \\in \\mathcal{N}(i)} k_j,

    where :math:`N_k` is the number of nodes with degree :math:`k`.
    On log-log axes, assortative networks produce an increasing trend
    (slope :math:`\\mu > 0`); disassortative networks a decreasing trend
    (:math:`\\mu < 0`).  The slope :math:`\\mu` is the *degree mixing exponent*
    (Pastor-Satorras *et al.* 2001).

    For directed networks the function is evaluated using *total* degree
    (in + out) so the result is comparable with the undirected coefficient.

    Parameters
    ----------
    G : nx.DiGraph
        Network (directed or undirected).
    title : str
        Figure title suffix.

    References
    ----------
    Pastor-Satorras R., Vázquez A. & Vespignani A. (2001). Dynamical and
    correlation properties of the Internet. *PRL* 87, 258701.

    Newman M.E.J. (2003). Mixing patterns in networks.
    *Phys. Rev. E* 67, 026126.
    """
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    knn = nx.average_neighbor_degree(G_und, weight=None)
    degrees = dict(G_und.degree())

    # Bin by degree, compute mean k_nn per bin
    from collections import defaultdict
    bin_knn: dict[int, list[float]] = defaultdict(list)
    for n, knn_val in knn.items():
        bin_knn[degrees[n]].append(knn_val)

    ks = np.array(sorted(bin_knn.keys()), dtype=float)
    knn_means = np.array([np.mean(bin_knn[k]) for k in ks.astype(int)])

    # Power-law fit in log-log space (only for k ≥ 2 to avoid log(0))
    mask = ks >= 2
    mu, log_c = np.polyfit(np.log10(ks[mask]), np.log10(knn_means[mask]), 1)
    k_fit = np.logspace(np.log10(ks[mask].min()), np.log10(ks[mask].max()), 200)
    knn_fit = 10 ** log_c * k_fit ** mu

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ks, knn_means, s=15, alpha=0.6, color="steelblue", zorder=3,
               label=r"$\bar{k}_{\mathrm{nn}}(k)$ binned mean")
    ax.plot(k_fit, knn_fit, "r--", linewidth=1.8,
            label=rf"Power-law fit $\mu = {mu:.3f}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree $k$", fontsize=12)
    ax.set_ylabel(r"$\bar{k}_{\mathrm{nn}}(k)$", fontsize=12)
    ax.set_title(
        rf"Average nearest-neighbour degree — {title}" + "\n"
        rf"$\mu = {mu:.3f}$ ({'disassortative' if mu < 0 else 'assortative'})",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save:
        savefig(f"knn_degree_mixing_{_slug(title)}")
    plt.show()


def plot_directed_mixing(
    G: nx.DiGraph,
    title: str = "",
    n_bins: int = 20,
    n_edge_samples: int = 5000,
    seed: int = 42,
    save: bool = True,
) -> None:
    """
    Directed degree-mixing heat maps: four panels for all in/out combinations.

    In a directed earthquake network each directed edge :math:`u \\to v`
    represents an event in cell :math:`u` immediately preceding an event in
    cell :math:`v`.  The four degree-mixing panels are:

    * **out→out**: do prolific-triggering cells (high :math:`k^{\\text{out}}`)
      predominantly trigger other prolific cells?
    * **out→in**: do triggering cells connect to cells that are themselves
      heavily triggered?
    * **in→out**: do heavily-triggered cells generate further triggers?
    * **in→in**: are frequently-triggered cells triggered by other
      frequently-triggered cells?

    Each panel is a 2-D histogram (source degree vs target degree) in log-log
    space.  The Pearson :math:`r` is computed on the full edge list before
    subsampling and printed as a subtitle.

    Parameters
    ----------
    G : nx.DiGraph
        Directed earthquake network (self-loops removed internally).
    title : str
        Figure title suffix.
    n_bins : int
        Number of log-spaced bins per axis for the 2-D histogram.
    n_edge_samples : int
        Maximum edges used for the heat map (all edges used for r).
    seed : int
        RNG seed for edge subsampling.

    References
    ----------
    Foster J.G., Foster D.V., Grassberger P. & Paczuski M. (2010).
    Edge direction and the structure of networks. *PNAS* 107, 10815–10820.
    """
    from scipy.stats import pearsonr

    G_dir = G.copy()
    G_dir.remove_edges_from(nx.selfloop_edges(G_dir))

    in_deg  = dict(G_dir.in_degree())
    out_deg = dict(G_dir.out_degree())

    edges_all = list(G_dir.edges())
    rng = np.random.default_rng(seed)
    if len(edges_all) > n_edge_samples:
        idx   = rng.choice(len(edges_all), size=n_edge_samples, replace=False)
        edges_sub = [edges_all[i] for i in idx]
    else:
        edges_sub = edges_all

    panels = [
        ("out", "out", r"$k^{\mathrm{out}}_{\mathrm{src}}$", r"$k^{\mathrm{out}}_{\mathrm{tgt}}$"),
        ("out", "in",  r"$k^{\mathrm{out}}_{\mathrm{src}}$", r"$k^{\mathrm{in}}_{\mathrm{tgt}}$"),
        ("in",  "out", r"$k^{\mathrm{in}}_{\mathrm{src}}$",  r"$k^{\mathrm{out}}_{\mathrm{tgt}}$"),
        ("in",  "in",  r"$k^{\mathrm{in}}_{\mathrm{src}}$",  r"$k^{\mathrm{in}}_{\mathrm{tgt}}$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes_flat = axes.flatten()

    for ax, (src_type, tgt_type, xlabel, ylabel) in zip(axes_flat, panels):
        src_dict = out_deg if src_type == "out" else in_deg
        tgt_dict = out_deg if tgt_type == "out" else in_deg

        src_all = np.array([src_dict[u] for u, _ in edges_all], dtype=float)
        tgt_all = np.array([tgt_dict[v] for _, v in edges_all], dtype=float)
        mask_r  = (src_all > 0) & (tgt_all > 0)
        r = pearsonr(src_all[mask_r], tgt_all[mask_r])[0] if mask_r.sum() > 5 else float("nan")

        src_vals = np.array([src_dict[u] for u, _ in edges_sub], dtype=float)
        tgt_vals = np.array([tgt_dict[v] for _, v in edges_sub], dtype=float)
        mask = (src_vals > 0) & (tgt_vals > 0)
        src_vals, tgt_vals = src_vals[mask], tgt_vals[mask]

        if len(src_vals) < 5:
            ax.set_visible(False)
            continue

        bins_x = np.logspace(np.log10(max(src_vals.min(), 1)),
                             np.log10(src_vals.max() + 1), n_bins + 1)
        bins_y = np.logspace(np.log10(max(tgt_vals.min(), 1)),
                             np.log10(tgt_vals.max() + 1), n_bins + 1)

        h, xedges, yedges = np.histogram2d(src_vals, tgt_vals, bins=[bins_x, bins_y])
        h_log = np.log1p(h.T)
        mesh = ax.pcolormesh(xedges, yedges, h_log, cmap="Blues")
        fig.colorbar(mesh, ax=ax, label="log(1+count)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(rf"{src_type}→{tgt_type} mixing, $r = {r:.3f}$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.2)

    fig.suptitle(f"Directed degree mixing — {title}", fontsize=13)
    plt.tight_layout()
    if save:
        savefig(f"directed_degree_mixing_{_slug(title)}")
    plt.show()


def plot_rich_club(
    G: nx.DiGraph,
    n_null: int = 50,
    seed: int = 42,
    title: str = "",
    save: bool = True,
) -> None:
    """
    Rich-club coefficient :math:`\\phi(k)` with random-null normalisation.

    The rich-club coefficient at degree threshold :math:`k` measures the
    density of edges among the nodes with degree strictly greater than
    :math:`k`:

    .. math::

        \\phi(k) = \\frac{2\\,E_{>k}}{N_{>k}(N_{>k}-1)},

    where :math:`E_{>k}` is the number of edges between such nodes and
    :math:`N_{>k}` their count.  Because high-degree nodes have more
    *potential* edges, :math:`\\phi(k)` is normalised by its expectation
    under a degree-preserving random null (configuration model):

    .. math::

        \\phi_{\\mathrm{norm}}(k) = \\frac{\\phi(k)}{\\langle\\phi_{\\mathrm{rand}}(k)\\rangle}.

    Values :math:`> 1` indicate a **rich club** (hubs cluster together);
    values :math:`< 1` indicate **rich-club absence** (consistent with
    disassortative hub-periphery structure).  For earthquake networks
    dominated by aftershock trees, we expect :math:`\\phi_{\\mathrm{norm}} < 1`
    across all :math:`k` — hubs trigger leaves, not other hubs.

    Parameters
    ----------
    G : nx.DiGraph
        Directed network; converted to undirected internally (NetworkX
        `rich_club_coefficient` is defined for undirected graphs).
    n_null : int
        Number of configuration-model rewirings used to estimate
        :math:`\\langle\\phi_{\\mathrm{rand}}(k)\\rangle`.  Each rewiring uses
        :math:`100 \\times |E|` double-edge swaps.
    seed : int
        RNG seed (passed to ``nx.double_edge_swap``).
    title : str
        Figure title suffix.

    References
    ----------
    Colizza V., Flammini A., Serrano M.A. & Vespignani A. (2006).
    Detecting rich-club ordering in complex networks.
    *Nature Physics* 2, 110–115.

    Zhou S. & Mondragon R.J. (2004). The rich-club phenomenon in the
    Internet topology. *IEEE Commun. Lett.* 8, 180–182.
    """
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    rc_obs = nx.rich_club_coefficient(G_und, normalized=False)
    ks = np.array(sorted(rc_obs.keys()), dtype=int)
    phi_obs = np.array([rc_obs[k] for k in ks], dtype=float)

    log.info("Rich-club: computing %d null rewirings...", n_null)
    rng = np.random.default_rng(seed)
    null_curves: list[np.ndarray] = []
    for _ in range(n_null):
        H = G_und.copy()
        n_swaps = max(100 * H.number_of_edges(), 1)
        try:
            nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 10,
                                seed=int(rng.integers(1 << 31)))
        except nx.NetworkXError:
            pass
        rc_null = nx.rich_club_coefficient(H, normalized=False)
        null_curves.append(np.array([rc_null.get(k, 0.0) for k in ks]))

    null_arr  = np.vstack(null_curves)          # (n_null, len(ks))
    phi_rand  = null_arr.mean(axis=0)
    phi_rand_std = null_arr.std(axis=0)

    # Avoid division by zero
    denom = np.where(phi_rand > 0, phi_rand, np.nan)
    phi_norm = phi_obs / denom
    phi_norm_lo = (phi_obs - phi_rand_std) / denom
    phi_norm_hi = (phi_obs + phi_rand_std) / denom

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: raw phi(k)
    ax = axes[0]
    ax.plot(ks, phi_obs, "o-", color="steelblue", linewidth=1.5, markersize=4,
            label=r"Observed $\phi(k)$")
    ax.fill_between(ks, phi_rand - phi_rand_std, phi_rand + phi_rand_std,
                    alpha=0.3, color="gray", label=r"Null mean ± 1σ")
    ax.plot(ks, phi_rand, "--", color="gray", linewidth=1.2)
    ax.set_xlabel("Degree threshold $k$", fontsize=11)
    ax.set_ylabel(r"$\phi(k)$", fontsize=11)
    ax.set_title("Raw rich-club coefficient", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Right: normalised phi_norm(k)
    ax = axes[1]
    valid = ~np.isnan(phi_norm)
    ax.plot(ks[valid], phi_norm[valid], "o-", color="tomato", linewidth=1.5,
            markersize=4, label=r"$\phi_{\mathrm{norm}}(k)$")
    ax.fill_between(ks[valid], phi_norm_lo[valid], phi_norm_hi[valid],
                    alpha=0.25, color="tomato")
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="Null baseline")
    ax.set_xlabel("Degree threshold $k$", fontsize=11)
    ax.set_ylabel(r"$\phi_{\mathrm{norm}}(k)$", fontsize=11)
    ax.set_title("Normalised rich-club coefficient", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    rich_club_present = bool(np.nanmean(phi_norm[valid]) > 1.0)
    verdict = "rich-club ordering present" if rich_club_present else "rich-club absent (hub–periphery structure)"
    fig.suptitle(f"Rich-club analysis — {title}\n({verdict})", fontsize=12)
    plt.tight_layout()
    if save:
        savefig(f"rich_club_{_slug(title)}")
    plt.show()


def plot_assortativity(
    G: nx.DiGraph,
    title: str = "",
    n_edge_samples: int = 5000,
    seed: int = 42,
    save: bool = True,
) -> None:
    """
    Visualise mixing patterns as edge-source vs edge-target scatter plots.

    Three panels: degree mixing, depth mixing, magnitude mixing.
    Large graphs are subsampled to ``n_edge_samples`` edges for speed.

    Parameters
    ----------
    G : nx.DiGraph
        Network with ``mean_depth`` and ``mean_magnitude`` node attributes.
    title : str
        Figure title suffix.
    n_edge_samples : int
        Max edges to plot per panel.
    seed : int
        RNG seed for edge subsampling.
    """
    G_und = G.to_undirected()
    G_und.remove_edges_from(nx.selfloop_edges(G_und))

    rng   = np.random.default_rng(seed)
    edges = list(G_und.edges())
    if len(edges) > n_edge_samples:
        idx   = rng.choice(len(edges), size=n_edge_samples, replace=False)
        edges = [edges[i] for i in idx]

    panels = [
        ("degree",         "Degree $k$",          lambda n: G_und.degree(n)),
        ("mean_depth",     "Mean Depth (km)",      lambda n: G_und.nodes[n].get("mean_depth", None)),
        ("mean_magnitude", "Mean Magnitude",       lambda n: G_und.nodes[n].get("mean_magnitude", None)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (attr, label, getter) in zip(axes, panels):
        src_vals, tgt_vals = [], []
        for u, v in edges:
            sv, tv = getter(u), getter(v)
            if sv is not None and tv is not None:
                src_vals.append(sv)
                tgt_vals.append(tv)

        if not src_vals:
            ax.set_visible(False)
            continue

        src = np.array(src_vals, dtype=float)
        tgt = np.array(tgt_vals, dtype=float)

        # 2-D histogram for readability
        _, _, _, mesh = ax.hist2d(src, tgt, bins=40, cmap="Blues",
                                  norm=plt.matplotlib.colors.LogNorm())
        fig.colorbar(mesh, ax=ax, label="Count (log)")
        # Regression line
        m, b = np.polyfit(src, tgt, 1)
        x_line = np.linspace(src.min(), src.max(), 100)
        ax.plot(x_line, m * x_line + b, "r-", linewidth=1.5, alpha=0.8)

        # Clip view to 99th percentile — outliers otherwise dominate axis range
        ax.set_xlim(src.min(), np.percentile(src, 99))
        ax.set_ylim(tgt.min(), np.percentile(tgt, 99))

        r, _ = pearsonr(src, tgt)
        ax.set_xlabel(f"Source node — {label}", fontsize=10)
        ax.set_ylabel(f"Target node — {label}", fontsize=10)
        ax.set_title(f"{label} mixing\n$r = {r:.3f}$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(f"Assortativity Mixing Patterns: {title}", fontsize=13)
    plt.tight_layout()
    if save:
        savefig(f"assortativity_{_slug(title)}")
    plt.show()
