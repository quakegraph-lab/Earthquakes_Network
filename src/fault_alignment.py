"""
Quantitative alignment between network communities and DISS seismogenic sources.

Two complementary metrics answer two distinct questions:

A. *Are communities located on faults?* → on-fault fraction (location).
B. *Do communities correspond to fault systems?* → fault-zone purity/entropy,
   tested against a label-permutation null.

Metric definitions
------------------
1. **On-fault fraction** – share of a community's cells within ``buffer_km`` of a
   DISS Composite Seismogenic Source (CSS). Largely a property of where the
   earthquakes are (context), reported per community and globally.

2. **Fault-zone purity / normalised entropy** – each on-fault cell is assigned to
   the nearest *coarsened* CSS macro-zone (KMeans on CSS centroids, so the zone
   cardinality matches the community count and avoids the NMI cardinality trap).
   ``purity`` = fraction of a community's on-fault cells in its dominant zone;
   ``norm_entropy`` ∈ [0, 1] = spread across zones. High purity + low entropy ⇒
   the community sits on a single fault system. Tested against a label-permutation
   null (random partition, same community sizes).

References
----------
DISS Working Group (2021), *Database of Individual Seismogenic Sources (DISS)
v3.3.1*, INGV. Basili et al. (2008) *Tectonophysics* 453, 20–43.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.community_custom import load_diss_faults
from src.plotutils import savefig, _slug

log = logging.getLogger(__name__)

Partition = dict


def _weighted_mean_purity(
    labels: np.ndarray,
    zone: np.ndarray,
    on_fault: np.ndarray,
) -> float:
    """Size-weighted mean fault-zone purity for a (possibly shuffled) labelling."""
    df = pd.DataFrame({"c": labels, "zone": zone, "on": on_fault})
    pur, w = [], []
    for _, g in df.groupby("c"):
        gon = g[g["on"]]
        if len(gon) > 0:
            vc = gon["zone"].value_counts()
            pur.append(vc.iloc[0] / len(gon))
            w.append(len(gon))
    return float(np.average(pur, weights=w)) if pur else np.nan


def _community_cells_gdf(
    G: nx.Graph,
    community_map: Partition,
    min_community_size: int,
    target_crs: str,
):
    """Community cells as a projected GeoDataFrame (community, point geometry)."""
    import geopandas as gpd

    rows = []
    for n in G.nodes():
        if "lat" not in G.nodes[n]:
            continue
        c = community_map.get(n, -1)
        if c == -1:
            continue
        rows.append({"cell_id": n, "community": int(c),
                     "lat": G.nodes[n]["lat"], "lon": G.nodes[n]["lon"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return gpd.GeoDataFrame(df)
    vc = df["community"].value_counts()
    keep = vc[vc >= min_community_size].index
    df = df[df["community"].isin(keep)].copy()
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326,
    ).to_crs(target_crs)
    return gdf


def _coarsen_fault_zones(css_m, n_zones: int, seed: int) -> np.ndarray:
    """KMeans on CSS centroids → macro-zone label per source (cardinality match)."""
    from sklearn.cluster import KMeans

    cent = np.c_[css_m.geometry.centroid.x, css_m.geometry.centroid.y]
    k = int(min(max(n_zones, 1), len(css_m)))
    if k <= 1:
        return np.zeros(len(css_m), dtype=int)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(cent)
    return km.labels_


def analyze_community_fault_alignment(
    G: nx.Graph,
    community_map: Partition,
    diss_dir,
    buffer_km: float = 15.0,
    min_community_size: int = 10,
    n_zones: int | None = None,
    target_crs: str = "epsg:32632",
    n_permutations: int = 300,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Quantify community–fault alignment with on-fault fraction and fault-zone purity.

    Parameters
    ----------
    G : nx.Graph
        Network whose nodes carry ``lat``/``lon`` (the community network).
    community_map : dict
        ``{node: community_id}`` partition.
    diss_dir : str or Path
        Directory with the DISS GeoJSON (needs ``csspln331.geojson``).
    buffer_km : float
        A cell is "on-fault" if within this distance of a CSS polygon.
    min_community_size : int
        Communities smaller than this are dropped (consistent with the maps).
    n_zones : int or None
        Number of coarsened fault macro-zones. Defaults to the community count
        so the two partitions are cardinality-matched.
    target_crs : str
        Projected CRS (metres) for distances and buffering. Use the same CRS as
        the network construction (Italy: ``epsg:32632``).
    n_permutations : int
        Label-shuffle permutations for the purity null.
    seed : int
        RNG seed.

    Returns
    -------
    per_comm : pd.DataFrame
        One row per community: ``n_cells``, ``n_on_fault``, ``on_fault_frac``,
        ``dominant_zone``, ``purity``, ``norm_entropy``, ``n_zones_touched``.
    summary : dict
        Global ``on_fault_frac`` plus observed vs null (mean ± std, z, p) for
        ``purity`` (one-sided high).
    """
    import geopandas as gpd

    faults = load_diss_faults(diss_dir, italy_only=True, with_iss=False)
    css = faults.get("css")
    if css is None or css.empty:
        raise ValueError("DISS CSS layer (csspln331.geojson) not found or empty.")

    css_m = css.to_crs(target_crs).reset_index(drop=True)

    cells = _community_cells_gdf(G, community_map, min_community_size, target_crs)
    if cells.empty:
        raise ValueError("No communities large enough to analyse.")

    n_comm = cells["community"].nunique()
    n_zones = n_zones or n_comm
    css_m["zone"] = _coarsen_fault_zones(css_m, n_zones, seed)

    join = gpd.sjoin_nearest(
        cells, css_m[["zone", "geometry"]], distance_col="dist_m", how="left",
    )
    join = join[~join.index.duplicated(keep="first")].copy()
    join["dist_km"] = join["dist_m"] / 1000.0
    join["on_fault"] = join["dist_km"] <= buffer_km

    # ── per-community metrics ────────────────────────────────────────────────
    rows = []
    for c, g in join.groupby("community"):
        gon = g[g["on_fault"]]
        n_cells, n_on = len(g), len(gon)
        row = {"community": int(c), "n_cells": n_cells, "n_on_fault": n_on,
               "on_fault_frac": n_on / n_cells if n_cells else np.nan}
        if n_on > 0:
            vc = gon["zone"].value_counts()
            p = vc / vc.sum()
            ent = float(-(p * np.log(p)).sum())
            row.update(
                dominant_zone=int(vc.index[0]),
                purity=float(vc.iloc[0] / n_on),
                n_zones_touched=int(len(vc)),
                norm_entropy=ent / np.log(len(vc)) if len(vc) > 1 else 0.0,
            )
        else:
            row.update(dominant_zone=-1, purity=np.nan, n_zones_touched=0,
                       norm_entropy=np.nan)
        rows.append(row)
    per_comm = pd.DataFrame(rows).sort_values("n_cells", ascending=False)

    # ── observed purity + label-permutation null ─────────────────────────────
    labels = join["community"].to_numpy()
    zone = join["zone"].to_numpy()
    on_f = join["on_fault"].to_numpy()

    obs_pur = _weighted_mean_purity(labels, zone, on_f)

    rng = np.random.default_rng(seed)
    null_pur = np.asarray(
        [_weighted_mean_purity(rng.permutation(labels), zone, on_f)
         for _ in range(n_permutations)], dtype=float)

    pur_mu, pur_sd = float(np.nanmean(null_pur)), float(np.nanstd(null_pur))
    pur_z = (obs_pur - pur_mu) / pur_sd if pur_sd > 0 else np.nan
    pur_p = float((np.sum(null_pur >= obs_pur) + 1) / (n_permutations + 1))

    summary = {
        "n_communities": int(n_comm),
        "n_zones": int(css_m["zone"].nunique()),
        "buffer_km": buffer_km,
        "overall_on_fault_frac": float(on_f.mean()),
        "purity_obs": obs_pur, "purity_null_mean": pur_mu,
        "purity_null_std": pur_sd, "purity_z": float(pur_z), "purity_p": pur_p,
    }
    return per_comm, summary


def print_alignment_summary(summary: dict, method_name: str = "") -> None:
    """Readable console summary of :func:`analyze_community_fault_alignment`."""
    s = summary
    print(f"=== Community–fault alignment: {method_name} ===")
    print(f"  communities = {s['n_communities']} | fault macro-zones = {s['n_zones']} "
          f"| buffer = {s['buffer_km']:.0f} km")
    print(f"  on-fault fraction (network on mapped faults): {s['overall_on_fault_frac']:.1%}")
    print(f"  fault-zone purity : obs={s['purity_obs']:.3f}  "
          f"null={s['purity_null_mean']:.3f}±{s['purity_null_std']:.3f}  "
          f"z={s['purity_z']:+.2f}  p={s['purity_p']:.3g}")
    print("  → " + ("communities concentrate on single fault systems"
                     if s["purity_p"] < 0.05 else "no alignment beyond chance") + ".")


def plot_fault_alignment(
    per_comm: pd.DataFrame,
    summary: dict,
    method_name: str = "",
    title: str = "",
    save: bool = True,
) -> None:
    """
    Two-panel bar chart: per-community on-fault fraction and fault-zone purity,
    with the network mean / permutation-null mean drawn as reference lines.
    """
    d = per_comm.copy()
    d["community"] = d["community"].astype(str)
    x = np.arange(len(d))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x, d["on_fault_frac"], color="#5c6bc0")
    axes[0].axhline(summary["overall_on_fault_frac"], ls="--", color="grey",
                    label=f"network mean ({summary['overall_on_fault_frac']:.0%})")
    axes[0].set_ylabel("on-fault fraction")
    axes[0].set_title("A. On-fault location")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, d["purity"], color="#ef6c00")
    axes[1].axhline(summary["purity_null_mean"], ls="--", color="grey",
                    label=f"null ({summary['purity_null_mean']:.2f})")
    axes[1].set_ylabel("fault-zone purity")
    axes[1].set_title(f"B. Fault-zone concentration  (p={summary['purity_p']:.3g})")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(d["community"], rotation=0, fontsize=8)
        ax.set_xlabel("community")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Community–fault alignment – {method_name} – {title}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    if save:
        savefig(f"community_fault_alignment_{_slug(method_name)}_{_slug(title)}")
    plt.show()
