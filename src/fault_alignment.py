"""
Alignment between network communities and DISS seismogenic fault zones.

The DISS Composite Seismogenic Sources are coarsened into a fixed set of macro
fault zones (KMeans on CSS centroids); each on-fault network cell is labelled by
its nearest zone. Normalised Mutual Information between a community partition and
this fault-zone labelling then measures how well the detected communities
reproduce the mapped tectonic segmentation – an external-validation analogue of
NMI-vs-ground-truth.

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


def _coarsen_fault_zones(css_m, n_zones: int, seed: int) -> np.ndarray:
    """KMeans on CSS centroids → macro-zone label per source (cardinality match)."""
    from sklearn.cluster import KMeans

    cent = np.c_[css_m.geometry.centroid.x, css_m.geometry.centroid.y]
    k = int(min(max(n_zones, 1), len(css_m)))
    if k <= 1:
        return np.zeros(len(css_m), dtype=int)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(cent)
    return km.labels_


def fault_zone_labels(
    G: nx.Graph,
    diss_dir,
    n_zones: int = 12,
    buffer_km: float = 15.0,
    target_crs: str = "epsg:32632",
    seed: int = 42,
) -> dict:
    """
    Ground-truth fault-zone label for every on-fault network cell.

    The DISS Composite Seismogenic Sources are coarsened into ``n_zones`` macro
    fault systems (KMeans on CSS centroids); each network cell is assigned the
    zone of its nearest source and kept only if within ``buffer_km`` of a fault.
    A single fixed zoning is returned so the label set is identical for every
    detection method (a fair external ground truth).

    Returns
    -------
    dict
        ``{cell_id: zone_int}`` for on-fault cells only.
    """
    import geopandas as gpd

    faults = load_diss_faults(diss_dir, italy_only=True, with_iss=False)
    css = faults.get("css")
    if css is None or css.empty:
        raise ValueError("DISS CSS layer (csspln331.geojson) not found or empty.")
    css_m = css.to_crs(target_crs).reset_index(drop=True)
    css_m["zone"] = _coarsen_fault_zones(css_m, n_zones, seed)

    rows = [{"cell_id": n, "lat": G.nodes[n]["lat"], "lon": G.nodes[n]["lon"]}
            for n in G.nodes() if "lat" in G.nodes[n]]
    cells = gpd.GeoDataFrame(
        pd.DataFrame(rows),
        geometry=gpd.points_from_xy([r["lon"] for r in rows],
                                    [r["lat"] for r in rows]),
        crs=4326,
    ).to_crs(target_crs)

    join = gpd.sjoin_nearest(
        cells, css_m[["zone", "geometry"]], distance_col="dist_m", how="left")
    join = join[~join.index.duplicated(keep="first")].copy()
    on = join[join["dist_m"] / 1000.0 <= buffer_km]
    return dict(zip(on["cell_id"], on["zone"].astype(int)))


def nmi_vs_fault_zones(
    G: nx.Graph,
    partitions: dict,
    diss_dir,
    n_zones: int = 12,
    buffer_km: float = 15.0,
    target_crs: str = "epsg:32632",
    seed: int = 42,
) -> tuple[pd.Series, int, int]:
    """
    NMI between each method's partition and the DISS fault-zone ground truth.

    Computed on the on-fault cells common to the partition and the fault zoning,
    so it answers "how well does each method reproduce the mapped tectonic
    segmentation?" – the external-validation analogue of NMI-vs-ground-truth.

    Returns
    -------
    (pd.Series, int, int)
        NMI per method (indexed by method name), the number of on-fault cells
        scored, and the number of fault macro-zones used.
    """
    from sklearn.metrics import normalized_mutual_info_score

    zlab = fault_zone_labels(G, diss_dir, n_zones, buffer_km, target_crs, seed)
    scores = {}
    for name, part in partitions.items():
        cells = [c for c in zlab if c in part]
        if len(cells) < 2:
            scores[name] = np.nan
            continue
        truth = [zlab[c] for c in cells]
        pred = [part[c] for c in cells]
        scores[name] = normalized_mutual_info_score(truth, pred)
    return pd.Series(scores, name="nmi_vs_faults"), len(zlab), n_zones


def plot_nmi_vs_fault_zones(
    nmi: pd.Series,
    title: str = "",
    n_cells: int | None = None,
    n_zones: int | None = None,
    save: bool = True,
) -> None:
    """Bar chart of NMI(partition, DISS fault zones) across detection methods."""
    x = np.arange(len(nmi))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, nmi.to_numpy(), color="#5c6bc0", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(nmi.index, rotation=25, ha="right")
    ax.set_ylabel("NMI")
    ax.set_ylim(0, 1)
    ax.set_title("Agreement with DISS fault zones")
    ax.grid(axis="y", alpha=0.3)

    # context (NET_LABEL + sample sizes) as a discreet footnote, not over the title
    foot = " · ".join(s for s in (
        title,
        f"{n_cells:,} on-fault cells" if n_cells is not None else "",
        f"{n_zones} fault zones" if n_zones is not None else "",
    ) if s)
    if foot:
        fig.text(0.99, 0.005, foot, ha="right", va="bottom",
                 fontsize=9, color="#777777")
    fig.tight_layout()
    if save:
        savefig(f"nmi_vs_fault_zones_{_slug(title)}")
    plt.show()

