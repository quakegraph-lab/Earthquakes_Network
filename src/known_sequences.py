"""
Validate community detection against known earthquake sequences.

A community partition is only useful if its communities mean something
physical.  This module treats well-documented Italian sequences (L'Aquila 2009,
Amatrice-Norcia 2016, …) as ground truth: each is isolated by a space-time box
around its mainshock, and we then check whether the detected communities
*recover* that sequence – i.e. whether the sequence's events land in a single
community rather than being scattered across many.

Pipeline
--------
1. :func:`label_known_sequences` – tag each event with the sequence whose
   space-time box it falls in (``None`` otherwise).
2. :func:`sequence_community_concentration` – per sequence, measure how
   concentrated its events are in one community (purity, normalised entropy,
   and what fraction of the dominant community the sequence actually makes up).
3. :func:`plot_sequence_community_geo` – map a sequence's events coloured by
   their detected community, to see at a glance whether it is one community.

A sequence with **high purity** and a **high share of its dominant community**
is recovered as a single community.  High purity but a tiny share means the sequence was
absorbed into a much larger flow basin (common with InfoMap).
"""

import logging
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px

from src.plotutils import save_plotly, _slug, pres_title

log = logging.getLogger(__name__)

Partition = dict


def label_known_sequences(
    df: pd.DataFrame,
    sequences: list[dict],
    pre_days: float = 2.0,
    seq_col: str = "sequence",
) -> pd.DataFrame:
    """
    Tag each event with the known sequence whose space-time box contains it.

    Each sequence dict must provide ``name``, ``mainshock_time`` (tz-aware
    Timestamp), ``days`` (aftershock window length), ``lat_range`` and
    ``lon_range``.  An event is labelled if it falls within
    ``[t0 - pre_days, t0 + days]`` **and** inside the lat/lon box.  Boxes are
    assumed disjoint; on overlap the first matching sequence in ``sequences``
    wins.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with ``time`` (tz-aware), ``latitude``, ``longitude``.
    sequences : list of dict
        Known-sequence definitions (see above).
    pre_days : float
        Days before the mainshock to include (catches foreshocks).
    seq_col : str
        Name of the output label column.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an added ``seq_col`` column (sequence name or None).
    """
    label = pd.Series(np.full(len(df), None, dtype=object), index=df.index)
    for s in sequences:
        t0 = s["mainshock_time"]
        tmin = t0 - pd.Timedelta(days=pre_days)
        tmax = t0 + pd.Timedelta(days=s["days"])
        la0, la1 = s["lat_range"]
        lo0, lo1 = s["lon_range"]
        mask = (
            (df["time"] >= tmin) & (df["time"] <= tmax)
            & df["latitude"].between(la0, la1)
            & df["longitude"].between(lo0, lo1)
        )
        label.loc[mask & label.isna()] = s["name"]
    return df.assign(**{seq_col: label})


def sequence_community_concentration(
    df_labeled: pd.DataFrame,
    community_map: Partition,
    comm_event_counts: dict | None = None,
    cell_col: str = "cell_id",
    seq_col: str = "sequence",
) -> pd.DataFrame:
    """
    Measure how concentrated each labelled sequence is within one community.

    Parameters
    ----------
    df_labeled : pd.DataFrame
        Output of :func:`label_known_sequences`, also carrying ``cell_col``
        (from ``discretize_space_3d`` at the network's cell size).
    community_map : dict
        ``{cell_id: community_int}`` partition.
    comm_event_counts : dict, optional
        ``{community: total_events_in_community}`` over the whole catalog. When
        given, adds ``seq_frac_of_community`` (how much of the dominant
        community the sequence makes up).
    cell_col, seq_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        One row per sequence: ``n_labeled``, ``n_in_partition``, ``coverage``,
        ``dominant_community``, ``purity``, ``n_communities``, ``norm_entropy``
        and (optionally) ``seq_frac_of_community``.
    """
    rows = []
    sub = df_labeled.dropna(subset=[seq_col])
    for name, g in sub.groupby(seq_col):
        comms = g[cell_col].map(community_map).dropna().astype(int)
        n_labeled, n_in = len(g), len(comms)
        row = {"sequence": name, "n_labeled": n_labeled, "n_in_partition": n_in,
               "coverage": n_in / n_labeled if n_labeled else np.nan}
        if n_in == 0:
            row.update(dominant_community=-1, purity=np.nan, n_communities=0,
                       norm_entropy=np.nan)
            if comm_event_counts is not None:
                row["seq_frac_of_community"] = np.nan
            rows.append(row)
            continue

        vc = comms.value_counts()
        dom, dom_n = int(vc.index[0]), int(vc.iloc[0])
        p = vc / vc.sum()
        ent = float(-(p * np.log(p)).sum())
        norm_ent = ent / np.log(len(vc)) if len(vc) > 1 else 0.0
        row.update(dominant_community=dom, purity=dom_n / n_in,
                   n_communities=int(len(vc)), norm_entropy=norm_ent)
        if comm_event_counts is not None:
            tot = comm_event_counts.get(dom, np.nan)
            row["seq_frac_of_community"] = dom_n / tot if tot else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sequence_community_geo(
    df_labeled: pd.DataFrame,
    community_map: Partition,
    sequence_name: str,
    title: str = "",
    method_name: str = "",
    cell_col: str = "cell_id",
    seq_col: str = "sequence",
    center_lat: float | None = None,
    center_lon: float | None = None,
    zoom: float = 7,
    height: int = 700,
    width: int = 770,
    save: bool = True,
    save_name: str | None = None,
    renderer: str | None = None,
) -> None:
    """
    Map one sequence's events, coloured by their detected community.

    If every point shares a colour the sequence maps to a single community
    (clean recovery); a mix of colours means the partition split the sequence.
    The view auto-centres on the sequence's events unless overridden.

    Pass ``renderer="png"`` (or ``"svg"`` / ``"pdf"``) to render a static image
    instead of the live figure (avoids exhausting the browser WebGL context cap).
    """
    g = df_labeled[df_labeled[seq_col] == sequence_name].copy()
    g["community"] = g[cell_col].map(community_map)
    g = g.dropna(subset=["community"])
    if g.empty:
        log.warning("No partitioned events for sequence '%s'", sequence_name)
        return
    g["community"] = g["community"].astype(int).astype(str)
    # Stringify the timestamp – pandas Timestamps (tz-aware especially) are not
    # JSON-serialisable, so Kaleido's PDF/JPG export pipeline fails on the
    # in-figure hover_data spec. ISO strings round-trip cleanly.
    g["time_str"] = g["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    clat = center_lat if center_lat is not None else float(g["latitude"].mean())
    clon = center_lon if center_lon is not None else float(g["longitude"].mean())

    fig = px.scatter_map(
        g, lat="latitude", lon="longitude",
        color="community",
        color_discrete_sequence=px.colors.qualitative.Bold,
        map_style="carto-positron",
        hover_data={"magnitude": True, "time_str": True},
        title=pres_title(
            f"{sequence_name}: events by community",
            f"{method_name} – {title}" if title else method_name,
        ),
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75))
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        width=width, height=height,
        map=dict(center={"lat": clat, "lon": clon}, zoom=zoom),
    )
    if save:
        save_plotly(fig, save_name or f"seq_{_slug(sequence_name)}_{_slug(method_name)}")
    fig.show(renderer)
