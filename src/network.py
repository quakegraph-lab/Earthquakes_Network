"""
Network construction functions for the Abe-Suzuki earthquake network.

References
----------
Abe, S., & Suzuki, N. (2004). Scale-free network of earthquakes.
Europhysics Letters, 65(4), 581-586.
"""

import logging
import time
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer

log = logging.getLogger(__name__)


def discretize_space_3d(
    df: pd.DataFrame,
    cell_size_km: float,
    target_crs: str = "epsg:5070",
    info: bool = True,
) -> pd.DataFrame:
    """
    Project geographic coordinates to metric space and assign each earthquake
    to a cubic grid cell.

    Parameters
    ----------
    df : pd.DataFrame
        Earthquake catalog with columns ``latitude``, ``longitude``,
        ``depth_km``.
    cell_size_km : float
        Edge length of each cubic cell in kilometres.
    target_crs : str
        Projected CRS for metric conversion. Default ``"epsg:5070"`` (NAD83 /
        CONUS Albers Equal Area) is correct for the US catalog. Use
        ``"epsg:32632"`` (UTM Zone 32N) for the Italy catalog.

    Returns
    -------
    pd.DataFrame
        New DataFrame (original unchanged) with added columns ``cell_x``,
        ``cell_y``, ``cell_z``, ``cell_id`` (string key ``"cx_cy_cz"``),
        and ``x_km`` / ``y_km`` (projected metric coordinates, unshifted).

    Notes
    -----
    Negative depths (surface-drift artefacts) are kept; they map to
    ``cell_z = -1``.
    Horizontal origin is shifted so that (cell_x, cell_y) ≥ 0 everywhere;
    depth is *not* shifted so that cell_z preserves physical meaning.
    """
    if info:
        log.info("Projecting to %s, cell size %d km ...", target_crs, cell_size_km)

    transformer = Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
    x_m, y_m = transformer.transform(df["longitude"].values, df["latitude"].values)

    x_km = x_m / 1000.0
    y_km = y_m / 1000.0
    z_km = df["depth_km"].values

    x_shifted = x_km - x_km.min()
    y_shifted = y_km - y_km.min()

    cx = pd.Series(np.floor(x_shifted / cell_size_km).astype(int), index=df.index)
    cy = pd.Series(np.floor(y_shifted / cell_size_km).astype(int), index=df.index)
    cz = pd.Series(np.floor(z_km      / cell_size_km).astype(int), index=df.index)

    return df.assign(
        x_km=pd.Series(x_km, index=df.index),
        y_km=pd.Series(y_km, index=df.index),
        cell_x=cx,
        cell_y=cy,
        cell_z=cz,
        cell_id=cx.astype(str) + "_" + cy.astype(str) + "_" + cz.astype(str),
    )

