# Earthquake Network Analysis — Italy (Abe–Suzuki Method)

Final project for the 2025–2026 **Network Science** course.

This project models four decades of Italian seismicity (INGV catalog, 1985–2025)
as a complex network. Following Abe & Suzuki (2004), space is divided into cubic
cells and consecutive earthquakes are linked into a directed, weighted network
whose structure reflects fault geometry and stress-transfer pathways. We then ask
what network science can say about seismicity: where the structural hubs are, how
the system breaks down under attack, and — above all — whether the communities
found by graph algorithms correspond to real tectonic structures.

## What's inside

- **`notebooks/network_custom_hybrid.ipynb`** — the main analysis: network
  construction, scale-free degree distribution, centrality, community detection
  (Louvain, InfoMap, HDBSCAN, mixed-membership SBM), assortativity, robustness,
  and validation against the DISS seismogenic-fault database.
- **`notebooks/community_known_eq.ipynb`** — a focused test: do the detected
  communities recover four documented Italian sequences (L'Aquila 2009,
  Amatrice–Norcia 2016, Emilia 2012, Umbria–Marche 1997)?

The notebooks hold the figures and the full commentary — start there.

## The network model

A hybrid Abe–Suzuki construction (30 km cells): two consecutive earthquakes are
linked only if they are close enough in space and time, and each link is weighted
by the events' magnitude and an exponential decay in space and time.

## Reproducing it

```bash
python -m venv venv_earthquakes
source venv_earthquakes/bin/activate
pip install -r requirements.txt
python network_custom_hybrid.py
```

The INGV catalog (`data/INGV/`, M ≥ 1.5) is not included in the repository; see
`data/INGV/data_info.txt` for the download parameters.

## Reference

Abe S. & Suzuki N. (2004). Scale-free network of earthquakes.
*Europhysics Letters* 65(4), 581–586.
