# Earthquake Network Analysis – Italy (Abe–Suzuki Method)

Final project for the 2025–2026 Network Science course.

This project represents four decades of Italian seismicity (INGV catalog,
1985–2025) as a complex network. Following Abe & Suzuki (2004), space is divided
into cubic cells and consecutive earthquakes are linked into a directed, weighted
network whose structure reflects fault geometry and stress-transfer pathways. The
network is then studied with standard network-analysis tools (degree
distribution, centrality, community detection), and the resulting communities are
compared with documented seismic sequences and the DISS fault database.

## What's inside
- `ITALY_preanalysis.ipynb` - the preanalysis of data: Temporal Distribution, 
   Magnitude and Depth Analysis, Gutenberg-Richter Law, Omori Law — Amatrice 2016.
- `abe_suzuki_network_analysis.ipynb` – the main analysis: network
  construction, degree distribution and power-law fit, centrality, community
  detection (Louvain, InfoMap, HDBSCAN, mixed-membership SBM), assortativity,
  robustness, and a comparison with the DISS seismogenic-fault database.
- `community_known_eq.ipynb` – a focused check of whether the detected
  communities recover four documented Italian sequences (L'Aquila 2009,
  Amatrice–Norcia 2016, Emilia 2012, Umbria–Marche 1997).
- `network_science_presentation.pdf` - the final presentation to expose.

The notebooks contain the figures and a brief discussion.

## The network model

A hybrid Abe–Suzuki construction (30 km cells): two consecutive earthquakes are
linked only if they are close enough in space and time, and each link is weighted
by the events' magnitude and an exponential decay in space and time.

## Reproducing it

```bash
python -m venv venv_earthquakes
source venv_earthquakes/bin/activate
pip install -r requirements.txt
jupyter-notebook . &
```

The INGV catalog (`data/INGV/`, M ≥ 1.5) is included in the repository; see
`data/INGV/data_info.txt` for the download parameters used.

## Reference

Abe S. & Suzuki N. (2004). Scale-free network of earthquakes.
*Europhysics Letters* 65(4), 581–586.
