"""
Global figure-save utilities for the earthquake network project.

Usage in every script
---------------------
1. At the top of the config block, add flags:
       SAVE_PDF: bool = True
       SAVE_JPG: bool = True

2. After imports, call once:
       setup_matplotlib()
       configure_saves(SAVE_JPG, SAVE_PDF, RESULTS_DIR / "figures" / "<country>" / "<method>")
       # country: "italy", "us", "japan", "comparison"
       # method:  "abe", "bp", "tl", "preanalysis"  (omit for "comparison")

3. In src/ functions, pass save=True (the default) to enable saving.
   Pass save=False to skip saving for that particular call.

Folder layout created automatically:
    results/figures/<country>/<method>/pdf/
    results/figures/<country>/<method>/jpg/
    results/figures/<country>/<method>/html/   (Plotly maps, created on demand)
"""

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# ── Module-level state set by configure_saves() ───────────────────────────────
_SAVE_JPG: bool = False
_SAVE_PDF: bool = False
_FIGURES_DIR: Path | None = None

# ── Presentation mode, set by setup_presentation_style() ──────────────────────
# When True, map/plot helpers render slide-ready figures: the project font
# (Latin Modern Sans, matching the UniPD Beamer deck), projection-scale fonts,
# and short titles (the small methodology sub-caption is dropped – see
# ``pres_title``). Left False for the normal screen/print notebooks.
_PRESENTATION: bool = False
# Font used by both matplotlib and the Plotly ``eqpres`` template.
# CMU Sans Serif (Computer Modern Unicode) is visually identical to the UniPD
# Beamer deck's Latin Modern Sans but, unlike lmsans, carries the full Greek
# block — labels use ρ, π, γ, σ, μ, τ, Δ etc., which plain LM Sans silently
# drops in both matplotlib and kaleido. The deck itself renders Greek from its
# *math* font (Computer Modern), so CMU Sans matches that too.
PRES_FONT: str = "CMU Sans Serif"


def pres_title(main: str, sub: str = "") -> str:
    """
    Build a figure title, dropping the methodology sub-caption in presentation mode.

    In the normal notebooks the maps carry a small ``<sup>`` line under the title
    (e.g. *"all 1,800 cells, colour = log₁₀(strength), top-2% links"*) so the
    figure is self-documenting. On a projected slide that line is unreadable and
    redundant with the slide header, so presentation mode returns ``main`` alone.

    Parameters
    ----------
    main : str
        Primary title text.
    sub : str
        Methodology caption (rendered as a smaller ``<sup>`` second line) – kept
        in normal mode, omitted in presentation mode.

    Returns
    -------
    str
        ``main`` in presentation mode (or when ``sub`` is empty); otherwise
        ``"{main}<br><sup>{sub}</sup>"``.
    """
    if _PRESENTATION or not sub:
        return main
    return f"{main}<br><sup>{sub}</sup>"


def setup_matplotlib() -> None:
    """
    Set global rcParams for consistent, publication-quality figures.

    Call once per script, after imports and before any plotting. Multi-panel
    functions may override figsize and individual fontsizes locally where
    space is constrained, but should not change the base sizes set here.
    """
    plt.rcParams.update({
        "font.size":            11,
        "axes.titlesize":       13,
        "axes.labelsize":       11,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "legend.fontsize":      10,
        "figure.titlesize":     14,
        "figure.figsize":       (10, 6),
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.1,
        "axes.grid":            True,
        "grid.alpha":           0.3,
        "grid.linestyle":       "--",
    })
    import warnings
    warnings.filterwarnings("ignore", message=".*tight_layout.*")
    warnings.filterwarnings("ignore", message=".*Axes that are not compatible.*")


def setup_presentation_style(font: str = PRES_FONT) -> None:
    """
    Switch all figures to slide-ready styling for the final presentation.

    Call once at the top of a script, *after* :func:`setup_matplotlib`. It is
    opt-in: scripts that do not call it keep the normal screen/print styling.
    It does three things:

    1. **Font match** – sets matplotlib and the Plotly ``eqpres`` template to
       ``font`` (default *CMU Sans Serif*, the Greek-complete twin of the UniPD
       Beamer deck's Latin Modern Sans – see :data:`PRES_FONT`), and points
       matplotlib mathtext at Computer Modern (``cm``) so in-figure math matches
       the slide math.
    2. **Projection sizes** – enlarges titles/labels/ticks/legends well beyond
       print sizes so they read from the back of the room.
    3. **Light theme** – white background, dark text, faint grid (the deck is a
       light theme; the maps already use the ``carto-positron`` light basemap).

    It also flips the module ``_PRESENTATION`` flag, which makes
    :func:`pres_title` drop the methodology sub-captions (see that function).

    Parameters
    ----------
    font : str
        Font family for every figure. Must be installed/visible to fontconfig
        (so both matplotlib and kaleido can resolve it). Defaults to
        :data:`PRES_FONT`.
    """
    global _PRESENTATION, PRES_FONT
    _PRESENTATION = True
    PRES_FONT = font

    # 1+2+3 – matplotlib: font, projection sizes, light theme, CM math
    plt.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      [font, "Latin Modern Sans", "DejaVu Sans"],
        "mathtext.fontset":     "cm",
        "font.size":            16,
        "axes.titlesize":       20,
        "axes.labelsize":       17,
        "xtick.labelsize":      14,
        "ytick.labelsize":      14,
        "legend.fontsize":      14,
        "figure.titlesize":     22,
        "axes.facecolor":       "white",
        "figure.facecolor":     "white",
        "axes.edgecolor":       "#333333",
        "text.color":           "#222222",
        "axes.labelcolor":      "#222222",
        "xtick.color":          "#222222",
        "ytick.color":          "#222222",
        "grid.alpha":           0.25,
    })

    # 1+2+3 – Plotly: register the eqpres template on top of plotly_white
    import plotly.graph_objects as go
    import plotly.io as pio
    import plotly.express as px

    eqpres = go.layout.Template()
    eqpres.layout.font = dict(family=font, size=16, color="#222222")
    # x=0.45 (not 0.5): nearly every Plotly figure here is a map with a right-side
    # colorbar/legend, so centring over the figure puts the title right of the map's
    # visual centre. The slight left nudge centres it over the map itself.
    eqpres.layout.title = dict(font=dict(family=font, size=22, color="#222222"),
                               x=0.45, xanchor="center")
    eqpres.layout.legend = dict(font=dict(family=font, size=15))
    eqpres.layout.paper_bgcolor = "white"
    eqpres.layout.plot_bgcolor = "white"
    eqpres.layout.colorway = px.colors.qualitative.Bold
    eqpres.layout.coloraxis = dict(colorscale="plasma")
    pio.templates["eqpres"] = eqpres
    pio.templates.default = "plotly_white+eqpres"

    # Note: kaleido default_scale is intentionally left at 1. Saved figures stay
    # crisp because save_plotly() passes scale explicitly (JPG scale=3, PDF is
    # vector). Bumping default_scale only enlarges the inline fig.show() render,
    # which makes notebook output oversized in presentation mode.

    log.info("Presentation style ON (font=%s, light theme, projection sizes)", font)


def configure_saves(
    save_jpg: bool,
    save_pdf: bool,
    figures_dir: Path,
) -> None:
    """
    Configure global save behaviour. Call once at the top of each script.

    Parameters
    ----------
    save_jpg : bool
        Save raster JPG (300 DPI) copies of every figure.
    save_pdf : bool
        Save vector PDF copies of every figure.
    figures_dir : Path
        Root directory for this script's figures, e.g.
        ``RESULTS_DIR / "figures" / "italy" / "abe"``.
        Sub-directories ``pdf/`` and ``jpg/`` are created automatically.
    """
    global _SAVE_JPG, _SAVE_PDF, _FIGURES_DIR
    _SAVE_JPG = save_jpg
    _SAVE_PDF = save_pdf
    _FIGURES_DIR = figures_dir
    if save_jpg or save_pdf:
        (figures_dir / "pdf").mkdir(parents=True, exist_ok=True)
        (figures_dir / "jpg").mkdir(parents=True, exist_ok=True)
        log.info("Figure saving ON → %s  (PDF=%s JPG=%s)",
                 figures_dir, save_pdf, save_jpg)


def _slug(text: str) -> str:
    """Convert a plot title to a safe filename component."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def savefig(name: str) -> None:
    """
    Save the current matplotlib figure before calling plt.show().

    Call this immediately before every ``plt.show()`` inside a src/ function
    or inline in a script. It is a no-op when neither SAVE_PDF nor SAVE_JPG
    is True, so it is safe to leave in place even when saving is off.

    Parameters
    ----------
    name : str
        Base filename (no extension). Descriptive slugs are preferred,
        e.g. ``"degree_distribution_log_binning_italy_10km"``.
    """
    if not (_SAVE_JPG or _SAVE_PDF) or _FIGURES_DIR is None:
        return
    name = re.sub(r"_+", "_", name).strip("_")   # no trailing/double '_' from empty slugs
    if _SAVE_PDF:
        path = _FIGURES_DIR / "pdf" / f"{name}.pdf"
        plt.savefig(path)
        log.info("Saved %s", path)
    if _SAVE_JPG:
        path = _FIGURES_DIR / "jpg" / f"{name}.jpg"
        plt.savefig(path)
        log.info("Saved %s", path)


def save_plotly(fig, name: str) -> None:
    """
    Save a Plotly figure as static PDF and/or JPG before calling fig.show().

    Requires ``kaleido`` (``pip install kaleido``). If kaleido is not
    installed, a warning is logged and the function returns silently so
    ``fig.show()`` still works.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to save.
    name : str
        Base filename (no extension).
    """
    if not (_SAVE_JPG or _SAVE_PDF) or _FIGURES_DIR is None:
        return
    name = re.sub(r"_+", "_", name).strip("_")   # no trailing/double '_' from empty slugs
    try:
        import kaleido  # noqa: F401  – presence check only
    except ImportError:
        log.warning("kaleido not installed – skipping static Plotly save for '%s'. "
                    "Run: pip install kaleido", name)
        return
    def _write(fmt: str, path: "Path", **kwargs: object) -> None:
        try:
            fig.write_image(str(path), **kwargs)
            log.info("Saved %s", path)
        except Exception as exc:
            # Tile-based scatter_map figures cannot be rasterised by Kaleido
            # (Error 525: Map error – headless browser has no tile access).
            # Fall back to HTML which preserves full interactivity.
            html_path = _FIGURES_DIR / "html" / f"{name}.html"
            html_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(html_path))
            log.warning(
                "Kaleido could not export '%s' as %s (%s). "
                "Saved interactive HTML → %s",
                name, fmt, exc, html_path,
            )

    if _SAVE_PDF:
        path = _FIGURES_DIR / "pdf" / f"{name}.pdf"
        _write("PDF", path)
    if _SAVE_JPG:
        path = _FIGURES_DIR / "jpg" / f"{name}.jpg"
        _write("JPG", path, scale=3)
