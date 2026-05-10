"""
Global figure-save utilities for the earthquake network project.

Usage in every script
---------------------
1. At the top of the config block, add flags:
       SAVE_PDF: bool = True
       SAVE_JPG: bool = True

2. After imports, call once:
       setup_matplotlib()
       configure_saves(SAVE_JPG, SAVE_PDF, RESULTS_DIR / "figures" / "<catalog>")
       # <catalog> is "italy", "us", or "comparison"

3. In src/ functions, pass save=True (the default) to enable saving.
   Pass save=False to skip saving for that particular call.

Folder layout created automatically:
    results/figures/<catalog>/pdf/
    results/figures/<catalog>/jpg/
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
        Root directory for this catalog's figures, e.g.
        ``RESULTS_DIR / "figures" / "italy"``.
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
    try:
        import kaleido  # noqa: F401  — presence check only
    except ImportError:
        log.warning("kaleido not installed — skipping static Plotly save for '%s'. "
                    "Run: pip install kaleido", name)
        return
    def _write(fmt: str, path: "Path", **kwargs: object) -> None:
        try:
            fig.write_image(str(path), **kwargs)
            log.info("Saved %s", path)
        except Exception as exc:
            # Tile-based scatter_map figures cannot be rasterised by Kaleido
            # (Error 525: Map error — headless browser has no tile access).
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
