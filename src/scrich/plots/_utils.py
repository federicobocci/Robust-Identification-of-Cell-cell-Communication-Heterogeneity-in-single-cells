"""Internal helpers shared by the plotting functions (colours, fraction matrices)."""

from __future__ import annotations

import functools
import importlib.util
import inspect
import os
import warnings
from collections.abc import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, ListedColormap, to_rgb

from .._settings import settings

_EXTENSION_PALETTES = ("tab20", "tab20b", "tab20c")


def _base_colors(cmap: str | Colormap | Sequence, n: int) -> list:
    """Colours of ``cmap`` in order: the listed colours of a qualitative palette,
    or ``n`` evenly spaced samples of a continuous colormap."""
    if isinstance(cmap, str):
        if cmap in mpl.color_sequences:
            return list(mpl.color_sequences[cmap])
        cmap = mpl.colormaps[cmap]
    if isinstance(cmap, Colormap):
        # Qualitative palettes (Set2, tab20, ...) are small ListedColormaps; large ones (viridis) are continuous.
        if isinstance(cmap, ListedColormap) and len(cmap.colors) <= 32:
            return list(cmap.colors)
        return [cmap(x) for x in np.linspace(0, 1, max(n, 1))]
    return list(cmap)  # explicit list of colours


def discrete_colors(cmap: str | Colormap | Sequence, n: int) -> list[tuple[float, float, float]]:
    """Return ``n`` colours (RGB tuples in 0-1) from a colormap, for ``n`` categories.

    Qualitative palettes are used in their original order. If they have fewer than
    ``n`` colours, colours from ``tab20``, ``tab20b`` and ``tab20c`` are appended,
    and beyond that evenly spaced colours from ``turbo``, so plots never fail or
    silently drop categories when there are many modes, patterns or cell types.

    Parameters
    ----------
    cmap
        Matplotlib colormap, its name (e.g. ``"Set2"``), or a list of colours.
    n
        Number of colours needed.

    Returns
    -------
    list of tuple
        ``n`` RGB colours.
    """
    colors = [to_rgb(c) for c in _base_colors(cmap, n)]
    for name in _EXTENSION_PALETTES:
        if len(colors) >= n:
            break
        colors += [c for c in map(to_rgb, mpl.colormaps[name].colors) if c not in colors]
    if len(colors) < n:
        colors += [to_rgb(c) for c in mpl.colormaps["turbo"](np.linspace(0.05, 0.95, n - len(colors)))]
    return colors[:n]


def _fraction_matrix(adata, pathway, key, order=None):
    """Fraction matrix of a pathway (modes x cell types) with its labels, columns in the requested order.

    Cell types are stored in sorted order (adata.uns['pathways'][pathway]['cell_types']); ``order``
    only changes how they are displayed.
    """
    frac = np.asarray(adata.uns['fraction_mat'][pathway])
    info = adata.uns['pathways'].get(pathway, {})
    cell_types = list(info.get('cell_types', sorted(set(adata.obs[key]))))
    modes = list(info.get('modes', range(frac.shape[0])))
    if order is None:
        return frac, modes, cell_types
    order = list(order)
    missing = [c for c in order if c not in cell_types]
    if missing:
        raise ValueError(f"order contains unknown cell types: {missing}")
    return frac[:, [cell_types.index(c) for c in order]], modes, order


# --------------------------------------------------------------------------- #
# Common plotting interface (scanpy-style): show / ax / return_fig
# --------------------------------------------------------------------------- #
_LEGACY = ("savefig", "showfig", "figname", "format", "dpi")


def _new_axes(ax=None, figsize=None):
    """Axes to draw on: the user's ``ax`` if given, otherwise a new figure with one axes."""
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    plt.sca(ax)
    return ax.figure, ax


def _is_plotly(obj) -> bool:
    return type(obj).__module__.startswith("plotly")


def plot_function(aliases: dict[str, str] | None = None):
    """Decorator giving every ``scrich.pl`` function the same interface.

    New arguments (scanpy style):

    - ``show=None``: show the figure; ``None`` follows ``scrich.settings.autoshow`` (True by default).
    - ``return_fig=False``: return the figure (matplotlib ``Figure`` or plotly figure).

    With ``show=False`` the axes (or list of axes; the plotly figure for alluvial plots) are returned,
    so plots can be modified or saved, e.g. ``ax.figure.savefig('plot.pdf')``.

    The previous arguments ``savefig``, ``showfig``, ``figname``, ``format`` and ``dpi`` still work but are
    deprecated (FutureWarning). ``aliases`` maps renamed arguments (old name -> new name).
    """
    aliases = aliases or {}

    def decorator(func):
        params = inspect.signature(func).parameters
        defaults = {k: params[k].default for k in ("figname", "format", "dpi") if k in params}

        @functools.wraps(func)
        def wrapper(*args, show=None, return_fig=False, **kwargs):
            for old, new in aliases.items():
                if old in kwargs:
                    warnings.warn(f"{func.__name__}: '{old}' is deprecated, use '{new}' instead.",
                                  FutureWarning, stacklevel=2)
                    kwargs.setdefault(new, kwargs.pop(old))
            legacy = {k: kwargs.pop(k) for k in _LEGACY if k in kwargs}
            if legacy:
                warnings.warn(
                    f"{func.__name__}: the arguments {sorted(legacy)} are deprecated. Use show=... and save the "
                    "returned figure instead, e.g. ax = scrich.pl.<plot>(..., show=False); "
                    "ax.figure.savefig('plot.pdf').", FutureWarning, stacklevel=2)
                if show is None and "showfig" in legacy:
                    show = bool(legacy["showfig"])
            # the plotting code itself never saves or shows
            for k in ("savefig", "showfig"):
                if k in params:
                    kwargs[k] = False
            out = func(*args, **kwargs)

            if _is_plotly(out):
                fig, axes = out, out
            else:
                ax = kwargs.get("ax")
                fig = ax.figure if ax is not None else plt.gcf()
                axes = fig.axes[0] if len(fig.axes) == 1 else list(fig.axes)

            if legacy.get("savefig"):
                _legacy_save(fig, {**defaults, **legacy})

            if settings.autoshow if show is None else show:
                fig.show() if _is_plotly(fig) else plt.show()
                result = None
            else:
                result = axes
            if return_fig:
                return fig
            if out is not None and not _is_plotly(out):
                return out  # functions that return data (e.g. return_curve=True)
            return result

        return wrapper

    return decorator


def _legacy_save(fig, opts):
    name = str(opts.get("figname") or "figure")
    fmt = opts.get("format") or None
    if fmt and not os.path.splitext(name)[1]:
        name = f"{name}.{fmt}"
    if _is_plotly(fig):
        if importlib.util.find_spec("kaleido") is None:
            raise ImportError("Saving alluvial plots as static images (pdf/png/svg) requires the 'kaleido' package "
                              "(pip install 'scrich[alluvial]'). Alternatively use fig.write_html('plot.html').")
        fig.write_image(name, format=fmt)
    else:
        dpi = opts.get("dpi")
        fig.savefig(name, format=fmt, dpi=dpi if isinstance(dpi, (int, float)) else 300)
