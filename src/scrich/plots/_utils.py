"""Internal helpers shared by the plotting functions (colours, fraction matrices)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, ListedColormap, to_rgb

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
