"""Global settings of scRICH (``scrich.settings``)."""

from __future__ import annotations


class _Settings:
    """Global settings.

    Attributes
    ----------
    autoshow
        If True (default), plotting functions show their figure when ``show`` is not given,
        as in scanpy. Set to False in scripts or on clusters without a display, and save
        figures from the returned objects instead (e.g. ``ax.figure.savefig('plot.pdf')``).
    """

    def __init__(self) -> None:
        self.autoshow: bool = True

    def __repr__(self) -> str:
        return f"scrich.settings(autoshow={self.autoshow})"


settings = _Settings()
