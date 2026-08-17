"""Immutable data structures returned by the reader.

The three things section 2.1 asks for are the three attributes of
:class:`FCSFile`: :attr:`~FCSFile.keywords`, :attr:`~FCSFile.channels` and
:attr:`~FCSFile.events`. Everything here is frozen and the event matrix is
returned read-only, because a result object that a caller can mutate is a result
object whose provenance record is a lie.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from prism_ex.errors import UnknownMarker
from prism_ex.provenance import Provenance


class Keywords(Mapping[str, str]):
    """Case-insensitive read-only view of the TEXT segment.

    FCS keywords are case-insensitive, so ``$par``, ``$PAR`` and ``$Par`` are the
    same keyword; storing them in a plain dict makes that the caller's problem.
    Original spellings are preserved for round-tripping.
    """

    __slots__ = ("_by_upper", "_original")

    def __init__(self, pairs: Mapping[str, str]):
        self._original: dict[str, str] = dict(pairs)
        self._by_upper: dict[str, str] = {k.upper(): v for k, v in pairs.items()}

    def __getitem__(self, key: str) -> str:
        return self._by_upper[key.upper()]

    def __iter__(self) -> Iterator[str]:
        return iter(self._original)

    def __len__(self) -> int:
        return len(self._original)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.upper() in self._by_upper

    def __repr__(self) -> str:
        return f"Keywords({len(self)} keywords)"

    def get_int(self, key: str) -> int:
        """Return a keyword parsed as an integer."""
        return int(self[key].strip())


@dataclass(frozen=True, slots=True)
class Channel:
    """Per-channel metadata, one instance per parameter in the file."""

    index: int
    """1-based parameter number, as used in the ``$Pn*`` keywords."""

    name: str
    """``$PnN``: the short name. Used as the column name of the event matrix."""

    stain: str | None = None
    """``$PnS``: the optional descriptive name, e.g. an antibody label."""

    bits: int = 32
    """``$PnB``: bits reserved per event for this parameter."""

    amplification: tuple[float, float] = (0.0, 0.0)
    """``$PnE``: (decades, offset). ``(0, 0)`` means the data are linear."""

    range_: float = 0.0
    """``$PnR``: the range of the parameter as declared by the acquisition software."""

    gain: float | None = None
    """``$PnG``: optional amplifier gain. Not applied by the reader; see notes."""

    @property
    def is_log(self) -> bool:
        """True when ``$PnE`` declares logarithmic amplification."""
        return self.amplification[0] > 0

    @property
    def label(self) -> str:
        """``name`` when there is no stain, otherwise ``name (stain)``."""
        return f"{self.name} ({self.stain})" if self.stain else self.name


@dataclass(frozen=True, slots=True)
class FCSFile:
    """A fully validated FCS 3.1 dataset.

    An instance of this class exists only if every check in
    :mod:`prism_ex.fcs.reader` passed. There is no partially populated state:
    construction is the last thing the reader does.
    """

    keywords: Keywords
    channels: tuple[Channel, ...]
    events: np.ndarray
    """``(n_events, n_channels)`` float64, read-only, columns ordered by ``$PnN``."""

    provenance: Provenance

    @property
    def n_events(self) -> int:
        return int(self.events.shape[0])

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def channel_names(self) -> tuple[str, ...]:
        """The ``$PnN`` values, in file order. These are the column names."""
        return tuple(channel.name for channel in self.channels)

    def index_of(self, marker: str) -> int:
        """Return the column index of ``marker``, matched on ``$PnN`` then ``$PnS``.

        Matching is case-insensitive and whitespace-insensitive. ``$PnN`` wins over
        ``$PnS`` so that a file where one channel's stain equals another channel's
        name resolves the way the acquisition software meant it.
        """
        wanted = marker.strip().casefold()
        for position, channel in enumerate(self.channels):
            if channel.name.strip().casefold() == wanted:
                return position
        for position, channel in enumerate(self.channels):
            if channel.stain and channel.stain.strip().casefold() == wanted:
                return position
        raise UnknownMarker(
            f"no channel named {marker!r}; available: {', '.join(self.channel_names)}"
        )

    def column(self, marker: str) -> np.ndarray:
        """Return one channel's values as a read-only 1-D array."""
        return self.events[:, self.index_of(marker)]

    def to_dataframe(self) -> Any:
        """Return the events as a :class:`pandas.DataFrame` with named columns.

        pandas is an optional dependency: the core of this package does not need it,
        and a dependency that only exists for a convenience method should not be
        mandatory for a caller who never calls it.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised only without pandas
            raise ImportError(
                "to_dataframe() needs pandas: pip install 'prism-ex[pandas]'"
            ) from exc
        return pd.DataFrame(np.asarray(self.events), columns=list(self.channel_names))

    def summary(self) -> str:
        """One-paragraph human-readable description, used by the CLI."""
        lines = [
            f"FCS 3.1: {self.n_events} events x {self.n_channels} channels",
            f"source: {self.provenance.source}",
        ]
        for channel in self.channels:
            scale = "log" if channel.is_log else "linear"
            lines.append(
                f"  P{channel.index:<3d} {channel.label:<28s} "
                f"{channel.bits:>2d} bit  {scale:<6s} range={channel.range_:g}"
            )
        return "\n".join(lines)
