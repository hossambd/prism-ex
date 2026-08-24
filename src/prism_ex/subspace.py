"""Marker subspace selection.

The subspace is a first-class object rather than an argument threaded through the
clustering functions, so that the subspace used for comparison need not be the one
communities were derived in. Comparing communities on markers held out of the
clustering is more informative than comparing them on the markers that separated
them by construction.

Transform and scaling live here rather than in the reader: cofactors are an
analysis choice, not a property of the file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from prism_ex.errors import AmbiguousMarker, ConfigurationError, UnknownMarker
from prism_ex.fcs.model import FCSFile
from prism_ex.provenance import Provenance

__all__ = ["Subspace", "select_subspace"]

TRANSFORMS = ("none", "asinh")
SCALINGS = ("none", "zscore", "robust")


@dataclass(frozen=True, slots=True)
class Subspace:
    """A matrix of transformed values over a chosen set of markers."""

    matrix: np.ndarray
    """``(n_events, n_markers)`` float64, in the order the caller asked for."""

    markers: tuple[str, ...]
    """Resolved ``$PnN`` names, one per column of :attr:`matrix`."""

    requested: tuple[str, ...]
    """What the caller asked for, before resolution against ``$PnN``/``$PnS``."""

    transform: str
    cofactor: float
    scaling: str
    provenance: Provenance

    @property
    def n_events(self) -> int:
        return int(self.matrix.shape[0])

    def subset(self, rows: np.ndarray) -> Subspace:
        """Return the same subspace restricted to ``rows``.

        Used by the stability analysis, which resamples events but must not
        re-derive the transform: re-standardising a subsample would make the
        resampled distances incomparable to the reference ones.
        """
        return Subspace(
            matrix=self.matrix[rows],
            markers=self.markers,
            requested=self.requested,
            transform=self.transform,
            cofactor=self.cofactor,
            scaling=self.scaling,
            provenance=self.provenance.derive(subset_size=int(np.size(rows))),
        )


def select_subspace(
    fcs: FCSFile,
    markers: Sequence[str],
    *,
    transform: str = "asinh",
    cofactor: float = 250.0,
    scaling: str = "zscore",
) -> Subspace:
    """Build the marker subspace the caller asked for.

    Parameters
    ----------
    fcs:
        A file returned by :func:`prism_ex.fcs.read_fcs`.
    markers:
        Channel names, matched against ``$PnN`` first and ``$PnS`` second,
        case-insensitively. Duplicates in the request are rejected rather than
        silently deduplicated, because a repeated marker doubles that marker's
        weight in every Euclidean distance downstream, and a caller who meant that
        should say so some other way.
    transform:
        ``"asinh"`` (default) or ``"none"``. Fluorescence spans several decades and
        is roughly log-normal within a population; on untransformed values a single
        bright population dominates every distance. ``asinh`` is preferred to
        ``log`` because it is defined at and below zero, where compensated data
        routinely sits.
    cofactor:
        The asinh cofactor. Values well below it are approximately linear, values
        well above approximately logarithmic.
    scaling:
        ``"zscore"`` (default), ``"robust"`` (median/IQR) or ``"none"``. Applied
        after the transform so that each marker contributes comparably to the
        distances the neighbourhood graph is built from.

    Returns
    -------
    Subspace
        The matrix plus the resolved marker names and the settings used.

    Raises
    ------
    UnknownMarker, AmbiguousMarker, ConfigurationError
    """
    if transform not in TRANSFORMS:
        raise ConfigurationError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
    if scaling not in SCALINGS:
        raise ConfigurationError(f"scaling must be one of {SCALINGS}, got {scaling!r}")
    if cofactor <= 0:
        raise ConfigurationError(f"cofactor must be positive, got {cofactor}")
    requested = tuple(markers)
    if not requested:
        raise ConfigurationError("at least one marker is required")

    indices: list[int] = []
    resolved: list[str] = []
    for marker in requested:
        position = _resolve(fcs, marker)
        if position in indices:
            raise AmbiguousMarker(
                f"{marker!r} resolves to channel {fcs.channel_names[position]!r}, "
                "which is already in the subspace"
            )
        indices.append(position)
        resolved.append(fcs.channel_names[position])

    matrix = np.asarray(fcs.events[:, indices], dtype=np.float64)
    if transform == "asinh":
        matrix = np.arcsinh(matrix / cofactor)
    matrix = _scale(matrix, scaling)

    return Subspace(
        matrix=matrix,
        markers=tuple(resolved),
        requested=requested,
        transform=transform,
        cofactor=cofactor,
        scaling=scaling,
        provenance=fcs.provenance.derive(
            markers=list(resolved), transform=transform, cofactor=cofactor, scaling=scaling
        ),
    )


def _resolve(fcs: FCSFile, marker: str) -> int:
    """Resolve one marker name to a column index, reporting ambiguity honestly."""
    wanted = marker.strip().casefold()
    exact = [i for i, c in enumerate(fcs.channels) if c.name.strip().casefold() == wanted]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:  # pragma: no cover - the reader rejects duplicate $PnN
        raise AmbiguousMarker(f"{marker!r} matches {len(exact)} channels by $PnN")

    by_stain = [
        i for i, c in enumerate(fcs.channels) if c.stain and c.stain.strip().casefold() == wanted
    ]
    if len(by_stain) == 1:
        return by_stain[0]
    if len(by_stain) > 1:
        names = ", ".join(fcs.channel_names[i] for i in by_stain)
        raise AmbiguousMarker(f"{marker!r} matches several channels by $PnS: {names}")

    raise UnknownMarker(f"no channel named {marker!r}; available: {', '.join(fcs.channel_names)}")


def _scale(matrix: np.ndarray, scaling: str) -> np.ndarray:
    if scaling == "none":
        return matrix
    if scaling == "zscore":
        centre = matrix.mean(axis=0)
        spread = matrix.std(axis=0)
    else:
        centre = np.median(matrix, axis=0)
        quartiles = np.percentile(matrix, [25, 75], axis=0)
        spread = (quartiles[1] - quartiles[0]) / 1.349  # IQR -> sigma for a normal
    # A constant channel has zero spread; dividing by 1 leaves it constant rather
    # than producing NaN, and a constant column contributes nothing to distances.
    spread = np.where(spread > 0, spread, 1.0)
    return (matrix - centre) / spread
