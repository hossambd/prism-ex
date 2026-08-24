"""Minimal FCS 3.1 writer.

Generates the fixtures used by the test suite and the synthetic datasets in
:mod:`prism_ex.synth`. Writing the format allows corruption tests to be targeted
mutations of a known-good file, so a rejection test demonstrates that the reader
caught the specific defect introduced.

Scope is deliberately narrow: list mode, one dataset, uniform parameter width, no
ANALYSIS segment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from prism_ex.fcs import spec

__all__ = ["build_fcs_bytes", "write_fcs"]

_DEFAULT_DELIMITER = "/"


def build_fcs_bytes(
    events: np.ndarray,
    channel_names: Sequence[str],
    *,
    stains: Sequence[str | None] | None = None,
    datatype: str = "F",
    delimiter: str = _DEFAULT_DELIMITER,
    extra_keywords: Mapping[str, str] | None = None,
    ranges: Sequence[float] | None = None,
) -> bytes:
    """Serialise an event matrix as a complete FCS 3.1 file.

    Parameters
    ----------
    events:
        ``(n_events, n_channels)`` array of measurements.
    channel_names:
        One ``$PnN`` per column; must be unique, since they name the columns.
    stains:
        Optional ``$PnS`` values, ``None`` to omit a given channel's.
    datatype:
        ``"F"`` (float32), ``"D"`` (float64) or ``"I"`` (uint32).
    delimiter:
        The TEXT delimiter. Occurrences inside keywords or values are doubled.
    extra_keywords:
        Additional keywords merged into TEXT, e.g. ``{"$CYT": "synthetic"}``.
        Also the hook the tests use to write deliberately contradictory files.
    ranges:
        Optional ``$PnR`` values; defaults to the observed per-channel maximum.

    Returns
    -------
    bytes
        The file, ready to be written to disk or handed to
        :func:`prism_ex.fcs.reader.read_fcs_bytes`.
    """
    events = np.asarray(events)
    if events.ndim != 2:
        raise ValueError(f"events must be 2-D, got shape {events.shape}")
    n_events, n_par = events.shape
    if len(channel_names) != n_par:
        raise ValueError(f"{len(channel_names)} channel names for {n_par} columns of events")
    datatype = datatype.upper()
    if datatype not in spec.SUPPORTED_DATATYPES:
        raise ValueError(f"datatype must be one of {spec.SUPPORTED_DATATYPES}")

    if datatype == "F":
        payload = events.astype("<f4").tobytes()
        bits = 32
    elif datatype == "D":
        payload = events.astype("<f8").tobytes()
        bits = 64
    else:
        payload = np.rint(events).astype("<u4").tobytes()
        bits = 32

    if ranges is None:
        maxima = events.max(axis=0) if n_events else np.ones(n_par)
        ranges = [float(max(value, 1.0)) for value in maxima]
    if datatype == "I":
        # $PnR is a channel count for integer data, and third-party readers parse it
        # with int(). Writing "4994.01" there is legal-looking and breaks them; the
        # differential test against flowio is what surfaced this.
        ranges = [float(np.ceil(value)) for value in ranges]

    keywords: dict[str, str] = {
        "$BEGINANALYSIS": "0",
        "$ENDANALYSIS": "0",
        "$BEGINSTEXT": "0",
        "$ENDSTEXT": "0",
        "$BYTEORD": spec.LITTLE_ENDIAN,
        "$DATATYPE": datatype,
        "$MODE": spec.LIST_MODE,
        "$NEXTDATA": "0",
        "$PAR": str(n_par),
        "$TOT": str(n_events),
    }
    for position, name in enumerate(channel_names, start=1):
        keywords[f"$P{position}B"] = str(bits)
        keywords[f"$P{position}E"] = "0,0"
        keywords[f"$P{position}N"] = name
        keywords[f"$P{position}R"] = f"{ranges[position - 1]:g}"
        stain = stains[position - 1] if stains else None
        if stain:
            keywords[f"$P{position}S"] = stain
    if extra_keywords:
        keywords.update(extra_keywords)

    data_begin, data_end = _solve_offsets(keywords, payload, delimiter)
    keywords["$BEGINDATA"] = str(data_begin)
    keywords["$ENDDATA"] = str(data_end)
    text = _encode_text(keywords, delimiter)

    text_begin = spec.HEADER_SIZE
    text_end = text_begin + len(text) - 1
    header = _encode_header(text_begin, text_end, data_begin, data_end)
    return header + text + payload


def write_fcs(path: str | Path, events: np.ndarray, channel_names: Sequence[str], **kwargs) -> Path:
    """Write an FCS 3.1 file to ``path`` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_fcs_bytes(events, channel_names, **kwargs))
    return path


def _solve_offsets(keywords: dict[str, str], payload: bytes, delimiter: str) -> tuple[int, int]:
    """Find DATA offsets consistent with a TEXT segment that contains them.

    The length of TEXT depends on the number of digits in ``$BEGINDATA``, which
    depends on the length of TEXT. Iterating to a fixed point takes two or three
    passes; the loop is bounded so a pathological case fails loudly.
    """
    trial = dict(keywords)
    begin = end = 0
    for _ in range(8):
        trial["$BEGINDATA"] = str(begin)
        trial["$ENDDATA"] = str(end)
        text_length = len(_encode_text(trial, delimiter))
        new_begin = spec.HEADER_SIZE + text_length
        new_end = new_begin + len(payload) - 1
        if (new_begin, new_end) == (begin, end):
            return begin, end
        begin, end = new_begin, new_end
    raise RuntimeError("DATA offsets did not converge")  # pragma: no cover


def _encode_text(keywords: Mapping[str, str], delimiter: str) -> bytes:
    if len(delimiter) != 1 or not (spec.MIN_DELIMITER <= ord(delimiter) <= spec.MAX_DELIMITER):
        raise ValueError(f"delimiter must be one ASCII byte in [1, 126], got {delimiter!r}")

    def escape(token: str) -> str:
        return token.replace(delimiter, delimiter * 2)

    body = "".join(
        f"{escape(key)}{delimiter}{escape(value)}{delimiter}" for key, value in keywords.items()
    )
    return (delimiter + body).encode("utf-8")


def _encode_header(text_begin: int, text_end: int, data_begin: int, data_end: int) -> bytes:
    def field(value: int) -> str:
        if value > spec.MAX_HEADER_OFFSET:
            value = 0  # the keywords carry it instead, as FCS 3.1 provides
        return f"{value:>{spec.OFFSET_FIELD_WIDTH}d}"

    header = (
        spec.VERSION
        + "    "
        + field(text_begin)
        + field(text_end)
        + field(data_begin)
        + field(data_end)
        + field(0)
        + field(0)
    )
    assert len(header) == spec.HEADER_SIZE
    return header.encode("ascii")
