"""Strict FCS 3.1 reader.

Parsing is implemented here rather than delegated to a general-purpose library
because the requirement is rejection semantics: a malformed, truncated, internally
inconsistent or wrong-version file must raise rather than return a partially
populated result. Established readers are deliberately lenient, since their users
need events out of imperfect instrument files.

``flowio`` is used in the test suite as a differential oracle: for well-formed
files both implementations must agree on the event matrix. It is not a runtime
dependency.

The read is atomic: parsing writes to locals, all validation runs, and
:class:`~prism_ex.fcs.model.FCSFile` is constructed as the final statement. No code
path exposes a half-built object.

Reference: Spidlen et al., Data File Standard for Flow Cytometry version FCS 3.1,
Cytometry Part A 77A:97-100 (2010).
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from prism_ex.errors import (
    InconsistentMetadata,
    MalformedHeader,
    MalformedText,
    MissingKeyword,
    TruncatedData,
    UnsupportedFCSFeature,
    UnsupportedFCSVersion,
)
from prism_ex.fcs import spec
from prism_ex.fcs.model import Channel, FCSFile, Keywords
from prism_ex.provenance import Provenance, sha256_bytes, sha256_file

__all__ = ["read_fcs", "read_fcs_bytes"]


@dataclass(frozen=True, slots=True)
class _Header:
    """The six byte offsets of the HEADER segment, as written on disk."""

    text_begin: int
    text_end: int
    data_begin: int
    data_end: int
    analysis_begin: int
    analysis_end: int


# ----------------------------------------------------------------- entry points


def read_fcs(path: str | Path) -> FCSFile:
    """Read and fully validate an FCS 3.1 file.

    Parameters
    ----------
    path:
        Path to the file.

    Returns
    -------
    FCSFile
        A complete, immutable dataset: keywords, per-channel metadata, and the
        event matrix with named columns.

    Raises
    ------
    UnsupportedFCSVersion
        The file declares a version other than FCS3.1.
    MalformedHeader, MalformedText, MissingKeyword
        The file's structure or required keywords are not readable as 3.1.
    InconsistentMetadata
        The file contradicts itself (offsets, event count, duplicate names).
    TruncatedData
        The DATA segment is shorter than the metadata requires.
    UnsupportedFCSFeature
        Valid 3.1 that this reader deliberately does not implement.

    Examples
    --------
    >>> from prism_ex.synth import write_demo_file          # doctest: +SKIP
    >>> path, _ = write_demo_file("demo.fcs")               # doctest: +SKIP
    >>> fcs = read_fcs(path)                                # doctest: +SKIP
    >>> fcs.n_events, fcs.channel_names[:2]                 # doctest: +SKIP
    (6000, ('FSC-A', 'SSC-A'))
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such FCS file: {path}")
    with open(path, "rb") as handle:
        return _read(handle, source=str(path), digest=sha256_file(path))


def read_fcs_bytes(payload: bytes, *, source: str = "<memory>") -> FCSFile:
    """Read and fully validate an FCS 3.1 file already in memory.

    Used by the HTTP endpoint, which receives an upload rather than a path, and by
    tests that build files without touching the filesystem.
    """
    return _read(io.BytesIO(payload), source=source, digest=sha256_bytes(payload))


# --------------------------------------------------------------------- the read


def _read(handle: BinaryIO, *, source: str, digest: str) -> FCSFile:
    file_size = _file_size(handle)
    header = _read_header(handle, file_size)
    raw_text = _read_segment(handle, header.text_begin, header.text_end, file_size, "TEXT")
    keywords = _parse_text(raw_text)

    # Supplemental TEXT, if the file declares one, is merged before validation so
    # that a file which puts required keywords there is still readable.
    keywords = _merge_supplemental(handle, keywords, file_size)

    _require(keywords, spec.REQUIRED_KEYWORDS)
    _check_global_semantics(keywords)

    n_par = _positive_int(keywords, "$PAR")
    n_tot = _non_negative_int(keywords, "$TOT")
    channels = _build_channels(keywords, n_par)

    data_begin, data_end = _resolve_data_offsets(header, keywords)
    event_bytes = _validate_widths(keywords, channels)
    _check_data_extent(data_begin, data_end, n_tot, event_bytes, file_size)

    # $TOT = 0 is legal -- an empty acquisition is a result, not an error -- and it
    # is the one case where the DATA segment has no bytes to read.
    raw_data = _read_segment(handle, data_begin, data_end, file_size, "DATA") if n_tot else b""
    events = _decode_events(raw_data, keywords, channels, n_tot, n_par)

    events.setflags(write=False)
    return FCSFile(
        keywords=Keywords(keywords),
        channels=tuple(channels),
        events=events,
        provenance=Provenance.build(
            source,
            source_sha256=digest,
            config={"reader": "prism_ex.fcs.reader", "strict": True},
        ),
    )


def _file_size(handle: BinaryIO) -> int:
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    handle.seek(0)
    return size


# ------------------------------------------------------------------------ HEADER


def _read_header(handle: BinaryIO, file_size: int) -> _Header:
    if file_size < spec.HEADER_SIZE:
        raise MalformedHeader(
            f"file is {file_size} bytes; an FCS 3.1 HEADER alone is {spec.HEADER_SIZE}"
        )
    handle.seek(0)
    raw = handle.read(spec.HEADER_SIZE)

    version = raw[:6].decode("ascii", errors="replace")
    if version != spec.VERSION:
        raise UnsupportedFCSVersion(
            f"file declares {version!r}; this package reads {spec.VERSION} only"
        )
    if raw[6:10] != b"    ":
        raise MalformedHeader("bytes 6-9 of the HEADER must be four spaces")

    fields = []
    for position in range(6):
        start = 10 + position * spec.OFFSET_FIELD_WIDTH
        chunk = raw[start : start + spec.OFFSET_FIELD_WIDTH].decode("ascii", "replace")
        text = chunk.strip()
        if not text.isdigit():
            raise MalformedHeader(
                f"HEADER offset field {position} is {chunk!r}, which is not a number"
            )
        fields.append(int(text))

    header = _Header(*fields)
    if header.text_begin < spec.HEADER_SIZE or header.text_end < header.text_begin:
        raise MalformedHeader(
            f"HEADER declares TEXT at [{header.text_begin}, {header.text_end}], "
            "which does not describe a segment after the header"
        )
    return header


def _read_segment(handle: BinaryIO, begin: int, end: int, file_size: int, name: str) -> bytes:
    """Read the inclusive byte range ``[begin, end]``, as FCS offsets are defined."""
    length = end - begin + 1
    if length <= 0:
        raise MalformedHeader(f"{name} segment has non-positive length {length}")
    if end >= file_size:
        raise TruncatedData(f"{name} segment ends at byte {end} but the file is {file_size} bytes")
    handle.seek(begin)
    payload = handle.read(length)
    if len(payload) != length:
        raise TruncatedData(f"{name} segment: expected {length} bytes, read {len(payload)}")
    return payload


# -------------------------------------------------------------------------- TEXT


def _parse_text(raw: bytes) -> dict[str, str]:
    """Parse a TEXT segment into keyword/value pairs.

    The grammar: the first byte is the delimiter, the segment ends with it, and a
    literal delimiter inside a keyword or value is written twice.
    """
    if len(raw) < 2:
        raise MalformedText("TEXT segment is too short to contain a delimiter and a pair")

    delimiter = raw[0:1]
    if not (spec.MIN_DELIMITER <= raw[0] <= spec.MAX_DELIMITER):
        raise MalformedText(
            f"TEXT delimiter is byte {raw[0]}, outside the permitted range "
            f"[{spec.MIN_DELIMITER}, {spec.MAX_DELIMITER}]"
        )
    if not raw.endswith(delimiter):
        raise MalformedText("TEXT segment does not end with its delimiter")

    tokens: list[bytes] = []
    current = bytearray()
    body = raw[1:-1]
    position = 0
    while position < len(body):
        byte = body[position : position + 1]
        if byte == delimiter:
            if body[position + 1 : position + 2] == delimiter:
                current += delimiter  # escaped literal delimiter
                position += 2
                continue
            tokens.append(bytes(current))
            current = bytearray()
            position += 1
            continue
        current += byte
        position += 1
    tokens.append(bytes(current))

    if len(tokens) % 2 != 0:
        raise MalformedText(
            f"TEXT segment contains {len(tokens)} delimited items; keyword/value "
            "pairs require an even number"
        )

    pairs: dict[str, str] = {}
    seen: set[str] = set()
    for index in range(0, len(tokens), 2):
        key = _decode(tokens[index], "keyword")
        value = _decode(tokens[index + 1], "value")
        if not key.strip():
            raise MalformedText("TEXT segment contains an empty keyword")
        if not value:
            # FCS 3.1 forbids empty values; the delimiter grammar cannot express one.
            raise MalformedText(f"keyword {key!r} has an empty value")
        if key.upper() in seen:
            raise InconsistentMetadata(f"keyword {key!r} appears more than once in TEXT")
        seen.add(key.upper())
        pairs[key] = value
    return pairs


def _decode(payload: bytes, what: str) -> str:
    try:
        return payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise MalformedText(f"TEXT {what} is not valid UTF-8: {payload!r}") from exc


def _merge_supplemental(
    handle: BinaryIO, keywords: dict[str, str], file_size: int
) -> dict[str, str]:
    """Merge the supplemental TEXT segment when ``$BEGINSTEXT``/``$ENDSTEXT`` say so."""
    begin = _optional_int(keywords, "$BEGINSTEXT", 0)
    end = _optional_int(keywords, "$ENDSTEXT", 0)
    if begin == 0 and end == 0:
        return keywords
    if begin == 0 or end < begin:
        raise InconsistentMetadata(
            f"supplemental TEXT offsets are [{begin}, {end}], which is not a segment"
        )
    supplemental = _parse_text(_read_segment(handle, begin, end, file_size, "supplemental TEXT"))
    upper = {k.upper() for k in keywords}
    for key, value in supplemental.items():
        if key.upper() in upper:
            raise InconsistentMetadata(
                f"keyword {key!r} appears in both primary and supplemental TEXT"
            )
        keywords[key] = value
    return keywords


# -------------------------------------------------------------------- semantics


def _require(keywords: dict[str, str], required: tuple[str, ...]) -> None:
    upper = {k.upper() for k in keywords}
    missing = [key for key in required if key not in upper]
    if missing:
        raise MissingKeyword(f"FCS 3.1 requires {', '.join(missing)}; not present in TEXT")


def _lookup(keywords: dict[str, str], key: str) -> str:
    wanted = key.upper()
    for name, value in keywords.items():
        if name.upper() == wanted:
            return value
    raise MissingKeyword(f"keyword {key} is required but absent")


def _optional(keywords: dict[str, str], key: str) -> str | None:
    try:
        return _lookup(keywords, key)
    except MissingKeyword:
        return None


def _optional_int(keywords: dict[str, str], key: str, default: int) -> int:
    raw = _optional(keywords, key)
    if raw is None:
        return default
    return _as_int(raw, key)


def _as_int(raw: str, key: str) -> int:
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise InconsistentMetadata(f"{key} is {raw!r}, which is not an integer") from exc


def _positive_int(keywords: dict[str, str], key: str) -> int:
    value = _as_int(_lookup(keywords, key), key)
    if value <= 0:
        raise InconsistentMetadata(f"{key} is {value}; it must be positive")
    return value


def _non_negative_int(keywords: dict[str, str], key: str) -> int:
    value = _as_int(_lookup(keywords, key), key)
    if value < 0:
        raise InconsistentMetadata(f"{key} is {value}; it must not be negative")
    return value


def _check_global_semantics(keywords: dict[str, str]) -> None:
    mode = _lookup(keywords, "$MODE").upper()
    if mode != spec.LIST_MODE:
        raise UnsupportedFCSFeature(
            f"$MODE is {mode!r}; FCS 3.1 deprecated histogram modes and this reader "
            "implements list mode only"
        )

    datatype = _lookup(keywords, "$DATATYPE").upper()
    if datatype == "A":
        raise UnsupportedFCSFeature(
            "$DATATYPE A (ASCII) is deprecated in FCS 3.1 and not implemented"
        )
    if datatype not in spec.SUPPORTED_DATATYPES:
        raise InconsistentMetadata(
            f"$DATATYPE is {datatype!r}; FCS 3.1 permits {', '.join(spec.SUPPORTED_DATATYPES)} or A"
        )

    byteord = _lookup(keywords, "$BYTEORD").strip()
    if byteord not in spec.VALID_BYTEORD:
        raise InconsistentMetadata(
            f"$BYTEORD is {byteord!r}; FCS 3.1 permits only "
            f"{spec.LITTLE_ENDIAN} or {spec.BIG_ENDIAN}"
        )

    if _as_int(_lookup(keywords, "$NEXTDATA"), "$NEXTDATA") != 0:
        raise UnsupportedFCSFeature(
            "$NEXTDATA is non-zero: this file contains multiple datasets, which this "
            "reader does not split. Reading only the first would silently discard data"
        )


def _build_channels(keywords: dict[str, str], n_par: int) -> list[Channel]:
    channels: list[Channel] = []
    for n in range(1, n_par + 1):
        for template in spec.REQUIRED_PARAMETER_KEYWORDS:
            key = template.format(n=n)
            if _optional(keywords, key) is None:
                raise MissingKeyword(f"{key} is required for parameter {n} of {n_par}")

        name = _lookup(keywords, f"$P{n}N")
        bits = _as_int(_lookup(keywords, f"$P{n}B"), f"$P{n}B")
        amplification = _parse_amplification(_lookup(keywords, f"$P{n}E"), n)
        range_ = _as_float(_lookup(keywords, f"$P{n}R"), f"$P{n}R")
        gain_raw = _optional(keywords, f"$P{n}G")
        stain = _optional(keywords, f"$P{n}S")

        channels.append(
            Channel(
                index=n,
                name=name,
                stain=stain or None,
                bits=bits,
                amplification=amplification,
                range_=range_,
                gain=_as_float(gain_raw, f"$P{n}G") if gain_raw is not None else None,
            )
        )

    lowered = [channel.name.casefold() for channel in channels]
    duplicates = sorted({name for name in lowered if lowered.count(name) > 1})
    if duplicates:
        raise InconsistentMetadata(
            f"$PnN values must be unique to name the event columns; repeated: "
            f"{', '.join(duplicates)}"
        )
    return channels


def _parse_amplification(raw: str, n: int) -> tuple[float, float]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise InconsistentMetadata(f"$P{n}E is {raw!r}; expected 'f1,f2'")
    decades = _as_float(parts[0], f"$P{n}E")
    offset = _as_float(parts[1], f"$P{n}E")
    if decades < 0 or offset < 0:
        raise InconsistentMetadata(f"$P{n}E is {raw!r}; both fields must be >= 0")
    if decades > 0 and offset == 0:
        raise InconsistentMetadata(
            f"$P{n}E is {raw!r}: FCS 3.1 requires a non-zero offset when the number "
            "of decades is non-zero, because 0 has no logarithm"
        )
    return decades, offset


def _as_float(raw: str, key: str) -> float:
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise InconsistentMetadata(f"{key} is {raw!r}, which is not a number") from exc


def _validate_widths(keywords: dict[str, str], channels: list[Channel]) -> int:
    """Check ``$PnB`` against ``$DATATYPE`` and return the bytes per event."""
    datatype = _lookup(keywords, "$DATATYPE").upper()

    if datatype in spec.FLOAT_BITS:
        required = spec.FLOAT_BITS[datatype]
        for channel in channels:
            if channel.bits != required:
                raise InconsistentMetadata(
                    f"$P{channel.index}B is {channel.bits} but $DATATYPE {datatype} "
                    f"requires {required}"
                )
            if channel.amplification != (0.0, 0.0):
                raise InconsistentMetadata(
                    f"$P{channel.index}E is {channel.amplification}; floating point "
                    "data must declare linear amplification '0,0'"
                )
    else:  # integer
        widths = {channel.bits for channel in channels}
        for width in widths:
            if width not in spec.INTEGER_BITS:
                raise UnsupportedFCSFeature(
                    f"$PnB of {width} bits is not byte-aligned; this reader implements "
                    f"widths {spec.INTEGER_BITS} and does not unpack bit fields"
                )
        if len(widths) > 1:
            raise UnsupportedFCSFeature(
                f"mixed integer widths {sorted(widths)} in one file are valid FCS but "
                "not implemented here; every parameter must share a $PnB"
            )

    total_bits = sum(channel.bits for channel in channels)
    if total_bits % 8 != 0:  # pragma: no cover - unreachable given the checks above
        raise InconsistentMetadata(f"event width {total_bits} bits is not a whole number of bytes")
    return total_bits // 8


def _resolve_data_offsets(header: _Header, keywords: dict[str, str]) -> tuple[int, int]:
    """Reconcile the HEADER offsets with ``$BEGINDATA``/``$ENDDATA``.

    FCS 3.1 allows the HEADER fields to be 0 when an offset exceeds eight digits,
    in which case the keywords are authoritative. When both are present they must
    agree: a file whose two statements of where its data live disagree is
    internally inconsistent and is rejected.
    """
    keyword_begin = _non_negative_int(keywords, "$BEGINDATA")
    keyword_end = _non_negative_int(keywords, "$ENDDATA")
    header_present = header.data_begin != 0 or header.data_end != 0
    keyword_present = keyword_begin != 0 or keyword_end != 0

    if not header_present and not keyword_present:
        raise InconsistentMetadata(
            "neither the HEADER nor $BEGINDATA/$ENDDATA locates the DATA segment"
        )
    if header_present and keyword_present:
        if (header.data_begin, header.data_end) != (keyword_begin, keyword_end):
            raise InconsistentMetadata(
                f"HEADER puts DATA at [{header.data_begin}, {header.data_end}] but "
                f"$BEGINDATA/$ENDDATA put it at [{keyword_begin}, {keyword_end}]"
            )
        return header.data_begin, header.data_end
    if header_present:
        if max(header.data_begin, header.data_end) > spec.MAX_HEADER_OFFSET:
            raise InconsistentMetadata("HEADER DATA offsets exceed the 8-digit field")
        return header.data_begin, header.data_end
    return keyword_begin, keyword_end


def _check_data_extent(
    data_begin: int, data_end: int, n_tot: int, event_bytes: int, file_size: int
) -> None:
    declared = data_end - data_begin + 1
    expected = n_tot * event_bytes
    if declared != expected:
        raise InconsistentMetadata(
            f"$TOT={n_tot} events of {event_bytes} bytes require {expected} bytes, "
            f"but the DATA segment spans {declared}"
        )
    if data_end >= file_size:
        raise TruncatedData(
            f"DATA ends at byte {data_end} but the file is {file_size} bytes: "
            f"{data_end + 1 - file_size} bytes are missing"
        )


def _decode_events(
    raw: bytes,
    keywords: dict[str, str],
    channels: list[Channel],
    n_tot: int,
    n_par: int,
) -> np.ndarray:
    datatype = _lookup(keywords, "$DATATYPE").upper()
    endian = "<" if _lookup(keywords, "$BYTEORD").strip() == spec.LITTLE_ENDIAN else ">"
    bits = channels[0].bits

    if datatype == "F":
        dtype = np.dtype(f"{endian}f4")
    elif datatype == "D":
        dtype = np.dtype(f"{endian}f8")
    else:
        dtype = np.dtype(f"{endian}u{bits // 8}")

    flat = np.frombuffer(raw, dtype=dtype)
    if flat.size != n_tot * n_par:  # pragma: no cover - width check makes this dead
        raise InconsistentMetadata(f"DATA holds {flat.size} values; $TOT x $PAR is {n_tot * n_par}")
    events = flat.reshape(n_tot, n_par).astype(np.float64, copy=True)

    if not np.all(np.isfinite(events)):
        raise InconsistentMetadata(
            "DATA contains NaN or infinite values, which no valid measurement produces"
        )
    return events
