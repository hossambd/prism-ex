"""Constants of the FCS 3.1 standard, shared by the reader and the writer.

Keeping them in one module ensures both agree on the specification: the corruption
tests mutate files the writer produced, and divergent constants would make those
tests pass for the wrong reason.

Reference: Spidlen et al., Data File Standard for Flow Cytometry version FCS 3.1,
Cytometry Part A 77A:97-100 (2010).
"""

from __future__ import annotations

VERSION = "FCS3.1"
"""The only version this package accepts."""

HEADER_SIZE = 58
"""6 bytes of version, 4 of padding, then six 8-byte ASCII offset fields."""

OFFSET_FIELD_WIDTH = 8
MAX_HEADER_OFFSET = 99_999_999
"""Offsets beyond this are written as 0 in HEADER and given only as keywords."""

REQUIRED_KEYWORDS = (
    "$BEGINANALYSIS",
    "$BEGINDATA",
    "$BEGINSTEXT",
    "$BYTEORD",
    "$DATATYPE",
    "$ENDANALYSIS",
    "$ENDDATA",
    "$ENDSTEXT",
    "$MODE",
    "$NEXTDATA",
    "$PAR",
    "$TOT",
)
"""Non-parameter keywords FCS 3.1 requires in every primary TEXT segment."""

REQUIRED_PARAMETER_KEYWORDS = ("$P{n}B", "$P{n}E", "$P{n}N", "$P{n}R")
"""Per-parameter keywords required for every n in 1..$PAR."""

LITTLE_ENDIAN = "1,2,3,4"
BIG_ENDIAN = "4,3,2,1"
VALID_BYTEORD = (LITTLE_ENDIAN, BIG_ENDIAN)
"""FCS 3.1 dropped the mixed byte orders that 3.0 permitted."""

SUPPORTED_DATATYPES = ("I", "F", "D")
"""``A`` (ASCII) is valid 3.1 but deprecated; this package rejects it explicitly."""

LIST_MODE = "L"
"""3.1 deprecated correlated and uncorrelated histogram modes."""

MIN_DELIMITER = 1
MAX_DELIMITER = 126
"""The TEXT delimiter must be a single ASCII byte in [1, 126]."""

FLOAT_BITS = {"F": 32, "D": 64}
INTEGER_BITS = (8, 16, 32, 64)
"""Byte-aligned integer widths. Sub-byte bit packing is rejected as unsupported."""
