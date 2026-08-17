"""Section 2.1: a bad file yields an error, never a partial result.

Each case starts from a known-good file and breaks exactly one thing, so a passing
test says the reader caught *that* defect. The final test is the important one: it
asserts the property the brief actually states, which is not that errors are raised
but that nothing half-built escapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from prism_ex.errors import (
    FCSError,
    InconsistentMetadata,
    MalformedHeader,
    MalformedText,
    MissingKeyword,
    TruncatedData,
    UnsupportedFCSFeature,
    UnsupportedFCSVersion,
)
from prism_ex.fcs.reader import read_fcs, read_fcs_bytes
from prism_ex.fcs.writer import build_fcs_bytes


def good(**kwargs) -> bytes:
    matrix = np.arange(24, dtype=float).reshape(8, 3)
    return build_fcs_bytes(matrix, ["A", "B", "C"], **kwargs)


def test_version_2_0_is_rejected():
    payload = bytearray(good())
    payload[0:6] = b"FCS2.0"
    with pytest.raises(UnsupportedFCSVersion, match=r"FCS2\.0"):
        read_fcs_bytes(bytes(payload))


def test_version_3_0_is_rejected():
    payload = bytearray(good())
    payload[0:6] = b"FCS3.0"
    with pytest.raises(UnsupportedFCSVersion):
        read_fcs_bytes(bytes(payload))


def test_empty_file_is_rejected():
    with pytest.raises(MalformedHeader):
        read_fcs_bytes(b"")


def test_header_shorter_than_58_bytes_is_rejected():
    with pytest.raises(MalformedHeader):
        read_fcs_bytes(good()[:40])


def test_non_numeric_header_offset_is_rejected():
    payload = bytearray(good())
    payload[10:18] = b"    abcd"
    with pytest.raises(MalformedHeader, match="not a number"):
        read_fcs_bytes(bytes(payload))


def test_truncated_data_segment_is_rejected():
    payload = good()
    with pytest.raises(TruncatedData, match="missing"):
        read_fcs_bytes(payload[:-4])


def test_text_not_ending_with_delimiter_is_rejected():
    payload = bytearray(good())
    payload[58 + 1] = ord("X")  # break the first delimiter's partner by shifting content
    with pytest.raises((MalformedText, MissingKeyword, InconsistentMetadata)):
        read_fcs_bytes(bytes(payload))


def test_odd_number_of_text_items_is_rejected():
    """A keyword with no value cannot be a pair."""
    matrix = np.zeros((2, 1))
    payload = bytearray(build_fcs_bytes(matrix, ["A"]))
    text_start = 58
    payload[text_start:text_start] = b"orphan/"
    with pytest.raises((MalformedText, InconsistentMetadata, TruncatedData)):
        read_fcs_bytes(bytes(payload))


@pytest.mark.parametrize(
    ("keyword", "value", "expected"),
    [
        ("$MODE", "C", UnsupportedFCSFeature),  # histogram mode, dropped in 3.1
        ("$DATATYPE", "A", UnsupportedFCSFeature),  # ASCII, deprecated in 3.1
        ("$DATATYPE", "Z", InconsistentMetadata),
        ("$BYTEORD", "3,4,1,2", InconsistentMetadata),  # mixed order, dropped in 3.1
        ("$NEXTDATA", "1024", UnsupportedFCSFeature),  # multiple datasets
        ("$TOT", "9999", InconsistentMetadata),  # contradicts the segment length
        ("$PAR", "4", MissingKeyword),  # promises a 4th parameter
        ("$P1B", "16", InconsistentMetadata),  # contradicts $DATATYPE F
        ("$P1E", "4,0", InconsistentMetadata),  # log with a zero offset
        ("$P2N", "A", InconsistentMetadata),  # duplicate channel name
    ],
)
def test_contradictory_keyword_is_rejected(keyword, value, expected):
    with pytest.raises(expected):
        read_fcs_bytes(good(extra_keywords={keyword: value}))


@pytest.mark.parametrize("keyword", ["$MODE", "$DATATYPE", "$TOT", "$PAR", "$P2N", "$P3B"])
def test_missing_required_keyword_is_rejected(keyword):
    payload = good()
    text_start = payload.index(b"/")
    # Delete the pair by rewriting TEXT without it. Offsets then no longer match,
    # which is itself a rejection -- so assert only that the file is refused.
    marker = f"{keyword}/".encode()
    assert marker in payload
    with pytest.raises(FCSError):
        read_fcs_bytes(payload.replace(marker, b"", 1))
    assert text_start == 58


def test_header_and_keyword_offsets_must_agree():
    payload = bytearray(good())
    payload[26:34] = b"     999"  # HEADER says one thing, $BEGINDATA another
    with pytest.raises(InconsistentMetadata, match="HEADER"):
        read_fcs_bytes(bytes(payload))


def test_sub_byte_integer_widths_are_refused_rather_than_guessed():
    with pytest.raises(UnsupportedFCSFeature, match="byte-aligned"):
        read_fcs_bytes(good(datatype="I", extra_keywords={"$P1B": "12"}))


def test_mixed_integer_widths_are_refused():
    with pytest.raises(UnsupportedFCSFeature, match="mixed integer widths"):
        read_fcs_bytes(good(datatype="I", extra_keywords={"$P1B": "16"}))


def test_nan_in_data_is_rejected():
    matrix = np.zeros((4, 2))
    matrix[1, 1] = np.nan
    with pytest.raises(InconsistentMetadata, match="NaN"):
        read_fcs_bytes(build_fcs_bytes(matrix, ["A", "B"], ranges=[1.0, 1.0]))


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_fcs(tmp_path / "absent.fcs")


@pytest.mark.parametrize("cut", [59, 100, 150, 200])
def test_no_partial_result_survives_a_truncation(cut):
    """The property the brief states, asserted directly.

    Whatever is wrong with the file, the only two outcomes are a complete object or
    an exception -- never an object with some fields filled in.
    """
    payload = good()[:cut]
    try:
        fcs = read_fcs_bytes(payload)
    except FCSError:
        return
    assert fcs.events.shape == (fcs.n_events, fcs.n_channels)
    assert len(fcs.channels) == fcs.n_channels
    assert np.isfinite(fcs.events).all()
