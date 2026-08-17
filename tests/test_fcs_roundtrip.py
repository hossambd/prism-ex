"""The reader accepts what the writer produces, and returns it unchanged."""

from __future__ import annotations

import numpy as np
import pytest

from prism_ex.fcs.reader import read_fcs, read_fcs_bytes
from prism_ex.fcs.writer import build_fcs_bytes, write_fcs


@pytest.mark.parametrize(
    ("datatype", "tolerance"),
    [("F", 1e-3), ("D", 1e-12)],
)
def test_float_roundtrip_preserves_values(tiny_matrix, datatype, tolerance):
    payload = build_fcs_bytes(tiny_matrix, ["A", "B", "C"], datatype=datatype)
    fcs = read_fcs_bytes(payload)

    assert fcs.n_events == 4
    assert fcs.channel_names == ("A", "B", "C")
    np.testing.assert_allclose(fcs.events, tiny_matrix, rtol=tolerance)


def test_integer_roundtrip(tiny_matrix):
    fcs = read_fcs_bytes(build_fcs_bytes(tiny_matrix, ["A", "B", "C"], datatype="I"))
    np.testing.assert_array_equal(fcs.events, np.rint(tiny_matrix))


@pytest.mark.parametrize("delimiter", ["/", "|", "\t", "!"])
def test_any_legal_delimiter_is_readable(tiny_matrix, delimiter):
    payload = build_fcs_bytes(tiny_matrix, ["A", "B", "C"], delimiter=delimiter)
    assert read_fcs_bytes(payload).n_events == 4


def test_delimiter_inside_a_value_is_escaped_and_recovered(tiny_matrix):
    """The doubling rule is the one part of the TEXT grammar that is easy to get wrong."""
    payload = build_fcs_bytes(
        tiny_matrix, ["A", "B", "C"], delimiter="/", extra_keywords={"$CYT": "a/b/c"}
    )
    assert read_fcs_bytes(payload).keywords["$CYT"] == "a/b/c"


def test_keywords_are_case_insensitive(tiny_bytes):
    keywords = read_fcs_bytes(tiny_bytes).keywords
    assert keywords["$par"] == keywords["$PAR"] == "3"
    assert "$Tot" in keywords


def test_stains_are_exposed_and_resolvable(tiny_matrix):
    payload = build_fcs_bytes(
        tiny_matrix, ["FL1-A", "FL2-A", "FL3-A"], stains=["CD3", None, "CD19"]
    )
    fcs = read_fcs_bytes(payload)

    assert fcs.channels[0].stain == "CD3"
    assert fcs.channels[1].stain is None
    assert fcs.index_of("CD19") == 2  # by $PnS
    assert fcs.index_of("fl1-a") == 0  # by $PnN, case-insensitively


def test_events_are_read_only(tiny_bytes):
    """Mutating a result would silently invalidate its provenance record."""
    fcs = read_fcs_bytes(tiny_bytes)
    with pytest.raises(ValueError):
        fcs.events[0, 0] = 42.0


def test_provenance_records_the_file_digest(tmp_path, tiny_matrix):
    path = write_fcs(tmp_path / "x.fcs", tiny_matrix, ["A", "B", "C"])
    fcs = read_fcs(path)

    assert (
        fcs.provenance.source_sha256 == __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    )


def test_zero_event_file_is_valid(tmp_path):
    """$TOT of 0 is legal: an empty tube is a result, not an error."""
    payload = build_fcs_bytes(np.empty((0, 2)), ["A", "B"], ranges=[1.0, 1.0])
    fcs = read_fcs_bytes(payload)
    assert fcs.n_events == 0 and fcs.n_channels == 2


def test_summary_mentions_every_channel(demo_file):
    path, _ = demo_file
    summary = read_fcs(path).summary()
    for name in read_fcs(path).channel_names:
        assert name in summary
