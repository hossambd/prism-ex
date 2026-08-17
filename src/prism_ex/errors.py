"""Typed error hierarchy.

Every failure mode the package can produce is a subclass of :class:`PrismExError`,
so a caller can distinguish "your file is not FCS 3.1" from "your marker name is
not in this file" without parsing message strings.

The FCS branch is deliberately fine-grained: section 2.1 of the brief requires
that a malformed, truncated, internally inconsistent or wrong-version file does
not yield a partially populated result, and a caller who wants to sort a drawer
of files into "readable" and "why not" needs the reason, not just the failure.
"""

from __future__ import annotations


class PrismExError(Exception):
    """Base class for every error raised by this package."""


# --------------------------------------------------------------------------- FCS


class FCSError(PrismExError):
    """Base class for anything wrong with an FCS file."""


class UnsupportedFCSVersion(FCSError):
    """The file declares a version other than FCS3.1.

    Earlier versions are out of scope (brief, section 2.1); they are rejected
    rather than best-effort parsed, because a 2.0 file read under 3.1 rules can
    succeed and be silently wrong.
    """


class MalformedHeader(FCSError):
    """The 58-byte HEADER is absent, short, or contains non-numeric offsets."""


class MalformedText(FCSError):
    """The TEXT segment does not obey the delimiter grammar of FCS 3.1."""


class MissingKeyword(FCSError):
    """A keyword required by FCS 3.1 is absent."""


class InconsistentMetadata(FCSError):
    """The file's own metadata contradicts itself.

    Examples: HEADER and ``$BEGINDATA`` disagree; ``$TOT`` * event width does not
    match the length of the DATA segment; two parameters share a ``$PnN``.
    """


class TruncatedData(FCSError):
    """The DATA segment is shorter than the metadata says it is."""


class UnsupportedFCSFeature(FCSError):
    """Valid FCS 3.1, but a corner this package deliberately does not implement.

    Distinct from :class:`InconsistentMetadata` on purpose: this one says "your
    file is fine, I am not", which is a different conversation.
    """


# ------------------------------------------------------------------- Downstream


class MarkerError(PrismExError):
    """Base class for marker-subspace selection problems."""


class UnknownMarker(MarkerError):
    """A requested marker matches no ``$PnN`` or ``$PnS`` in the file."""


class AmbiguousMarker(MarkerError):
    """A requested marker matches more than one channel."""


class ConfigurationError(PrismExError):
    """A parameter combination that cannot be honoured (e.g. k >= n_events)."""


class CommunityNotFound(PrismExError):
    """A community id was requested that the partition does not contain."""


class InsufficientData(PrismExError):
    """Not enough events remain to compute what was asked for."""
