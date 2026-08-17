"""FCS 3.1 reading and writing.

The reader is strict and atomic (see :mod:`prism_ex.fcs.reader`); the writer
exists to generate fixtures and synthetic data (see :mod:`prism_ex.fcs.writer`).
"""

from prism_ex.fcs.model import Channel, FCSFile, Keywords
from prism_ex.fcs.reader import read_fcs, read_fcs_bytes
from prism_ex.fcs.writer import build_fcs_bytes, write_fcs

__all__ = [
    "Channel",
    "FCSFile",
    "Keywords",
    "build_fcs_bytes",
    "read_fcs",
    "read_fcs_bytes",
    "write_fcs",
]
