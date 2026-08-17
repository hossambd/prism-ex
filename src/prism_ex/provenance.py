"""Provenance records.

Section 3.2 of the brief asks that the same input, the same configuration and the
same package version return the same answer. That is a property one can claim, or
a property one can make checkable; every result object in this package carries a
:class:`Provenance` naming exactly the three things the guarantee is quantified
over, so "same input" is a SHA-256 rather than a belief about a filename.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, read in chunks so large files are fine."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 of an in-memory buffer."""
    return hashlib.sha256(payload).hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable hash of a configuration mapping.

    Sorted keys and a canonical separator make the hash independent of dict
    insertion order, so two callers who spell the same configuration differently
    get the same identifier.
    """
    canonical = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class Provenance:
    """What produced a result, in enough detail to reproduce or blame it."""

    source: str
    """Origin of the data: a file path, or ``"<memory>"``."""

    source_sha256: str | None = None
    """Digest of the source file, when the source was a file."""

    package_version: str = ""
    """Version of :mod:`prism_ex` that produced the result."""

    config: dict[str, Any] = field(default_factory=dict)
    """The configuration actually used, after defaults were applied."""

    environment: dict[str, str] = field(default_factory=dict)
    """Interpreter and platform, for when a result differs across machines."""

    @property
    def config_id(self) -> str:
        """Short stable identifier for :attr:`config`."""
        return config_hash(self.config)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, with the config id materialised."""
        payload = asdict(self)
        payload["config_id"] = self.config_id
        return payload

    @classmethod
    def build(
        cls,
        source: str | Path,
        *,
        source_sha256: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> Provenance:
        """Construct a record, filling in version and environment automatically."""
        from prism_ex import __version__

        return cls(
            source=str(source),
            source_sha256=source_sha256,
            package_version=__version__,
            config=dict(config or {}),
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(terse=True),
            },
        )

    def derive(self, **config_updates: Any) -> Provenance:
        """Return a copy carrying the same source but an extended configuration."""
        merged = {**self.config, **config_updates}
        return Provenance(
            source=self.source,
            source_sha256=self.source_sha256,
            package_version=self.package_version,
            config=merged,
            environment=dict(self.environment),
        )
