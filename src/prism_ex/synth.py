"""Synthetic cytometry-shaped data with ground-truth labels.

Section 4.1 supplies no data and permits candidate-generated data. Generated data
is chosen here over a public FCS file for a reason that goes beyond licence
hygiene: the claim asked for in section 2.4 is about where a partition can be
relied on, and on a real file "reliable" can only mean "reproducible under
resampling". With labels in hand it can also mean "right", so the resampling
statistics can themselves be checked against the truth they are standing in for.

The generator is built so that the answer is not uniformly easy. It contains

* four well-separated populations, which any sane method recovers;
* a pair of populations separated by roughly one within-population standard
  deviation, which merge or split depending on resolution -- the honest content
  of the section 2.4 claim;
* a rare population at about 1.5% of events, where sampling noise dominates;
* two channels with no population structure at all, to punish blind inclusion of
  every channel in the subspace.

Marker names are borrowed from immunology only as familiar labels. Nothing here
models biology, and nothing downstream depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prism_ex.fcs.writer import write_fcs

__all__ = ["SyntheticDataset", "make_dataset", "write_demo_file"]

MARKERS = ("FSC-A", "SSC-A", "CD3", "CD4", "CD8", "CD19", "CD56", "Viability", "Time")
"""Column names of the generated files. ``Viability`` and ``Time`` carry no structure."""

CLUSTERING_MARKERS = ("CD3", "CD4", "CD8", "CD19", "CD56")
"""The subspace the README and CLI use by default."""

_COFACTOR = 250.0
"""Scale of the arcsinh transform the generator inverts, as used for fluorescence."""

# name, fraction of events, and mean position in arcsinh space for CD3/4/8/19/56
_POPULATIONS: tuple[tuple[str, float, tuple[float, float, float, float, float]], ...] = (
    ("CD4 T", 0.30, (3.4, 3.3, 0.3, 0.2, 0.2)),
    ("CD8 T", 0.22, (3.4, 0.3, 3.2, 0.2, 0.3)),
    ("B", 0.20, (0.3, 0.2, 0.2, 3.5, 0.2)),
    ("NK", 0.13, (0.4, 0.2, 0.9, 0.2, 3.3)),
    # Sits one within-population sigma from CD4 T: the deliberate ambiguity.
    ("CD4 T activated", 0.135, (3.4, 2.6, 0.4, 0.2, 0.6)),
    ("Rare NKT", 0.015, (3.2, 0.6, 2.0, 0.2, 2.6)),
)
_SPREAD = 0.55
"""Within-population standard deviation in arcsinh space, shared by all markers."""


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Generated events with the labels that produced them."""

    events: np.ndarray
    labels: np.ndarray
    """Integer ground-truth label per event, indexing :attr:`population_names`."""

    population_names: tuple[str, ...]
    markers: tuple[str, ...] = MARKERS

    @property
    def n_events(self) -> int:
        return int(self.events.shape[0])

    def sizes(self) -> dict[str, int]:
        """Ground-truth population sizes, by name."""
        return {
            name: int(np.sum(self.labels == index))
            for index, name in enumerate(self.population_names)
        }


def make_dataset(
    n_events: int = 6000,
    *,
    seed: int = 20260817,
    spread: float = _SPREAD,
) -> SyntheticDataset:
    """Generate ``n_events`` events drawn from the six populations described above.

    Parameters
    ----------
    n_events:
        Total number of events.
    seed:
        Seed for the generator. The same seed returns bit-identical events.
    spread:
        Within-population standard deviation in arcsinh space. Raising it makes
        every population overlap; the ambiguous pair goes first.

    Returns
    -------
    SyntheticDataset
        Events in linear (instrument-like) space, plus the true labels.
    """
    if n_events < len(_POPULATIONS):
        raise ValueError(f"n_events must be at least {len(_POPULATIONS)}")
    rng = np.random.default_rng(seed)

    fractions = np.array([fraction for _, fraction, _ in _POPULATIONS])
    fractions = fractions / fractions.sum()
    counts = _largest_remainder(fractions, n_events)

    label_blocks, marker_blocks = [], []
    for index, ((_, _, centre), count) in enumerate(zip(_POPULATIONS, counts, strict=True)):
        if count == 0:
            continue
        block = rng.normal(loc=centre, scale=spread, size=(count, len(centre)))
        marker_blocks.append(block)
        label_blocks.append(np.full(count, index, dtype=np.int32))

    arcsinh_space = np.vstack(marker_blocks)
    labels = np.concatenate(label_blocks)

    # Back to linear space, where an instrument would have written them.
    fluorescence = np.sinh(np.clip(arcsinh_space, -5.0, 12.0)) * _COFACTOR

    scatter = rng.normal(loc=(60000, 40000), scale=(9000, 7000), size=(n_events, 2))
    viability = rng.gamma(shape=2.0, scale=400.0, size=(n_events, 1))
    time = np.sort(rng.uniform(0, 3.6e5, size=n_events)).reshape(-1, 1)

    events = np.hstack([scatter, fluorescence, viability, time])

    # Shuffle so that event order carries no information about the labels; a
    # partition that recovers the populations from a shuffled file cannot have
    # recovered them from the row index.
    order = rng.permutation(n_events)
    return SyntheticDataset(
        events=events[order],
        labels=labels[order],
        population_names=tuple(name for name, _, _ in _POPULATIONS),
    )


def write_demo_file(
    path: str | Path, n_events: int = 6000, *, seed: int = 20260817
) -> tuple[Path, SyntheticDataset]:
    """Generate a dataset and write it as an FCS 3.1 file.

    Returns both the path and the dataset, so a caller that wants the ground truth
    does not have to regenerate it.
    """
    dataset = make_dataset(n_events, seed=seed)
    written = write_fcs(
        path,
        dataset.events,
        dataset.markers,
        stains=[
            None,
            None,
            "CD3-BV421",
            "CD4-FITC",
            "CD8-APC",
            "CD19-PE",
            "CD56-PE-Cy7",
            None,
            None,
        ],
        extra_keywords={
            "$CYT": "prism-ex synthetic generator",
            "$FIL": Path(path).name,
            "$SRC": f"prism_ex.synth seed={seed}",
        },
    )
    return written, dataset


def _largest_remainder(fractions: np.ndarray, total: int) -> np.ndarray:
    """Apportion ``total`` across ``fractions`` so the counts sum exactly.

    Rounding each fraction independently loses or gains events; the largest
    remainder method keeps the total exact, which matters because ``$TOT`` and the
    length of the DATA segment have to agree.
    """
    exact = fractions * total
    floors = np.floor(exact).astype(int)
    shortfall = total - int(floors.sum())
    if shortfall:
        order = np.argsort(-(exact - floors), kind="stable")
        floors[order[:shortfall]] += 1
    return floors
