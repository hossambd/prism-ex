"""prism-ex: FCS 3.1 ingestion, community detection, comparison and stability.

A small package answering the four results of the project,
in the order they build on each other:

* :func:`prism_ex.fcs.read_fcs` -- strict FCS 3.1 ingestion (2.1)
* :func:`prism_ex.pipeline.find_communities` -- neighbourhood graph and communities (2.2)
* :func:`prism_ex.compare.compare_communities` -- quantitative marker comparison (2.3)
* :func:`prism_ex.stability.assess_stability` -- evidence for the stability claim (2.4)

Example
-------
>>> from prism_ex import write_demo_file, find_communities_in_file
>>> path, truth = write_demo_file("demo.fcs")             # doctest: +SKIP
>>> result = find_communities_in_file(path, ["CD3", "CD4", "CD8", "CD19", "CD56"])
...                                                        # doctest: +SKIP
>>> result.sizes                                           # doctest: +SKIP
{0: 1812, 1: 1320, 2: 1200, 3: 810, 4: 780, 5: 78}
"""

from prism_ex.communities import Partition, detect_communities
from prism_ex.errors import PrismExError
from prism_ex.fcs import FCSFile, read_fcs, read_fcs_bytes, write_fcs
from prism_ex.graph import build_graph
from prism_ex.pipeline import (
    CommunityConfig,
    CommunityResult,
    find_communities,
    find_communities_in_file,
)
from prism_ex.subspace import select_subspace
from prism_ex.synth import make_dataset, write_demo_file

__version__ = "0.1.0"

__all__ = [
    "CommunityConfig",
    "CommunityResult",
    "FCSFile",
    "Partition",
    "PrismExError",
    "__version__",
    "build_graph",
    "detect_communities",
    "find_communities",
    "find_communities_in_file",
    "make_dataset",
    "read_fcs",
    "read_fcs_bytes",
    "select_subspace",
    "write_demo_file",
    "write_fcs",
]
