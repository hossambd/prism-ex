# prism-ex

Strict FCS 3.1 ingestion, community detection over a marker subspace, quantitative
comparison of two communities, and evidence about where the resulting partition can
be relied on.

[![CI](https://github.com/hossambd/prism-ex/actions/workflows/ci.yml/badge.svg)](https://github.com/hossambd/prism-ex/actions/workflows/ci.yml)

Four results, each built on the one before it:

| | what it gives you | entry point |
| --- | --- | --- |
| 1 | An FCS 3.1 file's keywords, per-channel metadata and named event matrix — or a typed error, never a half-read file | `prism_ex.read_fcs` |
| 2 | Communities in a neighbourhood graph over the events, with their sizes | `prism_ex.find_communities` |
| 3 | A quantitative comparison of any two of those communities' marker profiles | `prism_ex.compare.compare_communities` |
| 4 | The measurements a claim about partition stability can be made from | `prism_ex.stability.assess_stability` |

No cytometry knowledge is needed to use it. "Marker" means "column of the event
matrix", and the shipped example data is synthetic.

## Install

Into a clean virtual environment, from a checkout:

Linux/macOS:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
```

Optional extras: `pip install ".[api]"` for the HTTP endpoint, `".[pandas]"` for
`FCSFile.to_dataframe()`, `".[dev]"` for the test suite and linter.

Python 3.10 to 3.13. Most commands below finish on a laptop in under a minute;
the full stability analysis and evidence script intentionally take longer.

## Worked example

No data is shipped and none is needed: the package generates a valid FCS 3.1 file
with known populations, so every command here runs immediately after install.

### 1. Make a file, and read it

```bash
prism-ex demo demo.fcs --events 6000
prism-ex info demo.fcs
```

```
FCS 3.1: 6000 events x 9 channels
source: demo.fcs
  P1   FSC-A                        32 bit  linear range=93137
  P3   CD3 (CD3-BV421)              32 bit  linear range=28512
  ...
```

A file that is malformed, truncated, internally inconsistent, or that declares any
version other than 3.1 raises instead — and says which of those it was:

```bash
head -c 200 demo.fcs > truncated.fcs && prism-ex info truncated.fcs
# TruncatedData: TEXT segment ends at byte 732 but the file is 200 bytes
```

PowerShell equivalent:

```powershell
$bytes = [IO.File]::ReadAllBytes("demo.fcs")[0..199]
[IO.File]::WriteAllBytes("truncated.fcs", $bytes)
prism-ex info truncated.fcs
```

### 2. Find communities

```bash
prism-ex communities demo.fcs --markers CD3,CD4,CD8,CD19,CD56
```

```
6 communities over 6000 events (121474 edges, quality 48099.4)
  community 0        1620    27.0%
  community 1        1327    22.1%
  community 2        1200    20.0%
  community 3         989    16.5%
  community 4         781    13.0%
  community 5          83     1.4%
```

`--json` gives the same thing machine-readably, with the provenance record — the
input digest, the package version and the full configuration — attached.

### 3. Compare two of them

```sh
prism-ex compare demo.fcs --a 1 --b 2 --compare-markers CD3,CD19,Viability,FSC-A --inference split
```

```
Community 1 (n=1327) vs community 2 (n=1200) - energy distance 6.755

| marker      | median A | median B | Cliff's d | 95% CI         | magnitude  | q     |
| CD19*       |     0.18 |     3.51 |    -1.000 | [-1.00, -1.00] | large      | n/a   |
| CD3*        |     3.40 |     0.31 |    +1.000 | [+1.00, +1.00] | large      | n/a   |
| FSC-A       |     6.18 |     6.18 |    +0.019 | [-0.04, +0.07] | negligible | 0.917 |
| Viability   |     1.74 |     1.75 |    -0.011 | [-0.05, +0.03] | negligible | 0.917 |
```

Markers marked `*` were used to define the communities. Those get effect sizes but
never a p-value: the clustering separated the events on those markers, so a test of
whether they differ on them has no null hypothesis left to reject. Markers outside
the clustering subspace do get one, computed on held-out events after re-deriving
the communities on the other half. See the module docstring of `prism_ex.compare`
for why the split is necessary and why it is not sufficient.

### 4. Ask where the partition can be relied on

```bash
prism-ex stability demo.fcs --resamples 40
```

Per community: mean and 5th-percentile Jaccard against 80% resamples, the fraction
of resamples in which it survives, and the fraction of its events that return to it
reliably. Plus a seed-only check, and a sweep over `k` and `resolution`. The claim
those numbers support is in the accompanying report, not here — a stability number
without an interpretation is not a finding.

### As a library

```python
from prism_ex import find_communities_in_file
from prism_ex.compare import compare_communities
from prism_ex.stability import assess_stability

result = find_communities_in_file("demo.fcs", ["CD3", "CD4", "CD8", "CD19", "CD56"])
print(result.sizes)                       # {0: 1620, 1: 1327, ...}

comparison = compare_communities(result, 1, 2, markers=["CD3", "CD19", "Viability"])
print(comparison.to_markdown())

evidence = assess_stability(result, n_resamples=40)
print(evidence.to_markdown())
```

`result` keeps the file, the subspace, the graph and the partition together, so
anything downstream can re-run the same configuration under perturbation without
being handed the pieces again.

### Optional HTTP endpoint

Run the server in one terminal:

```sh
pip install ".[api]"
prism-ex serve
```

Then call it from another terminal (use `curl.exe` in PowerShell):

```sh
curl -F file=@demo.fcs -F markers=CD3,CD4,CD8,CD19,CD56 http://127.0.0.1:8000/communities/sizes
```

Returns the community sizes and the provenance record. One endpoint, no persistence,
no queue.

## What is where

```
src/prism_ex/
  fcs/reader.py     strict, atomic FCS 3.1 reader          (result 1)
  fcs/writer.py     FCS 3.1 writer: fixtures + synthetic data
  synth.py          generated data with ground-truth labels
  subspace.py       marker selection, asinh transform, scaling
  graph.py          exact kNN, shared-neighbour Jaccard weights  (result 2)
  communities.py    Leiden partition, sizes                      (result 2)
  pipeline.py       the four steps chained under one configuration
  compare.py        Cliff's delta, energy distance, honest inference (result 3)
  stability.py      resampling, sweeps, per-event core scores      (result 4)
  cli.py, api.py    interfaces
```

## Why these choices

`scripts/decision_evidence.py` re-derives the evidence behind every design decision
in the package — reader strictness against an independent reader, transform and
scaling, edge weighting, Leiden vs Louvain, the effect-size choice, why p-values are
refused inside the clustering subspace, and why resampling is sub-sampling rather
than bootstrap.

```bash
python scripts/decision_evidence.py           # ~4 minutes
python scripts/decision_evidence.py --quick   # ~1 minute
python scripts/decision_evidence.py --only 5  # just one section
```

Two of the eight sections report null results that do not support the default
chosen. They are kept.

For a narrated tour of every public feature and its output:

```sh
python scripts/tour_complet.py --rapide
python scripts/tour_complet.py --section 7
```

## Tests

```bash
pip install ".[dev]"
pytest                      # ~12 s
pytest -m "not slow"        # skips the resampling-heavy cases
ruff check src tests scripts && ruff format --check src tests scripts
```

Every fixture is generated in code — nothing depends on a file that exists only on
one machine. The suite includes a corruption matrix that breaks exactly one thing at
a time in a known-good file, and a differential test that requires an independent
implementation (`flowio`, a development dependency only) to extract the same event
matrix.

CI installs the package into a clean virtual environment on four Python versions,
lints, runs the tests with an 85% coverage floor, and exercises the representative
`demo`, `info`, `communities`, and guided-tour commands from this README.

## Reproducibility

The same input, the same configuration and the same package version give the same
answer. Neighbour search is exact with ties broken by index; Leiden's seed is a
parameter with a default rather than a clock read; and every result carries a
`Provenance` record naming the input's SHA-256, the configuration, the package
version and the interpreter — so "same input" is checkable rather than assumed.

## Licence

MIT. The synthetic data generator means no third-party data is redistributed here.
