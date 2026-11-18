# When Does Residual Learning Improve LoRa Path-Loss Prediction?

Companion code and reference results for the **IEEE Access manuscript** by
[Lorenzo Moreira Donatti](https://orcid.org/0009-0001-7563-3422),
[Tiago Bandeira Marchesan](https://orcid.org/0000-0002-5804-9531), and
[Carlos Henrique Barriquello](https://orcid.org/0000-0003-4454-6809),
Federal University of Santa Maria, Brazil.
Corresponding author: Carlos Henrique Barriquello — carlos.barriquello@ufsm.br.
The manuscript is under revision; no acceptance or publication is claimed.

The study compares direct and residual RF/XGBoost/KNN ensembles using the public
Bonn LoRa measurements. It controls receiver information, spatial separation,
training-set composition, and transfer to a held-out gateway.

## Start here

| Your goal | Where to go |
|---|---|
| Read the saved results without training | [Reference tables and table-to-file map](reference/README.md) |
| Obtain exact partitions, settings and run records | [Reproducibility v1 release](https://github.com/LorenzoDonatti/LoRa-Residual-Learning/releases/tag/reproducibility-v1) |
| Reproduce the revised campaign | Commands below and [frozen protocol](revision/ieee_access_resubmit/protocol.json) |
| Reproduce the original controlled comparisons | [Frozen original sources](reference/README.md#original-comparisons) |
| Explore the compact baseline implementation | [Original experiment guide](docs/original_experiments.md) |

## Reference findings

Mean test RMSE in dB; lower is better. The first two rows use the same four
split seeds. The last row averages the same eight feasible receiver sites.

| Prediction setting | Direct + height + OH | OH residual + height |
|---|---:|---:|
| 500-m spatial exclusion, fixed settings | 8.928 | 8.642 |
| 500-m spatial exclusion, validation-selected settings | 8.911 | 8.531 |
| Held-out receiver, shared-packet removal and 500-m position exclusion | 8.953 | 9.391 |

Supplying receiver height removes most of the apparent residual advantage under
local interpolation. A modest OH gain remains under spatial exclusion, depends
on the anchor and geographic training support, and reverses under strict receiver
transfer. These results concern one campaign, not independent cities or network
coverage. Conditional bootstrap intervals and refit variation have distinct meanings.

## Reproduce the revised campaign

Python 3.12 was used for the revised campaign. Install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_revision.py --check-only
```

The last command downloads the pinned public inputs and verifies their hashes
and the frozen sources, without fitting models. Then choose a stage or run all:

```bash
python run_revision.py --stages quality
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python run_revision.py --stages controls --jobs 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python run_revision.py --jobs 4
```

| Stage | Purpose | Ensemble configurations |
|---|---|---:|
| `controls` | Four feature/target configurations across four radii and four seeds | 64 |
| `tuning` | Equal-budget validation selection at 500 m | 16 |
| `spatial` | Five disjoint test folds with three model seeds | 60 |
| `gateways` | Packet and position exclusion for receiver holdouts | 104 |
| `counts` | Repeated count, gateway and geographic-composition controls | 360 |
| `quality` | Reconstructible FSPL filter; zero observations removed | 0 |

The full run is computationally substantial: **604 ensemble configurations**.
Outputs go to `results/revision_v1/`; completed configurations resume after
provenance checks. Each run records partitions, model candidates and selections,
metrics, and per-reception predictions. Keep `--jobs 4` for the archived setup;
small platform-dependent floating-point differences can remain.

The data release lacks satellite count, altitude and pre-alignment displacement,
so it cannot reproduce every quality filter used by the source publication.
The archived preprocessing audit documents this limitation.

## Verify saved artifacts without training

These checks use only Python's standard library:

```bash
python reference/verify.py
python reference/verify.py --download
```

The second command verifies the release archive and every contained file. It
downloads about 19 MB into the ignored `results/reference/` directory. The Git
repository holds only the small, browsable reference tables and source files.
See [reference documentation](reference/README.md) for coverage and provenance.

## Repository layout

- `run_revision.py`, `revision/`: entry point and frozen experiment sources.
- `reference/`: saved tables, integrity manifest, and archive verifier.
- `run.py`, `data.py`, `models.py`, `experiments.py`: compact exploratory runner.
- `docs/original_experiments.md`: options and output schemas for that runner.
- `CITATION.cff`: author and citation metadata.

The release preserves numerical evidence and partitions; the manuscript contains
the scientific results without a separate scientific supplement.
