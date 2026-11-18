# Reference results and provenance

These files are the saved outputs behind the controlled comparisons, including
unfavorable outcomes. They are not new fits. Table numbers below refer to the
revised manuscript; the originally submitted Tables 6–8 correspond to the
matched-information, exclusion, and receiver comparisons.

## Find a result

All RMSE/MAE values are in dB. CSV values retain their stored precision; the paper
rounds them for presentation. Gateway identifiers are strings, including leading zeros.

| Comparison | Revised table | Reference file |
|---|---|---|
| Original direct/residual input controls | 6 | [original_metrics.csv](tables/original_metrics.csv) — select `model=ensemble` |
| Original exclusion runs and individual seeds | 7 | [original_metrics.csv](tables/original_metrics.csv) — select buffered protocols |
| Expanded direct control and height-LDPL anchor | 7 | [exclusion_summary.csv](tables/exclusion_summary.csv), [revised_metrics.csv](tables/revised_metrics.csv), stage `controls` |
| Complexity sensitivity and equal-budget selection | 8 | [complexity_metrics.csv](tables/complexity_metrics.csv), [tuning_summary.csv](tables/tuning_summary.csv) |
| Disjoint spatial folds and refits | 9 | [spatial_refits.csv](tables/spatial_refits.csv) |
| Repeated count/composition controls | 10 | [count_controls.csv](tables/count_controls.csv); individual draws in [revised_metrics.csv](tables/revised_metrics.csv) |
| Earlier single-draw count controls | Revision history | [original_count_metrics.csv](tables/original_count_metrics.csv) |
| Original receiver holdouts, height and paired gains | 11 | [gateway_comparison.csv](tables/gateway_comparison.csv), [original_gateway_metrics.csv](tables/original_gateway_metrics.csv) |
| Strict receiver holdouts, aggregate and per-site | 12–13 | [gateway_transfer.csv](tables/gateway_transfer.csv), [revised_metrics.csv](tables/revised_metrics.csv), stage `gateways` |
| Conditional intervals for the revised comparisons | Results text | [conditional_intervals.csv](tables/conditional_intervals.csv) |
| Released data and available preprocessing checks | IV-A | [preprocessing.csv](tables/preprocessing.csv) |

The original metrics retain all five seeds where feasible. Cross-radius results
use **0, 1, 3, 4 at every radius**. Repeated count draws are averaged within each
split before averaging splits. Positive paired gain means direct RMSE minus
residual RMSE. Conditional intervals hold the fitted models fixed; overlapping
training folds and repeated subsamples are not independent campaigns.

Gateway `06df` has only 99 validation receptions after 500-m exclusion and is
infeasible under the fixed 100-row rule. Its split remains in the archive, with
no fitted score. Common-site summaries use the same eight feasible receivers.

## Exact partitions and run records

Download [reference_artifacts_v1.zip](https://github.com/LorenzoDonatti/LoRa-Residual-Learning/releases/download/reproducibility-v1/reference_artifacts_v1.zip)
from the [versioned release](https://github.com/LorenzoDonatti/LoRa-Residual-Learning/releases/tag/reproducibility-v1),
or run `python reference/verify.py --download` from the repository root.
[manifest.json](manifest.json) records its size, SHA-256, source hashes and table origins.
The archive is about **19 MB** and contains:

- **53 original and 242 revised partition files**, including infeasible splits.
- Exact per-run metric records, selected settings, validation scores, calibrated
  anchor coefficients and checksums from the **604 revised configurations**.
- Original controlled-run records, the fixed protocol, environment/provenance,
  preprocessing audit, and the numerical summaries used in the paper.
- Frozen source files whose hashes match the original and revised run manifests.
- `ARTIFACT_MANIFEST.json`, with a checksum for every member.

Paths inside the archive retain the experiment workspace layout. Each `.npz`
partition contains `train`, `validation` and `test`, as zero-based row positions
in the upstream measurement release pinned by the data hashes. Load them with
`numpy.load(..., allow_pickle=False)` after verification.

Large per-reception predictions and fitted models are not in this compact release.
The archived JSON records retain prediction hashes, and the frozen runners regenerate
prediction files. Raw inputs are downloaded from the original data repository.
Manuscripts and reviewer correspondence are not part of the release.

The original revised sources were published at
[`8f9564e`](https://github.com/LorenzoDonatti/LoRa-Residual-Learning/tree/8f9564ef22c0390191ac0d8cc9fadf44ee9f97f1).
Those sources remain unchanged. This release adds saved evidence and the exact
original count/receiver scripts. The recorded original campaign used Python 3.14.4;
the revised campaign used Python 3.12.3, with the core package versions listed in
the manifests. Platform/runtime differences can affect floating-point results.

## Original comparisons

For the exact original controlled comparisons, use the frozen sources under
`revision/`, not the compact exploratory runner. After installing requirements
and running `python run_revision.py --check-only`, execute from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python revision/revision_experiments.py --jobs 4 --resume
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python revision/size_matched_controls.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python revision/gateway_matched_controls.py
```

These commands write to `revision/results/` and can take substantial time.
The separate `run.py` interface remains available for exploration; its output
schema is described in [the original experiment guide](../docs/original_experiments.md).

## Integrity checks

```bash
python reference/verify.py
python reference/verify.py --archive /path/to/reference_artifacts_v1.zip
```

The verifier checks source and table hashes, the archive checksum, every archive
member, and agreement between browsable and archived tables. It does not refit
models or claim to independently recompute metrics without the prediction files.
