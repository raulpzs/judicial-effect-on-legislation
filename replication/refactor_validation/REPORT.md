# Readability refactor equivalence

Exact analytical equality: **True**. Checked-in results unchanged: **True**.

Compared 148 output/state files: all 17 original models, eight regime attempts, 24 prediction grids / 3,672 rows, storage/start diagnostics, reports and 48 generated figures.

Comparisons include row IDs, outcomes, cluster IDs, required fields, design matrices, optimizer state, coefficients, all covariance matrices (including failed attempts), SEs/CIs, LL/AIC/BIC, predictions, separation witnesses and all diagnostic flags. NPZ members are compared at stored precision; matching NaN and signed infinity masks are explicit. CSV/text/image bytes and every JSON leaf are checked. No tolerances or rounding are used; no whole metadata file is excluded.

## Differences

No unresolved analytical differences.
- Allowed provenance: `regime_subsets/analysis_manifest.json` / `['protected_file_sha256', 'replication/scripts/replicate.py']` — Hash of the specifically identified changed source artifact.
- Allowed provenance: `regime_subsets/analysis_manifest.json` / `['replication_script_sha256']` — Hash of the specifically identified changed source artifact.
- Allowed provenance: `regime_subsets/analysis_manifest.json` / `['script_sha256']` — Hash of the specifically identified changed source artifact.

The existing Stata discrepancies and four unsuccessful autocracy fits remain unchanged.

## Reproduce

```sh
.venv/bin/python replication/scripts/check_refactor_equivalence.py
```

Untouched scripts and baseline runs are preserved in `baseline/`; original results/figures in `preserved_results/`; input copies in `inputs/`. `baseline_record.json` records environment and hashes. Each project’s `commands/commands.json` and logs record actual commands. Repeated checks create a new `after_N/` directory and leave earlier evidence intact.
