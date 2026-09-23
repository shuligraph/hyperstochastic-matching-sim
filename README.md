# Hypergraph stochastic matching simulations

Simulation code accompanying *Greedy failure and a stabilizing partial group discipline on a nested hypergraph*.

The model has classes `{1, 2, 3, 4}`, hyperedges `{1,2}`, `{1,3}`, `{2,3}`, and `{1,2,3,4}`, and arrival probabilities `(0.3, 0.3, 0.3, 0.1)`. Arrivals are generated with NumPy PCG64. All policies start from the empty state.

## Files

| File | Purpose |
| --- | --- |
| `hyperstochastic_simulation.py` | Original greedy/WML simulator and plotting functions; also imported by the comparison script. See the tie-breaking note below. |
| `hyperstochastic_comparison.pdf` | Two-panel greedy/WML illustration. Its provenance should be checked against the manuscript-priority reproduction command below. |
| `hyperstochastic_fcfm.py` | Paired FCFM/WML experiments, using the manuscript's explicit WML priorities. |
| `test_hyperstochastic_fcfm.py` | Tests of FCFM order preservation, matching, WML priorities, conservation, and reproducibility. |
| `requirements.txt` | Simulation dependencies. |
| `output/fcfm_validated/summary.csv` | Per-replication summaries, when included with the repository. |
| `output/fcfm_validated/metadata.json` | Seeds, parameters, source hashes, software versions, and validation status for the reported experiment, when included. |

**Accessibility certificate:** `verify_accessibility.py`, referenced by Lemma 3, was not present in the local files used to prepare this README. These simulations and tests do not replace that proof-verification script. It must be supplied separately to reproduce the finite accessibility verification.

## Installation

Use Python 3.10 or newer. The reported experiments used Python 3.12.4, NumPy 2.0.1, and Matplotlib 3.9.1.

```console
python -m pip install -r requirements.txt
```

Run commands from the directory containing both simulation scripts.

## FCFM/WML comparison

```console
python hyperstochastic_fcfm.py --arrivals 100000 --repeats 20 --validate --output output/fcfm_validated
```

The default starting seed is 0, so this command uses seeds 0–19. Within each replication, both policies receive exactly the same arrival sequence. FCFM extends the oldest compatible committed group, keeps its original position after an incomplete extension, and removes completed groups immediately. WML uses the fixed priorities specified in the manuscript.

`--validate` checks state validity and class-wise item conservation after every arrival. It makes execution slower; omitting it leaves the policy transitions unchanged. For a quick run:

```console
python hyperstochastic_fcfm.py --arrivals 20000 --repeats 1 --output output/fcfm_quick
```

### Outputs

- `summary.csv`: one row per seed and policy, including mean, maximum and final waiting-item counts, class-4 counts, mean group count, and matching counts for each hyperedge.
- `trajectory_seed_<seed>.npz`: the arrival sequence and arrays named `fcfm` and `wml`. Each policy array has columns **waiting items, waiting class-4 items, stored groups**, measured after each arrival.
- `metadata.json`: parameters, seeds, WML priorities, versions, source hashes, execution time, and the mean and standard error of the paired difference in average waiting-item counts.

All time averages include all post-arrival states; no burn-in period is removed. Reusing an output directory overwrites files for matching seeds and the summary/metadata; use a new directory for a different experiment.

### Reported finite-horizon results

For 20 replications of 100,000 arrivals:

| Quantity | Value |
| --- | ---: |
| FCFM mean waiting-item population | 3.859935 |
| WML mean waiting-item population | 4.479192 |
| Mean paired difference, FCFM minus WML | -0.619257 |
| Standard error of the paired difference | 0.0106163447 |
| Approximate 95% paired Student-t interval | [-0.641477, -0.597037] |

The confidence interval uses the 20 independent replication differences, with 19 degrees of freedom, not the individual time steps. It is the paired mean plus or minus `2.0930240544` times its standard error. FCFM has a smaller time-average backlog in each of these replications. These results do not establish FCFM stability, stationary means, or a general performance ordering.

## WML tie-breaking and the two-panel figure

The original `hyperstochastic_simulation.py` resolves pressure ties lexicographically by group type. This differs from the manuscript's explicit priorities. The comparison script reuses its pressure calculations but replaces tie-breaking with `manuscript_wml_step`.

For seed 0 and 20,000 arrivals, the original script gives mean WML waiting-item population **4.0788**; the manuscript priorities give **4.2581**, with maximum **33**. The latter agrees with the manuscript's rounded value **4.26**. Do not use the original script's main entry point to reproduce the manuscript-priority WML trajectory.

To generate a separate two-panel figure with the manuscript priorities, run this command from the repository root (one line, supported by PowerShell):

```console
python -c "import numpy as np; import hyperstochastic_fcfm as f; a, t, _ = f.run_comparison(20000, 0); w = t['wml']; r = f.model.SimulationResult(a, np.cumsum(a == 4), w[:, 0], w[:, 2], w[:, 1]); f.model.make_figure(r, 'hyperstochastic_comparison_manuscript.pdf')"
```

Here the greedy class-4 count is the cumulative number of class-4 arrivals, as established by the paper's pathwise invariant. This command preserves the existing `hyperstochastic_comparison.pdf`.

## Tests

```console
python -m unittest test_hyperstochastic_fcfm -v
```

These tests check implementation semantics and bounded runs; they are not a proof of positive recurrence or an accessibility certificate.
