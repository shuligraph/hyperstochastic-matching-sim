"""Finite-horizon FCFM/WML comparison for the manuscript's four-class model.

Example: python hyperstochastic_fcfm.py --arrivals 100000 --repeats 20
FCFM retains creation order after extension. WML uses the manuscript's
explicit priorities (not the legacy simulator's lexicographic tie breaking).
Reported averages include all post-arrival states, with no burn-in removal.
Simulation does not establish stability or stationary performance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

import hyperstochastic_simulation as model


PRIORITIES = {
    1: ((2,), (3,), (4,), (2, 4), (3, 4), (2, 3, 4)),
    2: ((1,), (3,), (4,), (1, 4), (3, 4), (1, 3, 4)),
    3: ((1,), (2,), (4,), (1, 4), (2, 4), (1, 2, 4)),
    4: ((1,), (2,), (3,)),
}
PRIORITIES = {i: tuple(map(frozenset, groups)) for i, groups in PRIORITIES.items()}


def fcfm_step(groups, arrival_class):
    """Mutate oldest-first groups; return the completed edge, or None.

    Linear scan in the current number of groups. Incomplete extensions keep
    their position, and existing groups are never merged.
    """
    if arrival_class not in model.V:
        raise ValueError("Unknown arrival class")
    for pos, group in enumerate(groups):
        if arrival_class in group:
            continue
        extended = group | {arrival_class}
        if extended not in model.G_H_SET:
            continue
        if extended in model.COMPLETE:
            groups.pop(pos)
            return extended
        groups[pos] = extended
        return None
    groups.append(frozenset({arrival_class}))
    return None


def manuscript_wml_step(state, arrival_class):
    """Reuse pressure calculations with the paper's fixed priority order."""
    selected, best = None, 0
    for group in PRIORITIES[arrival_class]:
        if state[group] > 0:
            value = model.pressure(state, group, arrival_class)
            if value > best:  # Strict comparison preserves the first maximizer.
                selected, best = group, value
    if selected is None:
        model.store_singleton(state, arrival_class)
        return None
    extended = selected | {arrival_class}
    model.apply_extension(state, selected, arrival_class)
    return extended if extended in model.COMPLETE else None


def class_counts(groups):
    return np.array([sum(i in group for group in groups) for i in model.V])


def run_comparison(n_arrivals=20000, seed=0, validate=False):
    """Both policies start empty and receive the same PCG64 arrival sequence.

    validate=True checks state validity and class-wise conservation each step.
    This is intentionally expensive and intended for small validation runs.
    """
    if n_arrivals <= 0:
        raise ValueError("n_arrivals must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    arrivals = rng.choice(model.V, size=n_arrivals, p=model.MU)
    groups, state = [], model.empty_wml_state()
    matches = {name: {edge: 0 for edge in model.E_S} for name in ("fcfm", "wml")}
    traces = {name: np.empty((n_arrivals, 3), dtype=np.int64) for name in matches}
    waiting = {name: np.zeros(4, dtype=np.int64) for name in matches}
    arrived = np.zeros(4, dtype=np.int64)
    for epoch, raw in enumerate(arrivals):
        arrival = int(raw)
        arrived[arrival - 1] += 1
        edges = {"fcfm": fcfm_step(groups, arrival),
                 "wml": manuscript_wml_step(state, arrival)}
        for name, edge in edges.items():
            waiting[name][arrival - 1] += 1
            if edge is not None:
                matches[name][edge] += 1
                for i in edge:
                    waiting[name][i - 1] -= 1
            count = len(groups) if name == "fcfm" else sum(state.values())
            traces[name][epoch] = waiting[name].sum(), waiting[name][3], count
        if validate:
            assert all(g in model.G_H_SET and g not in model.COMPLETE for g in groups)
            assert all(v >= 0 for v in state.values())
            actual = {"fcfm": class_counts(groups), "wml": np.array([
                sum(v for g, v in state.items() if i in g) for i in model.V])}
            for name in matches:
                departed = np.array([sum(n for e, n in matches[name].items() if i in e)
                                     for i in model.V])
                np.testing.assert_array_equal(actual[name], waiting[name])
                np.testing.assert_array_equal(actual[name] + departed, arrived)
    rows = []
    for name, values in traces.items():
        row = {"seed": seed, "policy": name, "n_arrivals": n_arrivals,
               "mean_items": float(values[:, 0].mean()),
               "max_items": int(values[:, 0].max()),
               "final_items": int(values[-1, 0]),
               "mean_class4": float(values[:, 1].mean()),
               "final_class4": int(values[-1, 1]),
               "mean_groups": float(values[:, 2].mean())}
        row.update({"matches_" + "".join(map(str, sorted(e))): n
                    for e, n in matches[name].items()})
        rows.append(row)
    return arrivals, traces, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrivals", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("output/fcfm"))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.arrivals <= 0 or args.repeats <= 0 or args.seed < 0:
        parser.error("arrivals/repeats must be positive and seed nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    started, rows = perf_counter(), []
    for seed in range(args.seed, args.seed + args.repeats):
        arrivals, traces, run_rows = run_comparison(args.arrivals, seed, args.validate)
        rows.extend(run_rows)
        np.savez_compressed(args.output / f"trajectory_seed_{seed}.npz",
                            arrivals=arrivals, **traces)
        print(f"seed={seed}: FCFM mean items={run_rows[0]['mean_items']:.4f}, "
              f"WML={run_rows[1]['mean_items']:.4f}", flush=True)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    differences = np.array([rows[i]['mean_items'] - rows[i + 1]['mean_items']
                            for i in range(0, len(rows), 2)])
    metadata = {
        "interpretation": "Finite-horizon post-arrival summaries, no burn-in; not a stability proof",
        "n_arrivals": args.arrivals, "seeds": list(range(args.seed, args.seed + args.repeats)),
        "rng": "NumPy PCG64", "mu": model.MU.tolist(),
        "trajectory_columns": ["stored_items", "class4_items", "stored_groups"],
        "wml_ties": {str(i): [sorted(g) for g in order] for i, order in PRIORITIES.items()},
        "paired_mean_items_difference_fcfm_minus_wml": float(differences.mean()),
        "paired_standard_error": (float(differences.std(ddof=1) / np.sqrt(args.repeats))
                                  if args.repeats > 1 else None),
        "validation_enabled": args.validate, "elapsed_seconds": perf_counter() - started,
        "python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(model.__file__))},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved trajectories, summary.csv and metadata.json to", args.output)


if __name__ == "__main__":
    main()
