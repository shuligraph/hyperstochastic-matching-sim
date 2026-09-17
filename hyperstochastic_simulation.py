from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np

GroupType = FrozenSet[int]

V: Tuple[int, ...] = (1, 2, 3, 4)

E_S: Tuple[GroupType, ...] = (
    frozenset({1, 2, 3, 4}),
    frozenset({1, 2}),
    frozenset({1, 3}),
    frozenset({2, 3}),
)

MU = np.array([0.3, 0.3, 0.3, 0.1], dtype=float)

STORED_TYPES: Tuple[GroupType, ...] = (
    frozenset({1}),
    frozenset({2}),
    frozenset({3}),
    frozenset({4}),
    frozenset({1, 4}),
    frozenset({2, 4}),
    frozenset({3, 4}),
    frozenset({1, 2, 4}),
    frozenset({1, 3, 4}),
    frozenset({2, 3, 4}),
)

def canonical_type(group: GroupType) -> Tuple[int, ...]:
    return tuple(sorted(group))

def admissible_group_types(
    vertices: Sequence[int],
    hyperedges: Sequence[GroupType],
) -> Tuple[GroupType, ...]:
    """Return all nonempty subsets contained in at least one hyperedge."""
    types: List[GroupType] = []
    for size in range(1, len(vertices) + 1):
        for combo in combinations(vertices, size):
            group = frozenset(combo)
            if any(group <= edge for edge in hyperedges):
                types.append(group)
    return tuple(types)

G_H = admissible_group_types(V, E_S)
G_H_SET = frozenset(G_H)
COMPLETE = frozenset(E_S)
STORED_TYPE_SET = frozenset(STORED_TYPES)

if not np.isclose(MU.sum(), 1.0):
    raise ValueError("The arrival probabilities must sum to one.")
if np.any(MU < 0):
    raise ValueError("Arrival probabilities must be nonnegative.")
if not all(group in G_H_SET and group not in COMPLETE for group in STORED_TYPES):
    raise ValueError("STORED_TYPES must contain only admissible incomplete groups.")

def greedy_step(counts: np.ndarray, arrival_class: int) -> None:
    feasible_edges: List[GroupType] = []
    for edge in E_S:
        if arrival_class not in edge:
            continue
        if all(counts[j - 1] > 0 for j in edge if j != arrival_class):
            feasible_edges.append(edge)
    if not feasible_edges:
        counts[arrival_class - 1] += 1
        return
    selected_edge = min(
        feasible_edges,
        key=lambda edge: (len(edge), canonical_type(edge)),
    )
    for waiting_class in selected_edge:
        if waiting_class != arrival_class:
            counts[waiting_class - 1] -= 1

    if np.any(counts < 0):
        raise RuntimeError("The greedy transition produced a negative count.")

def empty_wml_state() -> Dict[GroupType, int]:
    return {group: 0 for group in STORED_TYPES}

def i_compatible_types(
    state: Dict[GroupType, int],
    arrival_class: int,
) -> List[GroupType]:
    """Return positive-population groups extendable by the arrival."""
    return [
        group
        for group in STORED_TYPES
        if state[group] > 0
        and arrival_class not in group
        and (group | {arrival_class}) in G_H_SET
    ]

def pressure(
    state: Dict[GroupType, int],
    group: GroupType,
    arrival_class: int,
) -> int:
    """Compute P_i(U,X)=|U|X_U-(|U|+1)Xbar_{U union {i}}."""
    extended_group = group | {arrival_class}
    if extended_group in COMPLETE:
        downstream_count = 0
    else:
        if extended_group not in STORED_TYPE_SET:
            raise RuntimeError(
                f"Unexpected incomplete downstream type: {sorted(extended_group)}"
            )
        downstream_count = state[extended_group]
    return (
        len(group) * state[group]
        - (len(group) + 1) * downstream_count
    )

def apply_extension(
    state: Dict[GroupType, int],
    group: GroupType,
    arrival_class: int,
) -> None:
    """Extend one stored group and remove a completed hyperedge immediately."""
    state[group] -= 1
    extended_group = group | {arrival_class}
    if extended_group not in COMPLETE:
        state[extended_group] += 1

    if any(count < 0 for count in state.values()):
        raise RuntimeError("The WML transition produced a negative count.")

def store_singleton(
    state: Dict[GroupType, int],
    arrival_class: int,
) -> None:
    state[frozenset({arrival_class})] += 1

def wml_step(
    state: Dict[GroupType, int],
    arrival_class: int,
) -> None:
    candidates = i_compatible_types(state, arrival_class)
    if not candidates:
        store_singleton(state, arrival_class)
        return
    pressures = {
        group: pressure(state, group, arrival_class)
        for group in candidates
    }
    maximum_pressure = max(pressures.values())
    if maximum_pressure <= 0:
        store_singleton(state, arrival_class)
        return
    maximizers = [
        group
        for group, value in pressures.items()
        if value == maximum_pressure
    ]
    selected_group = min(maximizers, key=canonical_type)
    apply_extension(state, selected_group, arrival_class)

def number_of_stored_groups(state: Dict[GroupType, int]) -> int:
    """Return ||X||_1, the number of committed partial groups."""
    return sum(state.values())


def number_of_stored_items(state: Dict[GroupType, int]) -> int:
    """Return the physical number of waiting items, sum_U |U| X_U."""
    return sum(len(group) * count for group, count in state.items())


def number_of_class4_items(state: Dict[GroupType, int]) -> int:
    """Each stored group contains at most one class-4 item."""
    return sum(
        count
        for group, count in state.items()
        if 4 in group
    )

@dataclass(frozen=True)
class SimulationResult:
    arrivals: np.ndarray
    class4_greedy: np.ndarray
    stored_items_wml: np.ndarray
    stored_groups_wml: np.ndarray
    class4_items_wml: np.ndarray

def run_paired_simulation(
    n_arrivals: int,
    seed: int = 0,
) -> SimulationResult:
    """Run both disciplines under the same i.i.d. arrival sequence."""
    if n_arrivals <= 0:
        raise ValueError("n_arrivals must be positive.")

    # Explicitly select PCG64 so that the pseudorandom generator is recorded.
    rng = np.random.Generator(np.random.PCG64(seed))
    arrivals = rng.choice(
        np.asarray(V, dtype=np.int64),
        size=n_arrivals,
        p=MU,
    )
    greedy_counts = np.zeros(len(V), dtype=np.int64)
    wml_state = empty_wml_state()
    class4_greedy = np.empty(n_arrivals, dtype=np.int64)
    stored_items_wml = np.empty(n_arrivals, dtype=np.int64)
    stored_groups_wml = np.empty(n_arrivals, dtype=np.int64)
    class4_items_wml = np.empty(n_arrivals, dtype=np.int64)
    for epoch, raw_class in enumerate(arrivals):
        arrival_class = int(raw_class)
        greedy_step(greedy_counts, arrival_class)
        wml_step(wml_state, arrival_class)
        class4_greedy[epoch] = greedy_counts[3]
        stored_items_wml[epoch] = number_of_stored_items(wml_state)
        stored_groups_wml[epoch] = number_of_stored_groups(wml_state)
        class4_items_wml[epoch] = number_of_class4_items(wml_state)

    return SimulationResult(
        arrivals=arrivals,
        class4_greedy=class4_greedy,
        stored_items_wml=stored_items_wml,
        stored_groups_wml=stored_groups_wml,
        class4_items_wml=class4_items_wml,
    )

def make_figure(
    result: SimulationResult,
    outfile: str = "hyperstochastic_comparison.pdf",
) -> None:
    """Generate the two-panel figure used in the manuscript."""
    n_arrivals = len(result.arrivals)
    epochs = np.arange(1, n_arrivals + 1)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.0, 5.0),
        sharex=True,
    )

    axes[0].plot(
        epochs,
        result.class4_greedy,
        linewidth=1.0,
        label=r"$Q_n(4)$",
    )
    axes[0].plot(
        epochs,
        MU[3] * epochs,
        linewidth=1.0,
        label=r"$\mu_4 n$",
    )
    axes[0].set_ylabel("class-4 items")
    axes[0].set_title("Greedy discipline: class-4 accumulation")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(
        epochs,
        result.stored_items_wml,
        linewidth=0.8,
        label=r"$N_{\mathrm{items}}(X_n)$",
    )
    axes[1].set_ylabel("stored items")
    axes[1].set_xlabel(r"arrival epoch $n$")
    axes[1].set_title("Positive-pressure WML: stored-item population")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def print_summary(result: SimulationResult) -> None:
    n_arrivals = len(result.arrivals)

    empirical_class4_rate = result.class4_greedy[-1] / n_arrivals

    print(f"Number of arrivals: {n_arrivals}")
    print("Pseudorandom generator: NumPy PCG64")
    print(
        "Empirical class-4 accumulation rate under greedy: "
        f"{empirical_class4_rate:.4f} "
        f"(arrival probability {MU[3]:.1f})"
    )
    print(
        "WML stored-item population along this trajectory: "
        f"time average {result.stored_items_wml.mean():.4f}, "
        f"maximum {result.stored_items_wml.max()}"
    )
    print(
        "WML stored-group count along this trajectory: "
        f"time average {result.stored_groups_wml.mean():.4f}, "
        f"maximum {result.stored_groups_wml.max()}"
    )
    print(
        "WML class-4 items along this trajectory: "
        f"time average {result.class4_items_wml.mean():.4f}, "
        f"maximum {result.class4_items_wml.max()}"
    )


if __name__ == "__main__":
    N_ARRIVALS = 20_000
    SEED = 0
    OUTPUT_FILE = "hyperstochastic_comparison.pdf"

    simulation = run_paired_simulation(
        n_arrivals=N_ARRIVALS,
        seed=SEED,
    )
    make_figure(simulation, outfile=OUTPUT_FILE)
    print_summary(simulation)
