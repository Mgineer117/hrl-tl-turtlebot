import itertools
import multiprocessing

import numpy as np
import spot
from numpy.typing import NDArray
from spot import twa as Twa
from torch import Tensor


def base_n(num_10: int, n: int, width: int | None = None) -> str:
    """
    Convert a base 10 number to a base n representation.

    Parameters
    ----------
    num_10 : int
        The base 10 number to convert.
    n : int
        The base to convert to.
    width : int | None = None
        The width of the output string. If None, the output will not be padded.

    Returns
    -------
    out_str : str
        The base n representation of the input number as a string.
    """
    str_n = ""
    while num_10:
        if num_10 % n >= 10:
            raise ValueError(
                f"Cannot convert {num_10} to base {n} as it contains digits >= 10."
            )
        str_n += str(num_10 % n)
        num_10 //= n

    out_str = str_n[::-1]
    if width is not None:
        out_str = out_str.zfill(width)
    return out_str


def sort_tl_weights(
    tl_weights: NDArray[np.integer] | Tensor,
    num_predicates: int,
    num_clauses: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Sort the task specification by weights and return the sorted weights for F and G clauses.

    Parameters
    ----------
    tl_weights : NDArray[np.integer] | Tensor
        The task specification weights, either as a NumPy array or a PyTorch tensor.
        Each element should be either 0 or 1, representing the presence or absence of a predicate in the task specification.
    num_predicates : int
        The number of predicates available to define the task specification.

    Returns
    -------
    sorted_f_weights : list[list[int]]
        The unique and sorted weights for the F clauses of the task specification.
    sorted_g_weights : list[list[int]]
        The unique and sorted weights for the G clauses of the task specification.
    """
    if isinstance(tl_weights, Tensor):
        tl_weights = tl_weights.cpu().detach().numpy()
    tl_weights = tl_weights.reshape(2 * num_clauses, num_predicates)
    sorted_f_weights: list[list[int]] = sort_temp_clause_weights(
        tl_weights[:num_clauses, :], num_predicates, num_clauses
    )
    sorted_g_weights: list[list[int]] = sort_temp_clause_weights(
        tl_weights[num_clauses:, :], num_predicates, num_clauses
    )

    return sorted_f_weights, sorted_g_weights


def sort_temp_clause_weights(
    temp_clause_weights: NDArray[np.integer],
    num_predicates: int,
    num_clauses: int,
) -> list[list[int]]:
    """
    Sort the temporary clause weights and return the unique and sorted weights.

    Parameters
    ----------
    temp_clause_weights : NDArray[np.integer] | Tensor
        The temporary clause weights, either as a NumPy array or a PyTorch tensor.
        Each element should be either 0 or 1, representing the presence or absence of a predicate in the temporary clause.
    num_predicates : int
        The number of predicates available to define the task specification.

    Returns
    -------
    sorted_temp_weights : list[list[int]]
        The unique and sorted weights for the temporary clause.
    """
    temp_clause_weights = temp_clause_weights.reshape(
        num_clauses, num_predicates
    )
    sorted_temp_weights: list[list[int]] = np.flipud(
        np.unique(temp_clause_weights, axis=0)
    ).tolist()

    return sorted_temp_weights


def weights2ltl(
    f_weights: list[list[int]],
    g_weights: list[list[int]],
    predicates: list[str],
) -> str:
    """
    Convert sorted task specification weights to a LTL with F and G clauses in CNF (Conjunctive Normal Form).

    Parameters
    ----------
    f_weights : list[list[int]]
        The unique and sorted weights for the F clauses of the task specification.
        Each element should be either 0 or 1, representing the presence or absence of a predicate in the F clause.
    g_weights : list[list[int]]
        The unique and sorted weights for the G clauses of the task specification.
        Each element should be either 0 or 1, representing the presence or absence of a predicate in the G clause.
    predicates : list[str]
        The list of predicates available to define the task specification.

    Returns
    -------
    tl_spec: str
        The task specification in LTL format, with F and G clauses in CNF.
        The format is: "F((p1 | p2) & ...) & G((q1 | q2) &  ...)" where p's are predicates for F and q's for G.
    """
    # Sort predicates alphabetically
    sorted_predicates = sorted(predicates)
    all_predicates = sorted_predicates

    f_clause: str = weights2temp_clause(f_weights, all_predicates)
    g_clause: str = weights2temp_clause(g_weights, all_predicates)

    f_spec: str = f"F({f_clause})" if f_clause else ""
    g_spec: str = f"G({g_clause})" if g_clause else ""
    tl_spec: str = (
        f"{f_spec} & {g_spec}" if f_spec and g_spec else f"{f_spec}{g_spec}"
    )

    return tl_spec


def weights2temp_clause(
    weights: list[list[int]], all_predicates: list[str]
) -> str:
    """
    Convert sorted task specification weights to a clause in CNF (Conjunctive Normal Form).

    Parameters
    ----------
    weights : list[list[int]]
        The unique and sorted weights for the task specification.
        Each element should be either 0 or 1, representing the presence or absence of a predicate in the clause.
    all_predicates : list[str]
        The list of all predicates available to define the task specification including negated ones.

    Returns
    -------
    temp_clause: str
        The clause in CNF format, where each predicate is separated by ' | '.
        If multiple predicates are present, they are enclosed in parentheses.
    """
    clauses: list[str] = []
    for weight in weights:
        used_preds: list[str] = []
        for i, w in enumerate(weight):
            match w:
                case 0:
                    continue
                case 1:
                    used_preds.append(all_predicates[i])
                case 2:
                    used_preds.append(f"!{all_predicates[i]}")
                case _:
                    raise ValueError(
                        f"Invalid weight {w} in weights2temp_clause. Expected 0, 1, or 2."
                    )
        # If no predicates are used, skip this clause
        if not used_preds:
            continue
        else:
            clause = " | ".join(used_preds)
            clauses.append(clause if len(used_preds) == 1 else f"({clause})")
    if len(clauses) == 1:
        clauses[0] = clauses[0].replace("(", "").replace(")", "")
    temp_clause: str = " & ".join(clauses)

    return temp_clause


def count_automaton_states(tl_spec: str) -> int:
    """
    Count the number of states in the automaton represented by the task specification.

    Parameters
    ----------
    tl_spec : str
        The task specification in LTL format.

    Returns
    -------
    num_states : int
        The number of states in the automaton.
    """
    aut: Twa = spot.translate(tl_spec, "Buchi", "state-based", "complete")
    num_states: int = int(aut.num_states())  # type: ignore

    return num_states


def check_overlapping_column_weights(weights: list[list[int]]) -> bool:
    # Return True if there are no overlapping column weights, False otherwise.
    weights_np: NDArray[np.integer] = np.array(weights)

    # Check if the weights are unique across columns
    _, nonzero_weights = weights_np.nonzero()
    # Check if there are any duplicate column indices
    has_duplicates: bool = len(nonzero_weights) != len(set(nonzero_weights))

    return not has_duplicates


def _generate_specification(
    i: int,
    num_elements: int,
    len_predicates: int,
    all_predicates: list[str],
    num_clauses: int,
    max_num_predicates: int,
) -> str | None:
    ter_rep: str = base_n(i, 3, width=num_elements)
    specification_weights: list[int] = [int(bit) for bit in ter_rep]
    clause_weights = sort_temp_clause_weights(
        np.array(specification_weights), len_predicates, num_clauses
    )
    if not check_overlapping_column_weights(clause_weights):
        return None
    clause_spec: str = weights2temp_clause(clause_weights, all_predicates)
    if not clause_spec:
        return None
    tl_spec = f"{spot.simplify(spot.formula(clause_spec))}"
    if tl_spec and (tl_spec == "0" or tl_spec == "1"):
        return None
    if len(get_used_predicates(tl_spec, all_predicates)) > max_num_predicates:
        return None
    return tl_spec


def _generate_reach_avoid_specification(
    i: int,
    num_elements: int,
    len_predicates: int,
    all_predicates: list[str],
    num_clauses: int,
    max_num_predicates: int,
) -> str | None:
    ter_rep: str = base_n(i, 2, width=num_elements)
    specification_weights: list[int] = [int(bit) for bit in ter_rep]
    clause_weights = sort_temp_clause_weights(
        np.array(specification_weights), len_predicates, num_clauses
    )
    if not check_overlapping_column_weights(clause_weights):
        return None
    clause_spec: str = weights2temp_clause(clause_weights, all_predicates)
    if not clause_spec:
        return None
    tl_spec = f"{spot.simplify(spot.formula(clause_spec))}"
    if tl_spec and (tl_spec == "0" or tl_spec == "1"):
        return None
    if len(get_used_predicates(tl_spec, all_predicates)) > max_num_predicates:
        return None
    return tl_spec


def get_used_predicates(tl_spec: str, predicates: list[str]) -> list[str]:
    used_preds: list[str] = []
    for pred in predicates:
        if pred in tl_spec:
            used_preds.append(pred)
    return used_preds


def _combine_specs(args: tuple[str, str, list[str]]) -> str | None:
    f_spec, g_spec, predicates = args
    if f_spec and g_spec:
        combined_spec: str = f"F({f_spec}) & G({g_spec})"
        f_used_preds: list[str] = get_used_predicates(f_spec, predicates)
        g_used_preds: list[str] = get_used_predicates(g_spec, predicates)
        # Return none if there are overlapping predicates in F and G
        if len(set(f_used_preds + g_used_preds)) != len(f_used_preds) + len(
            g_used_preds
        ):
            return None

    elif f_spec:
        combined_spec: str = f"F({f_spec})"
    elif g_spec:
        combined_spec: str = f"G({g_spec})"
    else:
        return None
    simplified_spec: str = f"{spot.simplify(spot.formula(combined_spec))}"
    if (
        simplified_spec
        and (simplified_spec != "0" or simplified_spec != "1")
        and count_automaton_states(simplified_spec) > 1
    ):
        return simplified_spec
    return None


def generate_all_specifications(
    predicates: list[str],
    num_processes: int,
    num_clauses: int | None = None,
    max_num_predicates: int | None = None,
) -> list[str]:
    if num_clauses is None:
        num_clauses = len(predicates)
    if max_num_predicates is None:
        max_num_predicates = len(predicates)
    num_elements: int = len(predicates) * num_clauses
    total_number_of_specifications: int = 3**num_elements

    all_predicates = sorted(predicates)

    print("Generating all specifications...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        specs = pool.starmap(
            _generate_specification,
            [
                (
                    i,
                    num_elements,
                    len(predicates),
                    all_predicates,
                    num_clauses,
                    max_num_predicates,
                )
                for i in range(total_number_of_specifications)
            ],
        )
    specs = set(filter(None, specs))
    specs.add("")
    specifications: list[str] = sorted(specs, key=len)

    num_clause_specs: int = len(specifications)
    print(
        f"Generated {num_clause_specs} specifications with {len(predicates)} predicates and {num_clauses} clauses."
    )
    indices: list[int] = list(range(num_clause_specs))
    perms: list[tuple[int, int]] = list(itertools.permutations(indices, 2))

    spec_pairs = [
        (specifications[i], specifications[j], all_predicates) for i, j in perms
    ]
    print(f"Generating all combinations of specifications...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        all_specs = pool.map(_combine_specs, spec_pairs)
    out_specs: list[str] = sorted(set(filter(None, all_specs)), key=len)
    return out_specs


def generate_all_reach_avoid_specifications(
    predicates: list[str],
    num_processes: int,
    num_clauses: int | None = None,
    max_num_predicates: int | None = None,
) -> list[str]:
    if num_clauses is None:
        num_clauses = len(predicates)
    if max_num_predicates is None:
        max_num_predicates = len(predicates)
    num_elements: int = len(predicates) * num_clauses
    total_number_of_specifications: int = 2**num_elements

    all_predicates = sorted(predicates)

    print("Generating all specifications...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        specs = pool.starmap(
            _generate_reach_avoid_specification,
            [
                (
                    i,
                    num_elements,
                    len(predicates),
                    all_predicates,
                    num_clauses,
                    max_num_predicates,
                )
                for i in range(total_number_of_specifications)
            ],
        )
    specs = set(filter(None, specs))
    specs.add("")
    specifications: list[str] = sorted(specs, key=len)

    num_clause_specs: int = len(specifications)
    print(
        f"Generated {num_clause_specs} specifications with {len(predicates)} predicates and {num_clauses} clauses."
    )
    indices: list[int] = list(range(num_clause_specs))
    perms: list[tuple[int, int]] = list(itertools.permutations(indices, 2))

    spec_pairs = [
        (specifications[i], specifications[j], all_predicates) for i, j in perms
    ]
    print(f"Generating all combinations of specifications...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        all_specs = pool.map(_combine_specs, spec_pairs)
    out_specs: list[str] = sorted(set(filter(None, all_specs)), key=len)
    return out_specs
