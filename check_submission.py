"""Check a submission and score it.

Run this on the directory holding your problem_*.npz files before you submit:

    uv run python check_submission.py --directory .

Problems 1, 2, 3 and 5 are scored exactly as we will score them, so what this
prints is what you will get. Problem 4 is only checked for format, since
printing its score would give away the answer.
"""

import argparse
from pathlib import Path

import numpy as np

from check_validity import (
    TOL,
    is_valid_povm,
    is_valid_state,
    repair_povm,
    repair_state,
)
from games import (
    DIM,
    MAX_WITNESSES,
    NUM_ANSWERS,
    NUM_GAMES_CHOICES,
    NUM_QUESTIONS,
    TOLERANCE,
    classical_values,
    expected_payoff,
    get_classical_best,
    get_game,
    score,
)


class Rejected(Exception):
    pass


def scored_payoff(V, pi, rho, M, N, dim_cap=DIM):
    """Gate the strategy on near-validity, repair it, and return its payoff.

    Raises Rejected with a reason if the submitted objects are not already a
    state and POVMs to within TOL.
    """
    rho, M, N = np.asarray(rho), np.asarray(M), np.asarray(N)
    if M.ndim != 4 or N.ndim != 4:
        raise Rejected("M and N must each have shape (questions, answers, d, d)")
    if M.shape[:2] != (V.shape[0], V.shape[2]) or N.shape[:2] != (V.shape[1], V.shape[3]):
        raise Rejected(
            f"POVM shapes {M.shape} and {N.shape} do not match the game {V.shape}"
        )
    if M.shape[-2:] != N.shape[-2:]:
        raise Rejected("Alice and Bob must use the same local dimension")

    d = M.shape[-1]
    if M.shape[-2] != d:
        raise Rejected("POVM elements must be square")
    if d > dim_cap:
        raise Rejected(f"local dimension {d} exceeds the cap of {dim_cap}")
    if rho.shape != (d * d, d * d):
        raise Rejected(f"rho has shape {rho.shape}, expected {(d * d, d * d)}")

    if not is_valid_state(rho):
        raise Rejected(f"rho is not a density matrix to within {TOL:g}")
    if not is_valid_povm(M):
        raise Rejected(f"M is not a set of POVMs to within {TOL:g}")
    if not is_valid_povm(N):
        raise Rejected(f"N is not a set of POVMs to within {TOL:g}")

    return expected_payoff(V, pi, repair_state(rho), repair_povm(M), repair_povm(N))


def load(directory, name, keys):
    path = directory / name
    if not path.is_file():
        raise Rejected(f"missing file: {path}")
    try:
        data = np.load(path)
        missing = [k for k in keys if k not in data]
        if missing:
            raise Rejected(f"{name} is missing key(s): {', '.join(missing)}")
        return [data[k] for k in keys]
    except Rejected:
        raise
    except Exception as exc:
        raise Rejected(f"could not read {name}: {exc}") from exc


# --- the games that are fixed by the problem statements --------------------

PI = np.ones((2, 2))/4

V_CHSH = np.array([
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
]).reshape(2, 2, 2, 2)

RHO_1 = np.array([
    [0.5, 0.0, 0.0, -0.5],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [-0.5, 0.0, 0.0, 0.5],
])

V_TRADING = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
]).reshape(2, 2, 2, 2)


def problem_1(directory):
    M, N = load(directory, "problem_1.npz", ("M", "N"))
    payoff = scored_payoff(V_CHSH, PI, RHO_1, M, N, dim_cap=2)
    return f"win probability {payoff:.6f} (classical 0.75)", float(payoff >= 0.75 + TOLERANCE)


def problem_2(directory):
    rho, M, N = load(directory, "problem_2.npz", ("rho", "M", "N"))
    payoff = scored_payoff(V_TRADING, PI, rho, M, N)
    advantage = payoff - get_classical_best(V_TRADING, PI)
    return f"payoff {payoff:.6f}, advantage {advantage:+.6f}", max(advantage, 0.0)


def problem_3(directory):
    V, rho, M, N = load(directory, "problem_3.npz", ("V", "rho", "M", "N"))
    if V.shape != (2, 2, 2, 2):
        raise Rejected(f"V has shape {V.shape}, expected (2, 2, 2, 2)")
    if not np.all(np.isin(V, (0, 1))):
        raise Rejected("every payoff in V must be 0 or 1")
    V = np.asarray(V).real.astype(float)
    advantage = scored_payoff(V, PI, rho, M, N) - get_classical_best(V, PI)
    return f"advantage {advantage:+.6f}", max(advantage, 0.0)


def problem_4_answer(directory):
    """The submitted count, validated but not scored."""
    (count,) = load(directory, "problem_4.npz", ("num_games_with_quantum_advantage",))
    count = np.asarray(count)
    if count.size != 1 or not np.all(np.isreal(count)):
        raise Rejected("the answer must be a single integer")
    count = float(count.reshape(()).real)
    if count != int(count):
        raise Rejected("the answer must be a whole number")
    if not 0 <= count <= 65536:
        raise Rejected("the answer must lie between 0 and 65536")
    return int(count)


def problem_4(directory):
    return f"answer {problem_4_answer(directory)} (scored separately)", None


def problem_5(directory):
    num_games, indices = load(directory, "problem_5.npz", ("num_games", "indices"))
    if np.asarray(num_games).size != 1:
        raise Rejected("num_games must be a single integer")
    num_games = int(np.asarray(num_games).reshape(()))
    if num_games not in NUM_GAMES_CHOICES:
        raise Rejected(f"num_games must be one of {NUM_GAMES_CHOICES}, got {num_games}")

    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.size > MAX_WITNESSES:
        raise Rejected(
            f"indices must be a one-dimensional array of at most {MAX_WITNESSES} games"
        )
    if not np.all(np.isreal(indices)) or not np.all(indices == np.round(indices.real)):
        raise Rejected("indices must be whole numbers")
    indices = indices.real.astype(np.int64)
    if len(np.unique(indices)) != len(indices):
        raise Rejected("every game may appear at most once")
    if len(indices) and (indices.min() < 0 or indices.max() >= num_games):
        raise Rejected(f"indices must lie in [0, {num_games})")

    rho, M, N = load(directory, "problem_5.npz", ("rho", "M", "N"))
    expected = {
        "rho": (len(indices), DIM**2, DIM**2),
        "M": (len(indices), NUM_QUESTIONS, NUM_ANSWERS, DIM, DIM),
        "N": (len(indices), NUM_QUESTIONS, NUM_ANSWERS, DIM, DIM),
    }
    for name, array in (("rho", rho), ("M", M), ("N", N)):
        if array.shape != expected[name]:
            raise Rejected(f"{name} has shape {array.shape}, expected {expected[name]}")

    omega_c = classical_values(num_games)
    pi = np.ones((NUM_QUESTIONS, NUM_QUESTIONS))/NUM_QUESTIONS**2
    verified, total_advantage, rejected = 0, 0.0, []
    for n, i in enumerate(indices):
        try:
            payoff = scored_payoff(get_game(i).astype(float), pi, rho[n], M[n], N[n])
        except Rejected as exc:
            rejected.append(f"game {i}: {exc}")
            continue
        if payoff >= omega_c[i] + TOLERANCE:
            verified += 1
            total_advantage += payoff - omega_c[i]
        else:
            rejected.append(
                f"game {i}: payoff {payoff:.12f} does not reach "
                f"{omega_c[i]:.12f} + {TOLERANCE:g}"
            )

    detail = (
        f"{verified}/{len(indices)} witnesses verified out of N = {num_games} games, "
        f"total advantage {total_advantage:.6f}"
    )
    if rejected:
        detail += "\n      rejected: " + "\n      rejected: ".join(rejected[:10])
        if len(rejected) > 10:
            detail += f"\n      ... and {len(rejected) - 10} more"
    return detail, score(verified, num_games)


PROBLEMS = [problem_1, problem_2, problem_3, problem_4, problem_5]


def report(directory, problems=None):
    """Print a line per problem and return the list of scores (None if unscored)."""
    scores = []
    for n, problem in enumerate(problems or PROBLEMS, start=1):
        try:
            detail, value = problem(directory)
        except Rejected as exc:
            print(f"Problem {n}: REJECTED  {exc}")
            scores.append(0.0)
            continue
        except Exception as exc:  # a malformed submission must not stop the other problems
            print(f"Problem {n}: REJECTED  unexpected {type(exc).__name__}: {exc}")
            scores.append(0.0)
            continue
        print(f"Problem {n}: score {'-' if value is None else f'{value:.6f}'}   {detail}")
        scores.append(value)
    return scores


def parse_directory():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    directory = parser.parse_args().directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    return directory


if __name__ == "__main__":
    directory = parse_directory()
    print(f"Checking submission in {directory}\n")
    report(directory)
