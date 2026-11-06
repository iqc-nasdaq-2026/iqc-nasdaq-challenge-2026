"""Helpers for the nonlocal games in this challenge.

A two-player nonlocal game is specified by

    V    payoff tensor, V[x, y, a, b] is the payoff when the referee asks
         (x, y) and the players answer (a, b)
    pi   question distribution, pi[x, y] = p(x, y)

A quantum strategy is specified by

    rho  density matrix of the shared state, shape (d_A*d_B, d_A*d_B)
    M    Alice's POVMs, M[x, a] is the element for answer a given question x
    N    Bob's POVMs, N[y, b] is the element for answer b given question y
"""

import hashlib
from pathlib import Path

import numpy as np
from scipy.special import betaincinv

# --- Problem 5 parameters (fixed; do not change) ---------------------------
MASTER_SEED = b"IQC-NASDAQ-2026"
NUM_QUESTIONS = 10  # questions per player
NUM_ANSWERS = 2  # answers per player
DIM = 5  # local dimension cap: d_A = d_B = DIM
TOLERANCE = 1e-9  # a witness must beat the classical value by this much
MAX_GAMES = 100_000
NUM_GAMES_CHOICES = (10, 100, 1_000, 10_000, 100_000)  # the allowed values of N
# covers the ~24% upper bound on the d <= 5 advantage rate at N = MAX_GAMES
MAX_WITNESSES = 25_000
# ---------------------------------------------------------------------------


def expected_payoff(V, pi, rho, M, N):
    """Expected payoff of the quantum strategy (rho, M, N) in the game (V, pi)."""
    V, pi = np.asarray(V, dtype=float), np.asarray(pi, dtype=float)
    rho, M, N = np.asarray(rho), np.asarray(M), np.asarray(N)
    d_A, d_B = M.shape[-1], N.shape[-1]
    # rho4[i, k, j, l] = <i,k| rho |j,l>, so tr(rho (M x N)) sums rho4 * M.T * N.T
    rho4 = rho.reshape(d_A, d_B, d_A, d_B)
    probs = np.einsum("ikjl,xaji,yblk->xyab", rho4, M, N).real
    return float(np.sum(pi[:, :, None, None]*V*probs))


def get_classical_best(V, pi):
    """Exact best expected payoff over all classical strategies.

    Enumerates Alice's deterministic strategies; Bob's best response to each is
    computed question by question. Shared randomness never beats the best
    deterministic pair, so this is exact.
    """
    V = np.asarray(V, dtype=float)
    m_A, m_B, n_A, n_B = V.shape
    W = V * np.asarray(pi, dtype=float)[:, :, None, None]

    # strategies[s, x] = Alice's answer to question x under strategy s
    strategies = (np.arange(n_A**m_A)[:, None] // n_A ** np.arange(m_A)) % n_A

    x = np.arange(m_A)[None, :, None, None]
    y = np.arange(m_B)[None, None, :, None]
    b = np.arange(n_B)[None, None, None, :]
    payoffs = W[x, y, strategies[:, :, None, None], b]  # (s, x, y, b)
    return float(payoffs.sum(axis=1).max(axis=-1).sum(axis=-1).max())



# --- the random games of Problem 5 -----------------------------------------


def _bits(index, num_bits):
    """A fixed pseudorandom bit stream for `index`, independent of any library."""
    blocks, counter = [], 0
    while 256 * len(blocks) < num_bits:
        seed = MASTER_SEED + int(index).to_bytes(8, "big") + counter.to_bytes(8, "big")
        blocks.append(hashlib.sha256(seed).digest())
        counter += 1
    stream = np.frombuffer(b"".join(blocks), dtype=np.uint8)
    return np.unpackbits(stream)[:num_bits]


def get_game(index):
    """Payoff tensor V[x, y, a, b] of random game number `index`. Entries are 0 or 1.

    Games are independent of each other and any game can be regenerated on its
    own, so `get_game(7)` is the same tensor no matter what you generated before.
    """
    num_bits = NUM_QUESTIONS**2 * NUM_ANSWERS**2
    shape = (NUM_QUESTIONS, NUM_QUESTIONS, NUM_ANSWERS, NUM_ANSWERS)
    return _bits(index, num_bits).reshape(shape)


def fingerprint(num_games=1000):
    """Checksum of the first `num_games` games, for verifying your copy of the data."""
    games = np.stack([get_game(i) for i in range(num_games)])
    return hashlib.sha256(games.tobytes()).hexdigest()[:16]


def classical_values(num_games=MAX_GAMES):
    """Our exact classical values for the first `num_games` random games."""
    if not 0 < num_games <= MAX_GAMES:
        raise ValueError(f"num_games must lie in [1, {MAX_GAMES}]")
    path = Path(__file__).with_name("classical_values.npz")
    return np.load(path)["classical_values"][:num_games]


def score(num_witnesses, num_games, confidence=0.05):
    """The Problem 5 score: a lower confidence bound on the advantage rate.

    This is the Clopper-Pearson lower bound for `num_witnesses` successes out of
    `num_games` trials. It rises with both, so examining more games and proving
    more of them both help, and it can never exceed the true rate.
    """
    if num_witnesses <= 0:
        return 0.0
    return float(betaincinv(num_witnesses, num_games - num_witnesses + 1, confidence))


# --- Problem 5 submissions -------------------------------------------------


def pad_strategy(rho, M, N, dim=DIM):
    """Re-express a strategy of local dimension d <= dim in local dimension dim.

    The extra dimensions are left unoccupied by the state and swept into answer
    0 of every POVM, so the expected payoff is unchanged.
    """
    rho, M, N = np.asarray(rho), np.asarray(M), np.asarray(N)
    d = M.shape[-1]
    if d > dim:
        raise ValueError(f"local dimension {d} exceeds the cap of {dim}")

    big_rho = np.zeros((dim, dim, dim, dim), dtype=complex)
    big_rho[:d, :d, :d, :d] = rho.reshape(d, d, d, d)

    def embed(P):
        big_P = np.zeros(P.shape[:-2] + (dim, dim), dtype=complex)
        big_P[..., :d, :d] = P
        big_P[..., 0, d:, d:] = np.eye(dim - d)
        return big_P

    return big_rho.reshape(dim * dim, dim * dim), embed(M), embed(N)


def save_witnesses(filename, witnesses, num_games):
    """Write a Problem 5 submission.

    `witnesses` maps a game index to a strategy `(rho, M, N)` for that game, and
    `num_games` is the number of games you examined: you are claiming witnesses
    for these games out of games 0, 1, ..., num_games - 1.

    Strategies of local dimension below the cap are padded up to it.
    """
    if num_games not in NUM_GAMES_CHOICES:
        raise ValueError(f"num_games must be one of {NUM_GAMES_CHOICES}")
    indices = sorted(witnesses)
    if len(indices) > MAX_WITNESSES:
        raise ValueError(f"at most {MAX_WITNESSES} witnesses, got {len(indices)}")
    if indices and (indices[0] < 0 or indices[-1] >= num_games):
        raise ValueError(f"game indices must lie in [0, {num_games})")

    povm_shape = (NUM_QUESTIONS, NUM_ANSWERS, DIM, DIM)
    all_rho = np.zeros((len(indices), DIM**2, DIM**2), dtype=complex)
    all_M = np.zeros((len(indices),) + povm_shape, dtype=complex)
    all_N = np.zeros((len(indices),) + povm_shape, dtype=complex)

    for n, i in enumerate(indices):
        rho, M, N = pad_strategy(*witnesses[i])
        if M.shape != povm_shape or N.shape != povm_shape:
            raise ValueError(
                f"game {i}: POVM shapes {M.shape} and {N.shape}, expected {povm_shape}"
            )
        all_rho[n], all_M[n], all_N[n] = rho, M, N

    np.savez_compressed(
        filename,
        num_games=np.int64(num_games),
        indices=np.array(indices, dtype=np.int64),
        rho=all_rho,
        M=all_M,
        N=all_N,
    )
