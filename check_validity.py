"""Checking and repairing quantum strategies.

Grading happens in two stages.

1. **Validity gate.** What you submit must already be a quantum state and a set
   of POVMs, to within `TOL`. Anything that fails is rejected outright and
   scores nothing. Building objects that really are states and measurements is
   part of the exercise, so this step is not done for you.

2. **Repair.** Whatever passes the gate is then projected onto the exactly
   valid state and POVMs nearest to it, and only the repaired objects are
   scored. This is purely a standardization step: it guarantees that a
   last-digit rounding error can never be turned into apparent quantum
   advantage. A strategy that is genuinely valid is unchanged to machine
   precision, so repair never costs you anything.
"""

import numpy as np

TOL = 1e-6  # how far from valid a submission may be before it is rejected


def is_valid_povm(M, tol=TOL):
    """Is M a family of POVMs, with the outcome index at axis -3?"""
    M = np.asarray(M)
    if not np.all(np.isfinite(M)):
        return False
    if M.ndim < 3 or M.shape[-1] != M.shape[-2]:
        return False

    # Hermitian
    if not np.allclose(M, M.conj().swapaxes(-2, -1), atol=tol, rtol=0):
        return False

    # elements sum to the identity
    if not np.allclose(M.sum(axis=-3), np.eye(M.shape[-1]), atol=tol, rtol=0):
        return False

    # positive semidefinite
    return bool(np.all(np.linalg.eigvalsh(M) > -tol))


def is_valid_state(rho, tol=TOL):
    """Is rho a density matrix: Hermitian, positive semidefinite, trace one?"""
    rho = np.asarray(rho)
    if not np.all(np.isfinite(rho)):
        return False
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        return False
    if not np.allclose(rho, rho.conj().T, atol=tol, rtol=0):
        return False
    if abs(np.trace(rho) - 1.0) > tol:
        return False
    return bool(np.all(np.linalg.eigvalsh(rho) > -tol))


def repair_state(rho):
    """The nearest density matrix: Hermitize, clip negative eigenvalues, renormalize."""
    rho = np.asarray(rho, dtype=complex)
    d = rho.shape[-1]
    rho = (rho + rho.conj().T) / 2
    w, U = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return np.eye(d, dtype=complex) / d
    return (U * (w / w.sum())) @ U.conj().T


def repair_povm(M, ridge=1e-12):
    """The nearest POVM family, with the outcome index at axis -3.

    Hermitize, clip negative eigenvalues, then rescale by the symmetric sandwich
    S^(-1/2) M S^(-1/2), where S is the sum of the clipped elements.
    """
    M = np.asarray(M, dtype=complex)
    d = M.shape[-1]

    M = (M + M.conj().swapaxes(-2, -1))/2
    w, U = np.linalg.eigh(M)
    M = (U * np.clip(w, 0.0, None)[..., None, :]) @ U.conj().swapaxes(-2, -1)

    S = M.sum(axis=-3)
    w_S, U_S = np.linalg.eigh(S)
    scale = np.clip(w_S, ridge, None)[..., None, :] ** -0.5
    inv_sqrt = (U_S * scale) @ U_S.conj().swapaxes(-2, -1)
    M = inv_sqrt[..., None, :, :] @ M @ inv_sqrt[..., None, :, :]

    # a singular S leaves the sandwich summing to a projector rather than the identity
    M[..., 0, :, :] += np.eye(d) - M.sum(axis=-3)
    return M
