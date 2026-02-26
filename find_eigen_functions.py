#-----------------------------------------------
#
# Routines to find 1d eigenfunctions given a potential.
#
#------------------------------------------------

import numpy as np
from typing import Dict, Tuple, Optional, Callable

Array = np.ndarray

#-------------------------------------------------
# Use imaginary time/relaxation to propagate to the ground state.
#-------------------------------------------------
def imaginary_time_propagate_fft(
    psi0: Array,
    V_of_y_tau: Callable[[Array, float], Array],
    tau_grid: Array,
    grid,
    return_all: bool = True,
    renormalize_each_step: bool = True,
    E_ref: float = 0.0,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Imaginary-time propagation for ground state (FFT / periodic BC).
    Solves: d/dtau psi = -(H - E_ref) psi, with H = -1/2 d^2/dy^2 + V(y,tau)

    Notes:
    - Requires V to be time-independent for strict eigenstate convergence;
      time-dependent V is still definable but is a different problem.
    - Renormalizes each step by default.
    """
    y, k, dy = grid.y, grid.k, float(grid.dy)

    psi = np.array(psi0, dtype=np.complex128, copy=True)
    if psi.shape != y.shape:
        raise ValueError(f"psi0 has shape {psi.shape}, expected {y.shape}.")

    # normalize
    nrm = np.sqrt(np.sum(np.abs(psi)**2) * dy)
    if nrm == 0:
        raise ValueError("psi0 has zero norm.")
    psi /= nrm

    tau_grid = np.asarray(tau_grid, dtype=float)
    Nt = tau_grid.size
    if Nt < 2:
        raise ValueError("tau_grid must have at least 2 points.")
    dtaus = np.diff(tau_grid)
    if np.any(dtaus <= 0):
        raise ValueError("tau_grid must be strictly increasing.")

    psi_out = np.empty((Nt, psi.size), dtype=np.complex128) if return_all else None
    norms = np.empty(Nt, dtype=float)
    if return_all:
        psi_out[0] = psi
    norms[0] = np.sum(np.abs(psi)**2) * dy

    for n in range(Nt - 1):
        dtau = float(dtaus[n])
        tau_mid = float(tau_grid[n] + 0.5 * dtau)

        Vmid = np.asarray(V_of_y_tau(y, tau_mid), dtype=float)
        if Vmid.shape != y.shape:
            raise ValueError(f"V returned shape {Vmid.shape}, expected {y.shape}.")

        # half potential: exp(-(V - E_ref) dtau / 2)
        psi *= np.exp(-0.5 * dtau * (Vmid - E_ref))

        # kinetic in k-space: exp(-(k^2/2) dtau)
        psi_k = np.fft.fft(psi)
        psi_k *= np.exp(-0.5 * (k**2) * dtau)
        psi = np.fft.ifft(psi_k)

        # half potential again
        psi *= np.exp(-0.5 * dtau * (Vmid - E_ref))

        if renormalize_each_step:
            nrm = np.sqrt(np.sum(np.abs(psi)**2) * dy)
            psi /= nrm

        if return_all:
            psi_out[n + 1] = psi
        norms[n + 1] = np.sum(np.abs(psi)**2) * dy

    diagnostics = {"tau": tau_grid.copy(), "norm": norms}
    return (psi_out if return_all else psi), diagnostics


#---------------------------------------------------------------
#
# Find first n eigenstates by G-S at each time step.
#
#---------------------------------------------------------------
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

Array = np.ndarray

def gs_inner_product(phi: np.ndarray, psi: np.ndarray, dy: float) -> complex:
    """
    Discrete approximation to <phi|psi> = ∫ phi*(y) psi(y) dy.
    """
    return np.sum(np.conj(phi) * psi) * dy

def normalize_inplace(psi: Array, dy: float) -> float:
    """
    Normalize psi so that ∑|psi|^2 dy = 1.
    Returns the norm before normalization.
    """
    nrm2 = float(np.sum(np.abs(psi) ** 2) * dy)
    if nrm2 <= 0.0:
        raise ValueError("Wavefunction has zero norm.")
    nrm = float(np.sqrt(nrm2))
    psi /= nrm
    return nrm


def gram_schmidt_inplace(psi: Array, basis: List[Array], dy: float, passes: int = 1) -> None:
    """
    Modified Gram–Schmidt orthogonalization of psi against an orthonormal basis.
    psi <- psi - Σ |phi_i><phi_i|psi>

    passes=2 is sometimes helpful for higher excited states.
    """
    for _ in range(max(1, int(passes))):
        for phi in basis:
            psi -= phi * gs_inner_product(phi, psi, dy)


def imag_time_step_inplace_fft(
    psi: Array,
    y: Array,
    k: Array,
    V_of_y: Callable[[Array], Array],
    dtau: float,
    tau: float = 0.0, # time in V_of_y if not specified it is zero, to be compatible with time-varying potentials
    E_ref: float = 0.0,
) -> None:
    """
    One imaginary-time Strang step (FFT / periodic):
        psi <- exp(- (V-Eref) dt/2) exp(-T dt) exp(- (V-Eref) dt/2) psi
    with T eigenvalues k^2/2 in Fourier space.
    """
    V = np.asarray(V_of_y(y,tau), dtype=float)
    if V.shape != y.shape:
        raise ValueError(f"V_of_y returned shape {V.shape}, expected {y.shape}.")

    psi *= np.exp(-0.5 * dtau * (V - E_ref))

    psi_k = np.fft.fft(psi)
    psi_k *= np.exp(-0.5 * (k ** 2) * dtau)  # exp(-(k^2/2) dtau)
    psi[:] = np.fft.ifft(psi_k)

    psi *= np.exp(-0.5 * dtau * (V - E_ref))


def imag_time_step_inplace_dst(
    psi: Array,
    y: Array,
    k: Array,
    V_of_y: Callable[[Array], Array],
    dtau: float,
    tau: float = 0.0,
    E_ref: float = 0.0,
) -> None:
    """
    One imaginary-time Strang step (DST / Dirichlet).
    Assumes DST-I with norm="ortho" and k_n = n*pi/L on the interior grid.
    """
    from scipy.fft import dst, idst

    V = np.asarray(V_of_y(y,tau), dtype=float)
    if V.shape != y.shape:
        raise ValueError(f"V_of_y returned shape {V.shape}, expected {y.shape}.")

    psi *= np.exp(-0.5 * dtau * (V - E_ref))

    # Real/imag DST separately
    a_re = dst(psi.real, type=1, norm="ortho")
    a_im = dst(psi.imag, type=1, norm="ortho")

    decay = np.exp(-0.5 * (k ** 2) * dtau)   # exp(-(k^2/2) dtau)
    a_re *= decay
    a_im *= decay

    psi[:] = idst(a_re, type=1, norm="ortho") + 1j * idst(a_im, type=1, norm="ortho")

    psi *= np.exp(-0.5 * dtau * (V - E_ref))


def find_states_imag_time(
    n_states: int,
    grid,
    V_of_y: Callable[[Array], Array],          # time-independent V(y)
    energy_fn: Callable[[Array], float],       # retur
    method: str = "fft",                       # "fft" or "dst"
    dtau: float = 5.0e-4,
    max_steps: int = 100_000,
    check_every: int = 200,
    energy_tol: float = 1.0e-8,
    reorth_every: int = 1,
    gs_passes: int = 2,                        # Gram–Schmidt passes (1 or 2)
    seed: int = 0,
    E_ref: float = 0.0,
    psi_guesses: Optional[List[Optional[np.ndarray]]] = None, # guesses for eigenfunctions if appropriate
) -> Dict[str, object]:
    """
    Find lowest n_states eigenstates using imaginary-time propagation with
    Gram–Schmidt orthogonalization against previously found states.

    Returns:
      - psis: list of eigenstates (each (Ny,) complex)
      - energies: array of <H> estimates
      - steps: array of step counts to convergence (-1 if not converged)
      - energy_traces: list of arrays of energies sampled during convergence
    """
    if n_states < 1:
        raise ValueError("n_states must be >= 1.")
    if dtau <= 0:
        raise ValueError("dtau must be > 0.")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    if check_every < 1:
        raise ValueError("check_every must be >= 1.")
    if reorth_every < 1:
        raise ValueError("reorth_every must be >= 1.")

    y, k, dy = grid.y, grid.k, float(grid.dy)
    Ny = y.size

    rng = np.random.default_rng(seed)

    # Choose stepper
    m = method.lower()
    if m == "fft":
        stepper = lambda psi: imag_time_step_inplace_fft(psi, y, k, V_of_y, dtau, E_ref=E_ref)
    elif m == "dst":
        stepper = lambda psi: imag_time_step_inplace_dst(psi, y, k, V_of_y, dtau, E_ref=E_ref)
    else:
        raise ValueError("method must be 'fft' or 'dst'.")

    psis: List[Array] = []
    energies = np.full(n_states, np.nan, dtype=float)
    steps = np.full(n_states, -1, dtype=int)
    energy_traces: List[Array] = []

    for s in range(n_states):
        # Initial guess: user-provided (if available), otherwise random
        use_guess = (
                        psi_guesses is not None
                    and s < len(psi_guesses)
                    and psi_guesses[s] is not None
                    )

        if use_guess:
            psi = np.asarray(psi_guesses[s], dtype=np.complex128).copy()
            if psi.shape != (Ny,):
                raise ValueError(f"psi_guesses[{s}] has shape {psi.shape}, expected {(Ny,)}")
        else:
            psi = (rng.normal(size=Ny) + 1j * rng.normal(size=Ny)).astype(np.complex128)

        # Orthogonalize to previous states + normalize
        if psis:
            gram_schmidt_inplace(psi, psis, dy, passes=gs_passes)
        normalize_inplace(psi, dy)

        last_E = None
        trace = []

        for step in range(1, max_steps + 1):
            stepper(psi)

            # Re-orthogonalize periodically
            if psis and (step % reorth_every == 0):
                gram_schmidt_inplace(psi, psis, dy, passes=gs_passes)

            normalize_inplace(psi, dy)

            if step % check_every == 0:
                E = float(energy_fn(psi))
                trace.append(E)
                if last_E is not None and abs(E - last_E) < energy_tol:
                    energies[s] = E
                    steps[s] = step
                    break
                last_E = E

        # Store the state
        psis.append(psi.copy())

        # Energy at final
        if np.isnan(energies[s]):
            energies[s] = float(energy_fn(psi))
        energy_traces.append(np.array(trace, dtype=float))

    return {"psis": psis, "energies": energies, "steps": steps, "energy_traces": energy_traces}
