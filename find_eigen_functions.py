#-----------------------------------------------
#
# Routines to find 1d eigenfunctions given a potential.
#
#------------------------------------------------

import numpy as np
from itertools import permutations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


#-------------------------------------------------------------
#
# Facilities for specific use on periodic potentials.
#
#-------------------------------------------------------------
# -----------------------------
# Bloch-sector projector
# -----------------------------
class BlochProjector:
    """
    Project onto the eigenspace of translation-by-a operator T_a with eigenvalue exp(i 2π m / Ncells).

    Assumptions (validated):
      - periodic grid with uniform spacing dy
      - domain length L = Ny * dy
      - L = Ncells * a (integer number of lattice cells in the box)
      - a/dy is (near) integer so T_a is a pure roll
    """
    def __init__(self, *, dy: float, Ny: int, a: float, m: int, tol: float = 1.0e-10):
        self.dy = float(dy)
        self.Ny = int(Ny)
        self.a = float(a)
        if self.a <= 0:
            raise ValueError("a must be > 0.")

        self.L = self.Ny * self.dy

        # commensurability: Ncells = L/a must be integer
        Ncells_f = self.L / self.a
        Ncells = int(np.rint(Ncells_f))
        if abs(Ncells_f - Ncells) > 1e-10:
            raise ValueError(
                f"Need L/a integer for Bloch projection. Got L={self.L}, a={self.a}, L/a={Ncells_f}."
            )
        if Ncells < 1:
            raise ValueError("Ncells must be >= 1.")
        self.Ncells = Ncells

        # grid shift per cell
        shift_f = self.a / self.dy
        shift = int(np.rint(shift_f))
        if abs(shift_f - shift) > 1e-10:
            raise ValueError(
                f"Need a/dy integer for roll-based translation. Got a={self.a}, dy={self.dy}, a/dy={shift_f}."
            )
        if (shift * self.Ncells) != self.Ny:
            # This should hold if both conditions above are met, but keep a crisp error if not.
            raise ValueError(
                f"Inconsistent: shift*Ncells != Ny. shift={shift}, Ncells={self.Ncells}, Ny={self.Ny}."
            )
        self.shift = shift

        # normalize m into [0, Ncells-1]
        self.m = int(m) % self.Ncells

        # phases for character projection
        n = np.arange(self.Ncells, dtype=int)
        self.phases = np.exp(-1j * 2.0*np.pi * self.m * n / self.Ncells).astype(np.complex128)
        self.invN = 1.0 / float(self.Ncells)

        # eigenvalue of T_a in this sector
        self.lambda_a = np.exp(1j * 2.0*np.pi * self.m / self.Ncells)

    @property
    def k(self) -> float:
        # discrete crystal momentum corresponding to this sector
        return 2.0*np.pi * self.m / self.L

    def __call__(self, psi: Array) -> None:
        """
        In-place projection: psi <- P_m psi
        """
        acc = np.zeros_like(psi, dtype=np.complex128)
        # P_m = (1/N) Σ_n exp(-i 2π m n/N) T_a^n
        for n, phase in enumerate(self.phases):
            acc += phase * np.roll(psi, n * self.shift)
        psi[:] = self.invN * acc


#------------------------------------
#
# FFT-based Bloch Projector, which can run much more quickly.
#
#------------------------------------
class BlochProjectorFFT:
    """
    Project onto the eigenspace of translation-by-a operator T_a with eigenvalue exp(i 2π m / Ncells),
    using an FFT along the cell index (much faster than summing rolls).

    Works in-place on:
      - psi: shape (Ny,) complex
      - Psi: shape (Ny, M) complex (projects each column) WITHOUT Python loops

    Assumptions (validated):
      - periodic grid with uniform spacing dy
      - domain length L = Ny * dy
      - L = Ncells * a (integer number of lattice cells in the box)
      - a/dy is (near) integer so each cell has exactly `shift` grid points
      - Ny = Ncells * shift
    """
    def __init__(self, *, dy: float, Ny: int, a: float, m: int, tol: float = 1.0e-10):
        self.dy = float(dy)
        self.Ny = int(Ny)
        self.a = float(a)
        if self.a <= 0:
            raise ValueError("a must be > 0.")

        self.L = self.Ny * self.dy

        # commensurability: Ncells = L/a must be integer
        Ncells_f = self.L / self.a
        Ncells = int(np.rint(Ncells_f))
        if abs(Ncells_f - Ncells) > tol:
            raise ValueError(
                f"Need L/a integer for Bloch projection. Got L={self.L}, a={self.a}, L/a={Ncells_f}."
            )
        if Ncells < 1:
            raise ValueError("Ncells must be >= 1.")
        self.Ncells = Ncells

        # grid shift per cell
        shift_f = self.a / self.dy
        shift = int(np.rint(shift_f))
        if abs(shift_f - shift) > tol:
            raise ValueError(
                f"Need a/dy integer for cell decomposition. Got a={self.a}, dy={self.dy}, a/dy={shift_f}."
            )
        if (shift * self.Ncells) != self.Ny:
            raise ValueError(
                f"Inconsistent: shift*Ncells != Ny. shift={shift}, Ncells={self.Ncells}, Ny={self.Ny}."
            )
        self.shift = shift

        # normalize m into [0, Ncells-1]
        self.m = int(m) % self.Ncells
        self.invN = 1.0 / float(self.Ncells)

        # eigenvalue of T_a in this sector
        self.lambda_a = np.exp(1j * 2.0*np.pi * self.m / self.Ncells)

        # reconstruction phase for inverse DFT of a single retained mode
        j = np.arange(self.Ncells, dtype=int)
        self._recon_phase = np.exp(1j * 2.0*np.pi * self.m * j / self.Ncells).astype(np.complex128)

    @property
    def k(self) -> float:
        return 2.0*np.pi * self.m / self.L

    def __call__(self, psi: Array) -> None:
        """
        In-place projection:
          - if psi.ndim==1: psi <- P_m psi
          - if psi.ndim==2: psi[:,j] <- P_m psi[:,j] for all columns j (vectorized)
        """
        if psi.ndim == 1:
            Psi = psi.reshape(self.Ncells, self.shift)              # (Ncells, shift)
            Psi_k = np.fft.fft(Psi, axis=0)                        # (Ncells, shift)
            mode = Psi_k[self.m]                                   # (shift,)
            Psi[:] = (self.invN * self._recon_phase[:, None]) * mode[None, :]
            return

        if psi.ndim == 2:
            Ny, M = psi.shape
            if Ny != self.Ny:
                raise ValueError(f"Expected Psi.shape[0]==Ny={self.Ny}, got {Ny}.")
            Psi = psi.reshape(self.Ncells, self.shift, M)           # (Ncells, shift, M)
            Psi_k = np.fft.fft(Psi, axis=0)                        # FFT along cell axis
            mode = Psi_k[self.m]                                   # (shift, M)
            Psi[:] = (self.invN * self._recon_phase[:, None, None]) * mode[None, :, :]
            return

        raise ValueError("psi must be a 1D or 2D array.")


# -----------------------------
# Periodic Bloch-sector eigenfinder
# -----------------------------
def find_bloch_states(
    *,
    n_bands: int,
    m_list: Sequence[int],
    grid,
    V_of_y: Callable[[Array, float], Array],     # V(y, tau) but for eigenstates should be time-independent in tau
    energy_fn: Callable[[Array], float],         # returns <H> estimate (should match your kinetic+V conventions)
    a: float = 1.0,
    dtau: float = 5.0e-4,
    max_steps: int = 200_000,
    check_every: int = 200,
    energy_tol: float = 1.0e-10,
    reorth_every: int = 1,
    gs_passes: int = 2,
    proj_every: int = 1,                          # how often to apply Bloch projection (1 = every step)
    seed: int = 0,
    E_ref: float = 0.0,
    psi_guesses: Optional[Dict[Tuple[int, int], Array]] = None,
    allow_repeat_restart: bool = True,
    overlap_tol: float = 1.0e-6,
    restart_tries: int = 5,
) -> Dict[str, object]:
    """
    Find Bloch eigenstates for a periodic potential by imaginary-time propagation
    *within each discrete Bloch sector* m (k = 2π m / L), using GS deflation.

    Returns a dict with:
      - k_vals: (Nk,) float array
      - energies: (Nk, n_bands) float array
      - psis: list of list: psis[ik][n] is (Ny,) complex array (band n at sector ik)
      - steps: (Nk, n_bands) int array
      - energy_traces: psis[ik][n] trace arrays

    Notes:
      - This targets the *lowest n_bands states within each k sector*, not a global ordering.
      - Uses FFT periodic stepper (your kinetic convention: T = k^2/2).
      - V_of_y is called as V_of_y(y, tau); tau is passed but can be ignored for static V.
    """
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1.")
    if dtau <= 0:
        raise ValueError("dtau must be > 0.")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")

    y, kgrid, dy = grid.y, grid.k, float(grid.dy)
    Ny = y.size
    rng = np.random.default_rng(seed)

    # Local stepper using your existing FFT Strang step
    def stepper(psi: Array) -> None:
        imag_time_step_inplace_fft(
            psi=psi, y=y, k=kgrid, V_of_y=V_of_y,
            dtau=dtau, tau=0.0, E_ref=E_ref
        )

    Nk = len(m_list)
    k_vals = np.zeros(Nk, dtype=float)
    energies = np.full((Nk, n_bands), np.nan, dtype=float)
    steps = np.full((Nk, n_bands), -1, dtype=int)
    psis_out: List[List[Array]] = []
    traces_out: List[List[Array]] = []

    for ik, m in enumerate(m_list):
        projector = BlochProjector(dy=dy, Ny=Ny, a=a, m=int(m))
        k_vals[ik] = projector.k

        band_states: List[Array] = []
        band_traces: List[Array] = []

        for band in range(n_bands):
            # --- build an initial guess ---
            key = (int(m), int(band))
            if psi_guesses is not None and key in psi_guesses:
                psi = np.asarray(psi_guesses[key], dtype=np.complex128).copy()
                if psi.shape != (Ny,):
                    raise ValueError(f"psi_guesses[{key}] has shape {psi.shape}, expected {(Ny,)}")
            else:
                # sector-biased random: start from plane wave in this sector + noise
                L = Ny * dy
                plane = np.exp(1j * (2.0*np.pi*projector.m/L) * y)
                noise = (rng.normal(size=Ny) + 1j*rng.normal(size=Ny)).astype(np.complex128)
                psi = plane + 0.05 * noise

            # enforce sector and orthogonality at start
            projector(psi)
            if band_states:
                gram_schmidt_inplace(psi, band_states, dy, passes=gs_passes)
            normalize_inplace(psi, dy)

            last_E = None
            trace: List[float] = []

            # helper: check if we've collapsed onto an existing band state
            def max_overlap_with_basis(p: Array, basis: List[Array]) -> float:
                if not basis:
                    return 0.0
                ovs = [abs(gs_inner_product(phi, p, dy)) for phi in basis]
                return float(np.max(ovs))

            tries = 0
            while True:
                tries += 1
                if tries > max(1, int(restart_tries)):
                    # accept whatever we got; user can inspect residual/overlaps externally
                    break

                last_E = None
                trace.clear()

                for step in range(1, max_steps + 1):
                    stepper(psi)

                    # optional: enforce Bloch sector during evolution
                    if proj_every > 0 and (step % proj_every == 0):
                        projector(psi)

                    # deflate within this sector (bands)
                    if band_states and (step % reorth_every == 0):
                        gram_schmidt_inplace(psi, band_states, dy, passes=gs_passes)

                    normalize_inplace(psi, dy)

                    if step % check_every == 0:
                        E = float(energy_fn(psi))
                        trace.append(E)

                        # overlap guard: if we’re numerically drifting back onto an earlier band, restart
                        if allow_repeat_restart and band_states:
                            ov = max_overlap_with_basis(psi, band_states)
                            if ov > overlap_tol:
                                # re-randomize and re-project in the same sector
                                noise = (rng.normal(size=Ny) + 1j*rng.normal(size=Ny)).astype(np.complex128)
                                psi[:] = psi + 0.2 * noise
                                projector(psi)
                                gram_schmidt_inplace(psi, band_states, dy, passes=gs_passes)
                                normalize_inplace(psi, dy)
                                last_E = None
                                trace.clear()
                                break  # break step loop, try again (tries loop)

                        if last_E is not None and abs(E - last_E) < energy_tol:
                            energies[ik, band] = E
                            steps[ik, band] = step
                            break
                        last_E = E
                else:
                    # max_steps exhausted
                    energies[ik, band] = float(energy_fn(psi))
                    steps[ik, band] = -1

                # if converged (or we ran out) and not obviously a repeat, we’re done
                if (not allow_repeat_restart) or (not band_states) or (max_overlap_with_basis(psi, band_states) <= overlap_tol):
                    break

                # otherwise restart with new noise and try again
                noise = (rng.normal(size=Ny) + 1j*rng.normal(size=Ny)).astype(np.complex128)
                psi[:] = np.exp(1j * projector.k * y) + 0.1 * noise
                projector(psi)
                gram_schmidt_inplace(psi, band_states, dy, passes=gs_passes)
                normalize_inplace(psi, dy)

            band_states.append(psi.copy())
            band_traces.append(np.array(trace, dtype=float))

        psis_out.append(band_states)
        traces_out.append(band_traces)

    return {
        "m_list": np.array([int(m) for m in m_list], dtype=int),
        "k_vals": k_vals,
        "energies": energies,
        "steps": steps,
        "psis": psis_out,
        "energy_traces": traces_out,
    }


#-----------------------------------------------------------------------|
#----- New Bloch state finder, which uses the FFT projection method ----|
#-----------------------------------------------------------------------|
def find_bloch_states_fft(
    *,
    n_bands: int,
    m_list: Sequence[int],
    grid,
    V_of_y: Callable[[Array, float], Array],
    energy_fn: Callable[[Array], float],
    a: float = 1.0,
    dtau: float = 5.0e-4,
    max_steps: int = 200_000,
    check_every: int = 200,
    energy_tol: float = 1.0e-10,
    reorth_every: int = 1,
    gs_passes: int = 2,
    proj_every: int = 1,
    seed: int = 0,
    E_ref: float = 0.0,
    psi_guesses: Optional[Dict[Tuple[int, int], Array]] = None,
    allow_repeat_restart: bool = True,
    overlap_tol: float = 1.0e-6,
    restart_tries: int = 5,
) -> Dict[str, object]:
    """
    Same as find_bloch_states(), but uses BlochProjectorFFT for much faster Bloch-sector projection.

    This preserves your output format and the overall algorithm (imag-time + GS deflation),
    but typically reduces wall-clock time substantially when Ncells > 1.
    """
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1.")
    if dtau <= 0:
        raise ValueError("dtau must be > 0.")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")

    y, kgrid, dy = grid.y, grid.k, float(grid.dy)
    Ny = y.size
    rng = np.random.default_rng(seed)

    # Local stepper using your existing FFT Strang step
    def stepper(psi: Array) -> None:
        imag_time_step_inplace_fft(
            psi=psi, y=y, k=kgrid, V_of_y=V_of_y,
            dtau=dtau, tau=0.0, E_ref=E_ref
        )

    Nk = len(m_list)
    k_vals = np.zeros(Nk, dtype=float)
    energies = np.full((Nk, n_bands), np.nan, dtype=float)
    steps = np.full((Nk, n_bands), -1, dtype=int)
    psis_out: List[List[Array]] = []
    traces_out: List[List[Array]] = []

    for ik, m in enumerate(m_list):
        projector = BlochProjectorFFT(dy=dy, Ny=Ny, a=a, m=int(m))
        k_vals[ik] = projector.k

        band_states: List[Array] = []
        band_traces: List[Array] = []

        for band in range(n_bands):
            # --- build an initial guess ---
            key = (int(m), int(band))
            if psi_guesses is not None and key in psi_guesses:
                psi = np.asarray(psi_guesses[key], dtype=np.complex128).copy()
                if psi.shape != (Ny,):
                    raise ValueError(f"psi_guesses[{key}] has shape {psi.shape}, expected {(Ny,)}")
            else:
                # sector-biased random: start from plane wave in this sector + noise
                L = Ny * dy
                plane = np.exp(1j * (2.0*np.pi*projector.m/L) * y)
                noise = (rng.normal(size=Ny) + 1j*rng.normal(size=Ny)).astype(np.complex128)
                psi = plane + 0.05 * noise

            # enforce sector and orthogonality at start
            projector(psi)
            if band_states:
                gram_schmidt_inplace(psi, band_states, dy, passes=gs_passes)
            normalize_inplace(psi, dy)

            last_E = None
            trace: List[float] = []

            # helper: check if we've collapsed onto an existing band state
            def max_overlap_with_basis(p: Array, basis: List[Array]) -> float:
                if not basis:
                    return 0.0
                ovs = [abs(gs_inner_product(phi, p, dy)) for phi in basis]
                return float(np.max(ovs))

            tries = 0
            while True:
                tries += 1
                if tries > max(1, int(restart_tries)):
                    break

                last_E = None
                trace.clear()

                for step in range(1, max_steps + 1):
                    stepper(psi)

                    # optional: enforce Bloch sector during evolution
                    if proj_every > 0 and (step % proj_every == 0):
                        projector(psi)

                    # deflate within this sector (bands)
                    if band_states and (step % reorth_every == 0):
                        gram_schmidt_inplace(psi, band_states, dy, passes=gs_passes)

                    normalize_inplace(psi, dy)

                    if step % check_every == 0:
                        E = float(energy_fn(psi))
                        trace.append(E)

                        # overlap guard: if we’re numerically drifting back onto an earlier band, restart
                        if allow_repeat_restart and band_states:
                            ov = max_overlap_with_basis(psi, band_states)
                            if ov > overlap_tol:
                                noise = (rng.normal(size=Ny) + 1j*rng.normal(size=Ny)).astype(np.complex128)
                                psi[:] = psi + 0.2 * noise
                                projector(psi)
                                gram_schmidt_inplace(psi, band_states, dy, passes=gs_passes)
                                normalize_inplace(psi, dy)
                                last_E = None
                                trace.clear()
                                break

                        if last_E is not None and abs(E - last_E) < energy_tol:
                            energies[ik, band] = E
                            steps[ik, band] = step
                            break
                        last_E = E
                else:
                    energies[ik, band] = float(energy_fn(psi))
                    steps[ik, band] = -1

                if (not allow_repeat_restart) or (not band_states) or (max_overlap_with_basis(psi, band_states) <= overlap_tol):
                    break

                # otherwise restart with new noise and try again
                noise = (rng.normal(size=Ny) + 1j*rng.normal(size=Ny)).astype(np.complex128)
                psi[:] = np.exp(1j * projector.k * y) + 0.1 * noise
                projector(psi)
                gram_schmidt_inplace(psi, band_states, dy, passes=gs_passes)
                normalize_inplace(psi, dy)

            band_states.append(psi.copy())
            band_traces.append(np.array(trace, dtype=float))

        psis_out.append(band_states)
        traces_out.append(band_traces)

    return {
        "m_list": np.array([int(m) for m in m_list], dtype=int),
        "k_vals": k_vals,
        "energies": energies,
        "steps": steps,
        "psis": psis_out,
        "energy_traces": traces_out,
    }
#---- End Bloch state finder with FFT ----------------------------------|


#---- QR Bloch Projector for speed -------------------------------------|

#--- helpers -----------------------
#-----------------------------------
def normalize_columns_inplace(Psi: Array, dy: float) -> Array:
    """
    Normalize each column of Psi so that sum(|Psi|^2) * dy = 1.
    Returns the vector of norms (before normalization).
    """
    # norms^2 = dy * sum_y |Psi|^2
    nrm2 = (np.sum(np.abs(Psi)**2, axis=0) * dy).astype(float)
    if np.any(nrm2 <= 0.0):
        raise ValueError("At least one column has zero norm.")
    nrm = np.sqrt(nrm2)
    Psi /= nrm[None, :]
    return nrm

def weighted_qr_inplace(Psi: Array, dy: float) -> None:
    """
    Replace Psi with an orthonormal (w.r.t. dy-weighted inner product) basis spanning the same columns.
    Uses QR on sqrt(dy)*Psi to enforce ∑ conj(Psi_i) Psi_j dy = δ_ij.
    """
    w = np.sqrt(dy)
    Q, _R = np.linalg.qr(w * Psi, mode="reduced")   # Q has orthonormal columns in Euclidean sense
    Psi[:] = Q / w

#-------------------------------------------------------------------------
# Bloch finder with QR projection ----------------------------------------
#-------------------------------------------------------------------------
def find_bloch_states_block(
    *,
    n_bands: int,
    m_list: Sequence[int],
    grid,
    V_of_y: Callable[[Array, float], Array],     # V(y, tau) but for eigenstates should be time-independent in tau
    energy_fn: Callable[[Array], float],         # returns <H> estimate (should match your kinetic+V conventions)
    a: float = 1.0,
    dtau: float = 5.0e-4,
    max_steps: int = 200_000,
    check_every: int = 200,
    energy_tol: float = 1.0e-10,
    reorth_every: int = 5,
    proj_every: int = 5,
    seed: int = 0,
    E_ref: float = 0.0,
    psi_guesses: Optional[Dict[Tuple[int, int], Array]] = None,
    projector: str = "fft",                      # "fft" (fast) or "roll" (original)
    cache_exponentials: bool = True,             # precompute exp(-dtau V/2) and exp(-dtau T) if V is static
) -> Dict[str, object]:
    """
    Block (subspace) imaginary-time Bloch state finder.

    Key differences vs find_bloch_states():
      - evolves all n_bands simultaneously as Psi shape (Ny, n_bands)
      - uses weighted QR (BLAS/LAPACK) instead of Python-loop Gram–Schmidt
      - can use BlochProjectorFFT for very fast projection (and it projects the whole block at once)

    Returns a dict with the same keys as find_bloch_states():
      - m_list, k_vals, energies, steps, psis, energy_traces
    """
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1.")
    if dtau <= 0:
        raise ValueError("dtau must be > 0.")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    if reorth_every < 1:
        raise ValueError("reorth_every must be >= 1.")
    if proj_every < 0:
        raise ValueError("proj_every must be >= 0.")
    if projector not in ("fft", "roll"):
        raise ValueError("projector must be 'fft' or 'roll'.")

    y, kgrid, dy = grid.y, grid.k, float(grid.dy)
    Ny = y.size
    rng = np.random.default_rng(seed)

    # Precompute kinetic exponential (always static)
    expT = np.exp(-0.5 * (kgrid**2) * dtau).astype(np.complex128)

    Nk = len(m_list)
    k_vals = np.zeros(Nk, dtype=float)
    energies = np.full((Nk, n_bands), np.nan, dtype=float)
    steps = np.full((Nk, n_bands), -1, dtype=int)
    psis_out: List[List[Array]] = []
    traces_out: List[List[Array]] = []

    for ik, m in enumerate(m_list):
        if projector == "fft":
            proj = BlochProjectorFFT(dy=dy, Ny=Ny, a=a, m=int(m))
        else:
            proj = BlochProjector(dy=dy, Ny=Ny, a=a, m=int(m))

        k_vals[ik] = proj.k

        # --- build initial block Psi (Ny, n_bands) ---
        Psi = np.zeros((Ny, n_bands), dtype=np.complex128)

        # If user provides guesses, use them; otherwise use plane wave + noise with different noise per band
        for band in range(n_bands):
            key = (int(m), int(band))
            if psi_guesses is not None and key in psi_guesses:
                psi = np.asarray(psi_guesses[key], dtype=np.complex128).copy()
                if psi.shape != (Ny,):
                    raise ValueError(f"psi_guesses[{key}] has shape {psi.shape}, expected {(Ny,)}")
            else:
                # sector-biased random
                L = Ny * dy
                plane = np.exp(1j * (2.0*np.pi*proj.m/L) * y)
                noise = (rng.normal(size=Ny) + 1j*rng.normal(size=Ny)).astype(np.complex128)
                psi = plane + 0.05 * noise
            Psi[:, band] = psi

        # enforce sector + orthonormalize
        proj(Psi)
        weighted_qr_inplace(Psi, dy)

        # cache exp(-dtau V/2) if requested (assumes static V for eigenproblem)
        expVhalf = None
        if cache_exponentials:
            V0 = np.asarray(V_of_y(y, 0.0), dtype=float)
            expVhalf = np.exp(-0.5 * dtau * (V0 - E_ref)).astype(np.complex128)

        # per-band energy traces
        trace_lists: List[List[float]] = [[] for _ in range(n_bands)]
        last_E: Optional[np.ndarray] = None

        def step_block(Psi: Array) -> None:
            """
            One Strang split imaginary-time step on all columns of Psi.
            """
            nonlocal expVhalf
            if expVhalf is None:
                V = np.asarray(V_of_y(y, 0.0), dtype=float)
                expVhalf_loc = np.exp(-0.5 * dtau * (V - E_ref)).astype(np.complex128)
            else:
                expVhalf_loc = expVhalf

            # V/2
            Psi *= expVhalf_loc[:, None]
            # kinetic in k-space (batched FFTs)
            Psi_k = np.fft.fft(Psi, axis=0)
            Psi_k *= expT[:, None]
            Psi[:] = np.fft.ifft(Psi_k, axis=0)
            # V/2
            Psi *= expVhalf_loc[:, None]

        # Main evolution loop
        for step in range(1, max_steps + 1):
            step_block(Psi)

            # keep norms controlled cheaply every step
            normalize_columns_inplace(Psi, dy)

            # optional: enforce Bloch sector
            if proj_every > 0 and (step % proj_every == 0):
                proj(Psi)

            # orthonormalize occasionally (this replaces GS deflation)
            if step % reorth_every == 0:
                weighted_qr_inplace(Psi, dy)

            # check convergence occasionally
            if step % check_every == 0:
                E = np.array([float(energy_fn(Psi[:, j])) for j in range(n_bands)], dtype=float)

                # keep columns ordered by energy (helps with band identity)
                order = np.argsort(E)
                E = E[order]
                Psi[:] = Psi[:, order]

                for j in range(n_bands):
                    trace_lists[j].append(E[j])

                if last_E is not None:
                    if np.max(np.abs(E - last_E)) < energy_tol:
                        energies[ik, :] = E
                        steps[ik, :] = step
                        break
                last_E = E.copy()

        # If no break (max_steps exhausted), record current estimates
        if steps[ik, 0] == -1:
            E = np.array([float(energy_fn(Psi[:, j])) for j in range(n_bands)], dtype=float)
            order = np.argsort(E)
            energies[ik, :] = E[order]
            Psi[:] = Psi[:, order]
            # steps remain -1 to flag non-convergence within max_steps

        # store output
        band_states = [Psi[:, j].copy() for j in range(n_bands)]
        band_traces = [np.array(trace_lists[j], dtype=float) for j in range(n_bands)]
        psis_out.append(band_states)
        traces_out.append(band_traces)

    return {
        "m_list": np.array([int(m) for m in m_list], dtype=int),
        "k_vals": k_vals,
        "energies": energies,
        "steps": steps,
        "psis": psis_out,
        "energy_traces": traces_out,
    }
#------ End of QR Bloch Projector ---------------------------------------|


#---- make a single BZ integer list 2pim/L --
def make_m_list_first_bz(grid, a=1.0):
    """
    Return m_list such that k_m = 2π m / L
    spans exactly one Brillouin zone.

    Assumes L = Ncells * a.
    """
    Ny = grid.y.size
    dy = grid.dy
    L = grid.L
    L_test = Ny * dy

    if ( np.abs(L-L_test) > 1.0e-10 ):
        raise ValueError(" The grid size is not commensurate.")

    Ncells = int(round(L / a))

    if abs(L / a - Ncells) > 1e-10:
        raise ValueError("L must be integer multiple of a.")

    # All allowed sectors
    m_all = np.arange(Ncells)

    return m_all.tolist()

#---- wrap momenta to the first BZ defined symmetrically about k=0.
def wrap_to_first_bz(k_vals, 
                     a=1.0):
    """Map k into [-π/a, π/a)."""
    return ((k_vals + np.pi/a) % (2*np.pi/a)) - np.pi/a



#---- band tracking so that bands are not mixed at a common momentum

def _normalize(psi: np.ndarray, dy: float) -> np.ndarray:
    nrm2 = np.vdot(psi, psi).real * dy
    if nrm2 <= 0:
        return psi
    return psi / np.sqrt(nrm2)

def _inner(psi: np.ndarray, phi: np.ndarray, dy: float) -> complex:
    # Discrete L2 inner product with uniform spacing dy
    return np.vdot(psi, phi) * dy

def _best_assignment(overlaps: np.ndarray) -> np.ndarray:
    """
    overlaps: (nb, nb) nonnegative matrix with overlaps[i,j] = |<prev_i | cand_j>|
    returns perm p of length nb: p[i] = chosen j for band i
    Uses exhaustive search for nb<=8; greedy otherwise.
    """
    nb = overlaps.shape[0]
    if nb <= 8:
        best_score = -1.0
        best_p = None
        for p in permutations(range(nb)):
            score = sum(overlaps[i, p[i]] for i in range(nb))
            if score > best_score:
                best_score = score
                best_p = np.array(p, dtype=int)
        return best_p
    else:
        # Greedy fallback
        p = -np.ones(nb, dtype=int)
        used = set()
        for i in range(nb):
            j = int(np.argmax(overlaps[i, :]))
            while j in used:
                overlaps[i, j] = -1.0
                j = int(np.argmax(overlaps[i, :]))
            p[i] = j
            used.add(j)
        return p

def track_bands_by_overlap(
    psis,               # list over ik, each is list/array of length nb: psis[ik][b] is vector
    energies: np.ndarray,  # shape (Nk, nb)
    dy: float,
    nbands: int | None = None,
    phase_fix: bool = True,
):
    """
    Returns:
      psis_tracked: list over ik, each list length nbands (tracked eigenvectors)
      energies_tracked: np.ndarray shape (Nk, nbands)
      perm_history: np.ndarray shape (Nk, nbands) where perm_history[ik,b] = original band index chosen
    """
    Nk = len(psis)
    nb0 = len(psis[0])
    nb = nb0 if nbands is None else min(nbands, nb0)

    # Copy first k-point as reference
    psis_tracked = [[None]*nb for _ in range(Nk)]
    energies_tracked = np.zeros((Nk, nb), dtype=float)
    perm_history = -np.ones((Nk, nb), dtype=int)

    # Normalize reference
    prev = [_normalize(np.array(psis[0][b], dtype=complex), dy) for b in range(nb)]
    for b in range(nb):
        psis_tracked[0][b] = prev[b]
        energies_tracked[0, b] = float(energies[0, b])
        perm_history[0, b] = b

    # March forward in k
    for ik in range(1, Nk):
        cands = [_normalize(np.array(psis[ik][j], dtype=complex), dy) for j in range(nb)]

        # Overlap matrix
        O = np.zeros((nb, nb), dtype=float)
        for i in range(nb):
            for j in range(nb):
                O[i, j] = abs(_inner(prev[i], cands[j], dy))

        p = _best_assignment(O.copy())  # p[i]=j
        # Reorder candidates accordingly, and phase-fix each band to be continuous
        curr = [None]*nb
        for i in range(nb):
            j = int(p[i])
            v = cands[j]
            if phase_fix:
                ov = _inner(prev[i], v, dy)
                if ov != 0:
                    v = v * np.exp(-1j * np.angle(ov))  # make <prev|v> real positive
            curr[i] = v
            psis_tracked[ik][i] = v
            energies_tracked[ik, i] = float(energies[ik, j])
            perm_history[ik, i] = j

        prev = curr

    return psis_tracked, energies_tracked, perm_history