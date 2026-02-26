from __future__ import annotations
import numpy as np
from scipy.fft import dst, idst  # requires SciPy
from dataclasses import dataclass
from typing import Callable, Dict, Protocol, Tuple, Optional

Array = np.ndarray

class Potential1D(Protocol):
    """
    A vectorized 1D potential energy V(y, tau).

    - y is a 1D numpy array of grid positions (dimensionless, unless you decide otherwise).
    - tau is a scalar time.
    - return value must be a 1D numpy array with same shape as y (real-valued preferred).
    """
    def __call__(self, y: Array, tau: float) -> Array: ...


@dataclass(frozen=True)
class Grid1D:
    y: Array   # position grid (Ny,)
    k: Array   # FFT wavenumbers conjugate to y (Ny,)
    kmax : Array # maximum wavenumber
    dy: float  # spacing


def make_grid(N: int, y_max: float) -> Grid1D:
    """
    Uniform grid on [-y_max, y_max) with FFT-friendly endpoint=False.
    """
    y = np.linspace(-y_max, y_max, N, endpoint=False)
    dy = float(y[1] - y[0])
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dy)
    kmax = np.max(np.abs(k))

    return Grid1D(y=y, k=k, kmax=kmax, dy=dy)


def normalize(psi: Array, dy: float) -> Array:
    """
    Normalize wavefunction so that ∫|psi|^2 dy = 1 (using trapezoid on uniform grid).
    """
    nrm = np.sqrt(np.sum(np.abs(psi) ** 2) * dy)
    if nrm == 0:
        raise ValueError("Wavefunction norm is zero.")
    return psi / nrm


def split_step_propagate(
    psi0: Array,
    V_of_y_tau: Potential1D,
    tau_grid: Array,
    grid: Grid1D,
    return_all: bool = True,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Time-dependent 1D Schrödinger equation in units where:
        i d/dtau psi = [ -1/2 d^2/dy^2 + V(y,tau) ] psi

    2nd-order Strang split:
        exp(-i V_mid dt/2) exp(-i T dt) exp(-i V_mid dt/2)
    where V_mid is evaluated at tau + dt/2.

    tau_grid may be nonuniform (and must be strictly increasing).
    """
    y, k, dy = grid.y, grid.k, grid.dy

    psi = np.array(psi0, dtype=np.complex128, copy=True)
    psi = normalize(psi, dy)

    tau_grid = np.asarray(tau_grid, dtype=float)
    Nt = int(tau_grid.size)
    if Nt < 2:
        raise ValueError("tau_grid must have at least 2 points.")

    dtaus = np.diff(tau_grid)
    if np.any(dtaus <= 0):
        raise ValueError("tau_grid must be strictly increasing.")

    psi_out = np.empty((Nt, psi.size), dtype=np.complex128) if return_all else None
    norms = np.empty(Nt, dtype=float)
    if return_all:
        psi_out[0] = psi
    norms[0] = np.sum(np.abs(psi) ** 2) * dy

    for n in range(Nt - 1):
        dt = float(dtaus[n])
        tau_mid = float(tau_grid[n] + 0.5 * dt)

        Vmid = np.asarray(V_of_y_tau(y, tau_mid), dtype=float)
        if Vmid.shape != y.shape:
            raise ValueError(f"V_of_y_tau returned shape {Vmid.shape}, expected {y.shape}.")

        phase_V_half = np.exp(-0.5j * dt * Vmid)

        # half potential kick
        psi *= phase_V_half

        # kinetic kick in k-space: T = k^2/2
        psi_k = np.fft.fft(psi)
        psi_k *= np.exp(-1.0j * dt * (k**2) / 2.0)
        psi = np.fft.ifft(psi_k)

        # half potential kick
        psi *= phase_V_half

        if return_all:
            psi_out[n + 1] = psi
        norms[n + 1] = np.sum(np.abs(psi) ** 2) * dy

    diagnostics = {"tau": tau_grid.copy(), "norm": norms}
    return (psi_out if return_all else psi), diagnostics


def yoshida_step_propagate(
    psi0: Array,
    V_of_y_tau: Potential1D,
    tau_grid: Array,
    grid: Grid1D,
    return_all: bool = True,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Same TDSE as split_step_propagate(), but with a 4th-order Suzuki–Yoshida
    composition of Strang steps.

    For time-dependent V, each sub-step evaluates V at the midpoint time of that sub-step.
    """
    y, k, dy = grid.y, grid.k, grid.dy

    psi = np.array(psi0, dtype=np.complex128, copy=True)
    psi = normalize(psi, dy)

    tau_grid = np.asarray(tau_grid, dtype=float)
    Nt = int(tau_grid.size)
    if Nt < 2:
        raise ValueError("tau_grid must have at least 2 points.")

    dtaus = np.diff(tau_grid)
    if np.any(dtaus <= 0):
        raise ValueError("tau_grid must be strictly increasing.")

    psi_out = np.empty((Nt, psi.size), dtype=np.complex128) if return_all else None
    norms = np.empty(Nt, dtype=float)
    if return_all:
        psi_out[0] = psi
    norms[0] = np.sum(np.abs(psi) ** 2) * dy

    # Yoshida coefficients
    cbrt2 = 2.0 ** (1.0 / 3.0)
    a = 1.0 / (2.0 - cbrt2)
    b = -cbrt2 / (2.0 - cbrt2)

    def strang_step_inplace(psi_arr: Array, tau_start: float, dt: float) -> None:
        tau_mid = float(tau_start + 0.5 * dt)
        Vmid = np.asarray(V_of_y_tau(y, tau_mid), dtype=float)
        if Vmid.shape != y.shape:
            raise ValueError(f"V_of_y_tau returned shape {Vmid.shape}, expected {y.shape}.")
        phase_V_half = np.exp(-0.5j * dt * Vmid)

        psi_arr *= phase_V_half
        psi_k = np.fft.fft(psi_arr)
        psi_k *= np.exp(-1.0j * dt * (k**2) / 2.0)
        psi_arr[:] = np.fft.ifft(psi_k)
        psi_arr *= phase_V_half

    for n in range(Nt - 1):
        dt = float(dtaus[n])
        tau0 = float(tau_grid[n])

        # Compose: S(a dt) S(b dt) S(a dt)
        strang_step_inplace(psi, tau0, a * dt)
        strang_step_inplace(psi, tau0 + a * dt, b * dt)
        strang_step_inplace(psi, tau0 + (a + b) * dt, a * dt)

        if return_all:
            psi_out[n + 1] = psi
        norms[n + 1] = np.sum(np.abs(psi) ** 2) * dy

    diagnostics = {"tau": tau_grid.copy(), "norm": norms}
    return (psi_out if return_all else psi), diagnostics


#--------------------------------------------------------------------------------
#
# Below is a DST based solver to enforce Dirichlet boundary conditions when 
# required.
#
#---------------------------------------------------------------------------------
def make_grid_dirichlet(N: int, y_max: float) -> Grid1D:
    """
    Uniform *interior* grid on (-y_max, y_max) with Dirichlet boundaries:
        psi(-y_max) = psi(+y_max) = 0

    IMPORTANT:
      - N here means the number of *interior* points (endpoints excluded).
      - This is the natural grid for sine (Dirichlet) spectral methods.

    Interior points:
        L  = 2*y_max
        dy = L/(N+1)
        y_j = -y_max + j*dy,  j = 1..N

    Sine wavenumbers:
        k_n = n*pi/L, n = 1..N
    """
    if N < 1:
        raise ValueError("N must be >= 1.")
    if y_max <= 0:
        raise ValueError("y_max must be > 0.")

    L = 2.0 * float(y_max)
    dy = L / (N + 1)

    j = np.arange(1, N + 1)
    y = (-y_max + j * dy).astype(float)

    n = np.arange(1, N + 1)
    k = (np.pi * n / L).astype(float)  # Dirichlet Laplacian eigen-wavenumbers
    kmax = np.max(np.abs(k))

    return Grid1D(y=y, k=k, kmax=kmax, dy=float(dy))


def _dst_ortho(x: Array) -> Array:
    # DST-I with orthonormal scaling; inverse is IDST-I with norm="ortho".
    return dst(x, type=1, norm="ortho")


def _idst_ortho(x: Array) -> Array:
    return idst(x, type=1, norm="ortho")


def split_step_propagate_dirichlet(
    psi0: Array,
    V_of_y_tau: Potential1D,
    tau_grid: Array,
    grid: Grid1D,
    return_all: bool = True,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Dirichlet/DST version of split_step_propagate().

    Solves TDSE:
        i d/dtau psi = [ -1/2 d^2/dy^2 + V(y,tau) ] psi

    with boundary conditions:
        psi(-y_max) = psi(+y_max) = 0

    Uses Strang splitting, evaluating V at the midpoint of each step.
    """
    y, k, dy = grid.y, grid.k, grid.dy

    psi = np.array(psi0, dtype=np.complex128, copy=True)
    if psi.shape != y.shape:
        raise ValueError(f"psi0 has shape {psi.shape}, expected {y.shape}.")
    psi = normalize(psi, dy)

    tau_grid = np.asarray(tau_grid, dtype=float)
    Nt = int(tau_grid.size)
    if Nt < 2:
        raise ValueError("tau_grid must have at least 2 points.")

    dtaus = np.diff(tau_grid)
    if np.any(dtaus <= 0):
        raise ValueError("tau_grid must be strictly increasing.")

    psi_out = np.empty((Nt, psi.size), dtype=np.complex128) if return_all else None
    norms = np.empty(Nt, dtype=float)
    if return_all:
        psi_out[0] = psi
    norms[0] = np.sum(np.abs(psi) ** 2) * dy

    for n in range(Nt - 1):
        dt = float(dtaus[n])
        tau_mid = float(tau_grid[n] + 0.5 * dt)

        Vmid = np.asarray(V_of_y_tau(y, tau_mid), dtype=float)
        if Vmid.shape != y.shape:
            raise ValueError(f"V_of_y_tau returned shape {Vmid.shape}, expected {y.shape}.")

        phase_V_half = np.exp(-0.5j * dt * Vmid)

        # half potential kick
        psi *= phase_V_half

        # kinetic kick in sine space: eigenvalues are k^2/2
        a = _dst_ortho(psi)
        a *= np.exp(-1.0j * dt * (k**2) / 2.0)
        psi = _idst_ortho(a)

        # half potential kick
        psi *= phase_V_half

        if return_all:
            psi_out[n + 1] = psi
        norms[n + 1] = np.sum(np.abs(psi) ** 2) * dy

    diagnostics = {"tau": tau_grid.copy(), "norm": norms}
    return (psi_out if return_all else psi), diagnostics


def yoshida_step_propagate_dirichlet(
    psi0: Array,
    V_of_y_tau: Potential1D,
    tau_grid: Array,
    grid: Grid1D,
    return_all: bool = True,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Dirichlet/DST version of yoshida_step_propagate() (4th-order Suzuki–Yoshida).
    """
    y, k, dy = grid.y, grid.k, grid.dy

    psi = np.array(psi0, dtype=np.complex128, copy=True)
    if psi.shape != y.shape:
        raise ValueError(f"psi0 has shape {psi.shape}, expected {y.shape}.")
    psi = normalize(psi, dy)

    tau_grid = np.asarray(tau_grid, dtype=float)
    Nt = int(tau_grid.size)
    if Nt < 2:
        raise ValueError("tau_grid must have at least 2 points.")

    dtaus = np.diff(tau_grid)
    if np.any(dtaus <= 0):
        raise ValueError("tau_grid must be strictly increasing.")

    psi_out = np.empty((Nt, psi.size), dtype=np.complex128) if return_all else None
    norms = np.empty(Nt, dtype=float)
    if return_all:
        psi_out[0] = psi
    norms[0] = np.sum(np.abs(psi) ** 2) * dy

    # Yoshida coefficients
    cbrt2 = 2.0 ** (1.0 / 3.0)
    a_coef = 1.0 / (2.0 - cbrt2)
    b_coef = -cbrt2 / (2.0 - cbrt2)

    def strang_step_inplace_dirichlet(psi_arr: Array, tau_start: float, dt: float) -> None:
        tau_mid = float(tau_start + 0.5 * dt)
        Vmid = np.asarray(V_of_y_tau(y, tau_mid), dtype=float)
        if Vmid.shape != y.shape:
            raise ValueError(f"V_of_y_tau returned shape {Vmid.shape}, expected {y.shape}.")
        phase_V_half = np.exp(-0.5j * dt * Vmid)

        psi_arr *= phase_V_half
        a = _dst_ortho(psi_arr)
        a *= np.exp(-1.0j * dt * (k**2) / 2.0)
        psi_arr[:] = _idst_ortho(a)
        psi_arr *= phase_V_half

    for n in range(Nt - 1):
        dt = float(dtaus[n])
        tau0 = float(tau_grid[n])

        # Compose: S(a dt) S(b dt) S(a dt)
        strang_step_inplace_dirichlet(psi, tau0, a_coef * dt)
        strang_step_inplace_dirichlet(psi, tau0 + a_coef * dt, b_coef * dt)
        strang_step_inplace_dirichlet(psi, tau0 + (a_coef + b_coef) * dt, a_coef * dt)

        if return_all:
            psi_out[n + 1] = psi
        norms[n + 1] = np.sum(np.abs(psi) ** 2) * dy

    diagnostics = {"tau": tau_grid.copy(), "norm": norms}
    return (psi_out if return_all else psi), diagnostics


#--------------------------------------------------------
#--- computes the expectation value of H ----------------#
#--- from a solution to the Schrodinger equation --------
#---------------------------------------------------------
def expectation_H(
    psi: Array,       # wavefunction computed on the grid
    grid,             # grid object
    Vx: Optional[Array] = None,
    V_of_y_tau: Optional[Callable[[Array, float], Array]] = None,
    tau: float = 0.0, # time
    bc: str = "fft",  # "fft" (periodic) or "dst" (Dirichlet)
) -> Dict[str, float]:
    """
    Compute <H>, <T>, <V> for 1D Hamiltonian
        H = -1/2 d^2/dy^2 + V(y)

    Parameters
    ----------
    psi : (Ny,) complex array
        Wavefunction on grid.y.
    grid : object with attributes y, k, dy
        Your Grid1D (or compatible).
    Vx : (Ny,) float array, optional
        Potential evaluated on grid.y at the desired time.
    V_of_y_tau : callable(y, tau)->(Ny,), optional
        Potential function; used if Vx not provided.
    tau : float, optional
        Time at which to evaluate V_of_y_tau.
    bc : {"fft","dst"}
        Which spectral convention to use for the kinetic operator.
        - "fft": periodic domain (your FFT solver)
        - "dst": Dirichlet walls (your DST solver)

    Returns
    -------
    out : dict with keys "H", "T", "V"
        Real-valued expectation values.
    """
    y = grid.y
    k = grid.k
    dy = float(grid.dy)

    psi = np.asarray(psi, dtype=np.complex128)
    if psi.shape != y.shape:
        raise ValueError(f"psi has shape {psi.shape}, expected {y.shape}.")

    # Potential on the grid
    if Vx is None:
        if V_of_y_tau is None:
            raise ValueError("Provide either Vx or V_of_y_tau.")
        if tau is None:
            raise ValueError("If using V_of_y_tau, you must provide tau.")
        Vx = np.asarray(V_of_y_tau(y, float(tau)), dtype=float)
    else:
        Vx = np.asarray(Vx, dtype=float)

    if Vx.shape != y.shape:
        raise ValueError(f"Vx has shape {Vx.shape}, expected {y.shape}.")

    # <V> = ∫ ψ* V ψ dy
    Vexp = float(np.real(np.sum(np.conj(psi) * (Vx * psi)) * dy))

    # <T> = ∫ ψ* (-1/2 ψ'') dy  (computed spectrally, consistent with your propagator)
    if bc.lower() == "fft":
        # Periodic spectral second derivative: ψ'' = ifft( -(k^2) fft(ψ) )
        psi_k = np.fft.fft(psi)
        psi_dd = np.fft.ifft(-(k**2) * psi_k)
        Texp = float(np.real(np.sum(np.conj(psi) * (-0.5 * psi_dd)) * dy))

    elif bc.lower() == "dst":
        # Dirichlet spectral second derivative via DST coefficients.
        # Requires that your DST propagator used DST-I with norm="ortho".

        # Do DST on real & imag separately (usually faster and clearer)
        a_re = dst(psi.real, type=1, norm="ortho")
        a_im = dst(psi.imag, type=1, norm="ortho")
        a = a_re + 1j * a_im

        # In sine basis, d^2/dy^2 -> -(k^2)
        dd_a = -(k**2) * a

        psi_dd_re = idst(dd_a.real, type=1, norm="ortho")
        psi_dd_im = idst(dd_a.imag, type=1, norm="ortho")
        psi_dd = psi_dd_re + 1j * psi_dd_im

        Texp = float(np.real(np.sum(np.conj(psi) * (-0.5 * psi_dd)) * dy))

    else:
        raise ValueError("bc must be 'fft' or 'dst'.")

    Hexp = Texp + Vexp
    return {"H": Hexp, "T": Texp, "V": Vexp}
#----------------------------------------------------------------------------------
