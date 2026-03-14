"""
GPU-enabled 1D Schrödinger split-step solver using PyTorch.

This module is intentionally separate from the NumPy/SciPy solver so that
existing code remains unchanged.

Supports:
    CPU
    Apple Silicon GPU (MPS) — float64/complex128 may fall back to CPU on MPS;
                               use complex64 for full native GPU throughput.
    CUDA GPUs
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np


# -------------------------------------------------------
# Grid
# -------------------------------------------------------

@dataclass(frozen=True)
class Grid1D:
    y: np.ndarray   # position grid (N,)
    k: np.ndarray   # FFT wavenumbers (N,)
    kmax: float     # maximum wavenumber  |k|_max
    dy: float       # grid spacing
    L: float        # domain length = 2 * y_max


def make_grid(N: int, y_max: float) -> Grid1D:
    """Uniform grid on [-y_max, y_max) with FFT-friendly endpoint=False."""
    y = np.linspace(-y_max, y_max, N, endpoint=False)
    dy = float(y[1] - y[0])
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dy)
    kmax = float(np.max(np.abs(k)))
    L = 2.0 * y_max
    return Grid1D(y=y, k=k, kmax=kmax, dy=dy, L=L)


# -------------------------------------------------------
# Device selection
# -------------------------------------------------------

def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------------------------------------
# Utility conversion
# -------------------------------------------------------

def to_torch(x, device, dtype=torch.complex128):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)   # convert both device AND dtype
    return torch.tensor(np.asarray(x), dtype=dtype, device=device)


# -------------------------------------------------------
# Split-step propagation
# -------------------------------------------------------

def split_step_propagate_torch(
    psi0: np.ndarray,
    V_of_y_tau,
    tau,
    grid: Grid1D,
    *,
    time_independent: bool = False,
    return_all: bool = False,
    device: str | None = None,
    cap: np.ndarray | None = None,
    dtype: torch.dtype = torch.complex128,
) -> tuple[np.ndarray, dict]:
    """
    Strang split-step Fourier propagation via PyTorch.

    Each step applies:
        psi(t+dt) ≈ exp(-i V dt/2) · F⁻¹[exp(-i k² dt/2) F[·]] · exp(-i V dt/2) psi(t)

    with an optional complex-absorbing-potential (CAP) damping factor
    exp(-W dt/2) merged into the half-kicks.

    Parameters
    ----------
    psi0 : (N,) complex array
        Initial wavefunction.
    V_of_y_tau : callable
        Potential: V(grid.y, t) → (N,) float array.
    tau : (Nt,) array
        Uniform time grid.
    grid : Grid1D
        Grid object with .y, .k, .dy attributes.
    time_independent : bool
        If True, V is evaluated once at tau[0] and reused every step (faster).
    return_all : bool
        If True  → return (Nt, N) wavefunction history.
        If False → return only the final (N,) wavefunction.
    device : str or None
        'cpu', 'mps', 'cuda', or None (auto-detect).
    cap : (N,) float array or None
        Real, non-negative absorption rate W(y) ≥ 0.
        Applied as exp(-W dt/2) half-kicks flanking the kinetic step.
    dtype : torch.dtype
        Complex dtype for all on-device tensors.
        torch.complex128 (default) — full double precision; not supported on MPS.
        torch.complex64             — single precision; required for MPS GPU speed.
        The corresponding real dtype (float64 / float32) is derived automatically.

    Returns
    -------
    psi_out : ndarray — shape (Nt, N) if return_all, else (N,)
              dtype is np.complex128 or np.complex64 matching the `dtype` argument.
    diag    : dict   — {'norm': list[float]}, one value per stored time step.
    """
    dev   = get_device(device)
    cdtype = dtype

    # Derive the matching real dtype (used for k, V, cap tensors before casting)
    if cdtype == torch.complex64:
        fdtype  = torch.float32
        np_cdtype = np.complex64
    elif cdtype == torch.complex128:
        fdtype  = torch.float64
        np_cdtype = np.complex128
    else:
        raise ValueError(f"dtype must be torch.complex64 or torch.complex128, got {cdtype}.")

    tau = np.asarray(tau, dtype=float)
    dt  = float(tau[1] - tau[0])
    Nt  = len(tau)
    N   = len(grid.y)

    # ---- wavefunction ----
    psi = to_torch(psi0, dev, cdtype)          # shape (N,)

    # ---- kinetic half-step phase (pre-computed; time-independent) ----
    k_t     = torch.tensor(grid.k, dtype=fdtype, device=dev)
    phase_k = torch.exp((-0.5j * dt) * (k_t ** 2).to(cdtype))

    # ---- potential half-kick phase ----
    _ph_V: torch.Tensor | None = None
    if time_independent:
        V0    = np.asarray(V_of_y_tau(grid.y, float(tau[0])), dtype=float)
        V0_t  = torch.tensor(V0, dtype=fdtype, device=dev)
        _ph_V = torch.exp((-0.5j * dt) * V0_t.to(cdtype))

    # ---- CAP half-kick (always pre-computable: W never depends on time) ----
    _ph_cap: torch.Tensor | None = None
    if cap is not None:
        cap = np.asarray(cap, dtype=float)
        if np.any(cap < 0.0):
            raise ValueError("cap must be non-negative everywhere.")
        cap_t   = torch.tensor(cap, dtype=fdtype, device=dev)
        _ph_cap = torch.exp(-0.5 * dt * cap_t.to(cdtype))
        if _ph_V is not None:
            _ph_V = _ph_V * _ph_cap   # merge into a single half-kick array

    # ---- output storage ----
    if return_all:
        psi_arr      = np.empty((Nt, N), dtype=np_cdtype)
        psi_arr[0]   = psi.cpu().numpy()

    norms: list[float] = [float((psi.abs() ** 2).sum().cpu()) * grid.dy]

    # ---- main time loop ----
    for n in range(Nt - 1):

        # -- first potential half-kick --
        if _ph_V is not None:                               # fast pre-computed path
            psi = psi * _ph_V
        else:
            V   = np.asarray(V_of_y_tau(grid.y, float(tau[n])), dtype=float)
            ph  = torch.exp(
                (-0.5j * dt) * torch.tensor(V, dtype=fdtype, device=dev).to(cdtype)
            )
            if _ph_cap is not None:
                ph = ph * _ph_cap
            psi = psi * ph

        # -- full kinetic step (in k-space) --
        psi = torch.fft.ifft(torch.fft.fft(psi) * phase_k)

        # -- second potential half-kick (same phase as first within this step) --
        if _ph_V is not None:
            psi = psi * _ph_V
        else:
            psi = psi * ph      # ph still valid; V doesn't change within a step

        # -- record --
        if return_all:
            psi_arr[n + 1] = psi.cpu().numpy()

        norms.append(float((psi.abs() ** 2).sum().cpu()) * grid.dy)

    diag = {'norm': norms}

    if return_all:
        return psi_arr, diag

    return psi.cpu().numpy(), diag
