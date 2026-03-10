import numpy as np
import math
from scipy.fft import dst, idst
#-----------------------------------------------------
#
# Harmonic Oscillator Wavefunction Utilities
#
#-----------------------------------------------------
def ho_eigenstate_ref(y: np.ndarray, n: int) -> np.ndarray:
    """
    Dimensionless harmonic-oscillator eigenstate for Omega=1.
    Returns psi_n(y), normalized in the continuum (up to discretization error):
      psi_n(y) = (pi^-1/4)/sqrt(2^n n!) * H_n(y) * exp(-y^2/2)

    Uses stable Hermite recursion rather than requiring SciPy.
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    y = np.asarray(y, dtype=float)
    gauss = np.exp(-0.5 * y**2)

    # Hermite polynomials H_n(y) (physicists' convention) via recursion:
    # H_0 = 1
    # H_1 = 2y
    # H_{n+1} = 2y H_n - 2n H_{n-1}
    if n == 0:
        Hn = np.ones_like(y)
    elif n == 1:
        Hn = 2.0 * y
    else:
        Hnm1 = np.ones_like(y)      # H_0
        Hn_ = 2.0 * y               # H_1
        for k in range(1, n):
            Hnp1 = 2.0 * y * Hn_ - 2.0 * k * Hnm1
            Hnm1, Hn_ = Hn_, Hnp1
        Hn = Hn_

    # normalization factor
    # norm = pi^-1/4 / sqrt(2^n n!)
    norm = (np.pi ** (-0.25)) / np.sqrt((2.0 ** n) * float(math.factorial(n)))

    psi = norm * Hn * gauss
    return psi.astype(np.complex128)

def ho_eigenstate(y: np.ndarray, n: int, Omega: float) -> np.ndarray:
    """
    Eigenstate for constant dimensionless frequency Omega (time-independent).
    psi_n(y;Omega) = Omega^(1/4) * psi_n(sqrt(Omega)*y; 1)
    """
    if Omega <= 0:
        raise ValueError("Omega must be > 0")
    s = np.sqrt(Omega)
    return (Omega ** 0.25) * ho_eigenstate_ref(s * y, n)

def ho_energy(n: int) -> float:
    """
    Harmonic oscillator eigenstate energies in units of hbar*omega.
    """
    e = float(n) + 0.5
    return e

#--- harmonic oscillator number operator
def number_expectation(
    psi_t: np.ndarray,      # (Nt, Ny)
    y: np.ndarray,          # (Ny,)
    k: np.ndarray,          # (Ny,)
    dy: float,
    tau: np.ndarray,        # (Nt,)
    Omega_of_tau=None,      # callable Omega(tau)->float; if None uses Omega=1
    Omega_ref: float = 1.0  # constant reference frequency when Omega_of_tau is None
) -> np.ndarray:
    """
    Returns n(tau) = <a^dagger a> either instantaneous (if Omega_of_tau provided)
    or reference (Omega=1).
    """
    if Omega_of_tau is None:
        if Omega_ref <= 0:
            raise ValueError("Omega_ref must be > 0")
    
    Nt = psi_t.shape[0]
    n = np.empty(Nt, dtype=float)

    for i in range(Nt):
        psi = psi_t[i]
        y2 = expect_y2(psi, y, dy)
        p2 = expect_p2(psi, k, dy)

        Om = float(Omega_of_tau(tau[i])) if Omega_of_tau is not None else float(Omega_ref)
        n[i] = 0.5*(p2/Om + Om*y2) - 0.5

    return n

#------------------------------------------------------------------
#
# Particle in a box.
#
# energy is in units of hbar^2/(m*L^2)
#
# lengths in units of L
#
#-------------------------------------------------------------------
def box_eigenstate(y: np.ndarray, n: int) -> np.ndarray:
    """
    Eigentates of the infinite square well from -L/2 to L/2 
    in the dimensionless variable y=x/L.
    """
    
    y = np.asarray(y, dtype=float)

    if ( n < 1 ):
        raise ValueError("n must be >= 1.")

    k = n*np.pi
    norm = np.sqrt(2)

    inside = ( y < 0.5 ) & ( y > -0.5 )
    outside = ~inside
    psi=np.zeros_like(y)

    if ( n % 2 == 1 ):  # even parity
        psi[inside] = norm * np.cos(k*y[inside])
    else:               # odd parity
        psi[inside] = norm * np.sin(k*y[inside])

    psi[outside]=0.0

    psi = psi.astype(np.complex128)

    return psi

def box_energy(n: int) -> float:
    """
    Eigen energies of the particle in the infinite box, 
    in units of hbar^2/(m*L^2)
    """
    e = 0.5*(float(n)**2)*(np.pi**2)
    return e

#-------------------------------------------------------------------
# free particle
#-------------------------------------------------------------------
def free_particle(y: np.ndarray, n: int, L: float=1.0) -> np.ndarray:
    psi = np.exp(-2.0j*np.pi*float(n)*y/L)/np.sqrt(L)
    return psi

#-------------------------------------------------------------------
# free particle energies (periodic box)
#-------------------------------------------------------------------
def free_particle_energy(n: int, L: float = 1.0) -> float:
    """
    Energy eigenvalue corresponding to free_particle(y, n, L)
    for Hamiltonian H = -1/2 d^2/dy^2.
    """
    k = 2.0 * np.pi * float(n) / L
    return 0.5 * k**2


#-------------------------------------------------------------------
#
# Gaussian wavepacket (moving)
#
# Useful as an initial state for 1D scattering problems.
#
# lengths in dimensionless units y = x / L_ref
# momenta in dimensionless units k = p * L_ref / hbar
#
#-------------------------------------------------------------------
def gaussian_wavepacket(y: np.ndarray,
                        y0: float,
                        k0: float,
                        sigma: float) -> np.ndarray:
    """
    Minimum-uncertainty Gaussian wavepacket:

        psi(y) = (2 pi sigma^2)^(-1/4)
                 * exp( -(y - y0)^2 / (4 sigma^2) )
                 * exp(  i k0 y )

    Parameters
    ----------
    y     : grid positions, shape (Ny,)
    y0    : center of the wavepacket
    k0    : central wavenumber (= mean momentum in units where hbar=m=1);
            positive -> moves right, negative -> moves left
    sigma : Gaussian width (position uncertainty Delta_y = sigma)

    The wavepacket satisfies Delta_y * Delta_p = 1/2 (minimum uncertainty).
    Mean energy under H = -1/2 d^2/dy^2:  E = k0^2/2 + 1/(8 sigma^2).
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")

    y = np.asarray(y, dtype=float)
    norm = (2.0 * np.pi * sigma**2) ** (-0.25)
    envelope = np.exp(-((y - y0) ** 2) / (4.0 * sigma**2))
    carrier  = np.exp(1.0j * k0 * y)
    return (norm * envelope * carrier).astype(np.complex128)


def gaussian_wavepacket_energy(k0: float, sigma: float) -> float:
    """
    Mean energy of gaussian_wavepacket under H = -1/2 d^2/dy^2:

        E = k0^2 / 2  +  1 / (8 sigma^2)

    The first term is the kinetic energy of the moving center of mass;
    the second is the zero-point kinetic energy from the momentum spread
    Delta_p = 1 / (2 sigma).
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    return 0.5 * k0**2 + 1.0 / (8.0 * sigma**2)


#-------------------------------------------------------------------
#
# General utilities
#
#--------------------------------------------------------------------
def p_of_psi(psi: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Compute p psi where p = -i d/dy, using FFT differentiation.
    psi: (Ny,) complex
    k:   (Ny,) FFT wavenumbers compatible with y-grid
    """
    psi_k = np.fft.fft(psi)
    dpsi = np.fft.ifft(1j * k * psi_k)   # d/dy psi
    return -1j * dpsi                    # p psi

#--- expectation value of position squared
def expect_y2(psi: np.ndarray, y: np.ndarray, dy: float) -> float:
    return float(np.sum((y**2) * (np.abs(psi)**2)) * dy)

#--- expectation value of p^2
def expect_p2(psi: np.ndarray, k: np.ndarray, dy: float) -> float:
    ppsi = p_of_psi(psi, k)
    return float(np.sum(np.abs(ppsi)**2) * dy)


def inner_product(phi: np.ndarray, psi: np.ndarray, dy: float) -> complex:
    """
    Discrete approximation to <phi|psi> = ∫ phi*(y) psi(y) dy.
    """
    return np.sum(np.conj(phi) * psi) * dy


def overlaps_with_basis(
    psi: np.ndarray,
    basis: np.ndarray,
    dy: float) -> np.ndarray:
    """
    Compute overlaps c_n = <basis_n | psi> for a single state psi.

    Parameters
    ----------
    psi   : shape (Ny,)
    basis : shape (Nbasis, Ny) where basis[n] is |n>
    dy    : grid spacing

    Returns
    -------
    c : shape (Nbasis,) complex overlaps
    """
    # c_n = sum_j basis[n,j]^* psi[j] dy
    return (basis.conj() @ psi) * dy


def overlaps_over_time(
    psi_t: np.ndarray,
    basis: np.ndarray,
    dy: float
) -> np.ndarray:
    """
    Compute overlaps C[t,n] = <basis_n | psi(t)> over a time series.

    Parameters
    ----------
    psi_t : shape (Nt, Ny)
    basis : shape (Nbasis, Ny)
    dy    : grid spacing

    Returns
    -------
    C : shape (Nt, Nbasis) complex
    """
    # For each time: c(t) = basis^* @ psi(t) * dy
    # We want (Nt, Nbasis): psi_t @ basis^T(conj) * dy
    return (psi_t @ basis.conj().T) * dy


def probabilities_from_overlaps(C: np.ndarray) -> np.ndarray:
    """
    Return |C|^2 as probabilities (shape matches C).
    """
    return np.abs(C)**2


def parity_expectation(psi: np.ndarray, dy: float) -> float:
    """
    Compute <psi|P|psi>, where P psi(y) = psi(-y).
    Returns a real number between -1 and 1.
    """
    psi_flip = psi[::-1]
    return float(np.real(np.sum(np.conj(psi) * psi_flip) * dy))

