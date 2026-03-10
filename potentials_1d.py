
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Callable
import numpy as np

Array = np.ndarray

# -------------------------
# Convenience potentials
# -------------------------

#---------------------------
# Harmonic oscillator potentials
#---------------------------

#---------------------------
# schedules for time-dependent frequencies
#---------------------------
class OmegaSchedule(Protocol):
    def __call__(self,tau: float) -> float: 
        pass


#--- Constant frequency -----
@dataclass(frozen=True)
class OmegaConstant:
    omega: float
    def __call__(self, tau: float) -> float:
        return float(self.omega)
    
#--- OmegaRamp -----
@dataclass(frozen=True)
class OmegaRamp:
    """
    Ramp the frequency from omega_i to omega_f at time tau0 with width delta.
    """
    omega_i: float
    omega_f: float
    tau0: float = 0.0
    delta: float = 1.0

    def __post_init__(self):
        if self.delta <= 1.0e-32:
            raise ValueError("delta must be positive and nonzero.")

    def __call__(self, tau: float) -> float:
        x = (tau - self.tau0)/self.delta
        s = 0.5*(1.0 + np.tanh(x))
        return float(self.omega_i + (self.omega_f-self.omega_i)*s)


#--- piecewise changing omega ---
@dataclass(frozen=True)
class OmegaPiecewise:
    """
    Piecewise-constant schedule:
      times = (t0, t1, ..., t_{M-1})
      omegas = (w0, w1, ..., w_{M-1}, wM)
    where for tau < t0 -> w0
          t0 <= tau < t1 -> w1
          ...
          tau >= t_{M-1} -> wM
    """
    times: tuple[float, ...]
    omegas: tuple[float, ...]

    def __post_init__(self):
        if len(self.omegas) != len(self.times) + 1:
            raise ValueError("omegas must have length len(times)+1")
        if any(self.times[i] >= self.times[i+1] for i in range(len(self.times)-1)):
            raise ValueError("times must be strictly increasing")

    def __call__(self, tau: float) -> float:
        # np.searchsorted gives interval index
        idx = int(np.searchsorted(np.asarray(self.times), tau, side="right"))
        return float(self.omegas[idx])


#----------------------------
# The harmonic potential
#----------------------------
@dataclass(frozen=True)
class HarmonicPotential:
    omega: OmegaSchedule

    def __call__(self,
                 y: Array,
                 tau: float) -> Array:
        w = float(self.omega(float(tau)))
        return 0.5* (w**2) * (y**2)
    



#---------------------------
# Quartic Potential
#---------------------------
@dataclass(frozen=True)
class QuarticPotential:
    lam: float

    def __call__(self, y:np.ndarray, 
                 tau: float) -> np.ndarray:
        y=np.asarray(y)
        v = self.lam*(y**4)
        return v
    

#------------------------------
#
# Symmetric square well.
#
# y = x/L
# V0 = V_0/E_L with E_L = hbar^2/(m*L^2)
#
#-------------------------------
@dataclass(frozen=True)
class SquareWell:
    
    V0: float # this is the height of the potential outside the box.

    def __call__(self,
                 y:np.ndarray, # dimensionless position
                 tau: float    # tau is dimensionless time, to be compatible with other functions.
                 ) -> np.ndarray:
        
        y = np.asarray(y, dtype=float)
        v_out = np.full_like(y,self.V0,dtype=float)
        mask = (y<=0.5) & (y>=-0.5) # inside the box
        v_out[mask] = 0.0

        return v_out   

#---------------------------------
#
# Multiple square wells with end caps.
#
# L = well width
# y = x/L 
# a = distance between wells in units of L.
# V0 = height of individual wells.
# Vend = height of endcaps.
#
#---------------------------------
@dataclass
class MultiSquareWells:
    """
    Dimensionless multiple square well potential.
    
    N_wells wells, each of width L=1.
    
    """
    a: float  # distance between well centers in units of L, so a>1 (or center distance > L)
    V0: float # height of interior well barriers
    Vend: float # height of endcap barriers, usually Vend >> V0
    N_wells: int = 1 # number of interior wells.
    center: float = 0.0 # center of the wells. It will usually be zero.

    def __post_init__(self):
        if (self.N_wells < 1):
            raise ValueError("Number of wells, N_wells, must be >= 1.")
        if (self.a < 1.0 ):
            raise ValueError("Well spacing must be more than well width, a > 1.0.")
        
    def well_centers(self) -> np.ndarray:
        """
        Give the centers of the wells in dimensionless units.
        """
        idx = np.arange(self.N_wells,dtype=float)
        return float(self.center) + (idx - 0.5*(self.N_wells-1))*self.a
    
    def __call__(self,
                 y:np.ndarray,
                 tau: float) -> np.ndarray:
        y=np.asarray(y,dtype=float)

        V0=float(self.V0)
        Vend=float(self.Vend)

        c = self.well_centers()
        left_edge=c[0]-0.5
        right_edge=c[-1]+0.5

        v_out = np.full_like(y,V0,dtype=float)

        # outside the "ends"
        v_out[y<left_edge] = Vend
        v_out[y>right_edge] = Vend

        # inside any well
        inside = (np.abs(y[:,None]-c[None,:]) <= 0.5 ).any(axis=1)
        v_out[inside]=0.0

        return v_out
    

#------------------------------
#
# Step barrier of height V0 at position L
#
#-------------------------------
@dataclass(frozen=True)
class StepBarrier:
    
    V0: float # this is the height of the potential outside the box.
    L: float=0.0 # position of the barrier

    def __call__(self,
                 y:np.ndarray, # dimensionless position
                 tau: float    # tau is dimensionless time, to be compatible with other functions.
                 ) -> np.ndarray:
        
        y = np.asarray(y, dtype=float)
        v_out = np.full_like(y,self.V0,dtype=float)
        v_out[y<=self.L]=0.0

        return v_out

#-----------------------------------
#
# Square barrier of height v0, width w (default 1.0), centred at a (default 0.0).
#
# The barrier occupies  a - w/2  <=  y  <=  a + w/2.
#
# y = x/L_ref  (dimensionless position)
# v0 = V0 / E_ref  (dimensionless height)
# w  = barrier width in the same units as y  (default 1.0)
# a  = barrier centre in the same units as y (default 0.0)
#
#-----------------------------------
@dataclass(frozen=True)
class SquareBarrier:
    v0: float
    a: float = 0.0   # centre of barrier
    w: float = 1.0   # width of barrier

    def __call__(self,
                 y: np.ndarray,
                 tau: float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        v_out = np.zeros_like(y, dtype=float)
        v_out[np.abs(y - float(self.a)) <= 0.5 * float(self.w)] = float(self.v0)
        return v_out
    



#---------------------------------------
#
# Cosine potential
#
# V = V0*cos(2pi*y) + Vshift, y=x/a
# 
# Energies are in units of hbar^2/(m*a^2)
#
# a is periodicity
#
#---------------------------------------
@dataclass(frozen=True)
class CosinePotential:
    V0: float = 1.0
    Vshift: float = 0.0 # Constant part of potential
    phi: float = 0.0 # phase offset if any
    a: float=1.0 # periodicity, assume 1

    def __call__(self,
                 y: np.ndarray,
                 tau: float ) -> np.ndarray:
        return ( self.Vshift + self.V0*np.cos(2.0*np.pi*y/self.a + self.phi) )
#----------------------------------------
    
#----------------------------------------
#
# Multi cosine potential
#
#----------------------------------------
@dataclass(frozen=True)
class MultiCosinePotential:
    amps: tuple[float, ...]          # (A1, A2, ...) list of amplitudes
    periods: tuple[float, ...]       # (a1, a2, ...) list of periodicities, a
    phis: tuple[float,...]           # phase offsets
    Vshift: float = 0.0

    def __call__(self, y: np.ndarray, tau: float) -> np.ndarray:
        v = np.full_like(y, self.Vshift, dtype=float)
        for A, a, phi in zip(self.amps, self.periods, self.phis):
            v += A * np.cos(2.0*np.pi*y/a + phi)
        return v
#----------------------------------------


#----------------------------------------
#
# Gaussian Well
#
#  
#
#----------------------------------------
@dataclass(frozen=True)
class GaussianWell:
    V0: float
    sigma: float


    def __post_init__(self):
        if self.sigma <= 1.0e-24:
            raise ValueError("sigma must be > 0.")
        if not np.isfinite(self.sigma):
            raise ValueError("sigma must be finite.")

    def __call__(self,
                 y: np.ndarray,
                 tau: float) -> np.ndarray:
        return -self.V0*np.exp(-0.5*(y/self.sigma)**2)
#------------------------------------------
    

#------------------------------------------
# A periodic Gaussian Well
#------------------------------------------
@dataclass(frozen=True)
class PeriodicGaussianWell:
    V0: float
    sigma: float
    a: float = 1.0
    x0: float = 0.0   # center of well inside cell

    def __post_init__(self):
        if self.a <= 1.0e-24:
            raise ValueError("a must be > 0.")
        if self.sigma <= 1.0e-24:
            raise ValueError("sigma must be > 0.")
        if not np.isfinite(self.a) or not np.isfinite(self.sigma) or not np.isfinite(self.x0):
            raise ValueError("parameters must be finite.")

    @staticmethod
    def _wrap(z: Array, a: float) -> Array:
        return ((z + 0.5*a) % a) - 0.5*a

    def __call__(self, y: Array, tau: float) -> Array:
        # Wrap relative coordinate (crucial fix)
        z = self._wrap(y - self.x0, self.a)
        return -self.V0 * np.exp(-0.5 * (z/self.sigma)**2)
#------------------------------------------

#-------------------------------------------
# Diatomic Gaussian Lattice
#-------------------------------------------
@dataclass(frozen=True)
class PeriodicDiatomicGaussianWell:
    V0_1: float
    sigma_1: float
    x1: float

    V0_2: float
    sigma_2: float
    x2: float

    a: float = 1.0

    def __post_init__(self):
        if self.a <= 1.0e-24:
            raise ValueError("a must be > 0.")
        if self.sigma_1 <= 1.0e-24 or self.sigma_2 <= 1.0e-24:
            raise ValueError("sigma must be > 0.")
        if not np.isfinite(self.a) or not np.isfinite(self.sigma_1) or not np.isfinite(self.sigma_2):
            raise ValueError("parameters must be finite.")
        if not np.isfinite(self.x1) or not np.isfinite(self.x2):
            raise ValueError("x1 and x2 must be finite.")

    @staticmethod
    def _wrap(z: Array, a: float) -> Array:
        return ((z + 0.5*a) % a) - 0.5*a

    def __call__(self, y: Array, tau: float) -> Array:
        # Wrap relative coordinates (crucial fix)
        z1 = self._wrap(y - self.x1, self.a)
        z2 = self._wrap(y - self.x2, self.a)

        v1 = -self.V0_1 * np.exp(-0.5 * (z1/self.sigma_1)**2)
        v2 = -self.V0_2 * np.exp(-0.5 * (z2/self.sigma_2)**2)
        return v1 + v2
#-----------------------------------------


#-----------------------------------------
#
# Softened Square Well
#
# Well width is 1, so if the well width is w, 
# then y = x/w by default when w=1.0.
#
#-----------------------------------------
@dataclass(frozen=True)
class SoftBarrier:
    V0: float            # height or depth (if negative)
    delta: float = 0.1   # transition length
    w: float = 1.0       # well width, defaults to 1

    def __post_init__(self):
        if ( self.delta <= 1.0e-24 ):
            raise ValueError(" Delta must be >= 0.")
        if (self.w <= 1.0e-24 ):
            raise ValueError(" a must be >= 0.")

    def __call__(self,
                 y: np.ndarray,
                 tau: float) -> np.ndarray:
        half_w=self.w/2.0
        v = 0.5*self.V0*(
            np.tanh((y+half_w)/self.delta)
            -np.tanh((y-half_w)/self.delta) )
        return v
#------------------------------------------


#------------------------------------------
# 
# Periodic Softened Barrier
#
#------------------------------------------
@dataclass(frozen=True)
class PeriodicSoftBarrier:
    V0: float
    delta: float = 0.1
    w: float = 1.0
    a: float = 2.5
    x0: float = 0.0

    def __post_init__(self):
        print(
            f" The potential width is w = {self.w:.2f} with periodicity a = {self.a:.2f} "
            f"and center x0 = {self.x0:.2f}."
        )
        if self.delta <= 1e-24:
            raise ValueError("delta must be > 0.")
        if self.w <= 1e-24:
            raise ValueError("w must be > 0.")
        if self.a <= 1e-24:
            raise ValueError("a must be > 0.")
        if self.a <= self.w:
            raise ValueError("Require a > w (feature must fit inside one period).")
        if not np.isfinite(self.x0):
            raise ValueError("x0 must be finite.")

    def __call__(self, y: np.ndarray, tau: float) -> np.ndarray:
        # Wrap RELATIVE coordinate so the feature always tiles correctly
        z = ((y - self.x0 + 0.5*self.a) % self.a) - 0.5*self.a

        half_w = self.w / 2.0
        return 0.5*self.V0 * (
            np.tanh((z + half_w)/self.delta) - np.tanh((z - half_w)/self.delta)
        )
#------------------------------------------



#------------------------------------------
# Diatomic Soft Barrier -- Two different wells.
#------------------------------------------
@dataclass(frozen=True)
class DiatomicSoftBarrierCell:
    """
    A single unit cell potential consisting of TWO softened square wells/barriers.

    V0 can be negative (well) or positive (barrier).
    x1, x2 are positions inside the cell coordinates (typically in [-a/2, a/2)).
    """
    V0_1: float
    delta_1: float
    w_1: float
    x1: float

    V0_2: float
    delta_2: float
    w_2: float
    x2: float

    def __post_init__(self):
        for name, d in (("delta_1", self.delta_1), ("delta_2", self.delta_2)):
            if d <= 1.0e-24:
                raise ValueError(f"{name} must be > 0.")
        for name, w in (("w_1", self.w_1), ("w_2", self.w_2)):
            if w <= 1.0e-24:
                raise ValueError(f"{name} must be > 0.")

    @staticmethod
    def _soft_barrier_centered(y: Array, V0: float, delta: float, w: float, x0: float) -> Array:
        half_w = 0.5 * w
        # same functional form as SoftBarrier, but centered at x0
        z = y - x0
        return 0.5 * V0 * (np.tanh((z + half_w)/delta) - np.tanh((z - half_w)/delta))

    def __call__(self, y_cell: Array, tau: float) -> Array:
        v1 = self._soft_barrier_centered(y_cell, self.V0_1, self.delta_1, self.w_1, self.x1)
        v2 = self._soft_barrier_centered(y_cell, self.V0_2, self.delta_2, self.w_2, self.x2)
        return v1 + v2
#------------------------------------------


#------------------------------------------
# Periodic Diatomic Soft Barrier
#------------------------------------------
@dataclass(frozen=True)
class PeriodicDiatomicSoftBarrier:
    """
    Robust periodic version of DiatomicSoftBarrierCell with lattice period a.

    IMPORTANT: wraps relative coordinates (y - x_i) into [-a/2, a/2),
    so the potential is truly periodic even if a feature straddles the cell boundary.
    """
    cell: DiatomicSoftBarrierCell
    a: float = 1.0

    def __post_init__(self):
        if self.a <= 1.0e-24:
            raise ValueError("a must be > 0.")
        if not np.isfinite(self.a):
            raise ValueError("a must be finite.")

        # Each feature must fit inside one period (same check you had elsewhere).
        if self.a <= self.cell.w_1 or self.a <= self.cell.w_2:
            raise ValueError("Require a > w_1 and a > w_2 (each feature must fit inside one period).")

        # Optional clarity check (not required for correctness, but helps avoid confusion)
        if abs(self.cell.x1) > 0.5*self.a or abs(self.cell.x2) > 0.5*self.a:
            raise ValueError("x1 and x2 should lie within [-a/2, a/2] for clarity.")

    @staticmethod
    def _wrap(z: Array, a: float) -> Array:
        """Wrap z into [-a/2, a/2)."""
        return ((z + 0.5*a) % a) - 0.5*a

    @staticmethod
    def _soft_barrier(z: Array, V0: float, delta: float, w: float) -> Array:
        """Softened square barrier/well centered at 0."""
        half_w = 0.5 * w
        return 0.5 * V0 * (
            np.tanh((z + half_w)/delta) - np.tanh((z - half_w)/delta)
        )

    def __call__(self, y: Array, tau: float) -> Array:
        # Wrap relative to each center (the crucial fix)
        z1 = self._wrap(y - self.cell.x1, self.a)
        z2 = self._wrap(y - self.cell.x2, self.a)

        v1 = self._soft_barrier(z1, self.cell.V0_1, self.cell.delta_1, self.cell.w_1)
        v2 = self._soft_barrier(z2, self.cell.V0_2, self.cell.delta_2, self.cell.w_2)
        return v1 + v2
#--------------------------------------------------------------


#===============================================================
#
# Make periodic potentials from individual "cell" 
# potentials.
#
#===============================================================
@dataclass(frozen=True)
class PeriodicFromCell:
    V_cell: callable # the individual cell potential
    a: float = 1.0 # the periodicity

    def __post_init__(self):
        if (self.a <= 0.0 ):
            raise ValueError("Spacing a must be positive.")
        if ( not np.isfinite(self.a) ):
            raise ValueError("Spacing a must be finite.")
        if (not callable(self.V_cell)):
            raise TypeError("V_cell must be callable.")
        
    def __call__(self,
                 y: np.ndarray,
                 tau: float) -> np.ndarray:
        y_wrapped = ( (y+0.5*self.a) % self.a ) - 0.5*self.a
        return self.V_cell(y_wrapped,tau)
#-------- Now periodic potentials can be build. ----------------|


#================================================================
#
#  Composition of potentials.
#
#================================================================
@dataclass(frozen=True)
class SumPotential:
    terms: tuple

    def __call__(self, y: np.ndarray, tau: float) -> np.ndarray:
        # Avoid building a long chain of temporaries: sum in one accumulator
        out = np.zeros_like(y, dtype=float)
        for pot in self.terms:
            out += pot(y, tau)
        return out

def Vsum(*pots):
    # Flatten nested sums so Vsum(Vsum(A,B),C) doesn't nest
    flat = []
    for p in pots:
        if isinstance(p, SumPotential):
            flat.extend(p.terms)
        else:
            flat.append(p)
    return SumPotential(tuple(flat))

    
#----------------------------------------------------------------
#
# Complex Absorbing Potential (CAP)
#
# Returns the real absorption profile W(y) >= 0.
#
# The propagators add -i*W to the effective Hamiltonian, causing
# probability to be smoothly absorbed near the domain boundaries
# instead of wrapping around (periodic BC artefact).
#
# Profile: quadratic ramp from 0 at the inner edge of the absorbing
# layer to `strength` at the hard boundary ±y_max.
#
#   W(y) = strength * ( distance_into_layer / width )^2
#
# Parameters
# ----------
# strength : peak absorption rate W0 > 0  (units: 1/time)
# width    : thickness of each absorbing layer  (same units as y)
# y_max    : half-width of the computational domain; walls at ±y_max
#
# Usage
# -----
#   cap = ComplexAbsorbingPotential(strength=0.5, width=2.0, y_max=12.0)
#   W   = cap(grid.y, 0.0)   # real array, shape (N,)
#   psi_t, diag = split_step_propagate(..., cap=W)
#
#----------------------------------------------------------------
@dataclass(frozen=True)
class ComplexAbsorbingPotential:
    strength: float   # peak absorption rate W0
    width: float      # thickness of each absorbing layer
    y_max: float      # domain half-width; boundaries at ±y_max

    def __post_init__(self):
        if self.strength <= 0.0:
            raise ValueError("strength must be > 0.")
        if self.width <= 0.0:
            raise ValueError("width must be > 0.")
        if self.y_max <= 0.0:
            raise ValueError("y_max must be > 0.")
        if self.width >= self.y_max:
            raise ValueError(
                "width must be less than y_max "
                "(absorbing layer must fit inside the domain)."
            )

    def __call__(self, y: np.ndarray, tau: float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        W = np.zeros_like(y, dtype=float)

        # Right absorbing layer: inner edge at y_max - width, wall at +y_max
        d_r = y - (self.y_max - self.width)
        mask_r = d_r > 0.0
        W[mask_r] += self.strength * (d_r[mask_r] / self.width) ** 2

        # Left absorbing layer: inner edge at -y_max + width, wall at -y_max
        d_l = (-self.y_max + self.width) - y
        mask_l = d_l > 0.0
        W[mask_l] += self.strength * (d_l[mask_l] / self.width) ** 2

        return W
#----------------------------------------------------------------


#----------------------------------------------------------------
#
# Use Cached Potentials for time-independent potentials to improve speed.
#
#----------------------------------------------------------------
@dataclass
class CachedPotential:
    pot: object
    _y_id: int | None = None
    _v: np.ndarray | None = None

    def __call__(self, y: np.ndarray, tau: float) -> np.ndarray:
        # (Re)compute cache if needed
        if self._v is None or self._y_id != id(y):
            v = np.array(self.pot(y, 0.0), copy=True)   # materialize a private copy
            v.setflags(write=False)                      # make read-only
            self._v = v
            self._y_id = id(y)
        return self._v
#------------------------------------------------------------------
