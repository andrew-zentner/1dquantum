
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
                 tau: float) -> float:
        v = lam*(y**4)
        return float(v)
    

#------------------------------
#
# Symmetric square well.
#
# y = x/L
# V0 = V_0/E_L with E_L = hbar^2/(2*m*L^2)
#
#-------------------------------
@dataclass(frozen=True)
class SymmetricBox:
    
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
        return float(self.center) + (idx - 0.5*(self.N_wells-1)*a)
    
    def __call__(self,
                 y:np.ndarray,
                 tau: float) -> np.npdarray:
        y=np.asarray(y,dtype=float)

        V0=float(self.V0)
        Vend=float(self.Vend)

        c = self.centers()
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
        v_out[y<=L]=0.0

        return v_out

#-----------------------------------
#
# Square barrier of height V0, width L=1, and center position a.
#
# y = x/L
# position is in units of L, a = x_center/L
# E_L = hbar^2(2*m*L^2)
# V0 = Potential/E_L
#
#-----------------------------------
@dataclass
class SquareBarrier:
    v0: float
    a: float = 0.0 # center of barrier

    def __call__(self,
                 y: np.ndarray,
                 tau: float ) -> np.ndarray:
        y=np.asarray(y,dtype=float)
        v_out=np.zeros_like(y,dtype=float)
        v_out[np.aps(y-float(self.a) <= 0.5)] = float(self.V0)
        return v_out
    


