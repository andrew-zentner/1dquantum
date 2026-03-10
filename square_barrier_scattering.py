"""
square_barrier_scattering.py
============================
Gaussian wavepacket scattering off a 1D square barrier.

  Domain   : y in [-12, +12)
  Initial  : Gaussian centered at y0=-6, moving right with k0=4
  Barrier  : height V0=6, width 1 (spanning y in [-0.5, +0.5])

Mean kinetic energy E = k0^2/2 = 8.0 > V0 = 6.0:
both quantum reflection and transmission are visible.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from schrodinger_solver_1d import make_grid, split_step_propagate
from wavefunctions import gaussian_wavepacket, gaussian_wavepacket_energy
from wavefunction_movies import make_movie
from potentials_1d import SquareBarrier, CachedPotential, ComplexAbsorbingPotential, SumPotential

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
N     = 2**12
y_max = 16.0
grid  = make_grid(N, y_max)

# ---------------------------------------------------------------------------
# Square barrier potential  (static)
# SquareBarrier(v0, a): height v0, width=1, centred at a.
# CachedPotential memoises the result so V is evaluated only once.
# ---------------------------------------------------------------------------
V0             = 6.0
barrier_center = 0.0
#barrier        = CachedPotential(SquareBarrier(v0=V0, a=barrier_center, w=1.0))
barrier        = CachedPotential( SumPotential([SquareBarrier(v0=0.9*V0,a=barrier_center, w=1.0),
                                                SquareBarrier(v0=1.1*V0,a=barrier_center+4.0,w=1.0)]) )

# ---------------------------------------------------------------------------
# Initial Gaussian wavepacket
# ---------------------------------------------------------------------------
y0    = -6.0   # center position
k0    = 4.0    # mean wavenumber / momentum  (positive = right-moving)
sigma = 0.8    # spatial width (Δy = sigma)

psi0   = gaussian_wavepacket(grid.y, y0, k0, sigma)
E_mean = gaussian_wavepacket_energy(k0, sigma)

print(f"Mean wavepacket energy : {E_mean:.3f}")
print(f"Barrier height V0      : {V0:.3f}")
print(f"E / V0                 : {E_mean / V0:.3f}  (>1 -> over-barrier, partial reflection)")

# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------
dt      = 0.001
tau_max = 10.0
tau     = np.arange(0.0, tau_max + 0.5 * dt, dt)
Nt      = tau.size
print(f"Grid   : N={N}, y_max={y_max}, dy={grid.dy:.5f}")
print(f"Time   : Nt={Nt}, dt={dt}, tau_max={tau[-1]:.3f}")
print()

#----------------------------------------------------------------------------
# imaginary cap potential to absorb wavefunctions that run off of the "edge" of 
# the computational grid.
#----------------------------------------------------------------------------
w_potential = ComplexAbsorbingPotential(strength=10.0,
                                  width=4.0,
                                  y_max=y_max)
w_cap = w_potential(grid.y,0.0)

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
print("Running split-step propagation (time_independent=True) ...")
psi_t, diag = split_step_propagate(
    psi0, barrier, tau, grid,
    time_independent=True,
    return_all=True,
    cap=w_cap
)
print(f"  Done.  Norm at final step: {diag['norm'][-1]:.10f}  (should be 1.0)")
print()

# ---------------------------------------------------------------------------
# Potential array for the movie overlay
# ---------------------------------------------------------------------------
V_arr = barrier(grid.y, 0.0)

# ---------------------------------------------------------------------------
# Movie
# ---------------------------------------------------------------------------
stride   = 8        # keep every 7th frame  → ~501 frames at 30 fps ≈ 17 s
outfile  = os.path.join(os.path.dirname(__file__), "square_barrier_scattering.mp4")

print("Rendering movie ...")
out = make_movie(
    psi_t, grid, tau,
    outfile=outfile,
    what="abs2",
    potential=V_arr,
    stride=stride,
    fps=40,
    dpi=150,
    figsize=(9, 5),
    xlim=(-12.0, 12.0),
    ylim=(0.0, 0.60),
    Vlim=(0.0, 8.0),
    psi_color="midnightblue",
    potential_color="firebrick",
    potential_alpha=0.75,
    psi_lw=2.0,
    potential_lw=2.0,
)
print(f"Movie saved to : {out}")
