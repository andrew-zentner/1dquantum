"""
benchmark_solvers.py
====================
Compare split_step_propagate (NumPy / scipy.fft workers=-1) against
split_step_propagate_torch (PyTorch — CPU and, if present, MPS).

Grid : N = 2^13 = 8192,  y in [-16, 16)
Steps: Nt = 1000,  dt = 0.001
Potential: SoftBarrier + ComplexAbsorbingPotential (same parameters as
           square_barrier_scattering.py)

For each variant we do:
    1 warm-up run  (not timed, ensures JIT / caches are hot)
    N_RUNS timed runs  → report mean ± std

Precision notes
---------------
* MPS does NOT support float64/complex128 — it raises a TypeError.
  MPS requires float32/complex64 for native GPU execution.
* torch CPU runs in complex128 (matching scipy) and also in complex64
  so you can see the precision-vs-speed trade-off on CPU.
* Memory for return_all=True (complex128): (Nt+1) * N * 16 bytes ≈ 131 MB.
"""

import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schrodinger_solver_1d       import (make_grid as make_grid_np,
                                          split_step_propagate)
from schrodinger_propagate_torch import (make_grid as make_grid_torch,
                                          split_step_propagate_torch)
from wavefunctions  import gaussian_wavepacket
from potentials_1d  import SoftBarrier, CachedPotential, ComplexAbsorbingPotential

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N      = 2**13          # 8192
y_max  = 16.0
Nt     = 1000           # number of propagation steps
dt     = 0.001
N_RUNS = 3              # timed repetitions per variant

# ---------------------------------------------------------------------------
# Grid, initial state, potential, CAP
# ---------------------------------------------------------------------------
grid_np    = make_grid_np   (N, y_max)
grid_torch = make_grid_torch(N, y_max)

tau  = np.linspace(0.0, Nt * dt, Nt + 1)   # Nt+1 points → Nt steps in solver loop

psi0    = gaussian_wavepacket(grid_np.y, y0=-6.0, k0=4.0, sigma=0.8)
barrier = CachedPotential(SoftBarrier(v0=9.0, delta=0.02, w=0.2, x0=0.0))

cap_obj = ComplexAbsorbingPotential(strength=12.0, width=3.0, y_max=y_max)
cap_arr = cap_obj(grid_np.y, 0.0)

# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
COL = 58   # width of name column

def bench(name: str, fn, n_runs: int = N_RUNS) -> float:
    fn()                                # warm-up (not timed)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mu  = float(np.mean(times))
    sig = float(np.std(times))
    tag = f"{mu:.3f} ± {sig:.3f} s"
    print(f"  {name:<{COL}}  {tag}")
    return mu

# ---------------------------------------------------------------------------
# Announce
# ---------------------------------------------------------------------------
mem_mb = (Nt + 1) * N * 16 / 1024**2   # complex128 history size

print()
print("=" * 82)
print(f"  Solver benchmark  N={N} (2^13)  y_max={y_max}  Nt={Nt} steps  dt={dt}")
print(f"  {N_RUNS} timed runs each after 1 warm-up.  History array ~{mem_mb:.0f} MB (c128).")
print("=" * 82)

# storage for summary table
results: dict[str, float | None] = {}

# ---------------------------------------------------------------------------
# 1.  NumPy / scipy.fft  (workers = -1)
# ---------------------------------------------------------------------------
print("\n[1] NumPy / scipy.fft  (workers=-1,  complex128)")

results['scipy  no-store  no-CAP'] = bench(
    "return_all=False   no CAP",
    lambda: split_step_propagate(
        psi0, barrier, tau, grid_np,
        time_independent=True, return_all=False))

results['scipy  no-store  CAP'] = bench(
    "return_all=False   with CAP",
    lambda: split_step_propagate(
        psi0, barrier, tau, grid_np,
        time_independent=True, return_all=False, cap=cap_arr))

results['scipy  store    no-CAP'] = bench(
    "return_all=True    no CAP",
    lambda: split_step_propagate(
        psi0, barrier, tau, grid_np,
        time_independent=True, return_all=True))

results['scipy  store    CAP'] = bench(
    "return_all=True    with CAP",
    lambda: split_step_propagate(
        psi0, barrier, tau, grid_np,
        time_independent=True, return_all=True, cap=cap_arr))

# ---------------------------------------------------------------------------
# 2.  PyTorch  CPU  complex128
# ---------------------------------------------------------------------------
print("\n[2] PyTorch  CPU  complex128")

results['cpu128  no-store  no-CAP'] = bench(
    "return_all=False   no CAP",
    lambda: split_step_propagate_torch(
        psi0, barrier, tau, grid_torch,
        time_independent=True, return_all=False,
        device='cpu', dtype=torch.complex128))

results['cpu128  no-store  CAP'] = bench(
    "return_all=False   with CAP",
    lambda: split_step_propagate_torch(
        psi0, barrier, tau, grid_torch,
        time_independent=True, return_all=False,
        device='cpu', dtype=torch.complex128, cap=cap_arr))

results['cpu128  store    no-CAP'] = bench(
    "return_all=True    no CAP",
    lambda: split_step_propagate_torch(
        psi0, barrier, tau, grid_torch,
        time_independent=True, return_all=True,
        device='cpu', dtype=torch.complex128))

results['cpu128  store    CAP'] = bench(
    "return_all=True    with CAP",
    lambda: split_step_propagate_torch(
        psi0, barrier, tau, grid_torch,
        time_independent=True, return_all=True,
        device='cpu', dtype=torch.complex128, cap=cap_arr))

# ---------------------------------------------------------------------------
# 3.  PyTorch  CPU  complex64  (to isolate float32 speed, same as MPS path)
# ---------------------------------------------------------------------------
print("\n[3] PyTorch  CPU  complex64  (float32 — same arithmetic as MPS)")

results['cpu64  no-store  no-CAP'] = bench(
    "return_all=False   no CAP",
    lambda: split_step_propagate_torch(
        psi0, barrier, tau, grid_torch,
        time_independent=True, return_all=False,
        device='cpu', dtype=torch.complex64))

results['cpu64  store    CAP'] = bench(
    "return_all=True    with CAP",
    lambda: split_step_propagate_torch(
        psi0, barrier, tau, grid_torch,
        time_independent=True, return_all=True,
        device='cpu', dtype=torch.complex64, cap=cap_arr))

# ---------------------------------------------------------------------------
# 4.  PyTorch  MPS  complex64  (Apple Silicon GPU — native float32)
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    print("\n[4] PyTorch  MPS  complex64  (Apple Silicon GPU — native float32)")
    print("    Note: return_all=True includes a GPU→CPU copy every step.")

    results['mps64  no-store  no-CAP'] = bench(
        "return_all=False   no CAP",
        lambda: split_step_propagate_torch(
            psi0, barrier, tau, grid_torch,
            time_independent=True, return_all=False,
            device='mps', dtype=torch.complex64))

    results['mps64  no-store  CAP'] = bench(
        "return_all=False   with CAP",
        lambda: split_step_propagate_torch(
            psi0, barrier, tau, grid_torch,
            time_independent=True, return_all=False,
            device='mps', dtype=torch.complex64, cap=cap_arr))

    results['mps64  store    no-CAP'] = bench(
        "return_all=True    no CAP",
        lambda: split_step_propagate_torch(
            psi0, barrier, tau, grid_torch,
            time_independent=True, return_all=True,
            device='mps', dtype=torch.complex64))

    results['mps64  store    CAP'] = bench(
        "return_all=True    with CAP",
        lambda: split_step_propagate_torch(
            psi0, barrier, tau, grid_torch,
            time_independent=True, return_all=True,
            device='mps', dtype=torch.complex64, cap=cap_arr))
else:
    for k in ('mps64  no-store  no-CAP', 'mps64  no-store  CAP',
              'mps64  store    no-CAP',   'mps64  store    CAP'):
        results[k] = None
    print("\n[4] MPS not available — skipping GPU variants.")

# ---------------------------------------------------------------------------
# 5.  Summary table
# ---------------------------------------------------------------------------
ref = results['scipy  no-store  no-CAP']   # baseline

print()
print("-" * 82)
print(f"  Summary  (baseline = scipy return_all=False no-CAP = {ref:.3f} s)")
print(f"  {'Variant':<42}  {'Time (s)':>10}  {'ratio':>8}")
print("-" * 82)

rows = [
    ("scipy         no-store  no-CAP  c128", 'scipy  no-store  no-CAP'),
    ("scipy         no-store  CAP     c128", 'scipy  no-store  CAP'),
    ("scipy         store     no-CAP  c128", 'scipy  store    no-CAP'),
    ("scipy         store     CAP     c128", 'scipy  store    CAP'),
    ("torch CPU     no-store  no-CAP  c128", 'cpu128  no-store  no-CAP'),
    ("torch CPU     no-store  CAP     c128", 'cpu128  no-store  CAP'),
    ("torch CPU     store     no-CAP  c128", 'cpu128  store    no-CAP'),
    ("torch CPU     store     CAP     c128", 'cpu128  store    CAP'),
    ("torch CPU     no-store  no-CAP  c64 ", 'cpu64  no-store  no-CAP'),
    ("torch CPU     store     CAP     c64 ", 'cpu64  store    CAP'),
    ("torch MPS     no-store  no-CAP  c64 ", 'mps64  no-store  no-CAP'),
    ("torch MPS     no-store  CAP     c64 ", 'mps64  no-store  CAP'),
    ("torch MPS     store     no-CAP  c64 ", 'mps64  store    no-CAP'),
    ("torch MPS     store     CAP     c64 ", 'mps64  store    CAP'),
]

for label, key in rows:
    t = results.get(key)
    if t is None:
        print(f"  {label:<42}  {'N/A':>10}  {'N/A':>8}")
    else:
        ratio = t / ref
        print(f"  {label:<42}  {t:>10.3f}  {ratio:>7.2f}x")

print("-" * 82)
print()
