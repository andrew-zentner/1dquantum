from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt

import numpy as np
import math

from matplotlib.animation import FuncAnimation
from IPython.display import Video


# =============================================================================
#
#  make_movie — general-purpose wavefunction movie with right-axis potential
#
#  Typical use:
#
#    psi_t, diag = split_step_propagate(psi0, V, tau_grid, grid, ...)
#    make_movie(psi_t, grid, diag["tau"], "scattering.mp4", potential=V)
#
# =============================================================================
def make_movie(
    psi_t: np.ndarray,
    grid,
    tau: np.ndarray,
    outfile: str = "movie.mp4",
    what: str = "abs2",
    potential=None,
    stride: int = 1,
    fps: int = 30,
    dpi: int = 150,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    Vlim: Optional[tuple] = None,
    figsize: tuple = (8, 4.5),
    psi_color: str = "tab:blue",
    potential_color: str = "tab:orange",
    potential_alpha: float = 0.8,
    psi_lw: float = 2.0,
    potential_lw: float = 2.0,
) -> str:
    """
    Save an animation of the wavefunction evolution to an .mp4 or .gif file.

    The wavefunction quantity is plotted on the left y-axis.  When a potential
    is provided it is drawn on a dedicated right-hand y-axis (using its own
    physical units, color-coded to distinguish it from the wavefunction).

    Parameters
    ----------
    psi_t     : (Nt, Ny) complex array returned by a propagator.
    grid      : Grid1D returned by make_grid or make_periodic_grid.
    tau       : (Nt,) time values, e.g. diagnostics["tau"].
    outfile   : output path; must end in ".mp4" or ".gif".
    what      : quantity to animate — one of:
                  "abs2"  |psi|^2  (default, most useful for scattering)
                  "abs"   |psi|
                  "real"  Re psi
                  "imag"  Im psi
    potential : how to display the potential — three options:
                  None          no potential shown (default)
                  (Ny,) array   static potential, plotted as a fixed line
                  callable      V(y, tau) -> (Ny,) for a time-varying
                                potential; the line is updated each frame
    stride    : plot every stride-th frame to control movie length / file size.
    fps       : frames per second in the output file.
    dpi       : output resolution in dots per inch.
    xlim      : (xmin, xmax) for the spatial axis; auto if None.
    ylim      : (ymin, ymax) for the wavefunction (left) axis; auto if None.
    Vlim      : (Vmin, Vmax) for the potential (right) axis; auto if None.
    figsize   : figure dimensions in inches, default (8, 4.5).
    psi_color       : color of the wavefunction line, default "tab:blue".
    potential_color : color of the potential line, default "tab:orange".
    potential_alpha : opacity of the potential line, default 0.8.
    psi_lw          : line width of the wavefunction line, default 2.0.
    potential_lw    : line width of the potential line, default 2.0.

    Returns
    -------
    outfile : the path that was written.

    Examples
    --------
    # Minimal call — no potential shown
    make_movie(psi_t, grid, diag["tau"], "free.mp4")

    # Static potential from a pre-evaluated array
    V_arr = SquareBarrier(v0=5.0)(grid.y, 0.0)
    make_movie(psi_t, grid, diag["tau"], "barrier.mp4", potential=V_arr)

    # Time-varying potential (callable)
    make_movie(psi_t, grid, diag["tau"], "ramp.mp4", potential=V_ho,
               what="abs2", stride=5, title=r"Harmonic ramp")
    """
    psi_t = np.asarray(psi_t)
    y     = np.asarray(grid.y, dtype=float)
    tau   = np.asarray(tau, dtype=float)

    # --- input validation ---
    if psi_t.ndim != 2:
        raise ValueError("psi_t must have shape (Nt, Ny).")
    Nt, Ny = psi_t.shape
    if y.size != Ny:
        raise ValueError(f"grid.y has size {y.size} but psi_t has Ny={Ny}.")
    if tau.size != Nt:
        raise ValueError(f"tau has size {tau.size} but psi_t has Nt={Nt}.")
    if stride < 1:
        raise ValueError("stride must be >= 1.")

    _labels = {
        "abs2": r"$|\psi|^2$",
        "abs":  r"$|\psi|$",
        "real": r"$\mathrm{Re}\,\psi$",
        "imag": r"$\mathrm{Im}\,\psi$",
    }
    if what not in _labels:
        raise ValueError(f"what must be one of {list(_labels)}.")

    def extract(arr: np.ndarray) -> np.ndarray:
        if what == "abs2": return np.abs(arr) ** 2
        if what == "abs":  return np.abs(arr)
        if what == "real": return arr.real
        return arr.imag

    frames = np.arange(0, Nt, stride)

    # --- potential mode detection ---
    _V_arr:      Optional[np.ndarray] = None  # set if static array provided
    _V_callable: Optional[Callable]   = None  # set if callable provided

    if potential is not None:
        if callable(potential):
            _V_callable = potential
        else:
            _V_arr = np.asarray(potential, dtype=float)
            if _V_arr.shape != y.shape:
                raise ValueError(
                    f"potential array shape {_V_arr.shape} does not match "
                    f"grid.y shape {y.shape}."
                )

    def _get_V(tau_val: float) -> Optional[np.ndarray]:
        if _V_callable is not None:
            return np.asarray(_V_callable(y, tau_val), dtype=float)
        return _V_arr  # None if no potential

    # --- figure and left axis ---
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.87 if potential is not None else 0.95)

    ax.set_xlabel(r"$y$", fontsize=13)
    ax.set_ylabel(_labels[what], fontsize=13, color=psi_color)
    ax.tick_params(axis="y", colors=psi_color, labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.yaxis.label.set_color(psi_color)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(float(y[0]), float(y[-1]))

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        sample = np.array([extract(psi_t[i]) for i in frames])
        ymin, ymax = float(sample.min()), float(sample.max())
        pad = 0.08 * (ymax - ymin + 1e-15)
        ax.set_ylim(ymin - pad, ymax + pad)

    # wavefunction line
    (psi_line,) = ax.plot(
        y, extract(psi_t[frames[0]]),
        color=psi_color, lw=psi_lw, zorder=3
    )

    # time label
    time_text = ax.text(
        0.02, 0.96, f"$\\tau = {tau[frames[0]]:.3f}$",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=11, color="0.35"
    )

    # --- right axis and potential line ---
    axV     = None
    V_line  = None

    if potential is not None:
        axV = ax.twinx()
        axV.set_ylabel(
            r"$V(y,\tau)$" if _V_callable is not None else r"$V(y)$",
            fontsize=13, color=potential_color
        )
        axV.tick_params(axis="y", colors=potential_color, labelsize=10)
        axV.yaxis.label.set_color(potential_color)

        # V-axis limits
        if Vlim is not None:
            axV.set_ylim(*Vlim)
        else:
            if _V_callable is not None:
                Vsample = np.array([_get_V(float(tau[i])) for i in frames])
            else:
                Vsample = _V_arr[np.newaxis, :]
            Vmin, Vmax = float(Vsample.min()), float(Vsample.max())
            Vpad = 0.08 * (Vmax - Vmin + 1e-15)
            axV.set_ylim(Vmin - Vpad, Vmax + Vpad)

        V0 = _get_V(float(tau[frames[0]]))
        (V_line,) = axV.plot(
            y, V0,
            color=potential_color, lw=potential_lw, ls="--",
            alpha=potential_alpha, zorder=2
        )

    # --- animation ---
    def update(fi: int):
        idx = frames[fi]

        psi_line.set_ydata(extract(psi_t[idx]))
        time_text.set_text(f"$\\tau = {tau[idx]:.3f}$")

        if V_line is not None and _V_callable is not None:
            V_line.set_ydata(_get_V(float(tau[idx])))

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False
    )

    # H.264 (and most video codecs) require both pixel dimensions to be even.
    # Enforce this by nudging the figure size up by 1 pixel if needed.
    if outfile.lower().endswith(".mp4"):
        _w_px = round(fig.get_figwidth()  * dpi)
        _h_px = round(fig.get_figheight() * dpi)
        if _w_px % 2 != 0:
            fig.set_size_inches((_w_px + 1) / dpi, fig.get_figheight())
        if _h_px % 2 != 0:
            fig.set_size_inches(fig.get_figwidth(), (_h_px + 1) / dpi)

    if outfile.lower().endswith(".gif"):
        anim.save(outfile, writer="pillow", fps=fps, dpi=dpi)
    elif outfile.lower().endswith(".mp4"):
        anim.save(outfile, writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        raise ValueError('outfile must end in ".mp4" or ".gif".')

    plt.close(fig)
    return outfile


# =============================================================================
#
#  make_two_panel_movie — wavefunction + potential (left) and <N(τ)> (right)
#
#  Typical use:
#
#    from wavefunctions import number_expectation
#    k      = np.fft.fftfreq(grid.Ny, d=grid.dy) * 2 * np.pi
#    N_vals = number_expectation(psi_t, grid.y, k, grid.dy,
#                                diag["tau"], Omega_of_tau=omega_ramp)
#    Omega_vals = np.array([omega_ramp(t) for t in diag["tau"]])
#    make_two_panel_movie(psi_t, grid, diag["tau"], N_vals,
#                         outfile="ramp.mp4", potential=V,
#                         Omega_vals=Omega_vals)
#
# =============================================================================
def make_two_panel_movie(
    psi_t: np.ndarray,
    grid,
    tau: np.ndarray,
    N_vals: np.ndarray,
    outfile: str = "movie.mp4",
    what: str = "abs2",
    potential=None,
    Omega_vals: Optional[np.ndarray] = None,
    stride: int = 1,
    fps: int = 30,
    dpi: int = 150,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    Vlim: Optional[tuple] = None,
    Nlim: Optional[tuple] = None,
    OmegaLim: Optional[tuple] = None,
    figsize: tuple = (12, 4.5),
    psi_color: str = "tab:blue",
    potential_color: str = "tab:orange",
    potential_alpha: float = 0.8,
    psi_lw: float = 2.0,
    potential_lw: float = 2.0,
    N_color: str = "tab:blue",
    N_lw: float = 2.0,
    Omega_color: str = "tab:orange",
    Omega_lw: float = 2.0,
) -> str:
    """
    Two-panel animation:
      Left:  wavefunction quantity on left y-axis, V(y,τ) on right y-axis
             (identical layout to make_movie).
      Right: <N̂(τ)> growing history on left y-axis; optional Ω(τ) growing
             history on right y-axis (pass Omega_vals to enable).

    N_vals must be pre-computed before calling this function, e.g. via
    wavefunctions.number_expectation.

    Parameters
    ----------
    psi_t     : (Nt, Ny) complex array returned by a propagator.
    grid      : Grid1D returned by make_grid or make_periodic_grid.
    tau       : (Nt,) time values, e.g. diagnostics["tau"].
    N_vals    : (Nt,) pre-computed <N̂(τ)> (from number_expectation).
    outfile   : output path; must end in ".mp4" or ".gif".
    what      : quantity to animate — "abs2", "abs", "real", or "imag".
    potential : None / (Ny,) array / callable V(y,τ); shown on right axis
                of the left panel.
    Omega_vals: (Nt,) optional Ω(τ) array; if provided it is shown on the
                right axis of the right panel.
    stride    : plot every stride-th frame.
    fps       : frames per second.
    dpi       : output resolution.
    xlim      : (xmin, xmax) for the spatial axis; auto if None.
    ylim      : (ymin, ymax) for the wavefunction axis; auto if None.
    Vlim      : (Vmin, Vmax) for the potential axis; auto if None.
    Nlim      : (Nmin, Nmax) for the <N̂> axis; auto if None.
    OmegaLim  : (Omin, Omax) for the Ω axis; auto if None.
    figsize   : figure size in inches, default (12, 4.5).
    psi_color       : color of the wavefunction line, default "tab:blue".
    potential_color : color of the potential line, default "tab:orange".
    potential_alpha : opacity of the potential line, default 0.8.
    psi_lw          : line width of the wavefunction line, default 2.0.
    potential_lw    : line width of the potential line, default 2.0.
    N_color    : color of the <N̂> history line, default "tab:blue".
    N_lw       : line width of the <N̂> line, default 2.0.
    Omega_color: color of the Ω history line, default "tab:orange".
    Omega_lw   : line width of the Ω line, default 2.0.

    Returns
    -------
    outfile : the path that was written.
    """
    psi_t  = np.asarray(psi_t)
    y      = np.asarray(grid.y, dtype=float)
    tau    = np.asarray(tau,    dtype=float)
    N_vals = np.asarray(N_vals, dtype=float)

    # --- input validation ---
    if psi_t.ndim != 2:
        raise ValueError("psi_t must have shape (Nt, Ny).")
    Nt, Ny = psi_t.shape
    if y.size != Ny:
        raise ValueError(f"grid.y has size {y.size} but psi_t has Ny={Ny}.")
    if tau.size != Nt:
        raise ValueError(f"tau has size {tau.size} but psi_t has Nt={Nt}.")
    if N_vals.size != Nt:
        raise ValueError(f"N_vals has size {N_vals.size} but psi_t has Nt={Nt}.")
    if Omega_vals is not None:
        Omega_vals = np.asarray(Omega_vals, dtype=float)
        if Omega_vals.size != Nt:
            raise ValueError(f"Omega_vals has size {Omega_vals.size} but psi_t has Nt={Nt}.")
    if stride < 1:
        raise ValueError("stride must be >= 1.")

    _labels = {
        "abs2": r"$|\psi|^2$",
        "abs":  r"$|\psi|$",
        "real": r"$\mathrm{Re}\,\psi$",
        "imag": r"$\mathrm{Im}\,\psi$",
    }
    if what not in _labels:
        raise ValueError(f"what must be one of {list(_labels)}.")

    def extract(arr: np.ndarray) -> np.ndarray:
        if what == "abs2": return np.abs(arr) ** 2
        if what == "abs":  return np.abs(arr)
        if what == "real": return arr.real
        return arr.imag

    frames = np.arange(0, Nt, stride)

    # --- potential mode detection (same as make_movie) ---
    _V_arr:      Optional[np.ndarray] = None
    _V_callable: Optional[Callable]   = None

    if potential is not None:
        if callable(potential):
            _V_callable = potential
        else:
            _V_arr = np.asarray(potential, dtype=float)
            if _V_arr.shape != y.shape:
                raise ValueError(
                    f"potential array shape {_V_arr.shape} does not match "
                    f"grid.y shape {y.shape}."
                )

    def _get_V(tau_val: float) -> Optional[np.ndarray]:
        if _V_callable is not None:
            return np.asarray(_V_callable(y, tau_val), dtype=float)
        return _V_arr

    # =========================================================================
    # Figure layout: two panels, left slightly wider to accommodate potential axis
    # =========================================================================
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [1.2, 1.0]}
    )
    fig.subplots_adjust(
        left=0.08, right=0.93, wspace=0.55,
        top=0.93, bottom=0.13
    )

    # =========================================================================
    # LEFT PANEL — wavefunction + potential (mirrors make_movie exactly)
    # =========================================================================
    axL.set_xlabel(r"$y$", fontsize=13)
    axL.set_ylabel(_labels[what], fontsize=13, color=psi_color)
    axL.tick_params(axis="y", colors=psi_color, labelsize=10)
    axL.tick_params(axis="x", labelsize=10)
    axL.yaxis.label.set_color(psi_color)

    if xlim is not None:
        axL.set_xlim(*xlim)
    else:
        axL.set_xlim(float(y[0]), float(y[-1]))

    if ylim is not None:
        axL.set_ylim(*ylim)
    else:
        sample = np.array([extract(psi_t[i]) for i in frames])
        ymin, ymax = float(sample.min()), float(sample.max())
        pad = 0.08 * (ymax - ymin + 1e-15)
        axL.set_ylim(ymin - pad, ymax + pad)

    (psi_line,) = axL.plot(
        y, extract(psi_t[frames[0]]),
        color=psi_color, lw=psi_lw, zorder=3
    )

    time_text = axL.text(
        0.02, 0.96, f"$\\tau = {tau[frames[0]]:.3f}$",
        transform=axL.transAxes, va="top", ha="left",
        fontsize=11, color="0.35"
    )

    # right axis of left panel (potential)
    axLV   = None
    V_line = None

    if potential is not None:
        axLV = axL.twinx()
        axLV.set_ylabel(
            r"$V(y,\tau)$" if _V_callable is not None else r"$V(y)$",
            fontsize=13, color=potential_color
        )
        axLV.tick_params(axis="y", colors=potential_color, labelsize=10)
        axLV.yaxis.label.set_color(potential_color)

        if Vlim is not None:
            axLV.set_ylim(*Vlim)
        else:
            if _V_callable is not None:
                Vsample = np.array([_get_V(float(tau[i])) for i in frames])
            else:
                Vsample = _V_arr[np.newaxis, :]
            Vmin, Vmax = float(Vsample.min()), float(Vsample.max())
            Vpad = 0.08 * (Vmax - Vmin + 1e-15)
            axLV.set_ylim(Vmin - Vpad, Vmax + Vpad)

        V0 = _get_V(float(tau[frames[0]]))
        (V_line,) = axLV.plot(
            y, V0,
            color=potential_color, lw=potential_lw, ls="--",
            alpha=potential_alpha, zorder=2
        )

    # =========================================================================
    # RIGHT PANEL — <N̂(τ)> growing history + optional Ω(τ)
    # =========================================================================
    axR.set_xlabel(r"$\tau$", fontsize=13)
    axR.set_ylabel(r"$\langle \hat{N} \rangle$", fontsize=13, color=N_color)
    axR.tick_params(axis="y", colors=N_color, labelsize=10)
    axR.tick_params(axis="x", labelsize=10)
    axR.yaxis.label.set_color(N_color)

    axR.set_xlim(float(tau[frames[0]]), float(tau[frames[-1]]))

    if Nlim is not None:
        axR.set_ylim(*Nlim)
    else:
        Nmin = float(N_vals[frames].min())
        Nmax = float(N_vals[frames].max())
        Npad = 0.08 * (Nmax - Nmin + 1e-15)
        axR.set_ylim(Nmin - Npad, Nmax + Npad)

    t0 = float(tau[frames[0]])
    (N_line,) = axR.plot(
        [t0], [N_vals[frames[0]]],
        color=N_color, lw=N_lw, zorder=3
    )

    # optional Omega on secondary right axis of right panel
    axRO       = None
    Omega_line = None

    if Omega_vals is not None:
        axRO = axR.twinx()
        axRO.set_ylabel(r"$\Omega(\tau)$", fontsize=13, color=Omega_color)
        axRO.tick_params(axis="y", colors=Omega_color, labelsize=10)
        axRO.yaxis.label.set_color(Omega_color)

        if OmegaLim is not None:
            axRO.set_ylim(*OmegaLim)
        else:
            Omin = float(Omega_vals[frames].min())
            Omax = float(Omega_vals[frames].max())
            Opad = 0.08 * (Omax - Omin + 1e-15)
            axRO.set_ylim(Omin - Opad, Omax + Opad)

        (Omega_line,) = axRO.plot(
            [t0], [Omega_vals[frames[0]]],
            color=Omega_color, lw=Omega_lw, ls="--", zorder=2
        )

    # =========================================================================
    # Animation update
    # =========================================================================
    def update(fi: int):
        idx      = frames[fi]
        hist_idx = frames[: fi + 1]

        # left panel
        psi_line.set_ydata(extract(psi_t[idx]))
        time_text.set_text(f"$\\tau = {tau[idx]:.3f}$")
        if V_line is not None and _V_callable is not None:
            V_line.set_ydata(_get_V(float(tau[idx])))

        # right panel: extend history
        t_hist = tau[hist_idx]
        N_line.set_data(t_hist, N_vals[hist_idx])
        if Omega_line is not None:
            Omega_line.set_data(t_hist, Omega_vals[hist_idx])

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False
    )

    # H.264 even-pixel fix
    if outfile.lower().endswith(".mp4"):
        _w_px = round(fig.get_figwidth()  * dpi)
        _h_px = round(fig.get_figheight() * dpi)
        if _w_px % 2 != 0:
            fig.set_size_inches((_w_px + 1) / dpi, fig.get_figheight())
        if _h_px % 2 != 0:
            fig.set_size_inches(fig.get_figwidth(), (_h_px + 1) / dpi)

    if outfile.lower().endswith(".gif"):
        anim.save(outfile, writer="pillow", fps=fps, dpi=dpi)
    elif outfile.lower().endswith(".mp4"):
        anim.save(outfile, writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        raise ValueError('outfile must end in ".mp4" or ".gif".')

    plt.close(fig)
    return outfile


############# Movie Making Routines
def save_wavefunction_movie(
    psi_t: np.ndarray,
    y: np.ndarray,
    tau: np.ndarray,
    potential_overlay=None,       # callable: V_of_tau(tau)->array(Ny), or None
    outfile: str = "psi.mp4",
    what: str = "abs2",           # "abs2", "real", "imag", "abs"
    stride: int = 1,              # plot every stride-th frame
    fps: int = 30,
    dpi: int = 150,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    overlay_alpha: float = 0.25,  # faintness of overlay
    overlay_scale: float = 0.9,   # overlay height as fraction of plot y-range
):
    """
    Save an animation of the wavefunction evolution.

    Parameters
    ----------
    psi_t : (Nt, Ny) complex array
    y     : (Ny,) grid
    tau   : (Nt,) times
    potential_overlay : callable or None
        If callable, must accept tau_val and return an array of shape (Ny,)
        representing (dimensionless) potential V(y, tau) on the y grid.
    outfile : "something.mp4" or "something.gif"
    what  : which quantity to plot: abs2, abs, real, imag
    stride: downsample frames to speed up / shrink output
    fps   : frames per second
    dpi   : output resolution
    ylim  : optional y-limits for the plot
    title : optional title prefix
    overlay_alpha : opacity for the overlay curve
    overlay_scale : scale overlay into plot y-range
    """
    psi_t = np.asarray(psi_t)
    y = np.asarray(y)
    tau = np.asarray(tau)

    if psi_t.ndim != 2:
        raise ValueError("psi_t must have shape (Nt, Ny)")
    if psi_t.shape[1] != y.size:
        raise ValueError("psi_t Ny dimension must match y.size")
    if psi_t.shape[0] != tau.size:
        raise ValueError("psi_t Nt dimension must match tau.size")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    frames = np.arange(0, psi_t.shape[0], stride)

    def extract(arr):
        if what == "abs2":
            return np.abs(arr) ** 2
        if what == "abs":
            return np.abs(arr)
        if what == "real":
            return arr.real
        if what == "imag":
            return arr.imag
        raise ValueError("what must be one of: 'abs2', 'abs', 'real', 'imag'")

    data0 = extract(psi_t[frames[0]])

    fig, ax = plt.subplots()
    (line,) = ax.plot(y, data0)

    # -- set x-axis limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(y[0], y[-1])

    # --- overlay line created only if a callable is provided ---
    overlay_line = None
    if potential_overlay is not None:
        if not callable(potential_overlay):
            raise ValueError("potential_overlay must be a callable V_of_tau(tau)->array(Ny) or None")
        (overlay_line,) = ax.plot(y, np.zeros_like(y), alpha=overlay_alpha)

    ax.set_xlabel("y")
    ax.set_ylabel({
        "abs2": r"$|\psi|^2$",
        "abs": r"$|\psi|$",
        "real": r"Re$\psi$",
        "imag": r"Im$\psi$"
    }[what])

    # --- set y-limits once ---
    if ylim is None:
        sample = extract(psi_t[frames])
        ymin = float(np.min(sample))
        ymax = float(np.max(sample))
        pad = 0.05 * (ymax - ymin + 1e-15)
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        ax.set_ylim(*ylim)

    yl0, yl1 = ax.get_ylim()

    # --- scaling helper for overlay into the plot range ---
    def scaled_overlay(tau_val):
        ov = np.asarray(potential_overlay(y, tau_val), dtype=float)
        if ov.shape != y.shape:
            raise ValueError("potential_overlay(tau) must return an array with the same shape as y")
        return yl0 + ov * (overlay_scale * (yl1 - yl0))

    # Initialize overlay at first frame (optional but avoids a blank line)
    if overlay_line is not None:
        overlay_line.set_ydata(scaled_overlay(tau[frames[0]]))

    # time label
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def update(fi):
        idx = frames[fi]

        line.set_ydata(extract(psi_t[idx]))

        prefix = title + " | " if title else ""
        txt.set_text(f"{prefix}tau = {tau[idx]:.4f}")

        if overlay_line is not None:
            overlay_line.set_ydata(scaled_overlay(tau[idx]))
            return line, overlay_line, txt

        return line, txt

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)

    if outfile.lower().endswith(".gif"):
        anim.save(outfile, writer="pillow", fps=fps, dpi=dpi)
    elif outfile.lower().endswith(".mp4"):
        anim.save(outfile, writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        raise ValueError("outfile must end in .mp4 or .gif")

    plt.close(fig)
    return outfile
#------------------------------------------------------------------------------

def save_two_panel_movie(
    psi_t: np.ndarray,
    y: np.ndarray,
    tau: np.ndarray,
    Omega_vals: np.ndarray,             # precomputed Omega(tau) on tau grid
    N_vals: np.ndarray,                 # precomputed <N(tau)> on tau grid
    potential_overlay=None,             # callable V_of_tau(y,tau)->array(Ny), or None
    outfile: str = "movie.mp4",
    what: str = "abs2",
    stride: int = 1,
    fps: int = 30,
    dpi: int = 150,
    xlim_left: tuple[float, float] | None = None,
    ylim_left: tuple[float, float] | None = None,
    title_left: str | None = None,
    title_right: str | None = None,
    overlay_alpha: float = 0.20,
    overlay_scale: float = 0.90,
    omega_color="forestgreen",                   # e.g. "firebrick" or "tab:blue"
    N_color="firebrick",                       # e.g. "forestgreen" or "tab:orange"
    psi_color="firebrick",
    overlay_color="forestgreen",
):
    """
    Two-panel animation:
      Left: wavefunction quantity + potential overlay (scaled into y-range)
      Right: live plot of Omega(tau) and <N(tau)> vs tau (two y-axes)

    Notes:
      - Omega_vals and N_vals must be on the SAME tau grid as psi_t.
      - potential_overlay should be V_of_tau(tau)->array(Ny) if provided.
      - Uses a single global normalization for the overlay so its amplitude evolves in time.
      - No moving vertical cursor line (removed).
    """
    psi_t = np.asarray(psi_t)
    y = np.asarray(y)
    tau = np.asarray(tau)
    Omega_vals = np.asarray(Omega_vals, dtype=float)
    N_vals = np.asarray(N_vals, dtype=float)

    if psi_t.ndim != 2:
        raise ValueError("psi_t must have shape (Nt, Ny)")
    Nt, Ny = psi_t.shape
    if y.size != Ny:
        raise ValueError("y must have length Ny matching psi_t.shape[1]")
    if tau.size != Nt:
        raise ValueError("tau must have length Nt matching psi_t.shape[0]")
    if Omega_vals.size != Nt:
        raise ValueError("Omega_vals must have length Nt")
    if N_vals.size != Nt:
        raise ValueError("N_vals must have length Nt")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    frames = np.arange(0, Nt, stride)

    def extract(arr):
        if what == "abs2":
            return np.abs(arr) ** 2
        if what == "abs":
            return np.abs(arr)
        if what == "real":
            return arr.real
        if what == "imag":
            return arr.imag
        raise ValueError("what must be one of: 'abs2', 'abs', 'real', 'imag'")

    # ---- figure layout ----
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.2, 1.0]}
    )

    # ---- LEFT PANEL: wavefunction ----
    data0 = extract(psi_t[frames[0]])
    (psi_line,) = axL.plot(y, data0, color=psi_color)

    overlay_line = None
    if potential_overlay is not None:
        if not callable(potential_overlay):
            raise ValueError("potential_overlay must be callable V_of_tau(tau)->array(Ny) or None")
        (overlay_line,) = axL.plot(y, np.zeros_like(y), alpha=overlay_alpha, color=overlay_color)

    axL.set_xlabel("y")
    axL.set_ylabel({
        "abs2": r"$|\psi|^2$",
        "abs": r"$|\psi|$",
        "real": r"Re$\psi$",
        "imag": r"Im$\psi$"
    }[what])

    if xlim_left is not None:
        axL.set_xlim(*xlim_left)
    else:
        axL.set_xlim(y[0], y[-1])

    if ylim_left is None:
        sample = extract(psi_t[frames])
        ymin = float(np.min(sample))
        ymax = float(np.max(sample))
        pad = 0.05 * (ymax - ymin + 1e-15)
        axL.set_ylim(ymin - pad, ymax + pad)
    else:
        axL.set_ylim(*ylim_left)

    if title_left:
        axL.set_title(title_left)

    yl0, yl1 = axL.get_ylim()

    # Global scaling for overlay amplitude so it visibly evolves
    Vmax_global = 1.0
    if overlay_line is not None:
        Vmax_global = max(float(np.max(potential_overlay(y,tau[i]))) for i in frames)
        if Vmax_global <= 0:
            Vmax_global = 1.0

        def scaled_overlay(tau_val):
            V = np.asarray(potential_overlay(y,tau_val), dtype=float)
            if V.shape != y.shape:
                raise ValueError("potential_overlay(y,tau) must return array with same shape as y")
            return yl0 + (V) * (overlay_scale * (yl1 - yl0))

        overlay_line.set_ydata(scaled_overlay(tau[frames[0]]))
    else:
        scaled_overlay = None  # not used

    txtL = axL.text(0.02, 0.95, "", transform=axL.transAxes, va="top")

    # ---- RIGHT PANEL: Omega and N vs time ----
    axR.set_xlabel(r"$\tau$")
    axR.set_ylabel(r"$\omega(\tau)$")
    axR2 = axR.twinx()
    axR2.set_ylabel(r"$\langle \hat{N}(\tau)\rangle$")

    # Full time window
    axR.set_xlim(tau[frames[0]], tau[frames[-1]])

    # y-limits with padding
    Om_min = float(np.min(Omega_vals[frames]))
    Om_max = float(np.max(Omega_vals[frames]))
    Om_pad = 0.05 * (Om_max - Om_min + 1e-15)
    axR.set_ylim(Om_min - Om_pad, Om_max + Om_pad)

    N_min = float(np.min(N_vals[frames]))
    N_max = float(np.max(N_vals[frames]))
    N_pad = 0.05 * (N_max - N_min + 1e-15)
    axR2.set_ylim(N_min - N_pad, N_max + N_pad)

    if title_right:
        axR.set_title(title_right)

    # Live lines, with labels attached to the artists (important with twinx)
    t0 = tau[frames[0]]
    (omega_line,) = axR.plot(
        [t0], [Omega_vals[frames[0]]],
        color=omega_color,
        label=r"$\Omega(\tau)$"
    )
    (N_line,) = axR2.plot(
        [t0], [N_vals[frames[0]]],
        color=N_color,
        label=r"$\langle \hat{N}(\tau)\rangle$"
    )

    # Legend built explicitly from handles so colors match
    lines = [omega_line, N_line]
    labels = [ln.get_label() for ln in lines]
    axR.legend(lines, labels, loc="upper left", frameon=False)

    # Optional polish: color y-axis ticks/labels to match lines
    axR.tick_params(axis="y", colors=omega_line.get_color())
    axR2.tick_params(axis="y", colors=N_line.get_color())
    axR.yaxis.label.set_color(omega_line.get_color())
    axR2.yaxis.label.set_color(N_line.get_color())

    # ---- animation update ----
    def update(fi):
        idx = frames[fi]

        # Left update
        psi_line.set_ydata(extract(psi_t[idx]))
        prefix = (title_left + " | ") if title_left else ""
        txtL.set_text(f"{prefix}tau = {tau[idx]:.4f}")

        artists = [psi_line, txtL]

        if overlay_line is not None:
            overlay_line.set_ydata(scaled_overlay(tau[idx]))
            artists.append(overlay_line)

        # Right update (extend history)
        hist = frames[: fi + 1]
        t_hist = tau[hist]
        omega_line.set_data(t_hist, Omega_vals[hist])
        N_line.set_data(t_hist, N_vals[hist])

        artists.extend([omega_line, N_line])

        return tuple(artists)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)

    if outfile.lower().endswith(".gif"):
        anim.save(outfile, writer="pillow", fps=fps, dpi=dpi)
    elif outfile.lower().endswith(".mp4"):
        anim.save(outfile, writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        raise ValueError("outfile must end in .mp4 or .gif")

    plt.close(fig)
    return outfile

def top_overlaps_with_final_state(psi_final, y, omega_final, n_max=30, top_k=4, m=1.0, hbar=1.0):
    """
    Return (n_list, prob_list, phi_list) for the HO eigenstates (omega_final)
    with largest overlaps with psi_final.
    """
    y = np.asarray(y, dtype=float)
    psi_final = np.asarray(psi_final)

    if y.size < 2:
        raise ValueError("y grid too small.")
    dy = y[1] - y[0]

    # normalize psi on this grid (recommended)
    norm = np.sqrt(np.sum(np.abs(psi_final)**2) * dy)
    psiN = psi_final / (norm + 1e-300)

    probs = []
    phis = []
    for n in range(n_max + 1):
        phi = ho_eigenstate(y, n, omega_final)
        c = np.sum(np.conjugate(phi) * psiN) * dy
        probs.append(np.abs(c)**2)
        phis.append(phi)

    probs = np.asarray(probs)
    idx = np.argsort(probs)[::-1][:top_k]
    return idx.tolist(), probs[idx].tolist(), [phis[i] for i in idx]

def build_postroll_schedule(n_overlays, fps, gap_s=1.0, fade_s=0.75):
    """
    Post-roll frames: for each overlay, wait gap_s, then fade-in over fade_s.
    Returns list of (j, alpha) where j is overlay index, or (None, None) during waits.
    """
    gap_frames = int(round(gap_s * fps))
    fade_frames = max(1, int(round(fade_s * fps)))

    sched = []
    for j in range(n_overlays):
        sched += [(None, None)] * gap_frames
        for k in range(fade_frames):
            a = (k + 1) / fade_frames
            sched.append((j, a))
    return sched



def save_wavefunction_overlay_movie(
    psi_t: np.ndarray,
    y: np.ndarray,
    tau: np.ndarray,
    potential_overlay=None,       # callable: V(y,tau)->array(Ny), or None
    outfile: str = "psi.mp4",
    what: str = "abs2",           # "abs2", "real", "imag", "abs"
    stride: int = 1,
    fps: int = 30,
    dpi: int = 150,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    overlay_alpha: float = 0.25,
    overlay_scale: float = 0.9,

    # --- NEW: eigen-overlay post-roll ---
    omega_final: float | None = None,          # final oscillator frequency (at end of evolution)
    eigen_overlay: bool = False,               # turn on/off
    eigen_top_k: int = 4,
    eigen_n_max: int = 30,
    eigen_gap_s: float = 1.0,
    eigen_fade_s: float = 0.75,
    eigen_m: float = 1.0,
    eigen_hbar: float = 1.0,
):
    psi_t = np.asarray(psi_t)
    y = np.asarray(y)
    tau = np.asarray(tau)

    if psi_t.ndim != 2:
        raise ValueError("psi_t must have shape (Nt, Ny)")
    if psi_t.shape[1] != y.size:
        raise ValueError("psi_t Ny dimension must match y.size")
    if psi_t.shape[0] != tau.size:
        raise ValueError("psi_t Nt dimension must match tau.size")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    frame_idx = np.arange(0, psi_t.shape[0], stride)   # indices into psi_t/tau
    n_main = len(frame_idx)

    def extract(arr):
        if what == "abs2":
            return np.abs(arr) ** 2
        if what == "abs":
            return np.abs(arr)
        if what == "real":
            return arr.real
        if what == "imag":
            return arr.imag
        raise ValueError("what must be one of: 'abs2', 'abs', 'real', 'imag'")

    data0 = extract(psi_t[frame_idx[0]])

    fig, ax = plt.subplots()
    (line,) = ax.plot(y, data0)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(y[0], y[-1])

    overlay_line = None
    if potential_overlay is not None:
        if not callable(potential_overlay):
            raise ValueError("potential_overlay must be callable V(y,tau)->array(Ny) or None")
        (overlay_line,) = ax.plot(y, np.zeros_like(y), alpha=overlay_alpha)

    ax.set_xlabel("y")
    ax.set_ylabel({
        "abs2": r"$|\psi|^2$",
        "abs": r"$|\psi|$",
        "real": r"Re$\psi$",
        "imag": r"Im$\psi$"
    }[what])

    # y-limits
    if ylim is None:
        sample = extract(psi_t[frame_idx])
        ymin = float(np.min(sample))
        ymax = float(np.max(sample))
        pad = 0.05 * (ymax - ymin + 1e-15)
        ax.set_ylim(ymin - pad, ymax + pad)
    else:
        ax.set_ylim(*ylim)

    yl0, yl1 = ax.get_ylim()

    def scaled_overlay(tau_val):
        ov = np.asarray(potential_overlay(y, tau_val), dtype=float)
        if ov.shape != y.shape:
            raise ValueError("potential_overlay(y,tau) must return array with shape y")
        return yl0 + ov * (overlay_scale * (yl1 - yl0))

    if overlay_line is not None:
        overlay_line.set_ydata(scaled_overlay(tau[frame_idx[0]]))

    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    # --- eigen overlays prepared here ---
    overlay_lines = []
    overlay_sched = []

    if eigen_overlay:
        if omega_final is None:
            raise ValueError("eigen_overlay=True requires omega_final (final oscillator frequency).")

        psi_final = psi_t[frame_idx[-1]]
        n_list, p_list, phi_list = top_overlaps_with_final_state(
            psi_final=psi_final,
            y=y,
            omega_final=float(omega_final),
            n_max=int(eigen_n_max),
            top_k=int(eigen_top_k),
            m=float(eigen_m),
            hbar=float(eigen_hbar),
        )

        # Physics scaling: |c_n|^2 * |phi_n|^2
        for n, p, phi in zip(n_list, p_list, phi_list):
            phi2 = np.abs(phi)**2
            overlay_y = p * phi2

            (ln,) = ax.plot(
                y,
                overlay_y,
                lw=2,
                alpha=0.0,
                label=rf"$n={n}$  ($|c_n|^2={p:.2e}$)"
            )
            overlay_lines.append(ln)

        # Legend text colors match line colors
        if overlay_lines:
            leg = ax.legend(loc="upper right", frameon=False)
            for text_obj, line_obj in zip(leg.get_texts(), overlay_lines):
                text_obj.set_color(line_obj.get_color())

        overlay_sched = build_postroll_schedule(
            n_overlays=len(overlay_lines),
            fps=fps,
            gap_s=float(eigen_gap_s),
            fade_s=float(eigen_fade_s),
        )

    n_total = n_main + len(overlay_sched)

    def update(fi):
        if fi < n_main:
            idx = frame_idx[fi]
            line.set_ydata(extract(psi_t[idx]))

            prefix = title + " | " if title else ""
            txt.set_text(f"{prefix}tau = {tau[idx]:.4f}")

            if overlay_line is not None:
                overlay_line.set_ydata(scaled_overlay(tau[idx]))

            # hide eigen overlays during evolution
            for ln in overlay_lines:
                ln.set_alpha(0.0)

        else:
            # post-roll: freeze on final state
            idx = frame_idx[-1]
            line.set_ydata(extract(psi_t[idx]))

            prefix = title + " | " if title else ""
            txt.set_text(f"{prefix}tau = {tau[idx]:.4f}  (final)")

            if overlay_line is not None:
                overlay_line.set_ydata(scaled_overlay(tau[idx]))

            j, a = overlay_sched[fi - n_main]
            if j is not None:
                for jj, ln in enumerate(overlay_lines):
                    if jj < j:
                        ln.set_alpha(1.0)
                    elif jj == j:
                        ln.set_alpha(a)
                    else:
                        ln.set_alpha(0.0)

        # With blit=False, we don't need to meticulously return artists,
        # but returning them is still fine.
        artists = [line, txt]
        if overlay_line is not None:
            artists.append(overlay_line)
        artists.extend(overlay_lines)
        return tuple(artists)

    # IMPORTANT FIX: use a concrete iterator for frames, and blit=False for robustness
    anim = FuncAnimation(
        fig,
        update,
        frames=range(n_total),
        interval=1000 / fps,
        blit=False
    )

    if outfile.lower().endswith(".gif"):
        anim.save(outfile, writer="pillow", fps=fps, dpi=dpi)
    elif outfile.lower().endswith(".mp4"):
        anim.save(outfile, writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        raise ValueError("outfile must end in .mp4 or .gif")

    plt.close(fig)
    return outfile