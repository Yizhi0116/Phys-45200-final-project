"""
Page 2 — Dipolar XY Hubbard Spin Squeezing Simulator

Full effective Hamiltonian (hard-core fermions on a 2-D square lattice):

    H = -t  sum_{<i,j>,sigma} ( c†_{i,sigma} c_{j,sigma} + h.c. )
        + J_perp  sum_{i≠j}  (a/r_ij)^3  S+_i S-_j

At t = 0 the model reduces to a pure dipolar XY model.
Simulated with DTWA + RK4.
"""

import time
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import streamlit as st


# ============================================================
# Protocol diagram  (State prep → Evolution → Readout)
# ============================================================

def draw_protocol(J_perp, t_hop):
    """
    Three Bloch-sphere cartoons showing the three stages of the
    spin-squeezing protocol, with a pulse-sequence bar below.
    """
    fig = plt.figure(figsize=(11, 4.2))
    fig.patch.set_facecolor("#f8f9fa")

    # ── helper: draw a minimal Bloch sphere in an axis ──────
    def bloch_axes(ax):
        ax.set_aspect("equal"); ax.axis("off")
        phi = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(phi), np.sin(phi), color="#bbb", lw=1.0, zorder=0)
        ax.plot(np.cos(phi), 0.3*np.sin(phi), color="#bbb", lw=0.7, ls="--", zorder=0)
        # axis stubs
        for xy, lab in [((0,1.15),"z"), ((1.18,0),"y"), ((-0.7,-0.55),"x")]:
            ax.annotate("", xy=xy, xytext=(0,0),
                        arrowprops=dict(arrowstyle="-|>", color="#888",
                                        lw=1.0, mutation_scale=8))
            ax.text(xy[0]*1.08, xy[1]*1.08, f"${lab}$", fontsize=8,
                    ha="center", va="center", color="#555")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.4, 1.4)

    # ── Stage 1: State preparation — CSS disk ───────────────
    ax1 = fig.add_axes([0.03, 0.22, 0.26, 0.70])
    bloch_axes(ax1)
    phi = np.linspace(0, 2*np.pi, 200)
    r = 0.18
    ax1.fill(r*np.cos(phi), r*np.sin(phi), color="#4fa3e0", alpha=0.55, zorder=2)
    ax1.plot(r*np.cos(phi), r*np.sin(phi), color="#1a6aa8", lw=1.2, zorder=3)
    # mean spin along +x (projected into yz plane appears as a dot; show arrow tip)
    ax1.annotate("", xy=(0.78, 0), xytext=(0.02, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#1a4a7a",
                                 lw=2.2, mutation_scale=12), zorder=4)
    ax1.text(0, -1.25, "State preparation\n(CSS along $+x$)",
             ha="center", va="top", fontsize=8.5, color="#222")
    ax1.set_title("", fontsize=9)

    # ── Stage 2: Evolution — sheared ellipse ─────────────────
    ax2 = fig.add_axes([0.37, 0.22, 0.26, 0.70])
    bloch_axes(ax2)
    # show a ring of arrows around equator to indicate precession
    for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
        ax2.annotate("",
                     xy =(0.92*np.cos(ang+0.35), 0.28*np.sin(ang+0.35)),
                     xytext=(0.92*np.cos(ang),   0.28*np.sin(ang)),
                     arrowprops=dict(arrowstyle="-|>", color="#e07b39",
                                     lw=1.0, mutation_scale=7), zorder=3)
    # sheared ellipse
    angle_sq = np.radians(35)
    a_sq, b_sq = 0.07, 0.26
    ex = a_sq*np.cos(phi)*np.cos(angle_sq) - b_sq*np.sin(phi)*np.sin(angle_sq)
    ey = a_sq*np.cos(phi)*np.sin(angle_sq) + b_sq*np.sin(phi)*np.cos(angle_sq)
    ax2.fill(ex, ey, color="#f4a261", alpha=0.55, zorder=2)
    ax2.plot(ex, ey, color="#c0622b", lw=1.5, zorder=3)
    ax2.annotate("", xy=(0.78, 0), xytext=(0.02, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#1a4a7a",
                                 lw=2.2, mutation_scale=12), zorder=4)
    coup = f"$J_\\perp$={J_perp:.2g} Hz" + (f", $t$={t_hop:.2g} Hz" if t_hop else "")
    ax2.text(0, -1.25, f"Evolution under $H$\n({coup})",
             ha="center", va="top", fontsize=8.5, color="#222")

    # ── Stage 3: Readout — rotated axis ──────────────────────
    ax3 = fig.add_axes([0.71, 0.22, 0.26, 0.70])
    bloch_axes(ax3)
    # squeezed ellipse same as stage 2
    ax3.fill(ex, ey, color="#f4a261", alpha=0.55, zorder=2)
    ax3.plot(ex, ey, color="#c0622b", lw=1.5, zorder=3)
    ax3.annotate("", xy=(0.78, 0), xytext=(0.02, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#1a4a7a",
                                 lw=2.2, mutation_scale=12), zorder=4)
    # readout angle arrow
    theta_ro = np.radians(40)
    ax3.annotate("", xy=(1.1*np.cos(theta_ro), 1.1*np.sin(theta_ro)),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#c0392b",
                                 lw=1.8, mutation_scale=11), zorder=5)
    ax3.annotate("", xy=(-1.1*np.cos(theta_ro), -1.1*np.sin(theta_ro)),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="#c0392b",
                                 lw=1.8, mutation_scale=11), zorder=5)
    # arc showing angle theta
    arc_r = 0.45
    arc_phi = np.linspace(0, theta_ro, 40)
    ax3.plot(arc_r*np.cos(arc_phi), arc_r*np.sin(arc_phi), color="#c0392b", lw=1.2)
    ax3.text(arc_r*np.cos(theta_ro/2)*1.35, arc_r*np.sin(theta_ro/2)*1.35,
             r"$\theta$", fontsize=10, color="#c0392b")
    ax3.text(0, -1.25, "Readout at angle $\\theta$\n(minimise $\\xi^2$)",
             ha="center", va="top", fontsize=8.5, color="#222")

    # ── Arrows between stages ────────────────────────────────
    for x in [0.305, 0.645]:
        fig.text(x, 0.57, "→", fontsize=22, color="#555",
                 ha="center", va="center", fontweight="bold")

    # ── Pulse sequence bar ───────────────────────────────────
    ax_seq = fig.add_axes([0.03, 0.04, 0.94, 0.13])
    ax_seq.set_xlim(0, 10); ax_seq.set_ylim(0, 1); ax_seq.axis("off")

    def pulse_box(ax, x0, w, h, y0, color, label, fontsize=8):
        rect = mpatches.FancyBboxPatch((x0, y0), w, h,
                                       boxstyle="round,pad=0.05",
                                       fc=color, ec="#555", lw=1.0)
        ax.add_patch(rect)
        ax.text(x0 + w/2, y0 + h/2, label, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold")

    # Timeline line
    ax_seq.plot([0.1, 9.9], [0.35, 0.35], color="#888", lw=1.5)
    # Pulses
    pulse_box(ax_seq, 0.2,  1.1, 0.55, 0.075, "#4fa3e0", r"$\frac{\pi}{2}\,y$",  9)
    pulse_box(ax_seq, 2.0,  2.5, 0.55, 0.075, "#e07b39", r"$\pi_y$  (evolve $\tau$)", 8)
    pulse_box(ax_seq, 5.0,  2.5, 0.55, 0.075, "#4fa3e0", r"$\pi_{-y}$  (spin echo)", 8)
    pulse_box(ax_seq, 8.1,  1.6, 0.55, 0.075, "#c0392b", r"$\theta_x$  readout",   8)
    ax_seq.text(5.0, 0.92, "Pulse sequence", ha="center", va="center",
                fontsize=8.5, color="#333")

    return fig


# ============================================================
# Lattice schematic
# ============================================================

def draw_lattice_schematic(t_hop, J_perp):
    fig, (ax_lat, ax_bloch) = plt.subplots(
        1, 2, figsize=(11, 4.2),
        gridspec_kw={"width_ratios": [1.1, 1]},
    )
    fig.patch.set_facecolor("#f8f9fa")

    ax_lat.set_facecolor("#f0f4f8")
    n_show = 5
    for i in range(n_show):
        ax_lat.axhline(i, color="#c8d0da", lw=0.8, zorder=0)
        ax_lat.axvline(i, color="#c8d0da", lw=0.8, zorder=0)

    if t_hop > 0:
        for x0, y0, x1, y1 in [(1,2,2,2),(2,2,3,2),(2,1,2,2),(2,2,2,3)]:
            ax_lat.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#e07b39", lw=2,
                                mutation_scale=14, connectionstyle="arc3,rad=0.18"),
                zorder=4,
            )

    cx, cy = 2, 2
    dipole_color = "#3a7abf"
    for ix in range(n_show):
        for iy in range(n_show):
            if ix == cx and iy == cy:
                continue
            r = np.hypot(ix-cx, iy-cy)
            ax_lat.plot([cx,ix],[cy,iy], color=dipole_color,
                        lw=max(0.4,min(2.2,2.2/r**1.5)),
                        alpha=max(0.08,min(0.7,0.7/r**2)), zorder=1)

    for ix in range(n_show):
        for iy in range(n_show):
            if ix == cx and iy == cy:
                continue
            ax_lat.plot(ix, iy, "o", ms=14, color="#8ab4d4",
                        mec="#3a7abf", mew=1.2, zorder=3)
            ax_lat.annotate("", xy=(ix,iy+0.30), xytext=(ix,iy-0.28),
                            arrowprops=dict(arrowstyle="-|>", color="#1a4a7a",
                                            lw=1.2, mutation_scale=8), zorder=5)

    ax_lat.plot(cx, cy, "o", ms=18, color="#f9c74f", mec="#e07b39", mew=2, zorder=6)
    ax_lat.annotate("", xy=(cx,cy+0.38), xytext=(cx,cy-0.36),
                    arrowprops=dict(arrowstyle="-|>", color="#7a3a00",
                                    lw=1.8, mutation_scale=11), zorder=7)
    ax_lat.text(cx, cy-0.62, "ref site", ha="center", va="top",
                fontsize=8, color="#7a3a00", fontweight="bold")

    handles = [
        mpatches.Patch(color="#8ab4d4", label=r"spin-$\frac{1}{2}$ site"),
        mpatches.Patch(color="#f9c74f", label="reference site"),
        plt.Line2D([0],[0], color=dipole_color, lw=2, label=r"$J_\perp/r^3$ coupling"),
    ]
    if t_hop > 0:
        handles.append(plt.Line2D([0],[0], color="#e07b39", lw=2,
                                   marker=">", markersize=7, label=r"hopping $t$"))
    ax_lat.legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.85)
    ax_lat.set_xlim(-0.6, n_show-0.4); ax_lat.set_ylim(-0.6, n_show-0.4)
    ax_lat.set_xticks(range(n_show)); ax_lat.set_yticks(range(n_show))
    ax_lat.set_xticklabels([f"{i}a" for i in range(n_show)], fontsize=8)
    ax_lat.set_yticklabels([f"{i}a" for i in range(n_show)], fontsize=8)
    ax_lat.set_title("2-D dipolar lattice", fontsize=11, fontweight="bold")
    hop_str = f"$t = {t_hop:.2g}$ Hz" if t_hop > 0 else "$t = 0$ (Mott insulator)"
    ax_lat.text(0.02, 0.02, hop_str + f"\n$J_\\perp = {J_perp:.3g}$ Hz",
                transform=ax_lat.transAxes, fontsize=8.5, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Bloch sphere
    ax_bloch.set_facecolor("#f0f4f8")
    ax_bloch.set_aspect("equal"); ax_bloch.axis("off")
    phi = np.linspace(0, 2*np.pi, 300)
    ax_bloch.plot(np.cos(phi), np.sin(phi), color="#aaa", lw=1.2, zorder=0)
    ax_bloch.plot(np.cos(phi), 0.28*np.sin(phi), color="#aaa", lw=0.8, ls="--", zorder=0)
    ax_bloch.annotate("", xy=(0,1.15), xytext=(0,-1.15),
                      arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2, mutation_scale=10))
    ax_bloch.annotate("", xy=(1.2,0), xytext=(-1.2,0),
                      arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2, mutation_scale=10))
    ax_bloch.text(0.02, 1.18, r"$S_z$", fontsize=10, color="#333")
    ax_bloch.text(1.22, -0.08, r"$S_y$", fontsize=10, color="#333")
    r_css = 0.18
    ax_bloch.fill(r_css*np.cos(phi), r_css*np.sin(phi), color="#aad4f5", alpha=0.5, zorder=2)
    ax_bloch.plot(r_css*np.cos(phi), r_css*np.sin(phi), color="#3a7abf", lw=1.2, zorder=3)
    angle_sq = np.radians(30)
    a_sq, b_sq = 0.08, 0.28
    ell_x = a_sq*np.cos(phi)*np.cos(angle_sq) - b_sq*np.sin(phi)*np.sin(angle_sq)
    ell_y = a_sq*np.cos(phi)*np.sin(angle_sq) + b_sq*np.sin(phi)*np.cos(angle_sq)
    ax_bloch.fill(ell_x, ell_y, color="#f4a261", alpha=0.55, zorder=2)
    ax_bloch.plot(ell_x, ell_y, color="#e07b39", lw=1.5, zorder=3)
    ax_bloch.annotate("", xy=(0.82,0.0), xytext=(0,0),
                      arrowprops=dict(arrowstyle="-|>", color="#1a4a7a",
                                      lw=2.5, mutation_scale=14), zorder=4)
    ax_bloch.text(0.85, 0.07, r"$\langle\vec{J}\rangle$", fontsize=10, color="#1a4a7a")
    ax_bloch.annotate("", xy=(0.2,0.45), xytext=(-0.2,-0.15),
                      arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.8,
                                      mutation_scale=12, connectionstyle="arc3,rad=-0.4"),
                      zorder=5)
    ax_bloch.text(-0.55,-0.3, "shearing\n(squeezing)", fontsize=8, color="#c0392b", ha="center")
    ax_bloch.text(-r_css-0.06, 0.01, "CSS", fontsize=8, color="#3a7abf", ha="right")
    ax_bloch.text(0.32, 0.3, "squeezed", fontsize=8, color="#e07b39")
    ax_bloch.set_xlim(-1.45,1.45); ax_bloch.set_ylim(-1.35,1.35)
    ax_bloch.set_title("Bloch-sphere picture of spin squeezing", fontsize=11, fontweight="bold")

    fig.tight_layout(pad=1.5)
    return fig


# ============================================================
# Lattice correlation heatmaps — one independent colorbar per panel
# ============================================================

def draw_lattice_correlations(pos, corr_sq, corr_asq, n_grid,
                               ref_site_idx, sq_deg, asq_deg, tau_ms, opt_tau_ms):
    xi = pos[:, 0].astype(int)
    yi = pos[:, 1].astype(int)

    def make_grid(corr):
        g = np.full((n_grid, n_grid), np.nan)
        for j in range(len(pos)):
            if j == ref_site_idx:          # exclude self-correlation
                continue
            if 0 <= xi[j] < n_grid and 0 <= yi[j] < n_grid:
                g[xi[j], yi[j]] = corr[j]
        return g

    grid_sq  = make_grid(corr_sq)
    grid_asq = make_grid(corr_asq)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.patch.set_facecolor("#f8f9fa")

    for ax, grid, label, deg in [
        (axes[0], grid_sq,  "Squeezed",      sq_deg),
        (axes[1], grid_asq, "Anti-squeezed", asq_deg),
    ]:
        ax.set_facecolor("#dde3ea")

        finite = grid[np.isfinite(grid)]
        if len(finite) == 0:
            vmin, vmax = -1e-9, 1e-9
        else:
            vabs = max(float(np.max(np.abs(finite))), 1e-9)
            vmin, vmax = -vabs, vabs

        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color="#dde3ea")
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

        im = ax.imshow(
            grid.T, origin="lower", cmap=cmap, norm=norm,
            extent=[-0.5, n_grid-0.5, -0.5, n_grid-0.5],
            interpolation="nearest", zorder=1,
        )

        ax.plot(xi[ref_site_idx], yi[ref_site_idx], "*",
                ms=16, color="#f9c74f", mec="#7a3a00", mew=1.5,
                zorder=4, label="ref site (★)")

        cb = fig.colorbar(im, ax=ax, shrink=0.82)
        cb.set_label(r"$g^{(2)}_{\mathrm{ref},j}$", fontsize=9)
        ax.set_title(f"{label}   θ = {deg:.0f}°", fontsize=10, fontweight="bold")
        ax.set_xlabel("x / a", fontsize=9); ax.set_ylabel("y / a", fontsize=9)
        ax.set_xticks(range(0, n_grid, max(1, n_grid//6)))
        ax.set_yticks(range(0, n_grid, max(1, n_grid//6)))
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    tau_label = (f"τ = {tau_ms:.0f} ms  (optimal)"
                 if abs(tau_ms - opt_tau_ms) < 1 else f"τ = {tau_ms:.0f} ms")
    fig.suptitle(
        r"Site-resolved correlations  "
        r"$g^{(2)}_{\mathrm{ref},j} = \langle s_{\mathrm{ref}}\,s_j\rangle"
        r"- \langle s_{\mathrm{ref}}\rangle\langle s_j\rangle$"
        f"    [{tau_label}]",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(pad=1.5)
    return fig


# ============================================================
# Physics helpers
# ============================================================

def build_lattice(n, filling, seed):
    rng = np.random.default_rng(seed)
    all_pos = np.array(list(product(range(n), range(n))), dtype=float)
    N_spins = max(1, int(round(filling * n * n)))
    idx = rng.choice(len(all_pos), size=N_spins, replace=False)
    return all_pos[idx]


def build_dipolar_J(pos, J_perp):
    N = len(pos)
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            J[i, j] = J_perp / np.linalg.norm(pos[i] - pos[j])**3
    return J


def build_hopping_J(pos, t_hop):
    N = len(pos)
    T = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if np.isclose(np.linalg.norm(pos[i] - pos[j]), 1.0):
                T[i, j] = t_hop
    return T


def sample_css_x(num_samples, N, rng):
    spins = np.zeros((num_samples, N, 3))
    spins[:, :, 0] = 0.5
    spins[:, :, 1] = rng.choice([-0.5, 0.5], size=(num_samples, N))
    spins[:, :, 2] = rng.choice([-0.5, 0.5], size=(num_samples, N))
    return spins


def eom_full(spins, K):
    Sx, Sy, Sz = spins[:,:,0], spins[:,:,1], spins[:,:,2]
    Bx = Sx @ K.T;  By = Sy @ K.T
    dS = np.empty_like(spins)
    dS[:,:,0] =  By * Sz
    dS[:,:,1] = -Bx * Sz
    dS[:,:,2] =  Bx * Sy - By * Sx
    return dS


def rk4_step(spins, dt, K):
    k1 = eom_full(spins,             K)
    k2 = eom_full(spins+0.5*dt*k1,  K)
    k3 = eom_full(spins+0.5*dt*k2,  K)
    k4 = eom_full(spins+   dt*k3,   K)
    return spins + (dt/6.0)*(k1+2*k2+2*k3+k4)


def project_theta(spins, theta):
    return np.sum(spins[:,:,2]*np.cos(theta) + spins[:,:,1]*np.sin(theta), axis=1)


def ramsey_contrast(spins, N):
    return np.linalg.norm(np.mean(np.sum(spins, axis=1), axis=0)) / (N/2.0)


def noise_squeezing_theta(spins, N, theta):
    return 4.0 * np.var(project_theta(spins, theta), ddof=1) / N


def site_correlations_at_theta(spins, pos, theta):
    s = spins[:,:,2]*np.cos(theta) + spins[:,:,1]*np.sin(theta)
    mean_s = np.mean(s, axis=0)
    C2 = np.mean(s[:,:,None]*s[:,None,:], axis=0) - np.outer(mean_s, mean_s)
    ref_idx = int(np.argmin(np.linalg.norm(pos - np.mean(pos, axis=0), axis=1)))
    return C2[ref_idx], ref_idx


# ============================================================
# Cached simulation
# ============================================================

@st.cache_data(show_spinner=False)
def run_dipolar_sim(n, filling, J_perp, t_hop, t_max, num_steps,
                    num_samples, seed, n_theta):
    rng   = np.random.default_rng(seed)
    pos   = build_lattice(n, filling, seed)
    N     = len(pos)
    K     = build_dipolar_J(pos, J_perp) + build_hopping_J(pos, t_hop)

    times  = np.linspace(0.0, t_max, num_steps+1)
    dt     = times[1] - times[0]
    thetas = np.linspace(0, np.pi, n_theta, endpoint=False)

    xi2_full    = np.full((num_steps+1, n_theta), np.nan)
    contrast_ts = np.zeros(num_steps+1)

    spins = sample_css_x(num_samples, N, rng)
    for step in range(num_steps+1):
        contrast_ts[step] = ramsey_contrast(spins, N)
        for m, th in enumerate(thetas):
            xi2_full[step, m] = noise_squeezing_theta(spins, N, th)
        if step < num_steps:
            spins = rk4_step(spins, dt, K)

    # Second pass to record spin snapshots for correlation plots
    rng2   = np.random.default_rng(seed)
    spins2 = sample_css_x(num_samples, N, rng2)
    spins_history = np.empty((num_steps+1, num_samples, N, 3), dtype=np.float32)
    spins_history[0] = spins2
    for step in range(num_steps):
        spins2 = rk4_step(spins2, dt, K)
        spins_history[step+1] = spins2

    return dict(
        times=times, contrast=contrast_ts,
        thetas=thetas, xi2_full=xi2_full,
        spins_history=spins_history,
        N=N, n=n, pos=pos,
    )


def xi2_at_step(xi2_full, step):
    """xi^2 vs theta at a given time step (in dB)."""
    row = xi2_full[step].copy()
    return 10.0 * np.log10(np.where(row > 0, row, np.nan))


def xi2_at_theta_idx(xi2_full, th_idx):
    """xi^2 vs time at a given theta index (in dB)."""
    col = xi2_full[:, th_idx].copy()
    return 10.0 * np.log10(np.where(col > 0, col, np.nan))


def safe_argmin(arr):
    v = np.isfinite(arr)
    return int(np.nanargmin(arr)) if v.any() else 0


def safe_argmax(arr):
    v = np.isfinite(arr)
    return int(np.nanargmax(arr)) if v.any() else 0


# ============================================================
# Page layout
# ============================================================

st.title("Dipolar XY Spin Squeezing — DTWA Simulator")

st.markdown("### Model Hamiltonian")
st.latex(
    r"""
    H = -t \sum_{\langle i,j\rangle,\sigma}
        \!\!\left(\hat{c}^\dagger_{i,\sigma}\hat{c}_{j,\sigma} + \mathrm{h.c.}\right)
    \;+\;
    J_\perp \sum_{i \neq j} \frac{a^3}{r_{ij}^3}
        \hat{S}^+_i \hat{S}^-_j
    """
)
st.markdown(
    r"""
**First term — hopping:** Hard-core fermions with spin $\sigma\in\{\uparrow,\downarrow\}$
tunnel between nearest-neighbour sites with amplitude $t$. In the Mott-insulating limit
(one particle per site) hopping is suppressed; in the DTWA treatment it enters as a
nearest-neighbour XY coupling of strength $t$.

**Second term — dipolar exchange:** The operators
$\hat{S}^\pm_i = \hat{c}^\dagger_{i,\uparrow}\hat{c}_{i,\downarrow}$ flip spin pairs.
Using $\hat{S}^+_i\hat{S}^-_j = \hat{S}^x_i\hat{S}^x_j+\hat{S}^y_i\hat{S}^y_j$
the exchange is a pure long-range XY interaction decaying as $r^{-3}$.

**How spin squeezing arises and is applied:** Starting from a coherent spin state (CSS)
polarised along $+x$, the XY interactions entangle the spins and shear the uncertainty
disk on the Bloch sphere into a tilted ellipse. One transverse direction has reduced
variance (squeezed), the orthogonal one is enlarged. The noise squeezing parameter
quantifies the noise reduction; values below 1 (negative dB) are below the
standard quantum limit (SQL). In an atomic interferometer the squeezed state replaces
the CSS: reduced quantum noise allows phase estimation beyond the shot-noise limit,
improving sensing of magnetic fields, gravitational waves, or time standards.
The metrologically relevant figure of merit is the Wineland parameter, which
accounts for contrast loss $C < 1$ caused by the same interactions.
Only states with $\xi_R^2 < 1$ provide genuine phase-sensitivity improvement.

**Simulation method:** DTWA samples stochastic spin trajectories and averages
observables — $O(N^2)$ cost per step instead of $O(2^N)$.
"""
)
st.latex(
    r"\xi^2 = \frac{4\,\min_\theta\,\mathrm{Var}[\hat{S}_\theta]}{N}, \qquad"
    r"\hat{S}_\theta = \cos\theta\,\hat{S}_z + \sin\theta\,\hat{S}_y"
)
st.latex(
    r"\xi_R^2 = \frac{\xi^2}{C^2}, \qquad C = \frac{|\langle\vec{J}\rangle|}{N/2}"
)

# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.header("Dipolar Simulation Inputs")

n = st.sidebar.slider("Lattice size n", min_value=4, max_value=24, value=12, step=1)
filling = st.sidebar.slider(
    "Filling fraction", min_value=0.1, max_value=1.0, value=0.80, step=0.05,
)
N_display = max(1, int(round(filling * n * n)))
st.sidebar.write(f"Spins: **N ≈ {N_display}**  ({n}×{n} lattice)")

st.sidebar.markdown("**Interaction parameters**")
J_perp = st.sidebar.number_input(
    "J⊥ (Hz) — dipolar exchange", min_value=0.01, max_value=500.0, value=1.00, step=0.01,
)
t_hop = st.sidebar.number_input(
    "t (Hz) — hopping amplitude", min_value=0.0, max_value=500.0, value=0.0, step=0.5,
    help="t = 0: Mott-insulating regime. Increase to add NN hopping.",
)
if t_hop > 0:
    ratio = t_hop / J_perp if J_perp > 0 else float("inf")
    st.sidebar.info(f"t / J⊥ = {ratio:.2g}  →  "
                    + ("hopping-dominated" if ratio>4 else
                       "comparable" if ratio>0.5 else "exchange-dominated"))

st.sidebar.markdown("**Time parameters**")
t_max = st.sidebar.slider(
    "Max evolution time (s)", min_value=0.05, max_value=3.0, value=0.5, step=0.033,
)

st.sidebar.markdown("**Numerical parameters**")
num_steps = st.sidebar.slider(
    "Time steps", min_value=50, max_value=600, value=200, step=50,
)
num_samples = st.sidebar.slider(
    "DTWA trajectories", min_value=200, max_value=8000, value=2000, step=200,
)
n_theta = st.sidebar.slider(
    "Readout angle resolution", min_value=18, max_value=180, value=60, step=6,
    help="Number of θ values swept 0–180°.",
)
seed = st.sidebar.number_input(
    "Random seed", min_value=0, max_value=999999, value=42, step=1,
)
run_button = st.sidebar.button("Run simulation", type="primary")

cost = num_samples * N_display**2 * num_steps
if cost > 2e9:
    st.warning(f"Estimated computation: {cost:.1e} ops — may be slow. "
               "Reduce trajectories, lattice size, or steps.")

# ── Protocol diagram (always visible) ────────────────────────
st.markdown("### Measurement protocol")
fig_proto = draw_protocol(J_perp, t_hop)
st.pyplot(fig_proto); plt.close(fig_proto)

# ── Lattice schematic (always visible) ───────────────────────
st.markdown("### System schematic")
fig_lat = draw_lattice_schematic(t_hop, J_perp)
st.pyplot(fig_lat); plt.close(fig_lat)

# ── Session-state storage so results survive widget interactions ──
if run_button:
    t0 = time.time()
    with st.spinner("Running DTWA simulation…"):
        res = run_dipolar_sim(
            n=n, filling=filling, J_perp=J_perp, t_hop=t_hop,
            t_max=t_max, num_steps=num_steps,
            num_samples=num_samples, seed=int(seed),
            n_theta=n_theta,
        )
    elapsed = time.time() - t0

    # Compute global optimum immediately so widget defaults can be set fresh
    _xi2_db = 10.0 * np.log10(np.where(res["xi2_full"] > 0, res["xi2_full"], np.nan))
    _valid  = np.isfinite(_xi2_db)
    _flat   = safe_argmin(_xi2_db) if _valid.any() else 0
    _os, _ot = np.unravel_index(_flat, _xi2_db.shape)
    _times_ms = res["times"] * 1000.0
    _theta_deg = np.degrees(res["thetas"])
    _dt_ms_new = float(_times_ms[1] - _times_ms[0]) if len(_times_ms) > 1 else 1.0
    _opt_tau   = float(np.clip(_times_ms[_os], 0.0, t_max*1000 - _dt_ms_new*0.5))
    _opt_theta = float(np.clip(_theta_deg[_ot], 0.0, 179.9))

    # Clear stale widget values so number_inputs re-initialize to new optimum
    for key in ("tau_c", "theta_d", "tau_e"):
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["sim_res"]       = res
    st.session_state["sim_elapsed"]   = elapsed
    st.session_state["sim_opt_tau"]   = _opt_tau
    st.session_state["sim_opt_theta"] = _opt_theta
    st.session_state["sim_params"]    = dict(
        n=n, t_max=t_max, n_theta=n_theta, J_perp=J_perp, t_hop=t_hop,
    )

# Render results if we have them (persists across widget interactions)
if "sim_res" in st.session_state:
    res     = st.session_state["sim_res"]
    elapsed = st.session_state["sim_elapsed"]
    params  = st.session_state["sim_params"]

    # Use stored params for labels (not sidebar — sidebar may have changed)
    _t_max   = params["t_max"]
    _n_theta = params["n_theta"]

    ratio_str = (f",  t/J⊥ = {params['t_hop']/params['J_perp']:.2g}"
                 if params["J_perp"] > 0 else "")
    st.success(f"Done in {elapsed:.1f} s  —  N = {res['N']} spins{ratio_str}")

    times         = res["times"]
    contrast      = res["contrast"]
    thetas        = res["thetas"]
    xi2_full      = res["xi2_full"]
    spins_history = res["spins_history"]
    N             = res["N"]
    pos           = res["pos"]
    theta_deg     = np.degrees(thetas)
    times_ms      = times * 1000.0
    n_steps       = len(times) - 1

    # ── Global optimum ──────────────────────────────────────
    xi2_db_full = 10.0 * np.log10(np.where(xi2_full > 0, xi2_full, np.nan))
    valid_all   = np.isfinite(xi2_db_full)
    opt_step    = safe_argmin(xi2_db_full.ravel() if valid_all.any()
                              else np.zeros(xi2_db_full.size))
    opt_step, opt_th_idx = np.unravel_index(
        safe_argmin(xi2_db_full) if valid_all.any() else 0,
        xi2_db_full.shape,
    )
    opt_tau_ms  = float(times_ms[opt_step])
    opt_theta_d = float(theta_deg[opt_th_idx])
    opt_xi2_db  = float(xi2_db_full[opt_step, opt_th_idx]) if valid_all.any() else float("nan")

    _dt_ms   = float(times_ms[1] - times_ms[0]) if len(times_ms) > 1 else 1.0
    _tau_max = float(_t_max * 1000)

    # Widget defaults come from session state (set fresh on each new run)
    _default_tau   = float(st.session_state.get("sim_opt_tau",   opt_tau_ms))
    _default_theta = float(st.session_state.get("sim_opt_theta", opt_theta_d))

    def _clamp_tau(v):
        return float(np.clip(v, 0.0, _tau_max - _dt_ms * 0.5))
    def _clamp_theta(v):
        return float(np.clip(v, 0.0, 179.9))

    # ── (a) Full 2-D noise squeezing map ────────────────────
    st.markdown("---")
    st.subheader(r"(a)  Noise squeezing $\xi^2(\tau,\,\theta)$")

    vmax_a = max(3.0, float(np.nanmax(np.abs(xi2_db_full))))
    fig_a, ax_a = plt.subplots(figsize=(9, 5))
    im_a = ax_a.pcolormesh(
        theta_deg, times_ms, xi2_db_full,
        cmap="RdBu_r", vmin=-vmax_a, vmax=vmax_a, shading="auto",
    )
    plt.colorbar(im_a, ax=ax_a, label=r"$\xi^2$ [dB]")
    if valid_all.any():
        ax_a.plot(opt_theta_d, opt_tau_ms, "k*", ms=14, zorder=5,
                  label=f"optimum: {opt_xi2_db:.1f} dB\n"
                        f"τ = {opt_tau_ms:.0f} ms,  θ = {opt_theta_d:.0f}°")
        ax_a.axvline(opt_theta_d, color="k", lw=0.8, ls="--", alpha=0.5)
        ax_a.axhline(opt_tau_ms,  color="k", lw=0.8, ls="--", alpha=0.5)
        ax_a.legend(fontsize=9, loc="upper right")
    ax_a.set_xlabel(r"Readout angle $\theta$ (deg)", fontsize=11)
    ax_a.set_ylabel(r"Evolution time $\tau$ (ms)", fontsize=11)
    ax_a.set_title(r"Noise squeezing $\xi^2(\tau,\theta)$  — full DTWA map", fontsize=12)
    st.pyplot(fig_a); plt.close(fig_a)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Best noise squeezing",  f"{opt_xi2_db:.2f} dB")
    with col2: st.metric("Optimal evolution time", f"{opt_tau_ms:.0f} ms")
    with col3: st.metric("Optimal readout angle",  f"{opt_theta_d:.0f}°")
    st.caption(
        r"Full $\xi^2(\tau,\theta)$ map for θ ∈ [0°, 180°). Blue = squeezed. "
        "Black star and dashed crosshairs mark the global optimum."
    )

    # ── (c) xi^2 vs θ ───────────────────────────────────────
    st.markdown("---")
    st.subheader(r"(c)  Squeezing parameter $\xi^2(\theta)$ at fixed $\tau$")

    c_col1, c_col2 = st.columns([3, 1])
    with c_col2:
        tau_c_ms = st.number_input(
            "τ (ms) for panel (c)",
            min_value=0.0, max_value=_tau_max,
            value=_clamp_tau(_default_tau),
            step=round(_dt_ms, 2), key="tau_c",
            help="Defaults to globally optimal τ. Edit to explore.",
        )
    step_c       = int(np.argmin(np.abs(times_ms - tau_c_ms)))
    tau_c_actual = float(times_ms[step_c])
    xi2_th_db    = xi2_at_step(xi2_full, step_c)
    valid_c      = np.isfinite(xi2_th_db)
    best_db_c    = float(np.nanmin(xi2_th_db)) if valid_c.any() else float("nan")
    best_th_c    = float(theta_deg[safe_argmin(xi2_th_db)]) if valid_c.any() else 0.0

    with c_col1:
        fig_c, ax_c = plt.subplots(figsize=(7, 4))
        ax_c.plot(theta_deg, xi2_th_db, color="darkorange", lw=2, label="DTWA")
        ax_c.axhline(0.0, color="gray", lw=1, ls="--", label="SQL (0 dB)")
        if valid_c.any():
            ax_c.axvline(best_th_c, color="tomato", lw=1.2, ls=":",
                         label=f"opt θ = {best_th_c:.0f}°  ({best_db_c:.1f} dB)")
        ax_c.set_xlabel(r"Readout angle $\theta$ (deg)", fontsize=11)
        ax_c.set_ylabel(r"$\xi^2$ [dB]", fontsize=11)
        ax_c.set_title(rf"$\xi^2(\theta)$  at  τ = {tau_c_actual:.0f} ms", fontsize=12)
        ax_c.legend(fontsize=9); ax_c.grid(True, alpha=0.3)
        st.pyplot(fig_c); plt.close(fig_c)

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("τ used", f"{tau_c_actual:.0f} ms", f"opt = {opt_tau_ms:.0f} ms")
    with m2: st.metric("Best ξ² at this τ", f"{best_db_c:.2f} dB")
    with m3: st.metric("Optimal θ at this τ", f"{best_th_c:.0f}°")
    st.caption(
        r"$\xi^2(\theta) = 4\,\mathrm{Var}[\hat{S}_\theta]/N$ at the selected time. "
        "Sinusoidal shape reflects the tilted ellipse on the Bloch sphere. "
        "Defaults to globally optimal τ; edit the box to explore."
    )

    # ── (d) xi^2 vs τ ───────────────────────────────────────
    st.subheader(r"(d)  Squeezing parameter $\xi^2(\tau)$ at fixed $\theta$")

    d_col1, d_col2 = st.columns([3, 1])
    with d_col2:
        theta_d_deg = st.number_input(
            "θ (deg) for panel (d)",
            min_value=0.0, max_value=179.9,
            value=_clamp_theta(_default_theta),
            step=round(180.0 / _n_theta, 2), key="theta_d",
            help="Defaults to globally optimal θ. Edit to explore.",
        )
    th_d_idx       = int(np.argmin(np.abs(theta_deg - theta_d_deg)))
    theta_d_actual = float(theta_deg[th_d_idx])
    xi2_t_db       = xi2_at_theta_idx(xi2_full, th_d_idx)
    valid_d        = np.isfinite(xi2_t_db)
    best_db_d      = float(np.nanmin(xi2_t_db))  if valid_d.any() else float("nan")
    best_tau_d     = float(times_ms[safe_argmin(xi2_t_db)]) if valid_d.any() else 0.0

    with d_col1:
        fig_d, ax_d = plt.subplots(figsize=(7, 4))
        ax_d.plot(times_ms, xi2_t_db, color="purple", lw=2, label="DTWA")
        ax_d.axhline(0.0, color="gray", lw=1, ls="--", label="SQL (0 dB)")
        if valid_d.any():
            ax_d.axvline(best_tau_d, color="tomato", lw=1.2, ls=":",
                         label=f"opt τ = {best_tau_d:.0f} ms  ({best_db_d:.1f} dB)")
        ax_d.set_xlabel(r"Evolution time $\tau$ (ms)", fontsize=11)
        ax_d.set_ylabel(r"$\xi^2$ [dB]", fontsize=11)
        ax_d.set_title(rf"$\xi^2(\tau)$  at  θ = {theta_d_actual:.0f}°", fontsize=12)
        ax_d.legend(fontsize=9); ax_d.grid(True, alpha=0.3)
        st.pyplot(fig_d); plt.close(fig_d)

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("θ used", f"{theta_d_actual:.0f}°", f"opt = {opt_theta_d:.0f}°")
    with m2: st.metric("Best ξ² at this θ",  f"{best_db_d:.2f} dB")
    with m3: st.metric("Optimal τ at this θ", f"{best_tau_d:.0f} ms")
    st.caption(
        r"$\xi^2(\tau)$ traces squeezing build-up and degradation with time. "
        "Defaults to globally optimal θ; edit the box to explore."
    )

    # ── (b) Ramsey contrast + Wineland parameter vs time ────
    st.markdown("---")
    st.subheader(r"(b)  Ramsey contrast $C(\tau)$ and Wineland parameter $\xi_R^2(\tau)$")

    fig_b, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(12, 4))

    # Ramsey contrast
    ax_b1.plot(times_ms, contrast, color="steelblue", lw=2, label="DTWA")
    ax_b1.axhline(1.0, color="gray", lw=1, ls="--", label="CSS ($C=1$)")
    ax_b1.axvline(opt_tau_ms, color="k", lw=1, ls="--", alpha=0.6,
                  label=f"opt τ = {opt_tau_ms:.0f} ms")
    ax_b1.set_xlabel(r"Evolution time $\tau$ (ms)", fontsize=11)
    ax_b1.set_ylabel(r"$C = |\langle\vec{J}\rangle| / (N/2)$", fontsize=11)
    ax_b1.set_title("Ramsey contrast decay", fontsize=12)
    ax_b1.set_ylim(0, 1.12)
    ax_b1.legend(fontsize=9); ax_b1.grid(True, alpha=0.3)

    # Wineland parameter vs time at optimal theta
    xi2_opt_t  = xi2_at_theta_idx(xi2_full, opt_th_idx)
    c_sq       = np.where(contrast > 1e-12, contrast**2, np.nan)
    xi2_opt_lin = np.where(xi2_full[:, opt_th_idx] > 0, xi2_full[:, opt_th_idx], np.nan)
    wineland_t  = xi2_opt_lin / c_sq
    wineland_db = 10.0 * np.log10(np.where(wineland_t > 0, wineland_t, np.nan))
    valid_w     = np.isfinite(wineland_db)
    best_w_db   = float(np.nanmin(wineland_db)) if valid_w.any() else float("nan")
    best_w_tau  = float(times_ms[safe_argmin(wineland_db)]) if valid_w.any() else 0.0

    ax_b2.plot(times_ms, wineland_db, color="darkorange", lw=2, label="DTWA")
    ax_b2.axhline(0.0, color="gray", lw=1, ls="--", label="SQL (0 dB)")
    if valid_w.any():
        ax_b2.axvline(best_w_tau, color="tomato", lw=1.2, ls=":",
                      label=f"best = {best_w_db:.1f} dB  (τ = {best_w_tau:.0f} ms)")
    ax_b2.set_xlabel(r"Evolution time $\tau$ (ms)", fontsize=11)
    ax_b2.set_ylabel(r"$\xi_R^2 = \xi^2/C^2$ [dB]", fontsize=11)
    ax_b2.set_title(rf"Wineland parameter at θ = {opt_theta_d:.0f}°", fontsize=12)
    ax_b2.legend(fontsize=9); ax_b2.grid(True, alpha=0.3)

    fig_b.tight_layout()
    st.pyplot(fig_b); plt.close(fig_b)

    bm1, bm2 = st.columns(2)
    with bm1: st.metric("Best Wineland squeezing", f"{best_w_db:.2f} dB")
    with bm2: st.metric("Optimal τ (Wineland)",     f"{best_w_tau:.0f} ms")
    st.caption(
        r"Left: Ramsey contrast $C(\tau)$ decay. "
        r"Right: Wineland parameter $\xi_R^2(\tau) = \xi^2(\tau)/C^2(\tau)$ "
        "at the globally optimal readout angle — the metrologically relevant figure of merit "
        "accounting for contrast loss."
    )

    # ── (e) Lattice correlation diagrams ─────────────────────
    st.markdown("---")
    st.subheader(r"(e)  Site-resolved two-point correlations")

    e_col1, e_col2 = st.columns([3, 1])
    with e_col2:
        tau_e_ms = st.number_input(
            "τ (ms) for panel (e)",
            min_value=0.0, max_value=_tau_max,
            value=_clamp_tau(_default_tau),
            step=round(_dt_ms, 2), key="tau_e",
            help="Defaults to globally optimal τ.",
        )
    step_e       = int(np.argmin(np.abs(times_ms - tau_e_ms)))
    tau_e_actual = float(times_ms[step_e])

    xi2_e_db  = xi2_at_step(xi2_full, step_e)
    valid_e   = np.isfinite(xi2_e_db)
    sq_idx_e  = safe_argmin(xi2_e_db) if valid_e.any() else 0
    asq_idx_e = safe_argmax(xi2_e_db) if valid_e.any() else min(_n_theta//2, _n_theta-1)
    sq_deg_e  = float(theta_deg[sq_idx_e])
    asq_deg_e = float(theta_deg[asq_idx_e])

    spins_e  = spins_history[step_e].astype(float)
    corr_sq,  ref_idx = site_correlations_at_theta(spins_e, pos, thetas[sq_idx_e])
    corr_asq, _       = site_correlations_at_theta(spins_e, pos, thetas[asq_idx_e])

    with e_col1:
        fig_e = draw_lattice_correlations(
            pos, corr_sq, corr_asq, n_grid=params["n"],
            ref_site_idx=ref_idx,
            sq_deg=sq_deg_e, asq_deg=asq_deg_e,
            tau_ms=tau_e_actual, opt_tau_ms=opt_tau_ms,
        )
        st.pyplot(fig_e); plt.close(fig_e)

    sq_finite  = corr_sq [np.isfinite(corr_sq)]
    asq_finite = corr_asq[np.isfinite(corr_asq)]
    sq_sign  = "positive" if len(sq_finite)  and np.mean(sq_finite [sq_finite !=0])>0 else "mixed/negative"
    asq_sign = "positive" if len(asq_finite) and np.mean(asq_finite[asq_finite!=0])>0 else "mixed/negative"

    st.caption(
        rf"Connected correlator $g^{{(2)}}_{{\mathrm{{ref}},j}}$ relative to the central "
        f"reference site (★). "
        "Each panel has its own colorbar scaled to its maximum absolute correlation value. "
        "Unoccupied sites appear as the background color."
    )

    # ── Export ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Export")
    import pandas as pd
    export_df = pd.DataFrame({"time_s": times, "contrast": contrast})
    st.download_button(
        "Download contrast data (CSV)",
        data=export_df.to_csv(index=False).encode(),
        file_name=(f"dipolar_n{params['n']}_fill{filling:.2f}"
                   f"_t{params['t_hop']:.1f}_J{params['J_perp']:.3f}.csv"),
        mime="text/csv",
    )

else:
    st.info("Set the parameters in the sidebar and click **Run simulation** to begin.")
