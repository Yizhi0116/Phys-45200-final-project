import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="2D Spin Squeezing Simulator",
    layout="wide",
)

st.title("2D Lattice Spin Squeezing Simulator")

st.markdown(
    r"""
This app compares **one-axis twisting (OAT)** and **generalized two-axis twisting (TAT)**
on an $n\times n$ spin-1/2 lattice using a stochastic DTWA / Monte Carlo approximation.

The total number of spins is
"""
)
st.latex(r"N=n^2")
st.markdown(
    r"""
Instead of evolving the full $2^N$-dimensional quantum state, the app samples many stochastic
spin trajectories and averages collective observables. This makes larger 2D lattices possible,
although it is an approximate semiclassical method rather than exact wave-function evolution.
"""
)


# ============================================================
# Lattice and coupling construction
# ============================================================

def lattice_positions_2d(n, spacing=1.0):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pos = np.column_stack([xs.ravel(), ys.ravel()]) * spacing
    return pos


def build_coupling_matrix(
    n,
    geometry="finite_range",
    coupling_range=2.0,
    alpha=3.0,
    normalize=True,
):
    """
    Build an N x N coupling matrix for an n x n square lattice.

    geometry options:
        all_to_all   : every spin couples to every other spin.
        nearest      : only nearest neighbors.
        finite_range : spins within distance coupling_range.
        power_law    : J_ij ~ 1 / r_ij^alpha, optionally cutoff by coupling_range.

    The matrix is symmetric with J_ii = 0.
    """
    pos = lattice_positions_2d(n)
    N = n * n
    J = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(i + 1, N):
            r = np.linalg.norm(pos[i] - pos[j])
            val = 0.0

            if geometry == "all_to_all":
                val = 1.0

            elif geometry == "nearest":
                if np.isclose(r, 1.0):
                    val = 1.0

            elif geometry == "finite_range":
                if r <= coupling_range:
                    val = 1.0

            elif geometry == "power_law":
                if r <= coupling_range:
                    val = 1.0 / (r ** alpha)

            else:
                raise ValueError("Unknown geometry.")

            J[i, j] = val
            J[j, i] = val

    if normalize:
        row_sums = np.sum(np.abs(J), axis=1)
        nonzero = row_sums[row_sums > 0]
        if len(nonzero) > 0:
            mean_row_sum = np.mean(nonzero)
            J = J / mean_row_sum

    return J


# ============================================================
# Initial state sampling
# ============================================================

def sample_initial_spins(num_samples, N, initial_axis="x", seed=None):
    """
    DTWA sampling for spin coherent states.

    For initial +x:
        Sx = 1/2 fixed
        Sy = +/- 1/2 sampled
        Sz = +/- 1/2 sampled

    For initial +y and +z, rotate the fixed component accordingly.
    """
    rng = np.random.default_rng(seed)
    spins = np.zeros((num_samples, N, 3), dtype=float)

    signs1 = rng.choice([-0.5, 0.5], size=(num_samples, N))
    signs2 = rng.choice([-0.5, 0.5], size=(num_samples, N))

    if initial_axis == "x":
        spins[:, :, 0] = 0.5
        spins[:, :, 1] = signs1
        spins[:, :, 2] = signs2

    elif initial_axis == "y":
        spins[:, :, 0] = signs1
        spins[:, :, 1] = 0.5
        spins[:, :, 2] = signs2

    elif initial_axis == "z":
        spins[:, :, 0] = signs1
        spins[:, :, 1] = signs2
        spins[:, :, 2] = 0.5

    else:
        raise ValueError("initial_axis must be x, y, or z.")

    return spins


# ============================================================
# Hamiltonian construction
# ============================================================

def make_hamiltonian_couplings(
    model,
    n,
    chi=1.0,
    geometry="finite_range",
    coupling_range=2.0,
    alpha=3.0,
    tat_jz_over_jy=1.0,
):
    """
    Return Jx, Jy, Jz coupling matrices.

    The classical DTWA equations are based on

        H = sum_{i<j} [
              Jx_ij Sx_i Sx_j
            + Jy_ij Sy_i Sy_j
            + Jz_ij Sz_i Sz_j
        ].

    OAT-like 2D lattice Hamiltonian:

        H_OAT = chi sum_{i<j} J_ij S_i^z S_j^z.

    Generalized TAT-like 2D lattice Hamiltonian:

        H_TAT = chi sum_{i<j} J_ij [r S_i^z S_j^z - S_i^y S_j^y],

    where r = |Jz/Jy|.

    Because the local field uses a full matrix multiplication over both i,j,
    the factor of 2 below compensates for the i<j convention.
    """
    base = build_coupling_matrix(
        n=n,
        geometry=geometry,
        coupling_range=coupling_range,
        alpha=alpha,
        normalize=True,
    )

    N = n * n
    Jx = np.zeros((N, N))
    Jy = np.zeros((N, N))
    Jz = np.zeros((N, N))

    if model == "OAT":
        Jz = 2.0 * chi * base

    elif model == "TAT":
        Jy = -2.0 * chi * base
        Jz = 2.0 * chi * tat_jz_over_jy * base

    else:
        raise ValueError("model must be OAT or TAT.")

    return Jx, Jy, Jz, base


# ============================================================
# Equations of motion and decoherence model
# ============================================================

def spin_derivative(spins, Jx, Jy, Jz):
    """
    Classical spin-precession equation:

        dS_i/dt = B_i x S_i,

    where B_i is the interaction-generated local effective field.
    """
    Sx = spins[:, :, 0]
    Sy = spins[:, :, 1]
    Sz = spins[:, :, 2]

    Bx = Sx @ Jx.T
    By = Sy @ Jy.T
    Bz = Sz @ Jz.T

    dS = np.empty_like(spins)

    # dS/dt = B x S
    dS[:, :, 0] = By * Sz - Bz * Sy
    dS[:, :, 1] = Bz * Sx - Bx * Sz
    dS[:, :, 2] = Bx * Sy - By * Sx

    return dS


def apply_phenomenological_decoherence(spins, dt, gamma_phi=0.0, gamma_relax=0.0):
    """
    Simple phenomenological decoherence model.

    gamma_phi:
        Dephasing around z. It damps local Sx and Sy while preserving Sz.

    gamma_relax:
        Crude spin-length relaxation / contrast decay. It damps all spin components.

    This is not a full Lindblad master-equation treatment. It is a practical DTWA-level
    way to show how decoherence reduces contrast and worsens squeezing.
    """
    if gamma_phi > 0.0:
        transverse_decay = np.exp(-gamma_phi * dt)
        spins[:, :, 0] *= transverse_decay
        spins[:, :, 1] *= transverse_decay

    if gamma_relax > 0.0:
        total_decay = np.exp(-gamma_relax * dt)
        spins *= total_decay

    return spins


def rk4_step(spins, dt, Jx, Jy, Jz, gamma_phi=0.0, gamma_relax=0.0):
    k1 = spin_derivative(spins, Jx, Jy, Jz)
    k2 = spin_derivative(spins + 0.5 * dt * k1, Jx, Jy, Jz)
    k3 = spin_derivative(spins + 0.5 * dt * k2, Jx, Jy, Jz)
    k4 = spin_derivative(spins + dt * k3, Jx, Jy, Jz)

    updated = spins + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    updated = apply_phenomenological_decoherence(
        updated,
        dt,
        gamma_phi=gamma_phi,
        gamma_relax=gamma_relax,
    )

    return updated


# ============================================================
# Squeezing and variance calculations
# ============================================================

def perpendicular_basis(mean_J):
    norm = np.linalg.norm(mean_J)

    if norm < 1e-12:
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])

    e0 = mean_J / norm

    trial = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(e0, trial)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])

    e1 = trial - np.dot(trial, e0) * e0
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(e0, e1)
    e2 = e2 / np.linalg.norm(e2)

    return e1, e2


def compute_observables(spins):
    """
    Compute collective spin, transverse variances, and Wineland squeezing.
    """
    num_samples, N, _ = spins.shape

    J_samples = np.sum(spins, axis=1)

    mean_J = np.mean(J_samples, axis=0)
    mean_norm = np.linalg.norm(mean_J)

    e1, e2 = perpendicular_basis(mean_J)

    J1 = J_samples @ e1
    J2 = J_samples @ e2

    cov_perp = np.cov(np.vstack([J1, J2]), bias=False)
    eigvals = np.linalg.eigvalsh(cov_perp)

    min_var = np.min(eigvals)
    max_var = np.max(eigvals)

    var_Jx = np.var(J_samples[:, 0], ddof=1)
    var_Jy = np.var(J_samples[:, 1], ddof=1)
    var_Jz = np.var(J_samples[:, 2], ddof=1)

    if mean_norm < 1e-12 or min_var <= 0.0:
        xi2 = np.nan
        xi_db = np.nan
    else:
        xi2 = N * min_var / (mean_norm ** 2)
        xi_db = 10.0 * np.log10(xi2)

    return {
        "xi2": xi2,
        "xi_db": xi_db,
        "mean_Jx": mean_J[0],
        "mean_Jy": mean_J[1],
        "mean_Jz": mean_J[2],
        "mean_J_norm": mean_norm,
        "mean_Jx_norm": mean_J[0] / (N / 2),
        "mean_Jy_norm": mean_J[1] / (N / 2),
        "mean_Jz_norm": mean_J[2] / (N / 2),
        "mean_J_norm_scaled": mean_norm / (N / 2),
        "min_var": min_var,
        "max_var": max_var,
        "var_Jx": var_Jx,
        "var_Jy": var_Jy,
        "var_Jz": var_Jz,
    }


# ============================================================
# Cached simulation
# ============================================================

@st.cache_data(show_spinner=False)
def simulate_squeezing_cached(
    n,
    model,
    chi,
    geometry,
    coupling_range,
    alpha,
    tat_jz_over_jy,
    num_samples,
    t_max,
    num_steps,
    seed,
    gamma_phi,
    gamma_relax,
):
    N = n * n
    initial_axis = "x"

    Jx, Jy, Jz, base = make_hamiltonian_couplings(
        model=model,
        n=n,
        chi=chi,
        geometry=geometry,
        coupling_range=coupling_range,
        alpha=alpha,
        tat_jz_over_jy=tat_jz_over_jy,
    )

    spins = sample_initial_spins(
        num_samples=num_samples,
        N=N,
        initial_axis=initial_axis,
        seed=seed,
    )

    times = np.linspace(0.0, t_max, num_steps + 1)
    dt = times[1] - times[0]

    rows = []

    for step, t in enumerate(times):
        obs = compute_observables(spins)
        obs["time"] = t
        obs["model"] = model
        obs["n"] = n
        obs["N"] = N
        obs["gamma_phi"] = gamma_phi
        obs["gamma_relax"] = gamma_relax
        rows.append(obs)

        if step < num_steps:
            spins = rk4_step(
                spins,
                dt,
                Jx,
                Jy,
                Jz,
                gamma_phi=gamma_phi,
                gamma_relax=gamma_relax,
            )

    df = pd.DataFrame(rows)

    if df["xi_db"].notna().any():
        best_idx = df["xi_db"].idxmin()
        best = df.loc[best_idx].to_dict()
    else:
        best = df.iloc[0].to_dict()

    return df, best, base


# ============================================================
# Plot helpers
# ============================================================

def plot_squeezing_db(combined, title="OAT vs TAT squeezing"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, df in combined.groupby("model"):
        ax.plot(df["time"], df["xi_db"], label=model_name)
    ax.axhline(0.0, linestyle="--", linewidth=1, label="SQL")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$10\log_{10}(\xi_R^2)$ [dB]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_squeezing_linear(combined):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, df in combined.groupby("model"):
        ax.plot(df["time"], df["xi2"], label=model_name)
    ax.axhline(1.0, linestyle="--", linewidth=1, label="SQL")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$\xi_R^2$")
    ax.set_title(r"Linear squeezing parameter $\xi_R^2$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_transverse_variances(combined):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, df in combined.groupby("model"):
        ax.plot(df["time"], df["min_var"], label=f"{model_name}: min transverse")
        ax.plot(df["time"], df["max_var"], linestyle="--", label=f"{model_name}: max transverse")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Collective spin variance")
    ax.set_title("Minimum and maximum transverse variances")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_raw_variances(df, selected_model):
    raw_df = df[df["model"] == selected_model]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(raw_df["time"], raw_df["var_Jx"], label=r"Var($J_x$)")
    ax.plot(raw_df["time"], raw_df["var_Jy"], label=r"Var($J_y$)")
    ax.plot(raw_df["time"], raw_df["var_Jz"], label=r"Var($J_z$)")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Variance")
    ax.set_title(f"Raw collective spin variances: {selected_model}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_mean_spin_components(combined):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, df in combined.groupby("model"):
        ax.plot(df["time"], df["mean_J_norm_scaled"], label=f"{model_name}: |<J>|/(N/2)")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Normalized mean spin length")
    ax.set_title("Mean spin contrast")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bloch_trajectory(combined, selected_model):
    df = combined[combined["model"] == selected_model]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Unit Bloch sphere wireframe
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.3, alpha=0.25)

    xs = df["mean_Jx_norm"].to_numpy()
    ys = df["mean_Jy_norm"].to_numpy()
    zs = df["mean_Jz_norm"].to_numpy()

    ax.plot(xs, ys, zs, linewidth=2, label=f"{selected_model} trajectory")
    ax.scatter(xs[0], ys[0], zs[0], s=50, label="start")
    ax.scatter(xs[-1], ys[-1], zs[-1], s=50, label="end")

    ax.set_xlabel(r"$\langle J_x\rangle/(N/2)$")
    ax.set_ylabel(r"$\langle J_y\rangle/(N/2)$")
    ax.set_zlabel(r"$\langle J_z\rangle/(N/2)$")
    ax.set_title(f"Mean-spin Bloch-sphere trajectory: {selected_model}")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.legend()
    fig.tight_layout()
    return fig


def plot_coupling_matrix(base):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(base, aspect="auto")
    ax.set_title("Normalized coupling matrix $J_{ij}$")
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    fig.colorbar(im, ax=ax, label=r"$J_{ij}$")
    fig.tight_layout()
    return fig


# ============================================================
# Sidebar controls
# ============================================================

st.sidebar.header("Simulation Inputs")

n = st.sidebar.slider(
    "Lattice size n",
    min_value=2,
    max_value=30,
    value=8,
    step=1,
)

N = n * n
st.sidebar.write(f"Total spins: **N = {N}**")

geometry = st.sidebar.selectbox(
    "Coupling geometry",
    ["finite_range", "nearest", "power_law", "all_to_all"],
    index=0,
)

coupling_range = st.sidebar.slider(
    "Coupling range Rc",
    min_value=1.0,
    max_value=float(max(2, n)),
    value=min(3.0, float(max(2, n))),
    step=0.5,
    disabled=(geometry in ["nearest", "all_to_all"]),
)

alpha = st.sidebar.slider(
    "Power-law exponent alpha",
    min_value=0.5,
    max_value=6.0,
    value=3.0,
    step=0.5,
    disabled=(geometry != "power_law"),
)

chi = st.sidebar.number_input(
    "Twisting strength chi",
    min_value=0.001,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

tat_jz_over_jy = st.sidebar.slider(
    "TAT magnitude ratio |Jz/Jy|",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
)

st.sidebar.divider()
st.sidebar.subheader("Time Window")

short_t_max = st.sidebar.slider(
    "Short-time plot range",
    min_value=0.5,
    max_value=20.0,
    value=5.0,
    step=0.5,
)

long_t_max = st.sidebar.slider(
    "Long-time simulation range",
    min_value=short_t_max,
    max_value=100.0,
    value=max(20.0, short_t_max),
    step=1.0,
)

num_steps = st.sidebar.slider(
    "Number of time steps",
    min_value=50,
    max_value=3000,
    value=600,
    step=50,
)

num_samples = st.sidebar.slider(
    "Monte Carlo samples",
    min_value=100,
    max_value=20000,
    value=1000,
    step=100,
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=999999,
    value=1234,
    step=1,
)

st.sidebar.divider()
st.sidebar.subheader("Decoherence")

enable_decoherence = st.sidebar.checkbox("Enable phenomenological decoherence", value=False)

gamma_phi = st.sidebar.number_input(
    "Dephasing rate gamma_phi",
    min_value=0.0,
    max_value=5.0,
    value=0.05 if enable_decoherence else 0.0,
    step=0.01,
    disabled=not enable_decoherence,
)

gamma_relax = st.sidebar.number_input(
    "Relaxation / contrast decay gamma_relax",
    min_value=0.0,
    max_value=5.0,
    value=0.00,
    step=0.01,
    disabled=not enable_decoherence,
)

if not enable_decoherence:
    gamma_phi = 0.0
    gamma_relax = 0.0

run_button = st.sidebar.button("Run simulation", type="primary")


# ============================================================
# Model description
# ============================================================

st.header("Current model")

st.markdown("OAT-like local 2D twisting:")
st.latex(
    r"""
    H_{\mathrm{OAT}}
    =
    \chi
    \sum_{i<j}
    J_{ij}
    S_i^z S_j^z
    """
)

st.markdown("Generalized TAT-like local 2D twisting:")
st.latex(
    r"""
    H_{\mathrm{TAT}}
    =
    \chi
    \sum_{i<j}
    J_{ij}
    \left(
    r S_i^z S_j^z
    -
    S_i^y S_j^y
    \right)
    """
)

st.markdown("where")
st.latex(r"r=\left|J_z/J_y\right|")
st.markdown("For symmetric TAT in the $y$-$z$ plane, use $r=1$.")

if enable_decoherence:
    st.markdown("Phenomenological decoherence is included after each RK4 step:")
    st.latex(r"S_x,S_y\rightarrow e^{-\gamma_\phi\Delta t}S_x,S_y")
    st.latex(r"\mathbf S\rightarrow e^{-\gamma_{\mathrm{relax}}\Delta t}\mathbf S")
else:
    st.info("Decoherence is currently disabled.")


# ============================================================
# Simulation warning
# ============================================================

estimated_cost = num_samples * (N ** 2) * num_steps

if estimated_cost > 5e9:
    st.warning(
        "This setting may be slow. Try reducing Monte Carlo samples, time steps, "
        "or lattice size for testing."
    )


# ============================================================
# Run simulation and tabs
# ============================================================

if run_button:
    start_time = time.time()

    with st.spinner("Running OAT simulation..."):
        oat_df, oat_best, base_oat = simulate_squeezing_cached(
            n=n,
            model="OAT",
            chi=chi,
            geometry=geometry,
            coupling_range=coupling_range,
            alpha=alpha,
            tat_jz_over_jy=tat_jz_over_jy,
            num_samples=num_samples,
            t_max=long_t_max,
            num_steps=num_steps,
            seed=seed,
            gamma_phi=gamma_phi,
            gamma_relax=gamma_relax,
        )

    with st.spinner("Running TAT simulation..."):
        tat_df, tat_best, base_tat = simulate_squeezing_cached(
            n=n,
            model="TAT",
            chi=chi,
            geometry=geometry,
            coupling_range=coupling_range,
            alpha=alpha,
            tat_jz_over_jy=tat_jz_over_jy,
            num_samples=num_samples,
            t_max=long_t_max,
            num_steps=num_steps,
            seed=seed + 1,
            gamma_phi=gamma_phi,
            gamma_relax=gamma_relax,
        )

    elapsed = time.time() - start_time
    combined = pd.concat([oat_df, tat_df], ignore_index=True)
    short_combined = combined[combined["time"] <= short_t_max].copy()

    st.success(f"Simulation finished in {elapsed:.2f} seconds.")

    tab_summary, tab_short, tab_long, tab_variance, tab_bloch, tab_decoh, tab_coupling, tab_export = st.tabs(
        [
            "Summary",
            "Short-time squeezing",
            "Long-time squeezing",
            "Variances",
            "Bloch dynamics",
            "Decoherence",
            "Coupling matrix",
            "Export",
        ]
    )

    # ========================================================
    # Summary tab
    # ========================================================
    with tab_summary:
        st.header("Best Squeezing Summary")

        summary_df = pd.DataFrame(
            [
                {
                    "Model": "OAT",
                    "Best squeezing dB": oat_best["xi_db"],
                    "Best xi^2": oat_best["xi2"],
                    "Optimal time": oat_best["time"],
                    "Min transverse variance": oat_best["min_var"],
                    "Mean spin norm": oat_best["mean_J_norm"],
                    "Normalized mean spin": oat_best["mean_J_norm_scaled"],
                },
                {
                    "Model": "TAT",
                    "Best squeezing dB": tat_best["xi_db"],
                    "Best xi^2": tat_best["xi2"],
                    "Optimal time": tat_best["time"],
                    "Min transverse variance": tat_best["min_var"],
                    "Mean spin norm": tat_best["mean_J_norm"],
                    "Normalized mean spin": tat_best["mean_J_norm_scaled"],
                },
            ]
        )

        st.dataframe(summary_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "OAT best squeezing",
                f"{oat_best['xi_db']:.3f} dB",
                f"t = {oat_best['time']:.3f}",
            )
        with col2:
            st.metric(
                "TAT best squeezing",
                f"{tat_best['xi_db']:.3f} dB",
                f"t = {tat_best['time']:.3f}",
            )
        with col3:
            improvement = oat_best["xi_db"] - tat_best["xi_db"]
            st.metric(
                "TAT improvement over OAT",
                f"{improvement:.3f} dB",
                "positive means TAT is better",
            )

        st.subheader("Simulation settings")
        settings_df = pd.DataFrame(
            [
                {
                    "n": n,
                    "N": N,
                    "geometry": geometry,
                    "coupling_range": coupling_range,
                    "alpha": alpha,
                    "chi": chi,
                    "TAT |Jz/Jy|": tat_jz_over_jy,
                    "long_t_max": long_t_max,
                    "num_steps": num_steps,
                    "num_samples": num_samples,
                    "gamma_phi": gamma_phi,
                    "gamma_relax": gamma_relax,
                }
            ]
        )
        st.dataframe(settings_df, use_container_width=True)

    # ========================================================
    # Short-time squeezing tab
    # ========================================================
    with tab_short:
        st.header("Short-time squeezing")
        st.pyplot(plot_squeezing_db(short_combined, title="Short-time OAT vs TAT squeezing"))
        st.pyplot(plot_squeezing_linear(short_combined))

    # ========================================================
    # Long-time squeezing tab
    # ========================================================
    with tab_long:
        st.header("Long-time squeezing")
        st.pyplot(plot_squeezing_db(combined, title="Long-time OAT vs TAT squeezing"))
        st.pyplot(plot_mean_spin_components(combined))

        st.markdown(
            r"""
Long-time dynamics are useful for seeing **over-twisting**, revivals, loss of mean spin contrast,
and the breakdown of useful squeezing after the optimal time.
"""
        )

    # ========================================================
    # Variances tab
    # ========================================================
    with tab_variance:
        st.header("Variance plots")
        st.pyplot(plot_transverse_variances(combined))

        selected_model_var = st.selectbox(
            "Choose model for raw variance plot",
            ["OAT", "TAT"],
            key="variance_model_selector",
        )
        st.pyplot(plot_raw_variances(combined, selected_model_var))

        st.markdown(
            r"""
The minimum transverse variance is the squeezed quadrature. The maximum transverse variance is the
anti-squeezed quadrature. The raw variances $\mathrm{Var}(J_x)$, $\mathrm{Var}(J_y)$, and
$\mathrm{Var}(J_z)$ are useful diagnostics, but the optimal squeezed direction is generally a
rotated direction perpendicular to $\langle \mathbf J\rangle$.
"""
        )

    # ========================================================
    # Bloch dynamics tab
    # ========================================================
    with tab_bloch:
        st.header("Bloch-sphere mean-spin dynamics")

        selected_model_bloch = st.selectbox(
            "Choose model for Bloch trajectory",
            ["OAT", "TAT"],
            key="bloch_model_selector",
        )
        st.pyplot(plot_bloch_trajectory(combined, selected_model_bloch))

        st.markdown(
            r"""
This plot shows the normalized collective mean-spin vector
$\langle\mathbf J\rangle/(N/2)$ on a Bloch sphere. It is not plotting every individual spin;
it is plotting the ensemble-averaged collective spin trajectory.
"""
        )

    # ========================================================
    # Decoherence tab
    # ========================================================
    with tab_decoh:
        st.header("Decoherence diagnostics")

        if enable_decoherence:
            st.warning(
                "Decoherence is implemented phenomenologically inside the DTWA trajectories. "
                "It is useful for qualitative comparison, but it is not a full Lindblad master-equation simulation."
            )
        else:
            st.info(
                "Decoherence is disabled. Turn it on in the sidebar to see contrast loss and squeezing degradation."
            )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.metric("gamma_phi", f"{gamma_phi:.4f}")
        with col_d2:
            st.metric("gamma_relax", f"{gamma_relax:.4f}")

        st.pyplot(plot_mean_spin_components(combined))
        st.pyplot(plot_squeezing_db(combined, title="Squeezing with selected decoherence settings"))

        st.markdown(
            r"""
Dephasing damps transverse spin components and therefore reduces the mean spin length in the
Wineland denominator. Since
"""
        )
        st.latex(r"\xi_R^2=\frac{N(\Delta J_\perp)^2_{\min}}{|\langle\mathbf J\rangle|^2}")
        st.markdown(
            r"""
a shorter mean-spin vector usually makes the squeezing parameter worse, even if some transverse
variance remains small.
"""
        )

    # ========================================================
    # Coupling matrix tab
    # ========================================================
    with tab_coupling:
        st.header("Coupling matrix")
        st.pyplot(plot_coupling_matrix(base_oat))

        st.markdown(
            r"""
The same normalized base matrix $J_{ij}$ is used for OAT and TAT. OAT applies it only in the
$z$-$z$ channel. TAT applies it with opposite signs in the $z$-$z$ and $y$-$y$ channels.
"""
        )

    # ========================================================
    # Export tab
    # ========================================================
    with tab_export:
        st.header("Export data")

        st.dataframe(combined.head(20), use_container_width=True)

        csv = combined.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download simulation data as CSV",
            data=csv,
            file_name=f"spin_squeezing_n{n}_{geometry}.csv",
            mime="text/csv",
        )

else:
    st.info("Set the parameters in the sidebar, then click **Run simulation**.")
